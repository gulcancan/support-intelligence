"""
Label correction via LLM distillation (async concurrent).

Uses a strong foundation model running locally to re-label subcategory,
priority, and sentiment on synthetic tickets. Sends multiple requests
concurrently to saturate vLLM's continuous batching for maximum throughput.

Sequential:  ~1 ticket/sec  (GPU idle between requests)
Concurrent:  ~10-30 tickets/sec  (GPU always busy with batched inference)

Supports two backends:
  1. vLLM (recommended): OpenAI-compatible API with continuous batching
  2. Ollama: simpler setup, lower throughput

Usage:
  # Run with 32 concurrent requests (default)
  python scripts/correct_labels.py --data data/raw/tickets.json

  # Adjust concurrency (higher = faster, until GPU saturates)
  python scripts/correct_labels.py --data data/raw/tickets.json --concurrency 64

  # Spot-check corrections
  python scripts/correct_labels.py --spot-check 200 --output data/raw/tickets_corrected.json

  # Use a smaller/faster model
  python scripts/correct_labels.py --model Qwen/Qwen2.5-32B-Instruct-AWQ --concurrency 64
"""
import json
import argparse
import asyncio
import time
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Label definitions ────────────────────────────────────────────────────────

CATEGORIES_AND_SUBCATEGORIES = {
    "Account Management": [
        "Access Control", "Billing", "License", "Subscription", "Upgrade"
    ],
    "Compliance / Security": [
        "Audit Trail", "Authentication", "Compliance", "Data Privacy", "Encryption"
    ],
    "Data Issue": [
        "Corruption", "Data Loss", "Import/Export", "Sync Error", "Validation"
    ],
    "Feature Request": [
        "API Extension", "Integration", "New Feature", "UI Enhancement"
    ],
    "How-To / Guidance": [
        "Best Practice", "Configuration", "Documentation", "Setup Help"
    ],
    "Outage / Downtime": [
        "Degraded Performance", "Full Outage", "Intermittent", "Regional"
    ],
    "Technical Issue": [
        "Crash/Bug", "Error Handling", "Performance", "Scalability"
    ],
}

PRIORITIES = ["critical", "high", "medium", "low"]
SENTIMENTS = ["frustrated", "angry", "neutral", "satisfied", "confused", "grateful"]


def build_prompt(ticket: dict) -> str:
    """Build correction prompt with all available ticket context."""
    category = ticket.get("category", "UNKNOWN")
    valid_subcats = CATEGORIES_AND_SUBCATEGORIES.get(category, [])

    fields = []
    fields.append(f"Subject: {ticket.get('subject', 'N/A')}")
    fields.append(f"Description: {ticket.get('description', 'N/A')}")
    fields.append(f"Category: {category}")
    fields.append(f"Product: {ticket.get('product', 'N/A')}")
    fields.append(f"Product module: {ticket.get('product_module', 'N/A')}")
    fields.append(f"Product version: {ticket.get('product_version', 'N/A')}")
    fields.append(f"Customer tier: {ticket.get('customer_tier', 'N/A')}")
    fields.append(f"Channel: {ticket.get('channel', 'N/A')}")
    fields.append(f"Environment: {ticket.get('environment', 'N/A')}")
    fields.append(f"Severity: {ticket.get('severity', 'N/A')}")
    fields.append(f"Affected users: {ticket.get('affected_users', 'N/A')}")
    fields.append(f"Business impact: {ticket.get('business_impact', 'N/A')}")

    if ticket.get("error_logs"):
        fields.append(f"Error logs: {str(ticket['error_logs'])[:300]}")
    if ticket.get("stack_trace"):
        fields.append(f"Stack trace (first 200 chars): {str(ticket['stack_trace'])[:200]}")
    if ticket.get("resolution"):
        fields.append(f"Resolution: {str(ticket['resolution'])[:300]}")
    if ticket.get("resolution_code"):
        fields.append(f"Resolution code: {ticket['resolution_code']}")
    if ticket.get("feedback_text"):
        fields.append(f"Customer feedback: {str(ticket['feedback_text'])[:200]}")
    if ticket.get("escalated"):
        fields.append(f"Escalated: {ticket['escalated']}")
        if ticket.get("escalation_reason"):
            fields.append(f"Escalation reason: {ticket['escalation_reason']}")
    if ticket.get("tags"):
        tags = ticket["tags"] if isinstance(ticket["tags"], list) else []
        if tags:
            fields.append(f"Tags: {', '.join(str(t) for t in tags[:10])}")

    ticket_text = "\n".join(fields)

    prompt = f"""You are a senior support operations analyst reviewing ticket labels for quality.

Given the support ticket below, determine the correct values for three fields.
The category "{category}" is already correct — do NOT change it.

TICKET:
{ticket_text}

TASK: Assign the correct labels based on the ticket content.

1. SUBCATEGORY — must be exactly one of: {', '.join(valid_subcats)}
   Choose based on what the ticket is actually about, not the current label.

2. PRIORITY — must be exactly one of: {', '.join(PRIORITIES)}
   Consider: affected users, business impact, severity, urgency language in the text,
   whether it's escalated, and the environment (production > staging > dev).

3. SENTIMENT — must be exactly one of: {', '.join(SENTIMENTS)}
   Infer from the description text tone, feedback text, and overall language.
   "frustrated" = repeated failed attempts, "angry" = accusatory/demanding,
   "neutral" = factual/business-like, "satisfied" = positive acknowledgment,
   "confused" = uncertainty/questions, "grateful" = explicit thanks.

Respond with ONLY a JSON object, no other text:
{{"subcategory": "...", "priority": "...", "sentiment": "..."}}"""

    return prompt


def parse_response(raw: str, category: str) -> Optional[Dict[str, str]]:
    """Parse LLM JSON response, validate labels against allowed values."""
    if not raw:
        return None

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    valid_subcats = CATEGORIES_AND_SUBCATEGORIES.get(category, [])
    parsed = {}

    subcat = result.get("subcategory", "")
    if subcat in valid_subcats:
        parsed["subcategory"] = subcat
    else:
        for vs in valid_subcats:
            if vs.lower() == subcat.lower():
                parsed["subcategory"] = vs
                break

    priority = result.get("priority", "").lower()
    if priority in PRIORITIES:
        parsed["priority"] = priority

    sentiment = result.get("sentiment", "").lower()
    if sentiment in SENTIMENTS:
        parsed["sentiment"] = sentiment

    return parsed if parsed else None


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Dict]:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, corrections: Dict[str, Dict]):
    with open(checkpoint_path, "w") as f:
        json.dump(corrections, f)


# ── Async API callers ────────────────────────────────────────────────────────

async def call_vllm_async(
    session: aiohttp.ClientSession,
    prompt: str,
    base_url: str,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Async vLLM call with concurrency control."""
    async with semaphore:
        try:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 100,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return None


async def call_ollama_async(
    session: aiohttp.ClientSession,
    prompt: str,
    base_url: str,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Async Ollama call with concurrency control."""
    async with semaphore:
        try:
            async with session.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": 100},
                },
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data["response"].strip()
        except Exception as e:
            return None


async def process_ticket(
    session: aiohttp.ClientSession,
    ticket: dict,
    call_fn,
    base_url: str,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> tuple:
    """Process a single ticket: build prompt → call LLM → parse response."""
    tid = ticket.get("ticket_id", "unknown")
    prompt = build_prompt(ticket)
    raw = await call_fn(session, prompt, base_url, model, temperature, semaphore)

    if raw is None:
        return tid, {}, "api_fail"

    parsed = parse_response(raw, ticket.get("category", ""))
    if parsed:
        return tid, parsed, "success"
    else:
        return tid, {}, "parse_fail"


async def process_batch(
    tickets: List[dict],
    call_fn,
    base_url: str,
    model: str,
    temperature: float,
    concurrency: int,
    corrections: Dict[str, Dict],
    checkpoint_path: Path,
    checkpoint_every: int,
):
    """Process all tickets with async concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    stats = {"success": 0, "parse_fail": 0, "api_fail": 0}
    t0 = time.time()
    processed = 0

    connector = aiohttp.TCPConnector(limit=concurrency + 10, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in chunks for checkpointing
        for chunk_start in range(0, len(tickets), checkpoint_every):
            chunk = tickets[chunk_start:chunk_start + checkpoint_every]

            tasks = [
                process_ticket(session, t, call_fn, base_url, model, temperature, semaphore)
                for t in chunk
            ]
            results = await asyncio.gather(*tasks)

            for tid, parsed, status in results:
                corrections[tid] = parsed
                stats[status] += 1
                processed += 1

            # Checkpoint after each chunk
            save_checkpoint(checkpoint_path, corrections)

            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(tickets) - processed) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {processed}/{len(tickets)} ({rate:.1f} tickets/sec, "
                f"ETA: {eta/60:.0f} min) | "
                f"success={stats['success']}, parse_fail={stats['parse_fail']}, "
                f"api_fail={stats['api_fail']}"
            )

    return stats


def spot_check(original: List[dict], corrections: Dict[str, Dict], n: int = 200):
    """Print a sample of corrections for manual review."""
    import random
    corrected_ids = [tid for tid in corrections if corrections[tid]]
    sample = random.sample(corrected_ids, min(n, len(corrected_ids)))

    changed = {"subcategory": 0, "priority": 0, "sentiment": 0}
    total = 0

    for tid in sample:
        ticket = next((t for t in original if t.get("ticket_id") == tid), None)
        if not ticket:
            continue
        corr = corrections[tid]
        total += 1

        diffs = []
        for field in ["subcategory", "priority", "sentiment"]:
            col = "customer_sentiment" if field == "sentiment" else field
            old_val = ticket.get(col, "N/A")
            new_val = corr.get(field)
            if new_val and new_val != old_val:
                diffs.append(f"  {field}: {old_val} → {new_val}")
                changed[field] += 1

        if diffs:
            print(f"\n{'='*70}")
            print(f"[{ticket.get('category')}] {ticket.get('subject', '')[:70]}")
            print(f"  Desc: {ticket.get('description', '')[:120]}")
            if ticket.get("error_logs"):
                print(f"  Error: {str(ticket['error_logs'])[:80]}")
            print(f"  Affected users: {ticket.get('affected_users')}, "
                  f"Tier: {ticket.get('customer_tier')}, "
                  f"Escalated: {ticket.get('escalated')}")
            for d in diffs:
                print(d)

    print(f"\n{'='*70}")
    print(f"Spot check: {total} tickets sampled")
    for field, count in changed.items():
        print(f"  {field}: {count}/{total} changed ({count/total*100:.0f}%)")


def main():
    p = argparse.ArgumentParser(description="Correct ticket labels using a local LLM (async)")
    p.add_argument("--data", default="data/raw/tickets.json", help="Input ticket JSON")
    p.add_argument("--output", default="data/raw/tickets_corrected.json", help="Output JSON")
    p.add_argument("--backend", choices=["vllm", "ollama"], default="vllm")
    p.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct", help="Model name")
    p.add_argument("--base-url", default=None, help="API base URL (auto-detected from backend)")
    p.add_argument("--concurrency", type=int, default=32,
                   help="Number of concurrent requests (default 32). "
                        "Higher = faster until GPU saturates. Try 16-64.")
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="Save checkpoint every N tickets (default 200)")
    p.add_argument("--max-tickets", type=int, default=None, help="Process only first N tickets")
    p.add_argument("--spot-check", type=int, default=0, help="Spot-check N corrected tickets and exit")
    p.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    args = p.parse_args()

    if args.base_url is None:
        args.base_url = "http://localhost:8001" if args.backend == "vllm" else "http://localhost:11434"

    # Load tickets
    logger.info(f"Loading {args.data}")
    with open(args.data) as f:
        tickets = json.load(f)
    logger.info(f"Loaded {len(tickets):,} tickets")

    if args.max_tickets:
        tickets = tickets[:args.max_tickets]
        logger.info(f"Processing first {len(tickets):,} tickets")

    # Load checkpoint
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    corrections = load_checkpoint(checkpoint_path)
    logger.info(f"Checkpoint: {len(corrections):,} tickets already processed")

    # Spot-check mode
    if args.spot_check > 0:
        if not corrections:
            logger.error("No corrections found. Run correction first.")
            sys.exit(1)
        spot_check(tickets, corrections, args.spot_check)
        sys.exit(0)

    # Test connectivity (synchronous, just once)
    import requests
    logger.info(f"Testing {args.backend} at {args.base_url}...")
    try:
        if args.backend == "vllm":
            r = requests.get(f"{args.base_url}/v1/models", timeout=10)
            r.raise_for_status()
            models = r.json()
            available = [m["id"] for m in models.get("data", [])]
            logger.info(f"Available models: {available}")
            if args.model not in available and available:
                logger.warning(f"Model '{args.model}' not found. Available: {available}")
                logger.warning(f"Using '{available[0]}' instead.")
                args.model = available[0]
        else:
            r = requests.get(f"{args.base_url}/api/tags", timeout=10)
            r.raise_for_status()
    except Exception as e:
        logger.error(f"Cannot reach {args.backend} at {args.base_url}: {e}")
        sys.exit(1)
    logger.info(f"Connected to {args.backend}, model={args.model}, concurrency={args.concurrency}")

    # Filter to remaining tickets
    remaining = [t for t in tickets if t.get("ticket_id") not in corrections]
    logger.info(f"Remaining: {len(remaining):,} tickets to process")

    if not remaining:
        logger.info("All tickets already processed. Use --spot-check to review.")
    else:
        # Choose async caller
        call_fn = call_vllm_async if args.backend == "vllm" else call_ollama_async

        # Run async processing
        stats = asyncio.run(process_batch(
            remaining, call_fn, args.base_url, args.model, args.temperature,
            args.concurrency, corrections, checkpoint_path, args.checkpoint_every,
        ))

    # Apply corrections to tickets
    logger.info("Applying corrections to ticket data...")
    label_changes = {"subcategory": 0, "priority": 0, "sentiment": 0}

    for ticket in tickets:
        tid = ticket.get("ticket_id")
        corr = corrections.get(tid, {})
        if not corr:
            continue
        if "subcategory" in corr and corr["subcategory"] != ticket.get("subcategory"):
            ticket["subcategory"] = corr["subcategory"]
            label_changes["subcategory"] += 1
        if "priority" in corr and corr["priority"] != ticket.get("priority"):
            ticket["priority"] = corr["priority"]
            label_changes["priority"] += 1
        if "sentiment" in corr and corr["sentiment"] != ticket.get("customer_sentiment"):
            ticket["customer_sentiment"] = corr["sentiment"]
            label_changes["sentiment"] += 1

    # Save corrected data
    logger.info(f"Writing corrected tickets to {args.output}")
    with open(args.output, "w") as f:
        json.dump(tickets, f, indent=2, default=str)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"LABEL CORRECTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total tickets: {len(tickets):,}")
    n_success = sum(1 for v in corrections.values() if v)
    logger.info(f"Successfully corrected: {n_success:,}")
    logger.info(f"Labels changed:")
    for field, count in label_changes.items():
        logger.info(f"  {field}: {count:,} ({count/len(tickets)*100:.1f}%)")
    logger.info(f"\nOutput: {args.output}")
    logger.info(f"Next steps:")
    logger.info(f"  1. Spot-check: python scripts/correct_labels.py --spot-check 200 --output {args.output} --data {args.data}")
    logger.info(f"  2. Retrain:    python scripts/train.py --data {args.output} --both")


if __name__ == "__main__":
    main()
