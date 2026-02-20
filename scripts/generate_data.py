"""Generate 100k synthetic support tickets."""
import json, random, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

random.seed(42); np.random.seed(42)

PRODUCTS = {
    "DataSync Pro": {"versions":["3.0.0","3.1.0","3.2.1","3.3.0-beta"],"modules":["sync_engine","scheduler","connector_hub","data_mapper","api_gateway"],"weight":0.30},
    "CloudStore": {"versions":["2.5.0","2.5.1","2.6.0","2.7.0"],"modules":["storage_core","access_control","cdn_manager","billing_module"],"weight":0.25},
    "AnalyticsHub": {"versions":["1.8.0","1.9.0","2.0.0","2.0.1"],"modules":["query_engine","dashboard","data_pipeline","ml_predictions","export"],"weight":0.20},
    "SecureAuth": {"versions":["4.0.0","4.1.0","4.2.0"],"modules":["sso_provider","mfa_engine","token_service","audit_logger"],"weight":0.15},
    "DevTools CLI": {"versions":["1.0.0","1.1.0","1.2.0","1.3.0"],"modules":["cli_core","plugin_manager","config_loader","deployment_agent"],"weight":0.10},
}
CATEGORIES = {
    "Technical Issue": {"weight":0.35,"subs":["Configuration","Integration","Performance","Data Loss","Crash/Bug"]},
    "Account & Billing": {"weight":0.15,"subs":["Billing Dispute","Subscription Change","Access Issue","Invoice Request"]},
    "Feature Request": {"weight":0.12,"subs":["New Feature","Enhancement","UI/UX Improvement","API Extension"]},
    "How-To / Guidance": {"weight":0.18,"subs":["Setup Help","Best Practices","Migration","Documentation Gap"]},
    "Bug Report": {"weight":0.10,"subs":["Functional Bug","UI Bug","API Bug","Security Vulnerability"]},
    "Outage / Downtime": {"weight":0.05,"subs":["Full Outage","Partial Degradation","Scheduled Maintenance"]},
    "Compliance / Security": {"weight":0.05,"subs":["Data Privacy","Audit Request","Certification"]},
}
ERROR_CODES = ["ERROR_TIMEOUT_429","ERROR_AUTH_401","ERROR_FORBIDDEN_403","ERROR_INTERNAL_500","ERROR_RATE_LIMIT_429","ERROR_CONN_REFUSED","ERROR_SSL_HANDSHAKE","ERROR_DISK_FULL","ERROR_OOM_KILLED","ERROR_DEADLOCK"]
RES_CODES = ["CONFIG_CHANGE","PATCH_APPLIED","DOCS_PROVIDED","ESCALATED_ENG","ACCOUNT_UPDATED","WORKAROUND_PROVIDED","BUG_CONFIRMED","FEATURE_LOGGED","INFRA_FIX","USER_ERROR"]
SENTIMENTS = ["frustrated","neutral","satisfied","angry","confused","anxious"]
PRIORITIES = ["low","medium","high","critical"]
CHANNELS = ["email","chat","phone","portal","slack"]
TIERS = ["free","starter","professional","enterprise"]
ENVS = ["production","staging","development","testing"]
REGIONS = ["NA","EU","APAC","LATAM"]
IMPACTS = ["low","medium","high","critical"]
KB = [f"KB-{i}" for i in range(100,1500)]
AGENTS = [{"id":f"AGENT-{i:03d}","exp":random.randint(3,120),"spec":random.choice(["database","networking","security","general","billing","api"])} for i in range(1,51)]

SUBJ = {
    "Technical Issue": ["{product} {module} throwing {ec} in {env}","Timeout errors using {module}","{module} crashes after upgrading to {ver}"],
    "Account & Billing": ["Unexpected charges on {product}","Cannot access {product} — license expired","Request to upgrade from {tier}"],
    "Feature Request": ["Request: batch processing for {module}","Would like API for {module} bulk ops","Need webhook support for {product}"],
    "How-To / Guidance": ["How to configure {module} for multi-region?","Best practices migrating to {ver}","Documentation unclear for {module} API"],
    "Bug Report": ["{module} UI shows incorrect data","API returns 500 when {module} receives null","Race condition in {module}"],
    "Outage / Downtime": ["{product} completely unreachable","Intermittent failures in {module} in {region}"],
    "Compliance / Security": ["Need SOC2 report for {product}","Data residency question for {product} in {region}"],
}
DESC = [
    "Experiencing issues with {product} {module} for 2 days. Affecting {aff} users. Tried restarting. Error: {ec}. Please advise.",
    "After upgrading to {ver}, failures occur processing large payloads. Error: {ec}. Blocking workflow.",
    "Our {tier} account has issues since 2 days ago. Impact: {aff} users. Urgent for production.",
]
RESOLUTIONS = [
    "Root cause: misconfigured timeout. Updated config. Customer confirmed resolved.",
    "Stale cache issue. Workaround: clear cache. Permanent fix in next release.",
    "Connection pool exhausted. Updated pool_size=50. Verified in {env}.",
]

def wchoice(opts):
    items = list(opts.keys()); weights = [opts[k]["weight"] for k in items]
    return random.choices(items, weights=weights, k=1)[0]

def gen(idx, base):
    days = random.randint(0,364); hour = max(0,min(23,int(np.random.normal(14,4))))
    created = base + timedelta(days=days,hours=hour,minutes=random.randint(0,59))
    product = wchoice(PRODUCTS); pi = PRODUCTS[product]
    ver = random.choice(pi["versions"]); module = random.choice(pi["modules"])
    cat = wchoice(CATEGORIES); ci = CATEGORIES[cat]; subcat = random.choice(ci["subs"])
    if cat in ("Outage / Downtime","Compliance / Security"): pri = random.choices(PRIORITIES,weights=[0.05,0.15,0.40,0.40])[0]
    elif cat == "Feature Request": pri = random.choices(PRIORITIES,weights=[0.40,0.40,0.15,0.05])[0]
    else: pri = random.choices(PRIORITIES,weights=[0.20,0.35,0.30,0.15])[0]
    sev = {"low":"P4","medium":"P3","high":"P2","critical":"P1"}[pri]
    tier = random.choices(TIERS,weights=[0.15,0.25,0.35,0.25])[0]
    chan = random.choices(CHANNELS,weights=[0.30,0.25,0.15,0.20,0.10])[0]
    env = random.choice(ENVS); region = random.choice(REGIONS)
    ec = random.choice(ERROR_CODES); has_err = cat in ("Technical Issue","Bug Report","Outage / Downtime")
    subj = random.choice(SUBJ.get(cat,SUBJ["Technical Issue"])).format(product=product,module=module,ec=ec,ver=ver,env=env,region=region,tier=tier)
    aff = random.randint(1,500) if pri in ("high","critical") else random.randint(1,20)
    desc = random.choice(DESC).format(product=product,module=module,ver=ver,ec=ec if has_err else "N/A",env=env,tier=tier,aff=aff)
    el,st = None,None
    if has_err and random.random()>0.3:
        ts = created.strftime("%Y-%m-%d %H:%M:%S")
        el = f"{ts} {ec}: Connection timeout after 30s\n{ts} RETRY_FAILED: Max retries exceeded"
        st = f"at {module}.execute({module}.py:{random.randint(50,500)})"
    if pri=="critical": sent = random.choices(SENTIMENTS,weights=[0.35,0.10,0.05,0.30,0.10,0.10])[0]
    elif cat=="Feature Request": sent = random.choices(SENTIMENTS,weights=[0.10,0.50,0.20,0.02,0.08,0.10])[0]
    else: sent = random.choices(SENTIMENTS,weights=[0.20,0.35,0.15,0.10,0.10,0.10])[0]
    res_hrs = max(0.5,np.random.lognormal(2.5,1.0))
    if pri=="critical": res_hrs*=0.5
    elif pri=="low": res_hrs*=1.5
    resolved = created+timedelta(hours=res_hrs); rc = random.choice(RES_CODES)
    res = random.choice(RESOLUTIONS).format(ver=ver,env=env)
    agent = random.choice(AGENTS); esc = random.random()<(0.15 if pri in ("high","critical") else 0.05)
    prev = max(0,int(np.random.exponential(3))); sat = max(1,min(5,int(np.random.normal(3.8 if not esc else 2.8,0.8))))
    nkb = random.randint(0,5); kbv = random.sample(KB,min(nkb,len(KB))); kbh = random.sample(kbv,min(random.randint(0,len(kbv)),len(kbv)))
    return {
        "ticket_id":f"TK-2024-{idx:06d}","created_at":created.isoformat(),"updated_at":resolved.isoformat(),
        "customer_id":f"CUST-{random.randint(1000,9999)}","customer_tier":tier,"organization_id":f"ORG-{random.randint(100,999)}",
        "product":product,"product_version":ver,"product_module":module,"category":cat,"subcategory":subcat,
        "priority":pri,"severity":sev,"channel":chan,"subject":subj,"description":desc,"error_logs":el,"stack_trace":st,
        "customer_sentiment":sent,"previous_tickets":prev,"resolution":res,"resolution_code":rc,
        "resolved_at":resolved.isoformat(),"resolution_time_hours":round(res_hrs,2),"resolution_attempts":random.randint(1,4),
        "agent_id":agent["id"],"agent_experience_months":agent["exp"],"agent_specialization":agent["spec"],
        "agent_actions":random.sample(["viewed_logs","checked_config","applied_fix","verified_resolution","contacted_customer","escalated"],k=random.randint(2,5)),
        "escalated":esc,"escalation_reason":"Complex issue" if esc else None,"transferred_count":random.randint(0,3),
        "satisfaction_score":sat,"feedback_text":random.choice(["Resolved quickly","Took too long","Great support",None]),
        "resolution_helpful":random.random()>0.2,
        "tags":random.sample(["database","sync","timeout","configuration","auth","performance","billing","upgrade","api","security"],k=random.randint(1,4)),
        "related_tickets":[f"TK-2024-{random.randint(1,max(1,idx-1)):06d}" for _ in range(random.randint(0,3))],
        "kb_articles_viewed":kbv,"kb_articles_helpful":kbh,"environment":env,
        "account_age_days":random.randint(30,1800),"account_monthly_value":round(random.choice([0,50,200,500,1000,5000,10000])*random.uniform(0.8,1.2),2),
        "similar_issues_last_30_days":max(0,int(np.random.exponential(15))),"product_version_age_days":random.randint(1,180),
        "known_issue":random.random()<0.1,"bug_report_filed":random.random()<0.08,
        "resolution_template_used":f"TEMPLATE-{rc}" if random.random()>0.4 else None,
        "auto_suggested_solutions":random.sample(KB,3),"auto_suggestion_accepted":random.random()<0.3,
        "ticket_text_length":len(desc),"response_count":random.randint(1,8),"attachments_count":random.randint(0,5),
        "contains_error_code":has_err and el is not None,"contains_stack_trace":has_err and st is not None,
        "business_impact":random.choices(IMPACTS,weights=[0.30,0.35,0.25,0.10])[0],"affected_users":aff,
        "weekend_ticket":created.weekday()>=5,"after_hours":hour<8 or hour>18,"language":"en","region":region,
    }

if __name__ == "__main__":
    print("Generating 100,000 tickets...")
    base = datetime(2024,1,1,tzinfo=timezone.utc)
    tickets = []
    for i in range(1,100001):
        tickets.append(gen(i,base))
        if i%25000==0: print(f"  {i:,}...")
    out = Path("data/raw/tickets_100k.json"); out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w") as f: json.dump(tickets,f)
    print(f"Done: {len(tickets):,} tickets -> {out}")
    from collections import Counter
    for c,n in Counter(t["category"] for t in tickets).most_common(): print(f"  {c}: {n:,} ({n/len(tickets)*100:.1f}%)")
