#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════
# Cloud Deployment Script for Support Intelligence System
#
# Tested on: Ubuntu 22.04/24.04 LTS
# Requirements: SSH access to a VM with ≥16GB RAM, 50GB disk
#
# Usage:
#   1. SSH into your VM
#   2. Upload:  scp support-intelligence.tar.gz user@server:~/
#   3. Extract: tar xzf support-intelligence.tar.gz
#   4. Run:     cd support-intelligence && bash deploy.sh
#   5. Upload data: scp tickets.json user@server:~/support-intelligence/data/raw/tickets.json
#   6. Run pipeline: bash run_pipeline.sh
# ═══════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════╗"
echo "║  Support Intelligence System — Cloud Setup       ║"
echo "╚══════════════════════════════════════════════════╝"

# ── Step 1: Install Docker Engine (official method) ──
if ! command -v docker &> /dev/null; then
    echo "→ Installing Docker Engine..."

    # Remove any old/conflicting packages
    for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
        sudo apt-get remove -y $pkg 2>/dev/null || true
    done

    # Install prerequisites
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl

    # Add Docker's official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the Docker apt repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
      https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine + Compose plugin
    sudo apt-get update
    sudo apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin

    # Let current user run docker without sudo
    sudo usermod -aG docker $USER

    echo ""
    echo "  ✓ Docker installed."
    echo ""
    echo "  ⚠  You need to log out and back in for group changes to take effect."
    echo "     Run:  exit"
    echo "     Then: ssh back in and re-run this script."
    echo ""
    echo "     Or run this to apply immediately (starts a new shell):"
    echo "       newgrp docker && bash deploy.sh"
    exit 0
fi

echo "→ Docker: $(docker --version)"
echo "→ Compose: $(docker compose version)"

# ── Step 2: Clone/extract project ──
PROJECT_DIR="$HOME/support-intelligence"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "→ Creating project directory..."
    # If you uploaded the tarball:
    if [ -f "$HOME/support-intelligence.tar.gz" ]; then
        cd "$HOME" && tar xzf support-intelligence.tar.gz
    else
        echo "  ERROR: No project found. Upload support-intelligence.tar.gz first:"
        echo "    scp support-intelligence.tar.gz user@your-server:~/"
        exit 1
    fi
fi

cd "$PROJECT_DIR"

# ── Step 3: Create data directory ──
mkdir -p data/raw

if [ ! -f "data/raw/tickets.json" ]; then
    echo ""
    echo "⚠  No ticket data found at data/raw/tickets.json"
    echo "   Upload your 300K ticket JSON file:"
    echo "     scp /path/to/tickets.json $(whoami)@$(hostname):$PROJECT_DIR/data/raw/tickets.json"
    echo ""
    echo "   Then run: bash run_pipeline.sh"
    echo ""
fi

# ── Step 4: Build and start infrastructure ──
echo "→ Building Docker images (this takes 3-5 minutes first time)..."
docker compose build

echo "→ Starting services (PostgreSQL, Qdrant, MLflow)..."
docker compose up -d postgres qdrant mlflow

echo "→ Waiting for services to be healthy..."
sleep 15

# Check health
for svc in postgres qdrant; do
    if docker compose ps $svc | grep -q "healthy"; then
        echo "  ✓ $svc is healthy"
    else
        echo "  ⚠ $svc may still be starting..."
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Infrastructure ready!                           ║"
echo "║                                                  ║"
echo "║  Next steps:                                     ║"
echo "║  1. Upload data (if not done):                   ║"
echo "║     scp tickets.json user@server:~/support-      ║"
echo "║       intelligence/data/raw/tickets.json         ║"
echo "║                                                  ║"
echo "║  2. Run the full pipeline:                       ║"
echo "║     cd ~/support-intelligence                    ║"
echo "║     bash run_pipeline.sh                         ║"
echo "║                                                  ║"
echo "║  Services:                                       ║"
echo "║    API:    http://your-server:8000/docs           ║"
echo "║    MLflow: http://your-server:5000                ║"
echo "╚══════════════════════════════════════════════════╝"
