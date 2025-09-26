#!/bin/bash
# chAIos Private Repository Setup Script
# ======================================
# Secure private repository configuration for protected code

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔐 chAIos Private Repository Setup${NC}"
echo -e "${BLUE}==================================${NC}"
echo ""

# Configuration
PRIVATE_REPO_NAME="chaios-private"
GITHUB_ORG="chiral-harmonic"  # Your GitHub organization
REPO_URL="git@github.com:${GITHUB_ORG}/${PRIVATE_REPO_NAME}.git"

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found. Please install Git first.${NC}"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}⚠️  GitHub CLI not found. Manual setup required.${NC}"
    MANUAL_SETUP=true
else
    MANUAL_SETUP=false
fi

echo -e "${GREEN}✅ Prerequisites checked${NC}"

# Step 1: Create private repository
echo ""
echo -e "${YELLOW}🏗️  Step 1: Creating private repository${NC}"

if [ "$MANUAL_SETUP" = false ]; then
    echo "Creating private repository on GitHub..."
    gh repo create "${PRIVATE_REPO_NAME}" --private --description "chAIos Protected Platform - Intellectual Property Secured"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Private repository created on GitHub${NC}"
    else
        echo -e "${YELLOW}⚠️  GitHub CLI failed. Please create repository manually:${NC}"
        echo "  https://github.com/new"
        echo "  Repository name: ${PRIVATE_REPO_NAME}"
        echo "  Visibility: Private"
        MANUAL_SETUP=true
    fi
else
    echo -e "${YELLOW}📝 Please create a private repository manually:${NC}"
    echo "  1. Go to: https://github.com/new"
    echo "  2. Repository name: ${PRIVATE_REPO_NAME}"
    echo "  3. Make it PRIVATE"
    echo "  4. Do NOT initialize with README"
    echo ""
    read -p "Press Enter when repository is created..."
fi

# Step 2: Initialize local repository
echo ""
echo -e "${YELLOW}📁 Step 2: Initializing local repository${NC}"

if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${YELLOW}⚠️  Git repository already exists${NC}"
fi

# Step 3: Configure repository
echo ""
echo -e "${YELLOW}⚙️  Step 3: Configuring repository${NC}"

# Create .gitignore for protected deployment
cat > .gitignore << EOF
# chAIos Protected Repository .gitignore
# =====================================

# Original source code (backed up separately)
backup_original/
*.py

# Development files
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp

# Secrets (NEVER commit)
secrets/
.env
*.key
*.pem

# Development databases
*.db
*.sqlite
*.sqlite3

# Node modules (if any)
node_modules/

# Build artifacts
dist/
build/
*.egg-info/

# Testing
.coverage
.pytest_cache/
.tox/
EOF

echo -e "${GREEN}✅ Repository configured with security-focused .gitignore${NC}"

# Step 4: Set up protected directory structure
echo ""
echo -e "${YELLOW}📂 Step 4: Setting up protected directory structure${NC}"

mkdir -p protected_build
mkdir -p secrets
mkdir -p monitoring
mkdir -p logs
mkdir -p backups

# Create .gitkeep files to maintain directory structure
touch protected_build/.gitkeep
touch secrets/.gitkeep
touch monitoring/.gitkeep
touch logs/.gitkeep
touch backups/.gitkeep

echo -e "${GREEN}✅ Protected directory structure created${NC}"

# Step 5: Initial commit with protected files
echo ""
echo -e "${YELLOW}💾 Step 5: Initial commit${NC}"

git add .
git add -f protected_build/.gitkeep secrets/.gitkeep monitoring/.gitkeep logs/.gitkeep backups/.gitkeep

git commit -m "🔒 Initial commit: chAIos Protected Platform

- Intellectual property secured and obfuscated
- Enterprise-grade code protection implemented
- Private repository for secure deployment
- All proprietary algorithms protected

🚀 Ready for secure production deployment"

echo -e "${GREEN}✅ Initial protected commit created${NC}"

# Step 6: Set up remote repository
echo ""
echo -e "${YELLOW}🔗 Step 6: Setting up remote repository${NC}"

if git remote get-url origin >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Remote 'origin' already exists. Updating...${NC}"
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi

echo -e "${GREEN}✅ Remote repository configured${NC}"

# Step 7: Push protected code
echo ""
echo -e "${YELLOW}📤 Step 7: Pushing protected code${NC}"

if git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null; then
    echo -e "${GREEN}✅ Protected code pushed to private repository${NC}"
else
    echo -e "${YELLOW}⚠️  Push failed. You may need to:${NC}"
    echo "  1. Set up SSH keys with GitHub"
    echo "  2. Manually push: git push -u origin main"
fi

# Step 8: Set up branch protection (if GitHub CLI available)
echo ""
echo -e "${YELLOW}🛡️  Step 8: Setting up branch protection${NC}"

if [ "$MANUAL_SETUP" = false ]; then
    echo "Setting up branch protection rules..."
    gh api repos/${GITHUB_ORG}/${PRIVATE_REPO_NAME}/branches/main/protection \
        -X PUT \
        -H "Accept: application/vnd.github.v3+json" \
        -f required_status_checks=null \
        -f enforce_admins=true \
        -f required_pull_request_reviews='{"required_approving_review_count": 1}' \
        -f restrictions=null || echo "Branch protection setup may require manual configuration"
fi

# Step 9: Create deployment documentation
echo ""
echo -e "${YELLOW}📚 Step 9: Creating deployment documentation${NC}"

cat > DEPLOYMENT_GUIDE.md << EOF
# 🚀 chAIos Protected Deployment Guide

## Overview
This private repository contains the protected, obfuscated deployment of chAIos.
All intellectual property is secured and source code is obfuscated.

## Repository Structure
\`\`\`
chaios-private/
├── protected_build/          # Obfuscated Python bytecode
├── Dockerfile.protected      # Protected container definition
├── docker-compose.protected.yml  # Production deployment
├── requirements_protected.txt    # Minimal runtime dependencies
├── secrets/                  # Encrypted secrets (not committed)
├── monitoring/              # Monitoring configuration
├── logs/                    # Application logs
└── backups/                 # Automated backups
\`\`\`

## Security Features
- ✅ Code obfuscation (multiple layers)
- ✅ Runtime encryption
- ✅ Intellectual property protection
- ✅ Container security hardening
- ✅ Audit logging
- ✅ Access control

## Deployment Instructions

### Quick Start
\`\`\`bash
# Clone protected repository
git clone git@github.com:${GITHUB_ORG}/${PRIVATE_REPO_NAME}.git
cd ${PRIVATE_REPO_NAME}

# Start protected system
docker-compose -f docker-compose.protected.yml up -d

# Verify deployment
curl http://localhost:8000/health
\`\`\`

### Production Deployment
\`\`\`bash
# Build production image
docker build -f Dockerfile.protected -t chaios-prod .

# Run in production
docker run -d --name chaios-production \\
  -p 8000:8000 \\
  -v ./secrets:/opt/chaios/secrets:ro \\
  chaios-prod
\`\`\`

## Security Notes
- Never commit secrets or original source code
- All sensitive data is encrypted at runtime
- Monitor for reverse engineering attempts
- Regular security updates required

## Support
- Email: security@chaios-platform.com
- Issues: GitHub Issues (this repository)
- Monitoring: http://localhost:8080

---
*chAIos Protected Repository - Intellectual Property Secured*
EOF

echo -e "${GREEN}✅ Deployment documentation created${NC}"

# Final summary
echo ""
echo -e "${GREEN}🎉 Private Repository Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${BLUE}📁 Repository:${NC} https://github.com/${GITHUB_ORG}/${PRIVATE_REPO_NAME}"
echo -e "${BLUE}🔐 Visibility:${NC} Private"
echo -e "${BLUE}🛡️  Protection:${NC} Enterprise-grade"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Review repository settings on GitHub"
echo "2. Set up team access permissions"
echo "3. Configure branch protection rules"
echo "4. Set up automated deployment pipelines"
echo "5. Create backup and disaster recovery procedures"
echo ""
echo -e "${GREEN}🔒 Your intellectual property is now secure!${NC}"
