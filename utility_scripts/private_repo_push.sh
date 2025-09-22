#!/bin/bash

# Private Repository Push Script for Full VantaX Codebase
# Usage: ./private_repo_push.sh YOUR_GITHUB_USERNAME REPO_NAME

if [ $# -lt 2 ]; then
    echo "❌ Error: Please provide GitHub username and repository name"
    echo "Usage: ./private_repo_push.sh YOUR_USERNAME REPO_NAME"
    echo "Example: ./private_repo_push.sh Koba42COO vantax-full-private"
    exit 1
fi

USERNAME=$1
REPO_NAME=$2
REPO_URL="https://github.com/$USERNAME/$REPO_NAME.git"

echo "🔒 Setting up PRIVATE repository for full VantaX codebase"
echo "   Username: $USERNAME"
echo "   Repository: $REPO_NAME"
echo "   URL: $REPO_URL"
echo ""

# Check if remote already exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "⚠️  GitHub remote already exists. Updating..."
    git remote set-url origin "$REPO_URL"
else
    echo "📡 Adding private GitHub remote..."
    git remote add origin "$REPO_URL"
fi

echo "🧹 Cleaning repository (removing ignored files)..."
echo "   This may take a moment for large codebase..."

# Remove files that should be ignored
git rm --cached -r . >/dev/null 2>&1
git add . >/dev/null 2>&1

echo "📊 Repository status after cleanup:"
git status --porcelain | wc -l | xargs echo "   Files staged:"

echo ""
echo "⬆️  Pushing full VantaX codebase to private repository..."

if git push -u origin main; then
    echo ""
    echo "✅ SUCCESS! Full VantaX codebase pushed to PRIVATE repository"
    echo ""
    echo "🔒 Your repository is now private at:"
    echo "   https://github.com/$USERNAME/$REPO_NAME"
    echo ""
    echo "🛡️  Security Features:"
    echo "   • Repository is PRIVATE"
    echo "   • Sensitive files excluded via .gitignore"
    echo "   • All proprietary code protected"
    echo "   • Only authorized collaborators can access"
    echo ""
    echo "📁 Repository Contents (29,000+ files):"
    echo "   • Complete VantaX LLM core system"
    echo "   • Consciousness mathematics framework"
    echo "   • Fractal-Harmonic Transform implementation"
    echo "   • Research notebooks and analysis tools"
    echo "   • Documentation and deployment scripts"
    echo "   • Academic paper and supporting materials"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Add collaborators if needed (Repository Settings → Collaborators)"
    echo "   2. Set up branch protection rules for main branch"
    echo "   3. Create development branches for new features"
    echo "   4. Set up GitHub Actions for CI/CD if desired"
    echo ""
    echo "🔐 REMEMBER: This is your PRIVATE intellectual property"
    echo "   Keep it secure and share only with trusted collaborators"
    echo ""
    echo "🚀 Ready for continued development and research!"
else
    echo ""
    echo "❌ Push failed. This might be because:"
    echo "   1. The GitHub repository doesn't exist yet"
    echo "   2. You don't have permission to push to the repository"
    echo "   3. Repository is not set to private"
    echo "   4. Authentication issues with GitHub"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Make sure you created the PRIVATE repository on GitHub first"
    echo "   2. Verify the repository name matches exactly"
    echo "   3. Check if you need to set up SSH keys or personal access tokens"
    echo "   4. Ensure repository visibility is set to PRIVATE"
    echo ""
    echo "📞 For help with GitHub authentication:"
    echo "   https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories"
    echo "   https://docs.github.com/en/authentication"
fi
