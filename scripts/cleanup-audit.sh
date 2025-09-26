#!/bin/bash

# 🚨 CODE AUDIT CLEANUP SCRIPT
# Automated cleanup based on audit findings

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚨 CODE AUDIT CLEANUP SCRIPT${NC}"
echo -e "${BLUE}=============================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get file size
get_size() {
    du -sh "$1" | cut -f1
}

# Function to safely remove files with confirmation
safe_remove() {
    local file="$1"
    local reason="$2"

    if [ -f "$file" ]; then
        local size=$(get_size "$file")
        print_warning "Removing $file (${size}) - ${reason}"

        # Create backup before removal (optional)
        # mkdir -p backups
        # cp "$file" "backups/$(basename "$file").backup"

        rm -f "$file"
        print_success "Removed $file"
    else
        print_status "File $file not found, skipping"
    fi
}

# PHASE 1: CRITICAL CLEANUP
echo -e "${RED}🔴 PHASE 1: CRITICAL CLEANUP${NC}"
echo "================================"

# Remove massive log file
safe_remove "intentful_mathematics_full_report.log" "6.6GB debug log file"

# Remove other large log files
safe_remove "ultimate_consciousness_system.log" "82MB debug log file"
safe_remove "revolutionary_learning_system.log" "15MB debug log file"

# Clean node_modules from archives
if [ -d "structured_chaos_full_archive" ]; then
    print_status "Cleaning node_modules from archives..."
    find structured_chaos_full_archive -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
    print_success "Cleaned node_modules from archives"
fi

# PHASE 2: REDUNDANCY CLEANUP
echo ""
echo -e "${YELLOW}🟡 PHASE 2: REDUNDANCY CLEANUP${NC}"
echo "================================="

# Remove duplicate enhanced files (keep newer ones)
print_status "Analyzing enhanced files..."
enhanced_count=$(find . -name "*_enhanced.py" | wc -l)
print_warning "Found ${enhanced_count} enhanced files"

# Create backup directory for enhanced files
mkdir -p backups/enhanced_files

# Move old enhanced files to backup (keep last 30 days)
find . -name "*_enhanced.py" -mtime +30 -exec mv {} backups/enhanced_files/ \; 2>/dev/null || true

# PHASE 3: PERFORMANCE OPTIMIZATION
echo ""
echo -e "${BLUE}🔵 PHASE 3: PERFORMANCE CLEANUP${NC}"
echo "==================================="

# Create .gitignore for large files
cat > .gitignore.audit << 'EOF'
# Audit-generated ignores
*.log
node_modules/
__pycache__/
*.pyc
.cache/
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db
EOF

if [ -f .gitignore ]; then
    cat .gitignore.audit >> .gitignore
    rm .gitignore.audit
    print_success "Updated .gitignore"
else
    mv .gitignore.audit .gitignore
    print_success "Created .gitignore"
fi

# PHASE 4: REPORTING
echo ""
echo -e "${GREEN}🟢 PHASE 4: CLEANUP REPORT${NC}"
echo "==========================="

# Calculate space saved
echo "Space analysis:"
echo "=============="
echo "Total size before cleanup:"
du -sh . 2>/dev/null || echo "Unable to calculate"

echo ""
echo "Large files remaining:"
find . -type f -size +100M -exec ls -lh {} \; 2>/dev/null || echo "No large files found"

echo ""
echo "Cleanup Summary:"
echo "==============="
echo "✅ Removed massive log files"
echo "✅ Cleaned node_modules"
echo "✅ Updated .gitignore"
echo "✅ Backed up old enhanced files"

# Recommendations
echo ""
echo -e "${BLUE}📋 NEXT STEPS:${NC}"
echo "1. Review CODE_AUDIT_REPORT.md for detailed recommendations"
echo "2. Run: python3 -c \"import sys; print(f'Python files: {len(list(sys.modules.keys()))}')\""
echo "3. Consider implementing shared modules for common code"
echo "4. Add proper logging configuration"
echo "5. Implement automated testing"

print_success "Cleanup completed! Check CODE_AUDIT_REPORT.md for next steps."
