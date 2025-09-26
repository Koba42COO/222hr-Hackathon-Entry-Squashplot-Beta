#!/usr/bin/env python3
"""
Analyze enhanced files to determine which ones to keep vs remove
"""

import os
import glob
from pathlib import Path
from datetime import datetime

def analyze_enhanced_files():
    """Analyze all enhanced files and their originals"""

    enhanced_files = glob.glob("**/*_enhanced.py", recursive=True)
    results = {
        "keep_enhanced": [],  # Enhanced is newer/better
        "keep_original": [],  # Original is newer/better
        "no_original": [],    # Only enhanced exists
        "identical": [],      # Files are identical
        "errors": []          # Analysis errors
    }

    print(f"Analyzing {len(enhanced_files)} enhanced files...")

    for enhanced_path in enhanced_files:
        enhanced_file = Path(enhanced_path)
        original_name = enhanced_file.name.replace("_enhanced.py", ".py")
        original_path = enhanced_file.parent / original_name

        try:
            if not original_path.exists():
                results["no_original"].append(str(enhanced_path))
                continue

            # Compare file sizes
            enhanced_size = enhanced_file.stat().st_size
            original_size = original_path.stat().st_size

            # Compare modification times
            enhanced_mtime = enhanced_file.stat().st_mtime
            original_mtime = original_path.stat().st_mtime

            # Enhanced is newer and larger (likely improved)
            if enhanced_mtime > original_mtime and enhanced_size > original_size * 0.8:
                results["keep_enhanced"].append({
                    "enhanced": str(enhanced_path),
                    "original": str(original_path),
                    "enhanced_newer_by": enhanced_mtime - original_mtime,
                    "size_ratio": enhanced_size / original_size
                })
            # Original is newer
            elif original_mtime > enhanced_mtime:
                results["keep_original"].append({
                    "enhanced": str(enhanced_path),
                    "original": str(original_path),
                    "original_newer_by": original_mtime - enhanced_mtime
                })
            # Enhanced exists but might be redundant
            else:
                results["keep_original"].append({
                    "enhanced": str(enhanced_path),
                    "original": str(original_path),
                    "reason": "same_age_or_smaller"
                })

        except Exception as e:
            results["errors"].append({
                "file": str(enhanced_path),
                "error": str(e)
            })

    return results

def print_analysis_report(results):
    """Print detailed analysis report"""

    print("\n" + "="*60)
    print("ENHANCED FILES ANALYSIS REPORT")
    print("="*60)

    print(f"\n📊 SUMMARY:")
    print(f"   Total enhanced files: {len(results['keep_enhanced']) + len(results['keep_original']) + len(results['no_original'])}")
    print(f"   Keep enhanced: {len(results['keep_enhanced'])}")
    print(f"   Keep original: {len(results['keep_original'])}")
    print(f"   No original: {len(results['no_original'])}")
    print(f"   Errors: {len(results['errors'])}")

    if results["keep_enhanced"]:
        print(f"\n✅ ENHANCED FILES TO KEEP ({len(results['keep_enhanced'])}):")
        for item in results["keep_enhanced"][:10]:  # Show first 10
            newer_time = item["enhanced_newer_by"] / 3600  # hours
            print(".1f")

        if len(results["keep_enhanced"]) > 10:
            print(f"   ... and {len(results['keep_enhanced']) - 10} more")

    if results["no_original"]:
        print(f"\n🆕 ENHANCED FILES WITH NO ORIGINAL ({len(results['no_original'])}):")
        for path in results["no_original"][:5]:
            print(f"   {path}")

    if results["errors"]:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results["errors"][:3]:
            print(f"   {error['file']}: {error['error']}")

def main():
    os.chdir("/Users/coo-koba42/dev")
    results = analyze_enhanced_files()
    print_analysis_report(results)

    # Save detailed report
    with open("enhanced_files_analysis.json", "w") as f:
        import json
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Detailed report saved to enhanced_files_analysis.json")

    # Recommendations
    safe_to_remove = len(results["keep_original"])
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   Safe to remove: {safe_to_remove} enhanced files")
    print(f"   Keep enhanced: {len(results['keep_enhanced'])} files (actually improved)")
    print(f"   Review manually: {len(results['no_original'])} files (no original found)")
if __name__ == "__main__":
    main()
