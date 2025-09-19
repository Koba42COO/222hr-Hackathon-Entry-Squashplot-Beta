#!/usr/bin/env python3
"""
🔍 DUPLICATE FILES ANALYSIS
==========================

Detailed analysis of the 681 potential duplicate files found during cleanup
to determine if they are actual duplicates or different builds/versions
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json

class DuplicateAnalyzer:
    """Analyze potential duplicate files to determine their nature"""

    def __init__(self, root_path="/Users/coo-koba42/dev"):
        self.root_path = Path(root_path)
        self.duplicate_analysis = {}

    def analyze_duplicates(self):
        """Analyze all potential duplicate files"""
        print("🔍 ANALYZING POTENTIAL DUPLICATE FILES")
        print("=" * 50)

        # Find all Python files
        python_files = list(self.root_path.rglob('*.py'))
        file_names = defaultdict(list)

        # Group files by name
        for file_path in python_files:
            if not any(part.startswith('.') for part in file_path.parts):
                file_names[file_path.name].append(file_path)

        # Analyze duplicates
        duplicates = {name: paths for name, paths in file_names.items() if len(paths) > 1}

        print(f"📊 Found {len(duplicates)} files with multiple copies")
        print(f"📄 Total duplicate instances: {sum(len(paths) for paths in duplicates.values())}")

        # Analyze each duplicate group
        duplicate_analysis = {}

        for file_name, file_paths in duplicates.items():
            print(f"\n🔍 Analyzing: {file_name}")
            print(f"   📂 Copies found: {len(file_paths)}")

            # Get file info for each copy
            file_info = []
            for i, file_path in enumerate(file_paths, 1):
                try:
                    stat = file_path.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime

                    # Calculate file hash
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    file_info.append({
                        'path': str(file_path),
                        'size': size,
                        'mtime': mtime,
                        'hash': file_hash,
                        'directory': str(file_path.parent)
                    })

                    print(f"   {i}. {file_path}")
                    print(f"      📏 Size: {size:,} bytes")
                    print(f"      📂 Dir: {file_path.parent.name}")

                except Exception as e:
                    print(f"   {i}. {file_path} - Error: {e}")

            # Analyze if they are identical or different
            hashes = [info['hash'] for info in file_info]
            unique_hashes = set(hashes)

            if len(unique_hashes) == 1:
                print("   ✅ IDENTICAL FILES - Safe to remove duplicates")
            elif len(unique_hashes) == len(file_info):
                print("   ⚠️ ALL DIFFERENT - Keep all versions")
            else:
                print(f"   🔄 MIXED - {len(unique_hashes)} unique versions out of {len(file_info)} copies")
            duplicate_analysis[file_name] = {
                'file_info': file_info,
                'unique_hashes': len(unique_hashes),
                'total_copies': len(file_info),
                'are_identical': len(unique_hashes) == 1,
                'all_different': len(unique_hashes) == len(file_info)
            }

        # Summary statistics
        total_files = sum(len(info['file_info']) for info in duplicate_analysis.values())
        identical_files = sum(1 for info in duplicate_analysis.values() if info['are_identical'])
        all_different_files = sum(1 for info in duplicate_analysis.values() if info['all_different'])
        mixed_files = len(duplicate_analysis) - identical_files - all_different_files

        print("\n📊 DUPLICATE ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"📁 Total files with duplicates: {len(duplicate_analysis)}")
        print(f"📄 Total duplicate instances: {total_files}")
        print(f"✅ Identical files (safe to dedupe): {identical_files}")
        print(f"⚠️ All different files (keep all): {all_different_files}")
        print(f"🔄 Mixed versions (review needed): {mixed_files}")

        # Show top duplicate files
        print("\n🔝 TOP DUPLICATE FILES:")
        sorted_duplicates = sorted(duplicate_analysis.items(),
                                  key=lambda x: x[1]['total_copies'], reverse=True)

        for file_name, info in sorted_duplicates[:10]:  # Show top 10
            status = "✅ Identical" if info['are_identical'] else "⚠️ Different" if info['all_different'] else "🔄 Mixed"
            print(f"   {status} {file_name}: {info['total_copies']} copies")

        # Save detailed analysis
        from datetime import datetime
        analysis_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_duplicate_groups': len(duplicate_analysis),
                'total_duplicate_instances': total_files,
                'identical_files': identical_files,
                'all_different_files': all_different_files,
                'mixed_files': mixed_files
            },
            'duplicate_details': duplicate_analysis
        }

        with open('duplicate_analysis_report.json', 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)

        print("\n📄 Detailed analysis saved: duplicate_analysis_report.json")
        return duplicate_analysis

def main():
    """Main function"""
    analyzer = DuplicateAnalyzer()
    duplicates = analyzer.analyze_duplicates()

    print("\n🎯 RECOMMENDATIONS:")
    print("   ✅ Identical files can be safely removed (keep 1 copy)")
    print("   ⚠️ Different files should be reviewed for versions/builds")
    print("   🔄 Mixed files need manual review to determine which to keep")

if __name__ == "__main__":
    main()
