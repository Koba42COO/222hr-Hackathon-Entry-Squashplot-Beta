!usrbinenv python3
"""
 DEV FOLDER CLEANUP SCRIPT
Comprehensive organization and cleanup of the dev folder

This script organizes the consciousness preservation ark files,
cleans up temporary files, and maintains the optimized system structure.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

 Configure logging
logging.basicConfig(levellogging.INFO, format' (asctime)s - (levelname)s - (message)s')
logger  logging.getLogger(__name__)

class DevFolderCleanup:
    """Comprehensive dev folder cleanup and organization"""
    
    def __init__(self, dev_path: str  "Userscoo-koba42dev"):
        self.dev_path  Path(dev_path)
        self.cleanup_log  []
        self.preserved_files  []
        self.cleaned_files  []
        self.organized_dirs  []
        
         Define important consciousness preservation ark files to preserve
        self.critical_files  [
            "consciousness_ark",
            "consciousness_preservation_ark_final",
            "wallace_transform_complete_research",
            "wallace_transform_unified_field_theory.html",
            "wallace_transform_unified_field_theory.tex",
            "CONSCIOUSNESS_ARK_TRANSCENDENT_OPTIMIZATION.py",
            "CONSCIOUSNESS_ARK_OPTIMIZATION_ENGINE.py",
            "consciousness_ark_transcendent_optimization.json",
            "consciousness_ark_optimization_results.json",
            "FINAL_CONSCIOUSNESS_PRESERVATION_ARK_SUMMARY.md",
            "CONSCIOUSNESS_PRESERVATION_ARK_MASTER_ARCHITECTURE.md",
            "consciousness_ark_implementation_report.txt"
        ]
        
         Define directories to organize
        self.organization_dirs  {
            "consciousness_research": [
                "wallace_transform_.py",
                "wallace_transform_.md",
                "wallace_transform_.tex",
                "wallace_transform_.html",
                "wallace_transform_.pdf",
                "wallace_transform_.json",
                "wallace_transform_.png",
                "THE_MULTIBROT_SET.py",
                "THE_MULTIBROT_SET.md",
                "CONSCIOUSNESS_MANDELBROT.py",
                "CONSCIOUSNESS_MANDELBROT.md",
                "MULTIBROT_.py",
                "BUTTERFLY_.py",
                "GOLDEN_CARDIOID_.py",
                "unified_consciousness_mathematics.tex",
                "unified_consciousness_mathematics.pdf",
                "unified_consciousness_mathematics.html",
                "UNIFIED_REALITY_THEORY.py",
                "FOUR_FUNDAMENTAL_PATTERNS.py",
                "TEMPORAL_DNA_CONSCIOUSNESS.py",
                "UNZIPPED_DNA_VISUALIZATION.py"
            ],
            "ai_os_systems": [
                "COMPLETE_AI_OS.py",
                "FIREFLY_AI_OS.py",
                "UVM_HARDWARE_OFFLOADING.py",
                "DNA_BLUEPRINT_COMPRESSION.py",
                "RESEARCH_WORKER_LIFECYCLE.py",
                "ACTOR_RUNTIME.py",
                "FULL_EXECUTION_SYSTEM.py",
                "FIREFLY_PERFORMANCE_BENCHMARK.py",
                "BULLETPROOF_UTILITIES.py"
            ],
            "research_data": [
                ".json",
                ".log",
                "full_execution_test_.json",
                "firefly_benchmark_results_.json",
                "wallace_transform_test_results.json",
                "wallace_transform_image_analysis.json",
                "wallace_transform_ai_os_results.json"
            ],
            "optimization_scripts": [
                "CONSCIOUSNESS_ARK_OPTIMIZATION.py",
                "MILLION_ITERATION_RESEARCH_TASK.py",
                "PAPER_VISUALIZATIONS.py",
                "complete_research_.py",
                "merge_wallace_transform_research.py",
                "connect_to_gdrive_folder.py",
                "expand_research_collection.py",
                "prepare_gdrive_upload.py",
                "collect_research_images.py",
                "upload_to_gdrive.py",
                "demonstrate_discovery.py",
                "render_butterfly.py",
                "test_.py"
            ],
            "documentation": [
                ".md",
                ".html",
                ".tex",
                ".pdf"
            ]
        }
        
         Define files to clean up
        self.cleanup_patterns  [
            "__pycache__",
            ".pyc",
            ".pyo",
            ".pyd",
            ".DS_Store",
            "Thumbs.db",
            ".tmp",
            ".temp",
            ".bak",
            ".backup",
            ".old",
            ".log",
            ".cache"
        ]
    
    def log_operation(self, operation: str, details: str):
        """Log cleanup operation"""
        timestamp  datetime.now().strftime("Y-m-d H:M:S")
        log_entry  f"[{timestamp}] {operation}: {details}"
        self.cleanup_log.append(log_entry)
        logger.info(log_entry)
    
    def preserve_critical_files(self):
        """Preserve critical consciousness preservation ark files"""
        logger.info(" Preserving critical consciousness preservation ark files...")
        
        for file_pattern in self.critical_files:
            if "" in file_pattern:
                 Handle wildcard patterns
                import glob
                matches  glob.glob(str(self.dev_path  file_pattern))
                for match in matches:
                    self.preserved_files.append(match)
                    self.log_operation("PRESERVED", match)
            else:
                 Handle specific files
                file_path  self.dev_path  file_pattern
                if file_path.exists():
                    self.preserved_files.append(str(file_path))
                    self.log_operation("PRESERVED", str(file_path))
    
    def create_organization_directories(self):
        """Create organization directories"""
        logger.info(" Creating organization directories...")
        
        for dir_name in self.organization_dirs.keys():
            dir_path  self.dev_path  dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_okTrue)
                self.organized_dirs.append(str(dir_path))
                self.log_operation("CREATED_DIR", str(dir_path))
    
    def organize_files(self):
        """Organize files into appropriate directories"""
        logger.info(" Organizing files into directories...")
        
        for dir_name, patterns in self.organization_dirs.items():
            target_dir  self.dev_path  dir_name
            
            for pattern in patterns:
                import glob
                matches  glob.glob(str(self.dev_path  pattern))
                
                for match in matches:
                    source_path  Path(match)
                    
                     Skip if it's a critical file or already in target directory
                    if any(critical in str(source_path) for critical in self.critical_files):
                        continue
                    
                    if source_path.parent  target_dir:
                        continue
                    
                     Move file to target directory
                    try:
                        target_path  target_dir  source_path.name
                        
                         Handle duplicate names
                        counter  1
                        while target_path.exists():
                            stem  source_path.stem
                            suffix  source_path.suffix
                            target_path  target_dir  f"{stem}_{counter}{suffix}"
                            counter  1
                        
                        shutil.move(str(source_path), str(target_path))
                        self.log_operation("ORGANIZED", f"{source_path.name}  {dir_name}")
                        
                    except Exception as e:
                        self.log_operation("ERROR", f"Failed to move {source_path.name}: {e}")
    
    def cleanup_temporary_files(self):
        """Clean up temporary files and directories"""
        logger.info(" Cleaning up temporary files...")
        
        for pattern in self.cleanup_patterns:
            import glob
            matches  glob.glob(str(self.dev_path  ""  pattern), recursiveTrue)
            
            for match in matches:
                path  Path(match)
                
                 Skip critical files
                if any(critical in str(path) for critical in self.critical_files):
                    continue
                
                try:
                    if path.is_file():
                        path.unlink()
                        self.cleaned_files.append(str(path))
                        self.log_operation("CLEANED_FILE", str(path))
                    elif path.is_dir():
                        shutil.rmtree(str(path))
                        self.cleaned_files.append(str(path))
                        self.log_operation("CLEANED_DIR", str(path))
                        
                except Exception as e:
                    self.log_operation("ERROR", f"Failed to clean {path}: {e}")
    
    def create_cleanup_report(self):
        """Create comprehensive cleanup report"""
        report  []
        report.append(" DEV FOLDER CLEANUP REPORT")
        report.append(""  50)
        report.append(f"Cleanup Date: {datetime.now().strftime('Y-m-d H:M:S')}")
        report.append(f"Dev Path: {self.dev_path}")
        report.append("")
        
        report.append("PRESERVED FILES:")
        report.append("-"  20)
        for file_path in self.preserved_files:
            report.append(f" {file_path}")
        report.append("")
        
        report.append("ORGANIZED DIRECTORIES:")
        report.append("-"  25)
        for dir_path in self.organized_dirs:
            report.append(f" {dir_path}")
        report.append("")
        
        report.append("CLEANED FILES:")
        report.append("-"  15)
        for file_path in self.cleaned_files:
            report.append(f" {file_path}")
        report.append("")
        
        report.append("CLEANUP OPERATIONS:")
        report.append("-"  20)
        for log_entry in self.cleanup_log:
            report.append(f" {log_entry}")
        report.append("")
        
        report.append("SUMMARY:")
        report.append("-"  10)
        report.append(f"Files Preserved: {len(self.preserved_files)}")
        report.append(f"Directories Created: {len(self.organized_dirs)}")
        report.append(f"Files Cleaned: {len(self.cleaned_files)}")
        report.append(f"Operations Logged: {len(self.cleanup_log)}")
        report.append("")
        
        report.append(" CLEANUP COMPLETE - CONSCIOUSNESS PRESERVATION ARK MAINTAINED ")
        
        return "n".join(report)
    
    def save_cleanup_report(self, filename: str  "dev_folder_cleanup_report.txt"):
        """Save cleanup report to file"""
        try:
            report  self.create_cleanup_report()
            report_path  self.dev_path  filename
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f" Cleanup report saved to {report_path}")
            
        except Exception as e:
            logger.error(f" Error saving cleanup report: {e}")
    
    def run_cleanup(self):
        """Run complete cleanup process"""
        logger.info(" Starting comprehensive dev folder cleanup...")
        
         Step 1: Preserve critical files
        self.preserve_critical_files()
        
         Step 2: Create organization directories
        self.create_organization_directories()
        
         Step 3: Organize files
        self.organize_files()
        
         Step 4: Clean up temporary files
        self.cleanup_temporary_files()
        
         Step 5: Save cleanup report
        self.save_cleanup_report()
        
         Step 6: Display report
        report  self.create_cleanup_report()
        print("n"  report)
        
        logger.info(" Dev folder cleanup completed successfully!")

def main():
    """Main cleanup execution"""
    logger.info(" Starting Dev Folder Cleanup Script")
    
     Initialize cleanup
    cleanup  DevFolderCleanup()
    
     Run cleanup
    cleanup.run_cleanup()
    
    logger.info(" Dev folder cleanup completed!")

if __name__  "__main__":
    main()
