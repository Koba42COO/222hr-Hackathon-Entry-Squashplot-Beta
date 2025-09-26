#!/usr/bin/env python3
"""
KOBA42 RECOVERY ANALYZER
========================
Analyze first run logs and resume from where we left off
=======================================================

Features:
1. Analyze first run logs and extract all information
2. Calculate overall training time and progress
3. Resume from exact point of interruption
4. Handle power loss, thunder storm, and disconnect scenarios
5. Provide comprehensive timing and completion estimates
"""

import numpy as np
import time
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from KOBA42_COMPREHENSIVE_TRAINING_LOGGER import ComprehensiveTrainingLogger, BatchF2Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecoveryAnalyzer:
    """Analyze training logs and provide recovery information."""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.training_log_file = Path("koba42_training.log")
        self.analysis_results = {}
        
    def analyze_first_run_logs(self) -> Dict[str, Any]:
        """Analyze the first run logs to extract all information."""
        print("🔍 ANALYZING FIRST RUN LOGS")
        print("=" * 50)
        
        analysis = {
            "first_run_start_time": None,
            "first_run_end_time": None,
            "first_run_duration": None,
            "configs_completed": 0,
            "total_configs": 0,
            "last_completed_config": None,
            "last_completed_matrix_size": None,
            "last_completed_optimization_level": None,
            "intentful_scores": [],
            "ml_accuracies": [],
            "execution_times": [],
            "interruption_reason": None,
            "interruption_time": None,
            "progress_percentage": 0.0,
            "estimated_completion_time": None,
            "can_resume": False
        }
        
        # Analyze training log file
        if self.training_log_file.exists():
            log_analysis = self._analyze_training_log()
            analysis.update(log_analysis)
        
        # Analyze any existing session files
        session_analysis = self._analyze_session_files()
        analysis.update(session_analysis)
        
        # Calculate overall progress and timing
        analysis = self._calculate_progress_and_timing(analysis)
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_training_log(self) -> Dict[str, Any]:
        """Analyze the training log file for timing and progress information."""
        analysis = {}
        
        try:
            with open(self.training_log_file, 'r') as f:
                log_lines = f.readlines()
            
            # Extract timing information
            start_times = []
            end_times = []
            intentful_scores = []
            ml_accuracies = []
            execution_times = []
            completed_configs = []
            
            for line in log_lines:
                # Look for start time patterns
                if "Started new training session" in line:
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        start_times.append(timestamp_match.group(1))
                
                # Look for completion patterns
                if "OPTIMIZATION" in line and "RESULTS" in line:
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        end_times.append(timestamp_match.group(1))
                
                # Look for intentful scores
                if "Intentful Score" in line:
                    score_match = re.search(r'Intentful Score = ([\d.]+)', line)
                    if score_match:
                        intentful_scores.append(float(score_match.group(1)))
                
                # Look for ML accuracies
                if "ML Accuracy" in line:
                    acc_match = re.search(r'ML Accuracy = ([\d.]+)', line)
                    if acc_match:
                        ml_accuracies.append(float(acc_match.group(1)))
                
                # Look for execution times
                if "Execution Time" in line:
                    time_match = re.search(r'Execution Time = ([\d.]+)s', line)
                    if time_match:
                        execution_times.append(float(time_match.group(1)))
                
                # Look for completed optimizations
                if "RUNNING OPTIMIZATION" in line:
                    config_match = re.search(r'OPTIMIZATION (\d+)/(\d+)', line)
                    if config_match:
                        completed_configs.append(int(config_match.group(1)))
                        analysis["total_configs"] = int(config_match.group(2))
                
                # Look for interruption patterns
                if any(keyword in line for keyword in ["interrupted", "error", "failed", "power", "storm"]):
                    analysis["interruption_reason"] = line.strip()
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        analysis["interruption_time"] = timestamp_match.group(1)
            
            # Set analysis values
            if start_times:
                analysis["first_run_start_time"] = start_times[0]
            if end_times:
                analysis["first_run_end_time"] = end_times[-1]
            if completed_configs:
                analysis["configs_completed"] = max(completed_configs)
                analysis["last_completed_config"] = max(completed_configs)
            if intentful_scores:
                analysis["intentful_scores"] = intentful_scores
            if ml_accuracies:
                analysis["ml_accuracies"] = ml_accuracies
            if execution_times:
                analysis["execution_times"] = execution_times
            
        except Exception as e:
            logger.error(f"Error analyzing training log: {e}")
        
        return analysis
    
    def _analyze_session_files(self) -> Dict[str, Any]:
        """Analyze existing session files for additional information."""
        analysis = {}
        
        session_file = self.log_dir / "training_session.json"
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                analysis.update({
                    "session_id": session_data.get("session_id"),
                    "session_start_time": session_data.get("start_time"),
                    "session_status": session_data.get("status"),
                    "session_configs_completed": session_data.get("configs_completed", 0),
                    "session_total_configs": session_data.get("total_configs", 0),
                    "session_intentful_scores": session_data.get("intentful_scores", []),
                    "session_ml_accuracies": session_data.get("ml_accuracies", []),
                    "session_execution_times": session_data.get("execution_times", [])
                })
                
            except Exception as e:
                logger.error(f"Error analyzing session file: {e}")
        
        return analysis
    
    def _calculate_progress_and_timing(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall progress and timing estimates."""
        # Calculate progress percentage
        if analysis.get("session_total_configs", 0) > 0:
            analysis["progress_percentage"] = (
                analysis.get("session_configs_completed", 0) / 
                analysis.get("session_total_configs", 1) * 100
            )
        elif analysis.get("total_configs", 0) > 0:
            analysis["progress_percentage"] = (
                analysis.get("configs_completed", 0) / 
                analysis.get("total_configs", 1) * 100
            )
        
        # Calculate first run duration
        if analysis.get("first_run_start_time") and analysis.get("first_run_end_time"):
            try:
                start_time = datetime.fromisoformat(analysis["first_run_start_time"])
                end_time = datetime.fromisoformat(analysis["first_run_end_time"])
                analysis["first_run_duration"] = (end_time - start_time).total_seconds()
            except Exception as e:
                logger.error(f"Error calculating duration: {e}")
        
        # Estimate completion time
        if analysis.get("session_execution_times"):
            avg_time_per_config = np.mean(analysis["session_execution_times"])
            remaining_configs = analysis.get("session_total_configs", 0) - analysis.get("session_configs_completed", 0)
            estimated_remaining_time = avg_time_per_config * remaining_configs
            
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            analysis["estimated_completion_time"] = estimated_completion.isoformat()
        
        # Determine if we can resume
        analysis["can_resume"] = (
            analysis.get("session_configs_completed", 0) > 0 and
            analysis.get("session_total_configs", 0) > analysis.get("session_configs_completed", 0)
        )
        
        return analysis
    
    def generate_recovery_plan(self) -> Dict[str, Any]:
        """Generate a recovery plan based on analysis."""
        analysis = self.analysis_results
        
        recovery_plan = {
            "recovery_timestamp": datetime.now().isoformat(),
            "analysis_summary": analysis,
            "recovery_strategy": "resume" if analysis.get("can_resume") else "restart",
            "resume_config": None,
            "remaining_configs": [],
            "estimated_completion": analysis.get("estimated_completion_time"),
            "recommendations": []
        }
        
        if analysis.get("can_resume"):
            # Create resume configuration
            recovery_plan["resume_config"] = {
                "start_config_index": analysis.get("session_configs_completed", 0),
                "session_id": analysis.get("session_id"),
                "progress_percentage": analysis.get("progress_percentage", 0)
            }
            
            # Define remaining configurations
            remaining_configs = [
                {
                    "matrix_size": 256,
                    "batch_size": 64,
                    "optimization_level": "advanced",
                    "ml_training_epochs": 50,
                    "intentful_enhancement": True,
                    "business_domain": "Blockchain Solutions"
                },
                {
                    "matrix_size": 512,
                    "batch_size": 128,
                    "optimization_level": "expert",
                    "ml_training_epochs": 75,
                    "intentful_enhancement": True,
                    "business_domain": "SaaS Platforms"
                }
            ]
            
            recovery_plan["remaining_configs"] = remaining_configs
            recovery_plan["recommendations"].extend([
                "Resume from checkpoint",
                "Use smaller batch sizes for stability",
                "Monitor power supply and weather conditions",
                "Implement UPS for power protection"
            ])
        else:
            recovery_plan["recommendations"].extend([
                "Start new training session",
                "Implement regular checkpointing",
                "Use UPS for power protection",
                "Monitor weather conditions"
            ])
        
        return recovery_plan
    
    def execute_recovery(self) -> bool:
        """Execute the recovery plan."""
        recovery_plan = self.generate_recovery_plan()
        
        print(f"\n🔄 EXECUTING RECOVERY PLAN")
        print(f"Strategy: {recovery_plan['recovery_strategy']}")
        print(f"Progress: {recovery_plan['analysis_summary'].get('progress_percentage', 0):.1f}%")
        print(f"Estimated Completion: {recovery_plan['estimated_completion']}")
        
        if recovery_plan["recovery_strategy"] == "resume":
            return self._resume_training(recovery_plan)
        else:
            return self._restart_training(recovery_plan)
    
    def _resume_training(self, recovery_plan: Dict[str, Any]) -> bool:
        """Resume training from where we left off."""
        try:
            # Initialize logger system
            logger_system = ComprehensiveTrainingLogger()
            
            # Log resume
            logger_system.log_resume()
            
            # Create remaining configurations
            remaining_configs = []
            for config_data in recovery_plan["remaining_configs"]:
                config = BatchF2Config(
                    matrix_size=config_data["matrix_size"],
                    batch_size=config_data["batch_size"],
                    optimization_level=config_data["optimization_level"],
                    ml_training_epochs=config_data["ml_training_epochs"],
                    intentful_enhancement=config_data["intentful_enhancement"],
                    business_domain=config_data["business_domain"],
                    timestamp=datetime.now().isoformat()
                )
                remaining_configs.append(config)
            
            # Resume training
            from KOBA42_COMPREHENSIVE_TRAINING_LOGGER import run_training_with_logging
            run_training_with_logging(
                logger_system, 
                remaining_configs, 
                start_config_index=recovery_plan["resume_config"]["start_config_index"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error resuming training: {e}")
            return False
    
    def _restart_training(self, recovery_plan: Dict[str, Any]) -> bool:
        """Restart training from beginning."""
        try:
            # Initialize logger system
            logger_system = ComprehensiveTrainingLogger()
            
            # Define new configurations
            configs = [
                BatchF2Config(
                    matrix_size=128,
                    batch_size=32,
                    optimization_level='basic',
                    ml_training_epochs=25,
                    intentful_enhancement=True,
                    business_domain='AI Development',
                    timestamp=datetime.now().isoformat()
                ),
                BatchF2Config(
                    matrix_size=256,
                    batch_size=64,
                    optimization_level='advanced',
                    ml_training_epochs=50,
                    intentful_enhancement=True,
                    business_domain='Blockchain Solutions',
                    timestamp=datetime.now().isoformat()
                ),
                BatchF2Config(
                    matrix_size=512,
                    batch_size=128,
                    optimization_level='expert',
                    ml_training_epochs=75,
                    intentful_enhancement=True,
                    business_domain='SaaS Platforms',
                    timestamp=datetime.now().isoformat()
                )
            ]
            
            # Start new session
            session_id = logger_system.start_new_session(configs)
            
            # Run training
            from KOBA42_COMPREHENSIVE_TRAINING_LOGGER import run_training_with_logging
            run_training_with_logging(logger_system, configs)
            
            return True
            
        except Exception as e:
            logger.error(f"Error restarting training: {e}")
            return False
    
    def save_analysis_report(self):
        """Save the analysis report to file."""
        analysis_report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_results": self.analysis_results,
            "recovery_plan": self.generate_recovery_plan()
        }
        
        report_file = self.log_dir / "recovery_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)
        
        print(f"📋 Analysis Report saved to: {report_file}")

def main():
    """Main recovery analysis and execution."""
    print("🚀 KOBA42 RECOVERY ANALYZER")
    print("=" * 60)
    print("Analyzing first run logs and executing recovery")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = RecoveryAnalyzer()
    
    # Analyze first run logs
    analysis = analyzer.analyze_first_run_logs()
    
    # Display analysis results
    print(f"\n📊 FIRST RUN ANALYSIS RESULTS:")
    print(f"   • First Run Start: {analysis.get('first_run_start_time', 'Unknown')}")
    print(f"   • First Run End: {analysis.get('first_run_end_time', 'Unknown')}")
    duration = analysis.get('first_run_duration', 0)
    print(f"   • First Run Duration: {duration:.2f} seconds" if duration else "   • First Run Duration: Unknown")
    print(f"   • Configs Completed: {analysis.get('configs_completed', 0)}")
    print(f"   • Total Configs: {analysis.get('total_configs', 0)}")
    print(f"   • Progress: {analysis.get('progress_percentage', 0):.1f}%")
    print(f"   • Can Resume: {analysis.get('can_resume', False)}")
    print(f"   • Interruption Reason: {analysis.get('interruption_reason', 'Unknown')}")
    print(f"   • Estimated Completion: {analysis.get('estimated_completion_time', 'Unknown')}")
    
    if analysis.get("intentful_scores"):
        print(f"   • Average Intentful Score: {np.mean(analysis['intentful_scores']):.6f}")
    if analysis.get("ml_accuracies"):
        print(f"   • Average ML Accuracy: {np.mean(analysis['ml_accuracies']):.6f}")
    
    # Save analysis report
    analyzer.save_analysis_report()
    
    # Execute recovery
    print(f"\n🔄 EXECUTING RECOVERY...")
    success = analyzer.execute_recovery()
    
    if success:
        print(f"\n✅ RECOVERY EXECUTED SUCCESSFULLY")
    else:
        print(f"\n❌ RECOVERY FAILED")
    
    return success

if __name__ == "__main__":
    main()
