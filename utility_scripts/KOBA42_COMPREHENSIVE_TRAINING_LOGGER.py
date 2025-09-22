#!/usr/bin/env python3
"""
KOBA42 COMPREHENSIVE TRAINING LOGGER
====================================
Comprehensive logging system for batch F2 matrix optimization
============================================================

Features:
1. First Run Logging and Tracking
2. Power Loss Recovery
3. Thunder Storm Disconnect Handling
4. Overall Training Time Tracking
5. Resume from Last Checkpoint
"""

import numpy as np
import time
import json
import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from KOBA42_BATCH_F2_MATRIX_OPTIMIZATION import BatchF2Config, BatchF2MatrixOptimizer

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('koba42_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingSession:
    """Training session information."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    total_duration: Optional[float]
    status: str  # 'running', 'completed', 'interrupted', 'resumed'
    interruption_reason: Optional[str]
    resume_time: Optional[str]
    configs_completed: int
    total_configs: int
    last_completed_batch: Optional[int]
    last_completed_matrix_size: Optional[int]
    last_completed_optimization_level: Optional[str]
    intentful_scores: List[float]
    ml_accuracies: List[float]
    execution_times: List[float]

@dataclass
class TrainingCheckpoint:
    """Training checkpoint for recovery."""
    checkpoint_id: str
    timestamp: str
    session_id: str
    current_config_index: int
    current_batch_id: int
    completed_results: List[Dict[str, Any]]
    pending_configs: List[Dict[str, Any]]
    overall_progress: float
    estimated_completion_time: Optional[str]

class ComprehensiveTrainingLogger:
    """Comprehensive training logger with recovery capabilities."""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.session_file = self.log_dir / "training_session.json"
        self.checkpoint_file = self.log_dir / "training_checkpoint.json"
        self.results_file = self.log_dir / "training_results.json"
        self.recovery_file = self.log_dir / "recovery_info.json"
        
        self.current_session: Optional[TrainingSession] = None
        self.current_checkpoint: Optional[TrainingCheckpoint] = None
        
        # Load existing session if available
        self._load_existing_session()
    
    def _load_existing_session(self):
        """Load existing training session if available."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                    self.current_session = TrainingSession(**session_data)
                    logger.info(f"Loaded existing session: {self.current_session.session_id}")
            except Exception as e:
                logger.error(f"Failed to load existing session: {e}")
        
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.current_checkpoint = TrainingCheckpoint(**checkpoint_data)
                    logger.info(f"Loaded checkpoint: {self.current_checkpoint.checkpoint_id}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
    
    def start_new_session(self, configs: List[BatchF2Config]) -> str:
        """Start a new training session."""
        session_id = f"session_{int(time.time())}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            total_duration=None,
            status='running',
            interruption_reason=None,
            resume_time=None,
            configs_completed=0,
            total_configs=len(configs),
            last_completed_batch=None,
            last_completed_matrix_size=None,
            last_completed_optimization_level=None,
            intentful_scores=[],
            ml_accuracies=[],
            execution_times=[]
        )
        
        self._save_session()
        logger.info(f"Started new training session: {session_id}")
        return session_id
    
    def log_interruption(self, reason: str):
        """Log training interruption (power loss, thunder storm, etc.)."""
        if self.current_session:
            self.current_session.status = 'interrupted'
            self.current_session.interruption_reason = reason
            self.current_session.end_time = datetime.now().isoformat()
            
            # Calculate duration so far
            start_time = datetime.fromisoformat(self.current_session.start_time)
            end_time = datetime.fromisoformat(self.current_session.end_time)
            self.current_session.total_duration = (end_time - start_time).total_seconds()
            
            self._save_session()
            logger.warning(f"Training interrupted: {reason}")
    
    def log_resume(self):
        """Log training resume."""
        if self.current_session:
            self.current_session.status = 'resumed'
            self.current_session.resume_time = datetime.now().isoformat()
            self._save_session()
            logger.info("Training resumed from interruption")
    
    def log_completion(self):
        """Log training completion."""
        if self.current_session:
            self.current_session.status = 'completed'
            self.current_session.end_time = datetime.now().isoformat()
            
            # Calculate total duration
            start_time = datetime.fromisoformat(self.current_session.start_time)
            end_time = datetime.fromisoformat(self.current_session.end_time)
            self.current_session.total_duration = (end_time - start_time).total_seconds()
            
            self._save_session()
            logger.info("Training completed successfully")
    
    def create_checkpoint(self, config_index: int, batch_id: int, 
                         completed_results: List[Dict[str, Any]], 
                         pending_configs: List[Dict[str, Any]]):
        """Create a training checkpoint."""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        # Calculate overall progress
        total_configs = len(completed_results) + len(pending_configs)
        progress = len(completed_results) / total_configs * 100
        
        # Estimate completion time
        if completed_results:
            avg_time_per_config = np.mean([r.get('execution_time', 0) for r in completed_results])
            remaining_configs = len(pending_configs)
            estimated_remaining_time = avg_time_per_config * remaining_configs
            
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            estimated_completion_str = estimated_completion.isoformat()
        else:
            estimated_completion_str = None
        
        self.current_checkpoint = TrainingCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            session_id=self.current_session.session_id if self.current_session else "unknown",
            current_config_index=config_index,
            current_batch_id=batch_id,
            completed_results=completed_results,
            pending_configs=pending_configs,
            overall_progress=progress,
            estimated_completion_time=estimated_completion_str
        )
        
        self._save_checkpoint()
        logger.info(f"Created checkpoint {checkpoint_id} at {progress:.1f}% progress")
    
    def log_result(self, config_index: int, result: Dict[str, Any]):
        """Log a completed optimization result."""
        if self.current_session:
            # Update session with result
            intentful_score = result.get('batch_optimization_results', {}).get('average_intentful_score', 0)
            ml_accuracy = result.get('ml_training_results', {}).get('average_accuracy', 0)
            execution_time = result.get('overall_performance', {}).get('total_execution_time', 0)
            
            self.current_session.intentful_scores.append(intentful_score)
            self.current_session.ml_accuracies.append(ml_accuracy)
            self.current_session.execution_times.append(execution_time)
            self.current_session.configs_completed += 1
            
            # Update last completed information
            config = result.get('optimization_config', {})
            self.current_session.last_completed_matrix_size = config.get('matrix_size')
            self.current_session.last_completed_optimization_level = config.get('optimization_level')
            
            self._save_session()
            logger.info(f"Logged result for config {config_index}: "
                       f"Intentful Score = {intentful_score:.6f}, "
                       f"ML Accuracy = {ml_accuracy:.6f}")
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get recovery information for resuming training."""
        recovery_info = {
            "session_exists": self.current_session is not None,
            "checkpoint_exists": self.current_checkpoint is not None,
            "can_resume": False,
            "resume_point": None,
            "estimated_completion": None,
            "total_progress": 0.0
        }
        
        if self.current_session and self.current_checkpoint:
            recovery_info["can_resume"] = True
            recovery_info["resume_point"] = {
                "config_index": self.current_checkpoint.current_config_index,
                "batch_id": self.current_checkpoint.current_batch_id,
                "session_id": self.current_session.session_id
            }
            recovery_info["estimated_completion"] = self.current_checkpoint.estimated_completion_time
            recovery_info["total_progress"] = self.current_checkpoint.overall_progress
        
        return recovery_info
    
    def _save_session(self):
        """Save current session to file."""
        if self.current_session:
            with open(self.session_file, 'w') as f:
                json.dump(asdict(self.current_session), f, indent=2, default=str)
    
    def _save_checkpoint(self):
        """Save current checkpoint to file."""
        if self.current_checkpoint:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(asdict(self.current_checkpoint), f, indent=2, default=str)
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save all training results."""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def generate_recovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive recovery report."""
        recovery_info = self.get_recovery_info()
        
        report = {
            "recovery_timestamp": datetime.now().isoformat(),
            "recovery_info": recovery_info,
            "session_summary": None,
            "checkpoint_summary": None,
            "recommendations": []
        }
        
        if self.current_session:
            report["session_summary"] = {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time,
                "status": self.current_session.status,
                "configs_completed": self.current_session.configs_completed,
                "total_configs": self.current_session.total_configs,
                "progress_percentage": (self.current_session.configs_completed / self.current_session.total_configs * 100) if self.current_session.total_configs > 0 else 0,
                "average_intentful_score": np.mean(self.current_session.intentful_scores) if self.current_session.intentful_scores else 0,
                "average_ml_accuracy": np.mean(self.current_session.ml_accuracies) if self.current_session.ml_accuracies else 0,
                "total_execution_time": sum(self.current_session.execution_times) if self.current_session.execution_times else 0
            }
        
        if self.current_checkpoint:
            report["checkpoint_summary"] = {
                "checkpoint_id": self.current_checkpoint.checkpoint_id,
                "timestamp": self.current_checkpoint.timestamp,
                "overall_progress": self.current_checkpoint.overall_progress,
                "estimated_completion_time": self.current_checkpoint.estimated_completion_time,
                "completed_results_count": len(self.current_checkpoint.completed_results),
                "pending_configs_count": len(self.current_checkpoint.pending_configs)
            }
        
        # Generate recommendations
        if recovery_info["can_resume"]:
            report["recommendations"].append("Resume training from checkpoint")
            report["recommendations"].append("Use smaller batch sizes for stability")
            report["recommendations"].append("Monitor power supply and weather conditions")
        else:
            report["recommendations"].append("Start new training session")
            report["recommendations"].append("Implement regular checkpointing")
            report["recommendations"].append("Use UPS for power protection")
        
        return report

def run_comprehensive_training_with_logging():
    """Run comprehensive training with full logging and recovery capabilities."""
    print("🚀 KOBA42 COMPREHENSIVE TRAINING LOGGER")
    print("=" * 60)
    print("Comprehensive training with logging and recovery")
    print("=" * 60)
    
    # Initialize logger
    logger_system = ComprehensiveTrainingLogger()
    
    # Check for existing session
    recovery_info = logger_system.get_recovery_info()
    
    if recovery_info["can_resume"]:
        print(f"🔄 RECOVERY MODE: Found existing session")
        print(f"   • Session ID: {recovery_info['resume_point']['session_id']}")
        print(f"   • Progress: {recovery_info['total_progress']:.1f}%")
        print(f"   • Estimated Completion: {recovery_info['estimated_completion']}")
        
        # Log resume
        logger_system.log_resume()
        
        # Resume from checkpoint
        resume_from_checkpoint(logger_system)
    else:
        print("🆕 STARTING NEW TRAINING SESSION")
        
        # Define configurations
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
        
        # Run training with logging
        run_training_with_logging(logger_system, configs)

def resume_from_checkpoint(logger_system: ComprehensiveTrainingLogger):
    """Resume training from checkpoint."""
    if not logger_system.current_checkpoint:
        print("❌ No checkpoint available for resume")
        return
    
    checkpoint = logger_system.current_checkpoint
    
    print(f"🔄 RESUMING FROM CHECKPOINT {checkpoint.checkpoint_id}")
    print(f"   • Config Index: {checkpoint.current_config_index}")
    print(f"   • Batch ID: {checkpoint.current_batch_id}")
    print(f"   • Progress: {checkpoint.overall_progress:.1f}%")
    
    # Convert pending configs back to BatchF2Config objects
    pending_configs = []
    for config_data in checkpoint.pending_configs:
        config = BatchF2Config(**config_data)
        pending_configs.append(config)
    
    # Resume training
    run_training_with_logging(logger_system, pending_configs, 
                            start_config_index=checkpoint.current_config_index)

def run_training_with_logging(logger_system: ComprehensiveTrainingLogger, 
                            configs: List[BatchF2Config], 
                            start_config_index: int = 0):
    """Run training with comprehensive logging."""
    all_results = []
    
    try:
        for i, config in enumerate(configs[start_config_index:], start=start_config_index):
            print(f"\n🔧 RUNNING OPTIMIZATION {i+1}/{len(configs)}")
            print(f"Matrix Size: {config.matrix_size}")
            print(f"Batch Size: {config.batch_size}")
            print(f"Optimization Level: {config.optimization_level}")
            
            # Create optimizer
            optimizer = BatchF2MatrixOptimizer(config)
            
            # Run optimization
            results = optimizer.run_batch_optimization()
            all_results.append(results)
            
            # Log result
            logger_system.log_result(i, results)
            
            # Create checkpoint
            pending_configs = [asdict(c) for c in configs[i+1:]]
            logger_system.create_checkpoint(i, 0, all_results, pending_configs)
            
            # Display results
            print(f"\n📊 OPTIMIZATION {i+1} RESULTS:")
            print(f"   • Intentful Score: {results['batch_optimization_results']['average_intentful_score']:.6f}")
            print(f"   • ML Accuracy: {results['ml_training_results']['average_accuracy']:.6f}")
            print(f"   • Execution Time: {results['overall_performance']['total_execution_time']:.2f}s")
            print(f"   • Total Batches: {results['batch_optimization_results']['total_batches']}")
            print(f"   • Total ML Models: {results['ml_training_results']['total_models_trained']}")
        
        # Training completed successfully
        logger_system.log_completion()
        logger_system.save_results(all_results)
        
        # Generate final report
        final_report = logger_system.generate_recovery_report()
        
        # Save final report
        with open(logger_system.log_dir / "final_training_report.json", 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n✅ COMPREHENSIVE TRAINING COMPLETED")
        print("🔧 Matrix Optimization: SUCCESSFUL")
        print("🤖 ML Training: COMPLETED")
        print("🧮 Intentful Mathematics: OPTIMIZED")
        print("🏆 KOBA42 Excellence: ACHIEVED")
        print(f"📋 Final Report: {logger_system.log_dir / 'final_training_report.json'}")
        
    except KeyboardInterrupt:
        print("\n⚠️ TRAINING INTERRUPTED BY USER")
        logger_system.log_interruption("User interruption")
        raise
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        logger_system.log_interruption(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Run comprehensive training with logging
    run_comprehensive_training_with_logging()
