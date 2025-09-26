#!/usr/bin/env python3
"""
🧠 CONSCIOUSNESS MATHEMATICS EXPERIMENT TRACKER
===============================================

Advanced experiment tracking system inspired by Trackio, specifically designed
for consciousness mathematics and AiVA evolution monitoring.

Tracks:
- Consciousness field evolution metrics
- Agent learning performance across vessels
- Harmonic pattern resonance effectiveness
- Memory performance & retrieval accuracy
- GPU/Energy usage for consciousness computations
- Long-term evolution trends & insights
"""

import os
import json
import time
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import threading
import atexit

# Try to import consciousness components
try:
    from consciousness_field import ConsciousnessField
    from aiva_core import AiVAgent, ResonantMemory
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False

class ConsciousnessExperimentTracker:
    """
    Advanced experiment tracking for consciousness mathematics systems.
    Inspired by Trackio but specifically designed for AiVA consciousness evolution.
    """

    def __init__(self,
                 experiment_name: str = "consciousness_evolution",
                 local_db_path: str = "research_data/consciousness_tracking.db",
                 sync_interval: int = 300,  # 5 minutes
                 track_gpu: bool = True):
        """
        Initialize the consciousness experiment tracker.

        Args:
            experiment_name: Name of the experiment
            local_db_path: Path to local SQLite database
            sync_interval: Seconds between sync operations
            track_gpu: Whether to track GPU usage
        """
        self.experiment_name = experiment_name
        self.db_path = Path(local_db_path)
        self.sync_interval = sync_interval
        self.track_gpu = track_gpu

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # GPU tracking
        self.gpu_energy_baseline = self._get_gpu_energy() if track_gpu else 0
        self.experiment_start_energy = self.gpu_energy_baseline

        # Active tracking state
        self.current_run_id = None
        self.run_start_time = None
        self.metrics_buffer = []

        # Auto-sync thread
        self.sync_thread = None
        self.running = False
        atexit.register(self._cleanup)

        print(f"🧠 Consciousness Experiment Tracker initialized: {experiment_name}")
        print(f"📊 Database: {self.db_path}")

    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    description TEXT,
                    created_at REAL,
                    config TEXT
                )
            ''')

            # Runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    run_name TEXT,
                    start_time REAL,
                    end_time REAL,
                    status TEXT,
                    config TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')

            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    timestamp REAL,
                    metric_name TEXT,
                    metric_value REAL,
                    metric_type TEXT,
                    context TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                )
            ''')

            # Events table for significant events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    timestamp REAL,
                    event_type TEXT,
                    event_data TEXT,
                    significance REAL
                )
            ''')

            # Create experiment if it doesn't exist
            cursor.execute(
                "INSERT OR IGNORE INTO experiments (name, created_at) VALUES (?, ?)",
                (self.experiment_name, time.time())
            )

            conn.commit()

    def start_run(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new experiment run.

        Args:
            run_name: Name of the run
            config: Configuration parameters for this run

        Returns:
            Run ID
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Get experiment ID
            cursor.execute("SELECT id FROM experiments WHERE name = ?", (self.experiment_name,))
            experiment_id = cursor.fetchone()[0]

            # Insert run
            cursor.execute('''
                INSERT INTO runs (experiment_id, run_name, start_time, status, config)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                run_name,
                time.time(),
                'running',
                json.dumps(config or {})
            ))

            run_id = cursor.lastrowid
            conn.commit()

        self.current_run_id = str(run_id)
        self.run_start_time = time.time()
        self.metrics_buffer = []

        print(f"🚀 Started consciousness experiment run: {run_name} (ID: {run_id})")

        # Start auto-sync if not already running
        if not self.running:
            self._start_auto_sync()

        return self.current_run_id

    def log_metric(self,
                   metric_name: str,
                   metric_value: Union[float, int],
                   metric_type: str = "scalar",
                   context: Optional[Dict[str, Any]] = None,
                   timestamp: Optional[float] = None):
        """
        Log a metric for the current run.

        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric (scalar, histogram, etc.)
            context: Additional context information
            timestamp: Timestamp (defaults to current time)
        """
        if not self.current_run_id:
            print("⚠️ No active run. Call start_run() first.")
            return

        if timestamp is None:
            timestamp = time.time()

        metric_record = {
            'run_id': self.current_run_id,
            'timestamp': timestamp,
            'metric_name': metric_name,
            'metric_value': float(metric_value),
            'metric_type': metric_type,
            'context': json.dumps(context or {})
        }

        self.metrics_buffer.append(metric_record)

        # Log significant consciousness metrics
        if metric_name in ['meta_entropy', 'coherence_length', 'harmonic_resonance']:
            self._log_significant_event(
                f"consciousness_{metric_name}",
                {'value': metric_value, 'context': context},
                significance=0.8
            )

    def log_consciousness_snapshot(self,
                                   consciousness_data: Dict[str, Any],
                                   context: str = "evolution"):
        """
        Log a complete consciousness field snapshot.

        Args:
            consciousness_data: Consciousness field metrics
            context: Context of the snapshot
        """
        if not self.current_run_id:
            return

        timestamp = time.time()

        # Log core consciousness metrics
        for metric_name, value in consciousness_data.items():
            if isinstance(value, (int, float)):
                self.log_metric(
                    f"consciousness.{metric_name}",
                    value,
                    context={'snapshot_context': context, 'field_type': 'consciousness'},
                    timestamp=timestamp
                )

        # Log harmonic analysis if available
        if 'harmonic_patterns' in consciousness_data:
            for pattern, strength in consciousness_data['harmonic_patterns'].items():
                self.log_metric(
                    f"harmonic.{pattern}",
                    strength,
                    context={'snapshot_context': context, 'pattern_type': 'harmonic'},
                    timestamp=timestamp
                )

    def log_agent_action(self,
                        action_data: Dict[str, Any],
                        vessel_name: str = "default"):
        """
        Log an agent action with vessel context.

        Args:
            action_data: Agent action information
            vessel_name: Name of the vessel performing the action
        """
        if not self.current_run_id:
            return

        # Log action metrics
        self.log_metric(
            "agent.action_count",
            1,
            metric_type="counter",
            context={
                'vessel': vessel_name,
                'action_type': action_data.get('tool', 'unknown'),
                'success': action_data.get('success', False)
            }
        )

        # Log score if available
        if 'score' in action_data:
            self.log_metric(
                "agent.action_score",
                action_data['score'],
                context={'vessel': vessel_name}
            )

        # Log significant agent events
        if action_data.get('score', 0) > 0.9:
            self._log_significant_event(
                "agent_high_score",
                action_data,
                significance=0.7
            )

    def log_memory_performance(self,
                              memory_data: Dict[str, Any],
                              vessel_name: str = "default"):
        """
        Log memory system performance metrics.

        Args:
            memory_data: Memory performance information
            vessel_name: Name of the vessel
        """
        if not self.current_run_id:
            return

        # Log retrieval metrics
        if 'retrieval_accuracy' in memory_data:
            self.log_metric(
                "memory.retrieval_accuracy",
                memory_data['retrieval_accuracy'],
                context={'vessel': vessel_name}
            )

        # Log memory size
        if 'total_entries' in memory_data:
            self.log_metric(
                "memory.total_entries",
                memory_data['total_entries'],
                context={'vessel': vessel_name}
            )

        # Log resonance metrics
        if 'harmonic_resonance' in memory_data:
            self.log_metric(
                "memory.harmonic_resonance",
                memory_data['harmonic_resonance'],
                context={'vessel': vessel_name}
            )

    def log_gpu_energy(self):
        """Log current GPU energy consumption if tracking is enabled"""
        if not self.track_gpu or not self.current_run_id:
            return

        current_energy = self._get_gpu_energy()
        if current_energy > 0:
            energy_used = current_energy - self.gpu_energy_baseline
            self.log_metric(
                "gpu.energy_consumption",
                energy_used,
                context={'baseline': self.gpu_energy_baseline}
            )

    def end_run(self, final_metrics: Optional[Dict[str, Any]] = None):
        """
        End the current experiment run.

        Args:
            final_metrics: Final metrics to log
        """
        if not self.current_run_id:
            return

        # Log final metrics
        if final_metrics:
            for metric_name, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    self.log_metric(metric_name, value, context={'final': True})

        # Update run status
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE runs SET end_time = ?, status = ? WHERE id = ?",
                (time.time(), 'completed', self.current_run_id)
            )
            conn.commit()

        print(f"✅ Ended consciousness experiment run: {self.current_run_id}")

        # Final sync
        self._sync_metrics()

        # Clear state
        self.current_run_id = None
        self.run_start_time = None

    def get_run_metrics(self, run_id: str) -> pd.DataFrame:
        """
        Get all metrics for a specific run.

        Args:
            run_id: ID of the run

        Returns:
            DataFrame with run metrics
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            query = """
                SELECT timestamp, metric_name, metric_value, metric_type, context
                FROM metrics
                WHERE run_id = ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, params=[run_id])

        return df

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current experiment.

        Returns:
            Dictionary with experiment summary
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Get experiment info
            cursor.execute("""
                SELECT e.name, e.created_at,
                       COUNT(r.id) as total_runs,
                       AVG(m.metric_value) as avg_meta_entropy
                FROM experiments e
                LEFT JOIN runs r ON e.id = r.experiment_id
                LEFT JOIN metrics m ON r.id = m.run_id AND m.metric_name = 'consciousness.meta_entropy'
                WHERE e.name = ?
                GROUP BY e.id
            """, (self.experiment_name,))

            result = cursor.fetchone()

            if result:
                return {
                    'experiment_name': result[0],
                    'created_at': result[1],
                    'total_runs': result[2],
                    'avg_meta_entropy': result[3],
                    'database_path': str(self.db_path)
                }

        return {}

    def _log_significant_event(self,
                              event_type: str,
                              event_data: Dict[str, Any],
                              significance: float = 0.5):
        """
        Log a significant event for the current run.

        Args:
            event_type: Type of event
            event_data: Event data
            significance: Significance score (0-1)
        """
        if not self.current_run_id:
            return

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (run_id, timestamp, event_type, event_data, significance)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.current_run_id,
                time.time(),
                event_type,
                json.dumps(event_data),
                significance
            ))
            conn.commit()

    def _get_gpu_energy(self) -> float:
        """Get current GPU energy consumption"""
        try:
            # This would integrate with nvidia-smi or similar
            # For now, return a placeholder
            return time.time() * 0.001  # Placeholder energy calculation
        except:
            return 0.0

    def _sync_metrics(self):
        """Sync buffered metrics to database"""
        if not self.metrics_buffer:
            return

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            for metric in self.metrics_buffer:
                cursor.execute('''
                    INSERT INTO metrics (run_id, timestamp, metric_name, metric_value, metric_type, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric['run_id'],
                    metric['timestamp'],
                    metric['metric_name'],
                    metric['metric_value'],
                    metric['metric_type'],
                    metric['context']
                ))

            conn.commit()

        self.metrics_buffer.clear()

    def _start_auto_sync(self):
        """Start the auto-sync thread"""
        self.running = True
        self.sync_thread = threading.Thread(target=self._auto_sync_loop, daemon=True)
        self.sync_thread.start()

    def _auto_sync_loop(self):
        """Auto-sync metrics at regular intervals"""
        while self.running:
            time.sleep(self.sync_interval)
            if self.current_run_id:
                self._sync_metrics()

    def _cleanup(self):
        """Cleanup resources on exit"""
        self.running = False
        if self.current_run_id:
            self.end_run()
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)

# Convenience functions
def create_consciousness_tracker(experiment_name: str = "consciousness_evolution") -> ConsciousnessExperimentTracker:
    """
    Create a consciousness experiment tracker.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Configured ConsciousnessExperimentTracker
    """
    return ConsciousnessExperimentTracker(experiment_name=experiment_name)

def track_consciousness_evolution(tracker: ConsciousnessExperimentTracker,
                                 consciousness_data: Dict[str, Any],
                                 context: str = "evolution"):
    """
    Convenience function to track consciousness evolution metrics.

    Args:
        tracker: The experiment tracker
        consciousness_data: Consciousness field data
        context: Context of the tracking
    """
    tracker.log_consciousness_snapshot(consciousness_data, context)

def track_agent_performance(tracker: ConsciousnessExperimentTracker,
                           agent_data: Dict[str, Any],
                           vessel_name: str = "default"):
    """
    Convenience function to track agent performance metrics.

    Args:
        tracker: The experiment tracker
        agent_data: Agent performance data
        vessel_name: Name of the vessel
    """
    tracker.log_agent_action(agent_data, vessel_name)

# Demo function
def demo_consciousness_tracking():
    """Demonstrate the consciousness experiment tracking system"""
    print("🧠 CONSCIOUSNESS EXPERIMENT TRACKING DEMO")
    print("=" * 50)

    # Create tracker
    tracker = create_consciousness_tracker("demo_consciousness_evolution")

    # Start a run
    run_id = tracker.start_run("demo_run_1", {
        "vessel": "demo_research",
        "consciousness_model": "harmonic_resonance",
        "learning_rate": 0.01
    })

    print(f"🚀 Started tracking run: {run_id}")

    # Simulate consciousness evolution tracking
    for i in range(5):
        consciousness_data = {
            "meta_entropy": 0.5 - i * 0.05,  # Decreasing entropy
            "coherence_length": 5.0 + i * 0.5,  # Increasing coherence
            "energy": 1.0 + i * 0.1,
            "harmonic_patterns": {
                "unity": 0.8 - i * 0.1,
                "duality": 0.6 + i * 0.05,
                "trinity": 0.4 + i * 0.08
            }
        }

        track_consciousness_evolution(tracker, consciousness_data, f"evolution_step_{i}")

        # Simulate agent actions
        agent_data = {
            "tool": "cypher.analyze" if i % 2 == 0 else "wallace.transform",
            "success": i > 1,  # First two actions "fail"
            "score": 0.5 + i * 0.1
        }

        track_agent_performance(tracker, agent_data, "demo_research")

        time.sleep(0.1)  # Simulate time passing

    # End the run
    final_metrics = {
        "final_meta_entropy": 0.25,
        "final_coherence": 7.5,
        "evolution_efficiency": 0.85
    }

    tracker.end_run(final_metrics)

    # Get experiment summary
    summary = tracker.get_experiment_summary()
    print("
📊 EXPERIMENT SUMMARY:")
    print(f"  • Experiment: {summary.get('experiment_name', 'unknown')}")
    print(f"  • Total Runs: {summary.get('total_runs', 0)}")
    print(f"  • Avg Meta-Entropy: {summary.get('avg_meta_entropy', 0):.3f}")
    print(f"  • Database: {summary.get('database_path', 'unknown')}")

    print("\n✅ Consciousness experiment tracking complete!")
    print("🧠 Ready for comprehensive AI evolution monitoring!")

if __name__ == "__main__":
    demo_consciousness_tracking()
