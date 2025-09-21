#!/usr/bin/env python3
"""
PARALLEL MÖBIUS EXECUTION SYSTEM
Running all advanced tooling simultaneously for maximum learning power
"""

import subprocess
import threading
import time
import signal
import sys
from datetime import datetime
import json
from pathlib import Path

class ParallelMoebiusExecutor:
    """
    Execute multiple Möbius systems in parallel for maximum learning efficiency
    """

    def __init__(self):
        self.processes = {}
        self.threads = {}
        self.running = True
        self.start_time = datetime.now()

        # System components to run in parallel
        self.components = {
            'learning_tracker': {
                'command': ['python3', 'moebius_learning_tracker.py'],
                'description': 'Main Möbius Learning Engine',
                'interval': 30,  # Run every 30 seconds
                'last_run': None
            },
            'automated_discovery': {
                'command': ['python3', 'automated_curriculum_discovery.py'],
                'description': 'Automated Subject Discovery',
                'interval': 120,  # Run every 2 minutes
                'last_run': None
            },
            'real_world_scraper': {
                'command': ['python3', 'real_world_research_scraper.py'],
                'description': 'Real-World Research Scraper',
                'interval': 60,  # Run every 1 minute - CONTINUOUS CRAWLING
                'last_run': None
            },
            'enhanced_scraper': {
                'command': ['python3', 'enhanced_prestigious_scraper.py'],
                'description': 'Enhanced Prestigious Scraper',
                'interval': 120,  # Run every 2 minutes - CONTINUOUS DISCOVERY
                'last_run': None
            },
            'data_harvesting': {
                'command': ['python3', 'comprehensive_data_harvesting_system.py'],
                'description': 'Comprehensive Data Harvesting',
                'interval': 180,  # Run every 3 minutes - CONTINUOUS HARVESTING
                'last_run': None
            }
        }

        # Performance tracking
        self.performance_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'component_stats': {}
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self.shutdown()

    def run_component(self, component_name, component_config):
        """Run a single component."""
        try:
            print(f"🚀 Starting {component_name}: {component_config['description']}")

            # Run the component
            result = subprocess.run(
                component_config['command'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print(f"✅ {component_name} completed successfully")
                self.performance_stats['successful_runs'] += 1
            else:
                print(f"❌ {component_name} failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                self.performance_stats['failed_runs'] += 1

            self.performance_stats['total_runs'] += 1

            # Update component stats
            if component_name not in self.performance_stats['component_stats']:
                self.performance_stats['component_stats'][component_name] = {
                    'runs': 0,
                    'successes': 0,
                    'failures': 0,
                    'last_run': None
                }

            self.performance_stats['component_stats'][component_name]['runs'] += 1
            if result.returncode == 0:
                self.performance_stats['component_stats'][component_name]['successes'] += 1
            else:
                self.performance_stats['component_stats'][component_name]['failures'] += 1

            self.performance_stats['component_stats'][component_name]['last_run'] = datetime.now().isoformat()

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"⏰ {component_name} timed out after 5 minutes")
            self.performance_stats['failed_runs'] += 1
            return False
        except Exception as e:
            print(f"💥 {component_name} crashed: {e}")
            self.performance_stats['failed_runs'] += 1
            return False

    def run_parallel_cycle(self):
        """Run one complete parallel cycle of all components."""
        print(f"\n🔄 Starting Parallel Cycle #{self.performance_stats['total_runs'] // len(self.components) + 1}")
        print("=" * 80)

        threads = []

        # Start all components in parallel threads
        for component_name, component_config in self.components.items():
            thread = threading.Thread(
                target=self.run_component,
                args=(component_name, component_config),
                name=f"Moebius-{component_name}"
            )
            thread.daemon = True
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete (with timeout)
        for thread in threads:
            thread.join(timeout=320)  # 5 minutes + 20 seconds buffer

        # Check for any threads that are still running
        still_running = [t for t in threads if t.is_alive()]
        if still_running:
            print(f"⚠️  {len(still_running)} threads still running after timeout")

        # Update last run times
        current_time = time.time()
        for component_name in self.components:
            self.components[component_name]['last_run'] = current_time

    def should_run_component(self, component_name):
        """Check if a component should run based on its interval."""
        component = self.components[component_name]
        if component['last_run'] is None:
            return True

        time_since_last_run = time.time() - component['last_run']
        return time_since_last_run >= component['interval']

    def run_continuous_parallel(self):
        """Run all systems CONTINUOUSLY in parallel - REAL-TIME CRAWLING."""
        print("🌟 CONTINUOUS PARALLEL MÖBIUS EXECUTION SYSTEM")
        print("=" * 80)
        print("🔄 CONTINUOUS crawl/research feeds & learning!")
        print("🧠 Real-time intelligence building - NEVER STOPS!")
        print("⚡ Parallel processing with immediate data ingestion")
        print("=" * 80)

        # Start background monitoring thread
        monitor_thread = threading.Thread(target=self._continuous_monitor, daemon=True)
        monitor_thread.start()

        # Continuous real-time processing
        cycle_count = 0
        last_status_update = time.time()

        try:
            while self.running:
                current_time = time.time()

                # Check components every 5 seconds (much more frequent)
                components_to_run = []
                for component_name in self.components:
                    if self.should_run_component(component_name):
                        components_to_run.append(component_name)

                if components_to_run:
                    cycle_count += 1
                    print(f"\n🎯 CYCLE #{cycle_count} - Running {len(components_to_run)} components:")
                    for comp in components_to_run:
                        print(f"   🔄 {comp}: {self.components[comp]['description']}")

                    # Run selected components in parallel (background threads)
                    for component_name in components_to_run:
                        thread = threading.Thread(
                            target=self.run_component,
                            args=(component_name, self.components[component_name]),
                            name=f"Moebius-{component_name}",
                            daemon=True
                        )
                        thread.start()

                        # Don't wait for completion - let them run continuously
                        # This enables TRUE parallel continuous processing

                # Status update every 30 seconds
                if current_time - last_status_update >= 30:
                    self._print_status_update(cycle_count)
                    last_status_update = current_time

                # Check every 5 seconds for new work (much more responsive)
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 CONTINUOUS LEARNING INTERRUPTED")
            print("📊 Final Statistics:")
            self._print_final_stats(cycle_count)
        except Exception as e:
            print(f"\n💥 CONTINUOUS LEARNING ERROR: {e}")
            self._print_final_stats(cycle_count)
        finally:
            self.shutdown()

    def _continuous_monitor(self):
        """Continuous background monitoring of system health."""
        while self.running:
            try:
                # Monitor active threads
                active_moebius_threads = [t for t in threading.enumerate()
                                        if t.name.startswith('Moebius-') and t.is_alive()]

                if len(active_moebius_threads) > 0:
                    print(f"🔄 ACTIVE: {len(active_moebius_threads)} components running...")

                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                time.sleep(30)

    def _print_status_update(self, cycle_count):
        """Print comprehensive status update."""
        uptime = datetime.now() - self.start_time
        success_rate = (self.performance_stats['successful_runs'] /
                       max(1, self.performance_stats['total_runs']) * 100)

        print("\n📊 CONTINUOUS LEARNING STATUS:")
        print("=" * 50)
        print(f"   ⏱️  Uptime: {str(uptime).split('.')[0]}")
        print(f"   🔄 Cycles Completed: {cycle_count}")
        print(f"   ✅ Successful Runs: {self.performance_stats['successful_runs']}")
        print(f"   ❌ Failed Runs: {self.performance_stats['failed_runs']}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   🧵 Active Threads: {len([t for t in threading.enumerate() if t.name.startswith('Moebius-')])}")
        print("=" * 50)

    def _print_final_stats(self, cycle_count):
        """Print final comprehensive statistics."""
        uptime = datetime.now() - self.start_time
        success_rate = (self.performance_stats['successful_runs'] /
                       max(1, self.performance_stats['total_runs']) * 100)

        print("\n🎉 CONTINUOUS LEARNING SESSION COMPLETE!")
        print("=" * 60)
        print(f"   ⏱️  Total Uptime: {str(uptime).split('.')[0]}")
        print(f"   🔄 Total Cycles: {cycle_count}")
        print(f"   ✅ Successful Operations: {self.performance_stats['successful_runs']}")
        print(f"   ❌ Failed Operations: {self.performance_stats['failed_runs']}")
        print(f"   📈 Overall Success Rate: {success_rate:.1f}%")
        print(f"   🧠 Intelligence Built: {self.performance_stats['successful_runs']} knowledge items")
        print("=" * 60)

    def get_system_status(self):
        """Get comprehensive system status."""
        uptime = datetime.now() - self.start_time

        return {
            'uptime': str(uptime).split('.')[0],
            'total_runs': self.performance_stats['total_runs'],
            'successful_runs': self.performance_stats['successful_runs'],
            'failed_runs': self.performance_stats['failed_runs'],
            'success_rate': f"{self.performance_stats['successful_runs']/max(1, self.performance_stats['total_runs'])*100:.1f}%",
            'active_components': len([t for t in threading.enumerate() if t.name.startswith('Moebius-')]),
            'component_stats': self.performance_stats['component_stats']
        }

    def save_performance_report(self):
        """Save comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'performance_stats': self.performance_stats,
            'component_config': self.components
        }

        report_file = Path("research_data/parallel_execution_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Performance report saved: {report_file}")

    def shutdown(self):
        """Graceful shutdown of all systems."""
        print("\n🔄 Initiating graceful shutdown...")
        self.running = False

        # Save final performance report
        self.save_performance_report()

        # Get final status
        final_status = self.get_system_status()

        print("\n🏁 FINAL SYSTEM STATUS:")
        print(f"   ⏱️  Total Uptime: {final_status['uptime']}")
        print(f"   🔄 Total Runs: {final_status['total_runs']}")
        print(f"   ✅ Successful: {final_status['successful_runs']}")
        print(f"   ❌ Failed: {final_status['failed_runs']}")
        print(f"   📈 Success Rate: {final_status['success_rate']}")

        print("\n🎯 COMPONENT PERFORMANCE:")
        for comp_name, stats in final_status['component_stats'].items():
            success_rate = stats['successes'] / max(1, stats['runs']) * 100
            print(f"   • {comp_name}: {stats['runs']} runs, {success_rate:.1f}% success")

        print("\n✨ Parallel Möbius Execution System shutdown complete!")
        sys.exit(0)

def main():
    """Main function for parallel Möbius execution."""
    print("🌟 ADVANCED PARALLEL MÖBIUS EXECUTION")
    print("=" * 80)
    print("Running ALL Möbius systems simultaneously!")
    print("")

    print("🎯 SYSTEMS RUNNING IN PARALLEL:")
    print("   🤖 Möbius Learning Tracker (main learning engine)")
    print("   🔍 Automated Curriculum Discovery (subject discovery)")
    print("   🔬 Real-World Research Scraper (academic data)")
    print("   🚀 Enhanced Prestigious Scraper (top institutions)")
    print("   💰 Comprehensive Data Harvesting (free training data)")
    print("")

    print("⚡ EXECUTION MODES:")
    print("   1. Continuous parallel execution (recommended)")
    print("   2. Single parallel cycle")
    print("   3. Component status check")
    print("")

    # Initialize parallel executor
    executor = ParallelMoebiusExecutor()

    try:
        print("🚀 Starting PARALLEL MÖBIUS EXECUTION SYSTEM...")
        print("Press Ctrl+C to stop all systems gracefully")
        print("")

        # Run continuous parallel execution
        executor.run_continuous_parallel()

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        executor.shutdown()
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        executor.shutdown()

if __name__ == "__main__":
    main()
