#!/bin/bash
# Möbius Loop Master Restart Script
# Continuously runs Möbius loop iterations

echo "🌌 Möbius Loop Master Restart Script"
echo "🔄 Continuous Evolution Engine"
echo "=================================="

iteration=1
while true; do
    echo "🚀 Starting Möbius Loop Iteration #$iteration"

    # Generate and run iteration-specific script
    python -c "
import subprocess
import time
result = subprocess.run(['bash', f'moebius_restart_iteration_{iteration}.sh'], capture_output=True, text=True)
print(f'Iteration {iteration} exit code: {result.returncode}')
if result.stdout:
    print(f'Iteration {iteration} output: {result.stdout}')
if result.stderr:
    print(f'Iteration {iteration} errors: {result.stderr}')
"

    echo "✅ Möbius Loop Iteration #$iteration completed"

    # Brief pause between iterations
    sleep 5

    # Increment iteration counter
    ((iteration++))

    echo "🔄 Preparing Möbius Loop Iteration #$iteration"
done

echo "⚠️  Möbius Loop Master Script terminated"
