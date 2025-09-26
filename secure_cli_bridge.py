#!/usr/bin/env python3
"""
SquashPlot Secure CLI Bridge
============================

A secure, easy-to-use bridge that allows users to execute CLI commands locally
while maintaining security and providing a professional interface.

Features:
- Secure command execution with validation
- Professional CLI templates
- Local file system integration
- Security sandboxing
- Easy deployment and usage
"""

import os
import sys
import json
import subprocess
import shlex
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecureCLIBridge:
    """Secure CLI Bridge for SquashPlot operations"""
    
    def __init__(self, config_file: str = "cli_bridge_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.allowed_commands = self._get_allowed_commands()
        self.working_directory = Path.cwd()
        
    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        default_config = {
            "security": {
                "max_execution_time": 300,  # 5 minutes
                "allowed_directories": ["./", "/tmp/", "/plots/"],
                "blocked_commands": ["rm", "del", "format", "fdisk", "mkfs"],
                "require_confirmation": True
            },
            "squashplot": {
                "executable": "python",
                "main_script": "main.py",
                "compression_script": "squashplot.py",
                "check_script": "check_server.py"
            },
            "templates": {
                "web_interface": "python main.py --web",
                "cli_mode": "python main.py --cli", 
                "demo_mode": "python main.py --demo",
                "server_check": "python check_server.py",
                "basic_plotting": "python squashplot.py -t /tmp/plot1 -d /plots -f {farmer_key} -p {pool_key}",
                "compressed_plotting": "python squashplot.py --compress 3 -t /tmp/plot1 -d /plots -f {farmer_key} -p {pool_key}",
                "batch_compression": "python squashplot.py --batch --input-dir {input_dir} --output-dir {output_dir} --level {level}"
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                
        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        return default_config
    
    def _get_allowed_commands(self) -> List[str]:
        """Get list of allowed commands for security"""
        return [
            "python", "python3",
            "main.py", "squashplot.py", "check_server.py",
            "squashplot_benchmark.py", "compression_validator.py",
            "ls", "dir", "pwd", "cd", "mkdir", "cp", "copy",
            "cat", "type", "head", "tail", "grep", "find"
        ]
    
    def validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate command for security"""
        try:
            # Parse command
            parts = shlex.split(command)
            if not parts:
                return False, "Empty command"
            
            # Check if command is in allowed list
            base_command = parts[0]
            if not any(allowed in base_command for allowed in self.allowed_commands):
                return False, f"Command '{base_command}' not allowed"
            
            # Check for blocked commands
            blocked = self.config["security"]["blocked_commands"]
            if any(blocked_cmd in command.lower() for blocked_cmd in blocked):
                return False, f"Command contains blocked operation"
            
            # Check directory restrictions
            allowed_dirs = self.config["security"]["allowed_directories"]
            for part in parts:
                if "/" in part or "\\" in part:
                    path = Path(part).resolve()
                    if not any(str(path).startswith(allowed) for allowed in allowed_dirs):
                        return False, f"Path '{part}' not in allowed directories"
            
            return True, "Command validated"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def execute_command(self, command: str, timeout: Optional[int] = None) -> Dict:
        """Execute command securely"""
        # Validate command
        is_valid, message = self.validate_command(command)
        if not is_valid:
            return {
                "success": False,
                "error": message,
                "output": "",
                "exit_code": -1
            }
        
        # Set timeout
        if timeout is None:
            timeout = self.config["security"]["max_execution_time"]
        
        try:
            logger.info(f"Executing command: {command}")
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode,
                "command": command
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "output": "",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1
            }
    
    def get_templates(self) -> Dict[str, str]:
        """Get available command templates"""
        return self.config["templates"]
    
    def execute_template(self, template_name: str, **kwargs) -> Dict:
        """Execute a command template with parameters"""
        templates = self.get_templates()
        
        if template_name not in templates:
            return {
                "success": False,
                "error": f"Template '{template_name}' not found",
                "output": "",
                "exit_code": -1
            }
        
        # Format template with parameters
        try:
            command = templates[template_name].format(**kwargs)
            return self.execute_command(command)
        except KeyError as e:
            return {
                "success": False,
                "error": f"Missing parameter: {e}",
                "output": "",
                "exit_code": -1
            }
    
    def interactive_mode(self):
        """Start interactive CLI mode"""
        print("🧠 SquashPlot Secure CLI Bridge")
        print("=" * 50)
        print("Available commands:")
        print("1. templates - Show available templates")
        print("2. execute <command> - Execute a command")
        print("3. template <name> - Execute a template")
        print("4. help - Show this help")
        print("5. exit - Exit the bridge")
        print()
        
        while True:
            try:
                user_input = input("squashplot> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye! 👋")
                    break
                    
                elif user_input.lower() == 'help':
                    print("Available commands:")
                    print("1. templates - Show available templates")
                    print("2. execute <command> - Execute a command")
                    print("3. template <name> - Execute a template")
                    print("4. help - Show this help")
                    print("5. exit - Exit the bridge")
                    
                elif user_input.lower() == 'templates':
                    templates = self.get_templates()
                    print("\nAvailable Templates:")
                    for name, template in templates.items():
                        print(f"  {name}: {template}")
                    print()
                    
                elif user_input.startswith('execute '):
                    command = user_input[8:].strip()
                    result = self.execute_command(command)
                    self._print_result(result)
                    
                elif user_input.startswith('template '):
                    template_name = user_input[9:].strip()
                    result = self.execute_template(template_name)
                    self._print_result(result)
                    
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_result(self, result: Dict):
        """Print command execution result"""
        if result["success"]:
            print("✅ Command executed successfully")
        else:
            print("❌ Command failed")
            
        if result["output"]:
            print("Output:")
            print(result["output"])
            
        if result["error"]:
            print("Error:")
            print(result["error"])
            
        print(f"Exit code: {result['exit_code']}")
        print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SquashPlot Secure CLI Bridge")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start interactive mode")
    parser.add_argument("--execute", "-e", type=str, 
                       help="Execute a single command")
    parser.add_argument("--template", "-t", type=str, 
                       help="Execute a template")
    parser.add_argument("--list-templates", "-l", action="store_true",
                       help="List available templates")
    parser.add_argument("--config", "-c", type=str, default="cli_bridge_config.json",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize bridge
    bridge = SecureCLIBridge(args.config)
    
    if args.list_templates:
        templates = bridge.get_templates()
        print("Available Templates:")
        for name, template in templates.items():
            print(f"  {name}: {template}")
        return
    
    if args.execute:
        result = bridge.execute_command(args.execute)
        bridge._print_result(result)
        return
        
    if args.template:
        result = bridge.execute_template(args.template)
        bridge._print_result(result)
        return
    
    # Default to interactive mode
    bridge.interactive_mode()

if __name__ == "__main__":
    main()
