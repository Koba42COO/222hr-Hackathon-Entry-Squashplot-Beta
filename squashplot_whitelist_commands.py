#!/usr/bin/env python3
"""
SquashPlot Whitelist Commands
Comprehensive whitelist of approved commands for the bridge app
"""

import json
from typing import Dict, List, Set
from datetime import datetime

class SquashPlotWhitelist:
    """Comprehensive whitelist of approved SquashPlot commands"""
    
    def __init__(self):
        self.whitelist_commands = {
            "dashboard_commands": {
                "description": "Dashboard and interface commands",
                "commands": [
                    # Basic SquashPlot commands
                    "squashplot --help",
                    "squashplot --version",
                    "squashplot --status",
                    "squashplot --info",
                    "squashplot --config",
                    "squashplot --list-plots",
                    "squashplot --list-farms",
                    
                    # Dashboard status commands
                    "squashplot dashboard --status",
                    "squashplot dashboard --info",
                    "squashplot dashboard --health",
                    "squashplot dashboard --metrics",
                    "squashplot dashboard --logs",
                    "squashplot dashboard --version",
                    
                    # System information
                    "squashplot system --info",
                    "squashplot system --status",
                    "squashplot system --health",
                    "squashplot system --disk-usage",
                    "squashplot system --memory-usage",
                    "squashplot system --cpu-usage",
                    "squashplot system --network-status",
                    
                    # Configuration management
                    "squashplot config --show",
                    "squashplot config --list",
                    "squashplot config --validate",
                    "squashplot config --backup",
                    "squashplot config --restore",
                    
                    # Logging and monitoring
                    "squashplot logs --show",
                    "squashplot logs --tail",
                    "squashplot logs --clear",
                    "squashplot logs --export",
                    "squashplot monitor --start",
                    "squashplot monitor --stop",
                    "squashplot monitor --status"
                ]
            },
            
            "plotting_commands": {
                "description": "Plot creation and management commands",
                "commands": [
                    # Plot creation
                    "squashplot plot --create",
                    "squashplot plot --create --size 32",
                    "squashplot plot --create --size 64",
                    "squashplot plot --create --size 128",
                    "squashplot plot --create --size 256",
                    "squashplot plot --create --size 512",
                    "squashplot plot --create --size 1024",
                    
                    # Plot management
                    "squashplot plot --list",
                    "squashplot plot --status",
                    "squashplot plot --info",
                    "squashplot plot --validate",
                    "squashplot plot --check",
                    "squashplot plot --verify",
                    "squashplot plot --optimize",
                    
                    # Plot operations
                    "squashplot plot --compress",
                    "squashplot plot --decompress",
                    "squashplot plot --backup",
                    "squashplot plot --restore",
                    "squashplot plot --move",
                    "squashplot plot --copy",
                    "squashplot plot --delete",
                    
                    # Plot compression
                    "squashplot compress --plot",
                    "squashplot compress --all",
                    "squashplot compress --batch",
                    "squashplot compress --status",
                    "squashplot compress --progress",
                    "squashplot compress --stop",
                    "squashplot compress --resume",
                    
                    # Plot decompression
                    "squashplot decompress --plot",
                    "squashplot decompress --all",
                    "squashplot decompress --batch",
                    "squashplot decompress --status",
                    "squashplot decompress --progress",
                    "squashplot decompress --stop",
                    "squashplot decompress --resume",
                    
                    # Plot validation
                    "squashplot validate --plot",
                    "squashplot validate --all",
                    "squashplot validate --batch",
                    "squashplot validate --status",
                    "squashplot validate --progress",
                    "squashplot validate --stop",
                    "squashplot validate --resume"
                ]
            },
            
            "farming_commands": {
                "description": "Farming and harvesting commands",
                "commands": [
                    # Farming operations
                    "squashplot farm --start",
                    "squashplot farm --stop",
                    "squashplot farm --restart",
                    "squashplot farm --status",
                    "squashplot farm --info",
                    "squashplot farm --health",
                    "squashplot farm --metrics",
                    
                    # Harvesting
                    "squashplot harvest --start",
                    "squashplot harvest --stop",
                    "squashplot harvest --status",
                    "squashplot harvest --progress",
                    "squashplot harvest --results",
                    "squashplot harvest --history",
                    
                    # Farm management
                    "squashplot farm --add-plot",
                    "squashplot farm --remove-plot",
                    "squashplot farm --list-plots",
                    "squashplot farm --optimize",
                    "squashplot farm --backup",
                    "squashplot farm --restore",
                    
                    # Farm monitoring
                    "squashplot farm --monitor",
                    "squashplot farm --logs",
                    "squashplot farm --alerts",
                    "squashplot farm --notifications"
                ]
            },
            
            "compression_commands": {
                "description": "Advanced compression and optimization commands",
                "commands": [
                    # Compression algorithms
                    "squashplot compress --algorithm lz4",
                    "squashplot compress --algorithm zstd",
                    "squashplot compress --algorithm gzip",
                    "squashplot compress --algorithm brotli",
                    "squashplot compress --algorithm lzma",
                    
                    # Compression levels
                    "squashplot compress --level 1",
                    "squashplot compress --level 3",
                    "squashplot compress --level 6",
                    "squashplot compress --level 9",
                    "squashplot compress --level max",
                    
                    # Compression optimization
                    "squashplot compress --optimize",
                    "squashplot compress --benchmark",
                    "squashplot compress --test",
                    "squashplot compress --analyze",
                    "squashplot compress --report",
                    
                    # Batch operations
                    "squashplot compress --batch --size 32",
                    "squashplot compress --batch --size 64",
                    "squashplot compress --batch --size 128",
                    "squashplot compress --batch --size 256",
                    "squashplot compress --batch --size 512",
                    "squashplot compress --batch --size 1024",
                    
                    # Parallel compression
                    "squashplot compress --parallel 2",
                    "squashplot compress --parallel 4",
                    "squashplot compress --parallel 8",
                    "squashplot compress --parallel 16",
                    "squashplot compress --parallel max"
                ]
            },
            
            "monitoring_commands": {
                "description": "System monitoring and health check commands",
                "commands": [
                    # System monitoring
                    "squashplot monitor --cpu",
                    "squashplot monitor --memory",
                    "squashplot monitor --disk",
                    "squashplot monitor --network",
                    "squashplot monitor --gpu",
                    "squashplot monitor --temperature",
                    "squashplot monitor --power",
                    
                    # Performance monitoring
                    "squashplot monitor --performance",
                    "squashplot monitor --throughput",
                    "squashplot monitor --latency",
                    "squashplot monitor --efficiency",
                    "squashplot monitor --utilization",
                    
                    # Health checks
                    "squashplot health --check",
                    "squashplot health --status",
                    "squashplot health --report",
                    "squashplot health --diagnose",
                    "squashplot health --fix",
                    
                    # Alerting
                    "squashplot alerts --list",
                    "squashplot alerts --status",
                    "squashplot alerts --configure",
                    "squashplot alerts --test",
                    "squashplot alerts --clear"
                ]
            },
            
            "utility_commands": {
                "description": "Utility and maintenance commands",
                "commands": [
                    # File operations
                    "squashplot files --list",
                    "squashplot files --info",
                    "squashplot files --size",
                    "squashplot files --checksum",
                    "squashplot files --verify",
                    "squashplot files --cleanup",
                    "squashplot files --organize",
                    
                    # Backup and restore
                    "squashplot backup --create",
                    "squashplot backup --list",
                    "squashplot backup --restore",
                    "squashplot backup --verify",
                    "squashplot backup --cleanup",
                    "squashplot backup --schedule",
                    
                    # Maintenance
                    "squashplot maintenance --start",
                    "squashplot maintenance --stop",
                    "squashplot maintenance --status",
                    "squashplot maintenance --schedule",
                    "squashplot maintenance --history",
                    
                    # Updates
                    "squashplot update --check",
                    "squashplot update --available",
                    "squashplot update --install",
                    "squashplot update --rollback",
                    "squashplot update --status"
                ]
            },
            
            "api_commands": {
                "description": "API and integration commands",
                "commands": [
                    # API management
                    "squashplot api --start",
                    "squashplot api --stop",
                    "squashplot api --restart",
                    "squashplot api --status",
                    "squashplot api --info",
                    "squashplot api --test",
                    "squashplot api --docs",
                    
                    # Web interface
                    "squashplot web --start",
                    "squashplot web --stop",
                    "squashplot web --restart",
                    "squashplot web --status",
                    "squashplot web --info",
                    "squashplot web --test",
                    
                    # Integration
                    "squashplot integrate --chia",
                    "squashplot integrate --madmax",
                    "squashplot integrate --bladebit",
                    "squashplot integrate --status",
                    "squashplot integrate --test",
                    "squashplot integrate --configure"
                ]
            }
        }
        
        self.dangerous_patterns = [
            # System commands
            r"rm\s+-rf",
            r"sudo\s+",
            r"chmod\s+777",
            r"chown\s+",
            r"format\s+",
            r"del\s+/f",
            r"rd\s+/s",
            
            # Network commands
            r"wget\s+",
            r"curl\s+",
            r"nc\s+",
            r"netcat\s+",
            r"telnet\s+",
            r"ssh\s+",
            r"scp\s+",
            
            # Process commands
            r"kill\s+",
            r"killall\s+",
            r"taskkill\s+",
            r"pkill\s+",
            r"ps\s+",
            r"top\s+",
            
            # File system commands
            r"dd\s+",
            r"mkfs\s+",
            r"fdisk\s+",
            r"mount\s+",
            r"umount\s+",
            
            # Shell commands
            r"bash\s+",
            r"sh\s+",
            r"cmd\s+",
            r"powershell\s+",
            r"python\s+-c",
            r"python\s+-m",
            
            # Dangerous characters
            r"[;&|`$]",
            r"\.\./",
            r"\.\.\\",
            r"<",
            r">",
            r"\*",
            r"\?",
            r"\[",
            r"\]"
        ]
    
    def get_all_commands(self) -> List[str]:
        """Get all whitelisted commands"""
        all_commands = []
        for category, data in self.whitelist_commands.items():
            all_commands.extend(data["commands"])
        return all_commands
    
    def get_commands_by_category(self, category: str) -> List[str]:
        """Get commands for a specific category"""
        if category in self.whitelist_commands:
            return self.whitelist_commands[category]["commands"]
        return []
    
    def is_command_whitelisted(self, command: str) -> bool:
        """Check if a command is whitelisted"""
        # Normalize command (remove extra spaces, convert to lowercase)
        normalized_command = " ".join(command.strip().split()).lower()
        
        # Check against whitelist
        for category, data in self.whitelist_commands.items():
            for whitelisted_command in data["commands"]:
                if normalized_command == whitelisted_command.lower():
                    return True
        
        return False
    
    def is_command_dangerous(self, command: str) -> bool:
        """Check if a command contains dangerous patterns"""
        import re
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        
        return False
    
    def validate_command(self, command: str) -> Dict[str, any]:
        """Validate a command against whitelist and dangerous patterns"""
        result = {
            "command": command,
            "is_whitelisted": False,
            "is_dangerous": False,
            "category": None,
            "safe": False,
            "reason": ""
        }
        
        # Check if dangerous
        if self.is_command_dangerous(command):
            result["is_dangerous"] = True
            result["reason"] = "Command contains dangerous patterns"
            return result
        
        # Check if whitelisted
        if self.is_command_whitelisted(command):
            result["is_whitelisted"] = True
            result["safe"] = True
            result["reason"] = "Command is whitelisted"
            
            # Find category
            for category, data in self.whitelist_commands.items():
                if command.lower() in [cmd.lower() for cmd in data["commands"]]:
                    result["category"] = category
                    break
        else:
            result["reason"] = "Command is not whitelisted"
        
        return result
    
    def get_whitelist_summary(self) -> Dict[str, any]:
        """Get summary of whitelist commands"""
        total_commands = len(self.get_all_commands())
        
        summary = {
            "total_commands": total_commands,
            "categories": {},
            "dangerous_patterns": len(self.dangerous_patterns),
            "last_updated": datetime.now().isoformat()
        }
        
        for category, data in self.whitelist_commands.items():
            summary["categories"][category] = {
                "description": data["description"],
                "command_count": len(data["commands"])
            }
        
        return summary
    
    def export_whitelist(self, format: str = "json") -> str:
        """Export whitelist in specified format"""
        if format == "json":
            return json.dumps({
                "whitelist_commands": self.whitelist_commands,
                "dangerous_patterns": self.dangerous_patterns,
                "summary": self.get_whitelist_summary()
            }, indent=2)
        elif format == "yaml":
            import yaml
            return yaml.dump({
                "whitelist_commands": self.whitelist_commands,
                "dangerous_patterns": self.dangerous_patterns,
                "summary": self.get_whitelist_summary()
            }, default_flow_style=False)
        else:
            return str(self.whitelist_commands)
    
    def generate_whitelist_config(self) -> Dict[str, any]:
        """Generate configuration for the bridge app"""
        return {
            "whitelist_commands": self.get_all_commands(),
            "dangerous_patterns": self.dangerous_patterns,
            "validation_rules": {
                "case_sensitive": False,
                "normalize_spaces": True,
                "allow_partial_matches": False,
                "strict_validation": True
            },
            "security_settings": {
                "max_command_length": 500,
                "timeout_seconds": 30,
                "max_concurrent_commands": 5,
                "rate_limit_per_minute": 10
            },
            "logging": {
                "log_all_commands": True,
                "log_dangerous_attempts": True,
                "log_validation_failures": True,
                "audit_trail": True
            }
        }

def main():
    """Main whitelist management entry point"""
    print("SquashPlot Whitelist Commands")
    print("=" * 50)
    
    whitelist = SquashPlotWhitelist()
    
    # Display summary
    summary = whitelist.get_whitelist_summary()
    print(f"Total Whitelisted Commands: {summary['total_commands']}")
    print(f"Dangerous Patterns: {summary['dangerous_patterns']}")
    print(f"Categories: {len(summary['categories'])}")
    
    print("\nCategories:")
    for category, info in summary["categories"].items():
        print(f"  {category}: {info['command_count']} commands - {info['description']}")
    
    # Test some commands
    test_commands = [
        "squashplot --help",
        "squashplot plot --create",
        "squashplot farm --status",
        "rm -rf /",
        "sudo rm -rf /"
    ]
    
    print("\nCommand Validation Tests:")
    for cmd in test_commands:
        result = whitelist.validate_command(cmd)
        status = "✅ SAFE" if result["safe"] else "❌ BLOCKED"
        print(f"  {status}: {cmd} - {result['reason']}")
    
    # Export configuration
    config = whitelist.generate_whitelist_config()
    with open("squashplot_whitelist_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nWhitelist configuration exported to: squashplot_whitelist_config.json")
    print(f"Total commands: {len(config['whitelist_commands'])}")
    print(f"Dangerous patterns: {len(config['dangerous_patterns'])}")

if __name__ == "__main__":
    main()
