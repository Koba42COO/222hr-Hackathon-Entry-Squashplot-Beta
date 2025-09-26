#!/usr/bin/env python3
"""
SquashPlot User Command Controller
Manages user preferences for command blocking/allowance
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Set, Optional
from pathlib import Path

class UserCommandController:
    """Manages user command preferences and integration with bridge app"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".squashplot"
        self.config_dir.mkdir(exist_ok=True)
        
        self.preferences_file = self.config_dir / "user_command_preferences.json"
        self.bridge_config_file = self.config_dir / "bridge_config.json"
        
        # Load user preferences
        self.user_preferences = self._load_user_preferences()
        
        # Default whitelist (developer-managed)
        self.default_whitelist = self._load_default_whitelist()
    
    def _load_user_preferences(self) -> Dict[str, bool]:
        """Load user command preferences"""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _load_default_whitelist(self) -> List[str]:
        """Load default whitelist from bridge config"""
        if self.bridge_config_file.exists():
            try:
                with open(self.bridge_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('security_settings', {}).get('whitelist_commands', [])
            except Exception:
                return []
        return []
    
    def get_user_allowed_commands(self) -> List[str]:
        """Get commands allowed by user preferences"""
        allowed_commands = []
        
        for command in self.default_whitelist:
            # If user has explicitly set preference, use it
            if command in self.user_preferences:
                if self.user_preferences[command]:
                    allowed_commands.append(command)
            else:
                # If no user preference, command is allowed by default
                allowed_commands.append(command)
        
        return allowed_commands
    
    def get_user_blocked_commands(self) -> List[str]:
        """Get commands blocked by user preferences"""
        blocked_commands = []
        
        for command in self.default_whitelist:
            if command in self.user_preferences and not self.user_preferences[command]:
                blocked_commands.append(command)
        
        return blocked_commands
    
    def set_command_preference(self, command: str, allowed: bool):
        """Set user preference for a specific command"""
        self.user_preferences[command] = allowed
        self._save_user_preferences()
    
    def set_category_preference(self, category: str, allowed: bool):
        """Set user preference for an entire category"""
        category_commands = self._get_category_commands(category)
        
        for command in category_commands:
            self.user_preferences[command] = allowed
        
        self._save_user_preferences()
    
    def _get_category_commands(self, category: str) -> List[str]:
        """Get commands for a specific category"""
        category_mapping = {
            "dashboard_commands": [
                "squashplot --help", "squashplot --version", "squashplot --status",
                "squashplot --info", "squashplot --config", "squashplot --list-plots",
                "squashplot dashboard --status", "squashplot dashboard --info",
                "squashplot dashboard --health", "squashplot dashboard --metrics",
                "squashplot system --info", "squashplot system --status",
                "squashplot system --health", "squashplot system --disk-usage",
                "squashplot config --show", "squashplot config --list",
                "squashplot logs --show", "squashplot logs --tail",
                "squashplot monitor --start", "squashplot monitor --stop"
            ],
            "plotting_commands": [
                "squashplot plot --create", "squashplot plot --list",
                "squashplot plot --status", "squashplot plot --info",
                "squashplot plot --validate", "squashplot plot --check",
                "squashplot plot --compress", "squashplot plot --decompress",
                "squashplot compress --plot", "squashplot compress --all",
                "squashplot compress --status", "squashplot compress --progress",
                "squashplot decompress --plot", "squashplot decompress --all",
                "squashplot validate --plot", "squashplot validate --all"
            ],
            "farming_commands": [
                "squashplot farm --start", "squashplot farm --stop",
                "squashplot farm --status", "squashplot farm --info",
                "squashplot farm --health", "squashplot farm --metrics",
                "squashplot harvest --start", "squashplot harvest --stop",
                "squashplot farm --add-plot", "squashplot farm --remove-plot",
                "squashplot farm --list-plots", "squashplot farm --optimize"
            ],
            "compression_commands": [
                "squashplot compress --algorithm lz4", "squashplot compress --algorithm zstd",
                "squashplot compress --level 1", "squashplot compress --level 3",
                "squashplot compress --level 6", "squashplot compress --level 9",
                "squashplot compress --optimize", "squashplot compress --benchmark",
                "squashplot compress --batch", "squashplot compress --parallel 2",
                "squashplot compress --parallel 4", "squashplot compress --parallel 8"
            ],
            "monitoring_commands": [
                "squashplot monitor --cpu", "squashplot monitor --memory",
                "squashplot monitor --disk", "squashplot monitor --network",
                "squashplot health --check", "squashplot health --status",
                "squashplot health --report", "squashplot alerts --list",
                "squashplot alerts --status", "squashplot alerts --test"
            ],
            "utility_commands": [
                "squashplot files --list", "squashplot files --info",
                "squashplot files --size", "squashplot files --checksum",
                "squashplot backup --create", "squashplot backup --list",
                "squashplot backup --restore", "squashplot maintenance --start",
                "squashplot maintenance --stop", "squashplot update --check",
                "squashplot update --available", "squashplot update --install"
            ],
            "api_commands": [
                "squashplot api --start", "squashplot api --stop",
                "squashplot api --status", "squashplot api --info",
                "squashplot web --start", "squashplot web --stop",
                "squashplot web --status", "squashplot integrate --chia",
                "squashplot integrate --madmax", "squashplot integrate --bladebit"
            ]
        }
        
        return category_mapping.get(category, [])
    
    def _save_user_preferences(self):
        """Save user preferences to file"""
        try:
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving user preferences: {e}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get command statistics"""
        total_commands = len(self.default_whitelist)
        allowed_commands = len(self.get_user_allowed_commands())
        blocked_commands = len(self.get_user_blocked_commands())
        
        return {
            "total_commands": total_commands,
            "allowed_commands": allowed_commands,
            "blocked_commands": blocked_commands,
            "categories": len(self._get_category_commands("dashboard_commands")),  # Placeholder
            "last_updated": datetime.now().isoformat()
        }
    
    def export_configuration(self) -> Dict[str, any]:
        """Export current configuration"""
        return {
            "user_preferences": self.user_preferences,
            "allowed_commands": self.get_user_allowed_commands(),
            "blocked_commands": self.get_user_blocked_commands(),
            "statistics": self.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }
    
    def import_configuration(self, config: Dict[str, any]):
        """Import configuration from file"""
        if "user_preferences" in config:
            self.user_preferences = config["user_preferences"]
            self._save_user_preferences()
    
    def reset_to_defaults(self):
        """Reset all user preferences to defaults"""
        self.user_preferences = {}
        self._save_user_preferences()
    
    def is_command_allowed(self, command: str) -> bool:
        """Check if a command is allowed by user preferences"""
        if command in self.user_preferences:
            return self.user_preferences[command]
        else:
            # If no user preference, command is allowed by default
            return True
    
    def update_bridge_config(self):
        """Update bridge app configuration with user preferences"""
        try:
            # Load current bridge config
            if self.bridge_config_file.exists():
                with open(self.bridge_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Update with user preferences
            allowed_commands = self.get_user_allowed_commands()
            
            if "security_settings" not in config:
                config["security_settings"] = {}
            
            config["security_settings"]["whitelist_commands"] = allowed_commands
            config["user_preferences"] = self.user_preferences
            config["last_updated"] = datetime.now().isoformat()
            
            # Save updated config
            with open(self.bridge_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error updating bridge config: {e}")
            return False
    
    def get_web_interface_data(self) -> Dict[str, any]:
        """Get data for web interface"""
        return {
            "categories": {
                "dashboard_commands": {
                    "name": "Dashboard Commands",
                    "description": "Dashboard and interface commands",
                    "commands": self._get_category_commands("dashboard_commands")
                },
                "plotting_commands": {
                    "name": "Plotting Commands", 
                    "description": "Plot creation and management commands",
                    "commands": self._get_category_commands("plotting_commands")
                },
                "farming_commands": {
                    "name": "Farming Commands",
                    "description": "Farming and harvesting commands", 
                    "commands": self._get_category_commands("farming_commands")
                },
                "compression_commands": {
                    "name": "Compression Commands",
                    "description": "Advanced compression and optimization commands",
                    "commands": self._get_category_commands("compression_commands")
                },
                "monitoring_commands": {
                    "name": "Monitoring Commands",
                    "description": "System monitoring and health check commands",
                    "commands": self._get_category_commands("monitoring_commands")
                },
                "utility_commands": {
                    "name": "Utility Commands",
                    "description": "Utility and maintenance commands",
                    "commands": self._get_category_commands("utility_commands")
                },
                "api_commands": {
                    "name": "API Commands",
                    "description": "API and integration commands",
                    "commands": self._get_category_commands("api_commands")
                }
            },
            "user_preferences": self.user_preferences,
            "statistics": self.get_statistics()
        }

def main():
    """Main controller entry point"""
    print("SquashPlot User Command Controller")
    print("=" * 50)
    
    controller = UserCommandController()
    
    # Display current statistics
    stats = controller.get_statistics()
    print(f"Total Commands: {stats['total_commands']}")
    print(f"Allowed Commands: {stats['allowed_commands']}")
    print(f"Blocked Commands: {stats['blocked_commands']}")
    
    # Test command allowance
    test_commands = [
        "squashplot --help",
        "squashplot plot --create", 
        "squashplot farm --status"
    ]
    
    print("\nCommand Allowance Test:")
    for cmd in test_commands:
        allowed = controller.is_command_allowed(cmd)
        status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
        print(f"  {status}: {cmd}")
    
    # Export configuration
    config = controller.export_configuration()
    print(f"\nConfiguration exported with {len(config['user_preferences'])} user preferences")

if __name__ == "__main__":
    main()
