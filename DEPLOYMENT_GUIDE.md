# 🚀 SquashPlot Deployment Guide

## Replit Deployment - Dashboard Functionality

### ✅ **YES, the dashboard will work on Replit!**

The SquashPlot system is specifically designed for Replit deployment with the following features:

#### **Dashboard Capabilities on Replit:**
- ✅ **Full Web Interface**: Professional dashboard with real-time monitoring
- ✅ **CLI Integration**: Andy's command templates and execution
- ✅ **System Monitoring**: CPU, memory, disk usage tracking
- ✅ **Compression Tools**: All compression levels and algorithms
- ✅ **API Endpoints**: RESTful API for external integrations
- ✅ **WebSocket Support**: Real-time updates and notifications

#### **Replit-Specific Optimizations:**
```python
# Automatic port configuration
PORT = int(os.getenv("PORT", "8080"))

# CORS middleware for Replit
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Replit mode detection
REPLIT_MODE = os.getenv("REPLIT", False)
```

### **How to Deploy on Replit:**

1. **Fork the Replit Template**
   ```bash
   # Click "Fork" on the Replit template
   # Or create new Replit and clone repository
   ```

2. **Automatic Setup**
   ```bash
   # Replit automatically installs dependencies
   pip install -r requirements.txt
   ```

3. **Start the Application**
   ```bash
   # Web interface (recommended)
   python main.py --web
   
   # Or use the API server directly
   python squashplot_api_server.py
   ```

4. **Access Your Dashboard**
   - **URL**: `https://your-replit-name.replit.dev`
   - **Local**: `http://localhost:8080`
   - **API Docs**: `https://your-replit-name.replit.dev/docs`

## 🔧 CLI Commands - Local Execution

### **Secure CLI Bridge System**

I've created a comprehensive CLI bridge that allows users to execute commands locally while maintaining security:

#### **Installation:**
```bash
# Make the bridge executable
chmod +x secure_cli_bridge.py

# Install dependencies
pip install -r requirements.txt
```

#### **Usage Options:**

**1. Interactive Mode (Recommended):**
```bash
python secure_cli_bridge.py --interactive
```

**2. Execute Single Command:**
```bash
python secure_cli_bridge.py --execute "python main.py --web"
```

**3. Execute Template:**
```bash
python secure_cli_bridge.py --template "web_interface"
```

**4. List Available Templates:**
```bash
python secure_cli_bridge.py --list-templates
```

### **Available CLI Templates:**

```bash
# Web Interface
python secure_cli_bridge.py --template "web_interface"

# CLI Mode  
python secure_cli_bridge.py --template "cli_mode"

# Demo Mode
python secure_cli_bridge.py --template "demo_mode"

# Server Check
python secure_cli_bridge.py --template "server_check"

# Basic Plotting (requires farmer/pool keys)
python secure_cli_bridge.py --template "basic_plotting" --farmer-key YOUR_KEY --pool-key YOUR_KEY

# Compressed Plotting
python secure_cli_bridge.py --template "compressed_plotting" --farmer-key YOUR_KEY --pool-key YOUR_KEY
```

## 🛡️ Security Features

### **Command Validation:**
- ✅ **Allowed Commands Only**: Whitelist of safe commands
- ✅ **Directory Restrictions**: Limited to safe directories
- ✅ **Blocked Operations**: Prevents dangerous commands (rm, format, etc.)
- ✅ **Timeout Protection**: Maximum execution time limits
- ✅ **Path Validation**: Prevents directory traversal attacks

### **Security Configuration:**
```json
{
  "security": {
    "max_execution_time": 300,
    "allowed_directories": ["./", "/tmp/", "/plots/"],
    "blocked_commands": ["rm", "del", "format", "fdisk", "mkfs"],
    "require_confirmation": true
  }
}
```

## 📊 Dashboard Features

### **Real-Time Monitoring:**
- **System Resources**: CPU, memory, disk usage
- **Server Status**: Andy's check_server integration
- **Compression Stats**: Space saved, compression ratios
- **Recent Activity**: Command history and logs

### **CLI Integration:**
- **Command Templates**: Pre-built professional commands
- **Custom Commands**: Secure execution of user commands
- **Output Display**: Real-time command output
- **Error Handling**: Comprehensive error reporting

### **Professional Interface:**
- **Black Glass Design**: Modern, sophisticated UI
- **Responsive Layout**: Works on all devices
- **Real-Time Updates**: WebSocket-powered live data
- **Interactive Elements**: Click-to-execute commands

## 🔄 Workflow Examples

### **1. Replit Deployment Workflow:**
```bash
# 1. Fork Replit template
# 2. Run the application
python main.py --web

# 3. Access dashboard at: https://your-replit-name.replit.dev
# 4. Use CLI integration for commands
```

### **2. Local Development Workflow:**
```bash
# 1. Clone repository
git clone https://github.com/Koba42COO/222hr-Hackathon-Entry-Squashplot-Beta.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start CLI bridge
python secure_cli_bridge.py --interactive

# 4. Execute commands through secure bridge
squashplot> template web_interface
squashplot> execute python check_server.py
```

### **3. Production Deployment:**
```bash
# 1. Use the API server directly
python squashplot_api_server.py

# 2. Access professional dashboard
# 3. Use CLI templates for operations
# 4. Monitor system through web interface
```

## 🎯 Key Benefits

### **For Replit Users:**
- ✅ **Zero Configuration**: Works out of the box
- ✅ **Professional Dashboard**: Enterprise-grade interface
- ✅ **CLI Integration**: Andy's command templates
- ✅ **Real-Time Monitoring**: Live system stats
- ✅ **Secure Execution**: Protected command execution

### **For Local Users:**
- ✅ **Secure Bridge**: Safe command execution
- ✅ **Template System**: Pre-built professional commands
- ✅ **Interactive Mode**: User-friendly CLI interface
- ✅ **Configuration**: Customizable security settings
- ✅ **Professional Commands**: Mad Max/BladeBit compatible

## 🚀 Quick Start Commands

### **Replit Deployment:**
```bash
# Start web interface
python main.py --web

# Access: https://your-replit-name.replit.dev
```

### **Local CLI Bridge:**
```bash
# Interactive mode
python secure_cli_bridge.py --interactive

# Execute web interface
python secure_cli_bridge.py --template web_interface

# Check server status
python secure_cli_bridge.py --template server_check
```

### **Direct Commands:**
```bash
# Web dashboard
python main.py --web

# CLI mode
python main.py --cli

# Demo mode
python main.py --demo

# Server check
python check_server.py
```

## 📝 Configuration

### **CLI Bridge Configuration:**
The system automatically creates `cli_bridge_config.json` with secure defaults:

```json
{
  "security": {
    "max_execution_time": 300,
    "allowed_directories": ["./", "/tmp/", "/plots/"],
    "blocked_commands": ["rm", "del", "format", "fdisk", "mkfs"],
    "require_confirmation": true
  },
  "squashplot": {
    "executable": "python",
    "main_script": "main.py",
    "compression_script": "squashplot.py",
    "check_script": "check_server.py"
  }
}
```

## 🎉 Summary

**YES, the dashboard works perfectly on Replit!** The system is specifically designed for cloud deployment with:

1. **Full Dashboard Functionality** - Professional web interface
2. **CLI Command Execution** - Secure bridge for local commands  
3. **Real-Time Monitoring** - Live system statistics
4. **Professional Templates** - Andy's command structures
5. **Security Features** - Protected command execution

The secure CLI bridge provides a professional, safe way for users to execute commands locally while maintaining security and providing an excellent user experience.
