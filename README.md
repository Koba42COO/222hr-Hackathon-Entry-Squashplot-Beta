# SquashPlot Bridge - Professional System

## 🚨 CRITICAL SECURITY NOTICE

**⚠️ READ SECURITY WARNING FIRST**: `squashplot_bridge_professional/SECURITY_WARNING_README.md`

This is an OPEN-SOURCE project. Users can fork and modify the code. Use only official versions.

## 🚀 Complete Setup Guide

### Step 1: Install Prerequisites

**If you don't have Python installed:**
1. Download Python from: https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation: Open Command Prompt and type `python --version`

**If you don't have Git installed:**
1. Download Git from: https://git-scm.com/downloads
2. Install with default settings
3. Verify installation: Open Command Prompt and type `git --version`

### Step 2: Get the Code

```bash
# Clone the repository
git clone https://github.com/your-repo/SquashPlot.git
cd SquashPlot

# Or download ZIP and extract to a folder
```

### Step 3: Install Dependencies

```bash
# Easy way - install all dependencies at once
pip install -r requirements.txt

# Or install manually if needed
pip install fastapi uvicorn requests numpy pandas matplotlib seaborn scikit-learn
```

### Step 4: Start the System

**Open TWO Command Prompt/Terminal windows:**

**Terminal 1:**
```bash
python squashplot_api_server.py
```

**Terminal 2:**
```bash
python main.py --web
```

### Step 5: Access the System

1. Open your browser
2. Go to: `http://localhost:8080`
3. Click "Developer Test Page" button in Command Details section

## Quick Start (If Already Set Up)

### 🚀 Full System Startup (Recommended)
```bash
# Terminal 1: Start the API Server (Bridge App)
python squashplot_api_server.py

# Terminal 2: Start the Main Web Interface
python main.py --web
```

Then open your browser to: `http://localhost:8080`

**Access Developer Bridge**: Click "Developer Test Page" button in the Command Details section

### 🔒 Maximum Security Version (Production Use)
```bash
cd squashplot_bridge_professional
python SquashPlotSecureBridge.py
```

Then open your browser to: `http://127.0.0.1:8080`

### 🔓 Developer Version (Development Only - NOT for Production)
```bash
cd squashplot_bridge_professional
python SquashPlotDeveloperBridge.py
```

Then open your browser to: `http://127.0.0.1:8081`

## What You Get

### 🚀 Full System (Main + Developer Bridge):
- **Complete SquashPlot Dashboard**: Full-featured web interface
- **Developer Bridge Integration**: Built-in developer testing tools
- **API Server**: Robust backend with all endpoints
- **Real-time Execution**: Live command execution and testing
- **Professional Interface**: Enterprise-grade web interface
- **Developer Tools**: Integrated testing and debugging tools

### 🔒 Maximum Security Version:
- **Quantum-Safe Cryptography**: 3-layer hybrid encryption
- **AI Threat Detection**: 10 AI models for threat analysis
- **Behavioral Biometrics**: 5-layer biometric authentication
- **Zero-Trust Architecture**: Advanced security controls
- **Professional Interface**: Enterprise-grade web interface
- **Authentication Required**: Secure token-based authentication

### 🔓 Developer Version (Locked Down):
- **Single Command Only**: Only "hello-world" allowed
- **No Authentication**: For development ease
- **Security Warnings**: Clear warnings about limitations
- **Development Interface**: Simple interface for testing

## Documentation

- **🚨 Security Warning**: `squashplot_bridge_professional/SECURITY_WARNING_README.md`
- **Professional Summary**: `squashplot_bridge_professional/PROFESSIONAL_SUMMARY.md`
- **Simple Summary**: `squashplot_bridge_professional/SIMPLE_SUMMARY.md`

## Architecture

### Maximum Security:
- **SquashPlotSecureBridge.py**: Maximum security server
- **SquashPlotSecureBridge.html**: Enterprise web interface
- **Security**: Quantum-safe crypto + AI threat detection

### Developer Version:
- **SquashPlotDeveloperBridge.py**: Locked-down developer server
- **SquashPlotDeveloperBridge.html**: Developer interface
- **Security**: Single command whitelist + warnings

## Security Features

### Maximum Security Version:
- Quantum-safe cryptography (3-layer hybrid encryption)
- AI threat detection (10 models)
- Behavioral biometrics (5 layers)
- Zero-trust architecture
- Advanced rate limiting
- Continuous authentication
- Comprehensive security logging

### Developer Version:
- Single command whitelist (hello-world only)
- No authentication (development ease)
- Security warnings and logging
- Rate limiting
- Development-focused interface

## ⚠️ IMPORTANT WARNINGS

- **Use SECURE version for production**
- **Developer version is NOT for production**
- **Always verify source authenticity**
- **Test thoroughly before deployment**
- **Monitor security logs**

## Perfect For

### 🚀 Full System:
- Complete development and testing
- Full-featured SquashPlot experience
- Integrated developer tools
- Production-ready applications
- Enterprise deployments

### 🔒 Maximum Security Version:
- Production deployments
- Enterprise applications
- High-security environments
- Commercial use
- Public-facing systems

### 🔓 Developer Version:
- Development testing
- Proof of concept
- Educational purposes
- Internal evaluation
- Learning and experimentation

## Troubleshooting

### Common Beginner Issues:

**"python is not recognized"**
- Python not installed or not in PATH
- Solution: Reinstall Python and check "Add to PATH"

**"pip is not recognized"**
- pip not installed with Python
- Solution: `python -m ensurepip --upgrade`

**"Module not found" errors**
- Dependencies not installed
- Solution: Run `pip install fastapi uvicorn requests numpy pandas matplotlib seaborn scikit-learn`

**"Port already in use"**
- Another process using port 8080
- Solution: 
```bash
# Check what's using port 8080
netstat -an | findstr :8080

# Kill processes if needed
taskkill /f /im python.exe
```

**"Connection refused" in browser**
- API server not running
- Solution: Make sure Terminal 1 shows "Uvicorn running on http://0.0.0.0:8080"

### Developer Bridge Not Working:
1. Ensure API server is running: `python squashplot_api_server.py`
2. Check browser console for errors (F12)
3. Verify connection status in the developer bridge interface
4. Test with "Test Connection" button first

### Still Having Issues?
- Check both terminals are running without errors
- Try refreshing the browser page
- Make sure you're using `http://localhost:8080` (not https)
- Check Windows Firewall isn't blocking the connection

---

**Status**: Production Ready (Secure Version) / Development Only (Developer Version)  
**Version**: 2.0.0-Secure / 1.0.0-Developer
