# SquashPlot Bridge - Developer Documentation

## 🛠️ **For Developers and Technical Users**

This documentation is for developers, system administrators, and technical users who want to understand, modify, or extend the SquashPlot Bridge system.

---

## 📁 **Architecture Overview**

### **System Components**
```
SquashPlot Bridge System
├── SquashPlotSecureBridge.py          # Maximum security bridge server
├── SquashPlotDeveloperBridge.py       # Locked-down developer version
├── SquashPlotSecureBridge.html        # Enterprise web interface
├── SquashPlotDeveloperBridge.html     # Developer web interface
├── SquashPlotBridgeInstaller.py       # Python installer (source)
├── SquashPlotBridgeEXEInstaller.py    # EXE installer generator
├── WebUninstaller.py                  # Web-based uninstaller
└── Documentation/
    ├── USER_DOCUMENTATION.md          # End-user guide
    ├── DEVELOPER_DOCUMENTATION.md     # This file
    ├── PROFESSIONAL_SUMMARY.md        # Technical specifications
    └── SECURITY_WARNING_README.md     # Security considerations
```

### **Security Levels**

#### **Maximum Security Version** (Production)
- **File**: `SquashPlotSecureBridge.py`
- **Port**: 8080
- **Features**:
  - Quantum-safe cryptography (3-layer hybrid encryption)
  - AI threat detection (10 models)
  - Behavioral biometrics (5 layers)
  - Zero-trust architecture
  - Advanced rate limiting
  - Continuous authentication
  - Token-based authentication
  - Comprehensive security logging

#### **Developer Version** (Development Only)
- **File**: `SquashPlotDeveloperBridge.py`
- **Port**: 8081
- **Features**:
  - Single command whitelist (hello-world only)
  - No authentication (development ease)
  - Security warnings and logging
  - Rate limiting
  - Development-focused interface

---

## 🔧 **Development Setup**

### **Prerequisites**
```bash
# Required Python version
Python 3.8+

# Required packages
pip install cryptography numpy tkinter
```

### **Running from Source**

#### **Secure Version (Production)**
```bash
cd squashplot_bridge_professional
python SquashPlotSecureBridge.py
# Access: http://127.0.0.1:8080
```

#### **Developer Version (Development)**
```bash
cd squashplot_bridge_professional
python SquashPlotDeveloperBridge.py
# Access: http://127.0.0.1:8081
```

### **Building Executable Installer**
```bash
# Install PyInstaller
pip install pyinstaller

# Generate executable installer
python SquashPlotBridgeEXEInstaller.py
# Output: executable_installer/package/SquashPlotBridgeInstaller.exe
```

---

## 🔒 **Security Implementation Details**

### **Cryptography Stack**
```python
# Quantum-safe hybrid encryption
class QuantumSafeCrypto:
    - Scrypt KDF (2^20 memory cost)
    - AES-256-GCM encryption
    - ChaCha20-Poly1305 encryption
    - Fernet (AES 128 + HMAC)
    - 3-layer hybrid encryption
```

### **AI Threat Detection**
```python
class AdvancedThreatDetection:
    - 10 AI threat detection models
    - Pattern analysis (command injection)
    - Behavioral analysis (anomalous behavior)
    - Rate limiting analysis
    - Threat intelligence correlation
    - Real-time threat scoring
```

### **Behavioral Biometrics**
```python
class BehavioralBiometrics:
    - 5 biometric authentication layers
    - Typing pattern analysis
    - Command sequence analysis
    - Timing pattern analysis
    - Session behavior analysis
    - Interaction pattern analysis
```

### **Authentication System**
```python
class SecureAuthentication:
    - Token-based authentication
    - Session timeout (1 hour)
    - Failed attempt tracking
    - Account lockout protection
    - Biometric validation
    - Continuous authentication
```

---

## 📡 **API Endpoints**

### **Secure Bridge (Port 8080)**

#### **GET /** - Web Interface
- **Purpose**: Serve the secure web interface
- **Response**: HTML page

#### **GET /status** - Status Check
- **Purpose**: Get bridge status and system information
- **Response**: JSON with status, platform, version, features

#### **POST /auth** - Authentication
- **Purpose**: Generate authentication token
- **Body**: `{"client_info": {...}}`
- **Response**: `{"success": true, "token": "...", "expires_in": 3600}`

#### **POST /execute** - Command Execution
- **Purpose**: Execute whitelisted commands
- **Headers**: `Authorization: Bearer <token>`
- **Body**: `{"command": "hello-world"}`
- **Response**: Execution result with security metrics

### **Developer Bridge (Port 8081)**

#### **GET /** - Developer Interface
- **Purpose**: Serve the developer web interface
- **Response**: HTML page with security warnings

#### **GET /status** - Developer Status
- **Purpose**: Get developer bridge status
- **Response**: JSON with status, warnings, allowed commands

#### **POST /execute** - Developer Command Execution
- **Purpose**: Execute single whitelisted command
- **Body**: `{"command": "hello-world"}`
- **Response**: Execution result (no authentication required)

#### **GET /security-warning** - Security Warning
- **Purpose**: Display security warnings
- **Response**: JSON with warnings and restrictions

### **Web Uninstaller (Port 8082)**

#### **GET /** - Uninstaller Interface
- **Purpose**: Serve the uninstaller web interface
- **Response**: HTML uninstaller page

#### **GET /status** - Installation Status
- **Purpose**: Check if SquashPlot Bridge is installed
- **Response**: JSON with installation status and config

#### **POST /uninstall** - Perform Uninstall
- **Purpose**: Execute uninstall process
- **Response**: JSON with uninstall results and log

---

## 🏗️ **Installation System**

### **Python Installer** (`SquashPlotBridgeInstaller.py`)
```python
class SquashPlotBridgeInstaller:
    - Cross-platform installation
    - Dependency checking and installation
    - File copying and validation
    - Startup integration setup
    - Configuration creation
    - Built-in uninstaller creation
    - Installation testing
    - GUI completion display
```

### **EXE Installer Generator** (`SquashPlotBridgeEXEInstaller.py`)
```python
class EXEInstallerGenerator:
    - PyInstaller integration
    - Executable building
    - Package creation
    - User documentation generation
    - Installer manifest creation
```

### **Installation Directory Structure**
```
# Windows
AppData/Local/SquashPlot/Bridge/

# macOS  
Library/Application Support/SquashPlot/Bridge/

# Linux
.local/share/squashplot/bridge/
```

---

## 🗑️ **Uninstall System**

### **Built-in Uninstaller**
- **File**: `uninstall.py` (created during installation)
- **Location**: Installation directory
- **Features**: Complete removal of files and startup integration

### **Web Uninstaller** (`WebUninstaller.py`)
- **Port**: 8082
- **Features**: Web interface for remote uninstallation
- **API**: RESTful endpoints for status and uninstall

### **Website Uninstaller** (`WebsiteUninstaller.html`)
- **Purpose**: Host on main SquashPlot website
- **Features**: Status check, uninstaller download, local server connection

---

## 🧪 **Testing and Development**

### **Testing Framework**
```python
# Test files available
test_professional_system.py    # Comprehensive system tests
test_bridge_connectivity.py    # Connectivity tests
test_security_features.py      # Security validation tests
```

### **Development Workflow**
1. **Use developer version** for development and testing
2. **Test security features** with the test suite
3. **Validate installation** with the installer
4. **Build executable** for distribution
5. **Test uninstaller** to ensure clean removal

### **Security Testing**
```bash
# Run security tests
python test_professional_system.py

# Test connectivity
python test_bridge_connectivity.py

# Validate security features
python test_security_features.py
```

---

## 🔐 **Security Considerations**

### **Open Source Security**
- **Fork Risk**: Anyone can fork and modify the code
- **Verification**: Always verify source authenticity
- **Red Flags**: Watch for modified security features
- **Distribution**: Only distribute official versions

### **Production Deployment**
- **Use Secure Version**: Never use developer version in production
- **Authentication**: Always enable authentication
- **Monitoring**: Monitor security logs regularly
- **Updates**: Keep dependencies updated

### **Development Security**
- **Developer Version**: Only use for development/testing
- **No Production**: Never deploy developer version
- **Limited Commands**: Only hello-world command allowed
- **Security Warnings**: Interface shows clear warnings

---

## 📊 **Performance and Monitoring**

### **Resource Usage**
- **Memory**: ~50-100 MB RAM usage
- **CPU**: Minimal when idle, spikes during execution
- **Disk**: ~50 MB installation size
- **Network**: Local only (127.0.0.1)

### **Logging**
- **Security Logs**: `squashplot_security.log`
- **Developer Logs**: `squashplot_developer_bridge.log`
- **Installation Logs**: Displayed in installer GUI
- **Uninstall Logs**: Displayed in uninstaller interface

### **Monitoring**
- **Health Checks**: `/status` endpoint for monitoring
- **Metrics**: Threat scores, biometric confidence, auth status
- **Alerts**: Security events logged and flagged

---

## 🔄 **Extending the System**

### **Adding New Commands**
1. **Update whitelist** in `CommandSanitizer` class
2. **Implement execution logic** in `execute_hello_world_demo` method
3. **Add security validation** for new commands
4. **Update documentation** and interfaces
5. **Test thoroughly** before deployment

### **Custom Security Features**
1. **Extend threat detection** models in `AdvancedThreatDetection`
2. **Add biometric layers** in `BehavioralBiometrics`
3. **Enhance authentication** in `SecureAuthentication`
4. **Implement custom encryption** in `QuantumSafeCrypto`

### **Platform-Specific Features**
1. **Modify platform detection** in `detect_platform` methods
2. **Add OS-specific commands** in execution logic
3. **Customize startup integration** for each platform
4. **Platform-specific security** implementations

---

## 📋 **Development Checklist**

### **Before Release**
- [ ] All tests pass
- [ ] Security features validated
- [ ] Documentation updated
- [ ] Installation tested on all platforms
- [ ] Uninstaller tested
- [ ] Performance validated
- [ ] Security audit completed

### **Code Quality**
- [ ] Type hints added
- [ ] Error handling implemented
- [ ] Logging comprehensive
- [ ] Documentation complete
- [ ] Security best practices followed

### **Distribution**
- [ ] Executable installer built
- [ ] User documentation created
- [ ] Developer documentation updated
- [ ] Security warnings included
- [ ] Installation manifest created

---

## 🚀 **Deployment Guide**

### **Development Environment**
```bash
# Clone repository
git clone [repository-url]
cd squashplot_bridge_professional

# Install dependencies
pip install -r requirements.txt

# Run developer version
python SquashPlotDeveloperBridge.py
```

### **Production Deployment**
```bash
# Build executable installer
python SquashPlotBridgeEXEInstaller.py

# Distribute installer
# - SquashPlotBridgeInstaller.exe
# - README.txt
# - QUICK_START.txt
```

### **Website Integration**
```html
<!-- Host on SquashPlot website -->
- WebsiteUninstaller.html (uninstall page)
- Download links to installer
- Status checking functionality
```

---

**SquashPlot Bridge** - Professional development and deployment system

---

**Version**: 2.0.0-Developer  
**Last Updated**: September 26, 2025  
**For Developers**: Technical implementation details
