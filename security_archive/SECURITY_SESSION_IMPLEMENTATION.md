# 🛡️ SquashPlot Bridge - Session Security Implementation

## Overview

This document outlines the comprehensive session-based security system implemented for the SquashPlot Bridge App. The system provides secure command execution with session management, timeout controls, and protection against various attack vectors.

## 🔐 Security Features Implemented

### 1. Session Management System

#### **Session Creation**
- **Secure Token Generation**: Uses `secrets.token_urlsafe(32)` for session IDs
- **CSRF Protection**: Each session includes a unique CSRF token
- **IP Binding**: Sessions are bound to client IP addresses
- **Configurable Timeout**: Default 15 minutes (900 seconds), user-configurable
- **Session Limits**: Maximum 3 sessions per IP address

#### **Session Validation**
- **Multi-layer Validation**: IP address, timeout, CSRF token, and activity checks
- **Automatic Cleanup**: Expired sessions are automatically removed
- **Warning System**: Users receive warnings 5 minutes before session expiry
- **Activity Tracking**: Last activity timestamp for each session

### 2. Enhanced Security Measures

#### **Input Sanitization**
```python
# Dangerous patterns blocked:
- Command injection: [;&|`$]
- Path traversal: ../, ~
- Dangerous commands: rm -rf, sudo, chmod 777
- System directory access: /etc/, /sys/, /proc/
```

#### **Rate Limiting**
- **Per-IP Limits**: Maximum 10 requests per minute per IP
- **Automatic Cleanup**: Old requests are automatically removed
- **DoS Protection**: Prevents abuse and resource exhaustion

#### **Command Execution Security**
- **Whitelist Commands**: Only approved SquashPlot commands allowed
- **Resource Limits**: CPU, memory, and file size restrictions
- **Process Isolation**: Commands run in restricted environment
- **Timeout Protection**: Maximum 30-second execution time

### 3. Web Interface Security

#### **Session Timeout Popup**
- **Visual Warning**: Prominent warning banner when session expires soon
- **Time Display**: Real-time countdown of remaining session time
- **Progress Bar**: Visual representation of session status
- **Auto-logout**: Automatic session termination on expiry

#### **CSRF Protection**
- **Token Validation**: All requests must include valid CSRF token
- **Session Binding**: CSRF tokens are bound to specific sessions
- **Request Verification**: Server validates CSRF tokens on each request

## 🚀 Usage Instructions

### 1. Starting the Secure Bridge

```bash
python secure_bridge_app.py
```

The bridge will start on `127.0.0.1:8443` with full security features enabled.

### 2. Creating a Session

**Via Web Interface:**
1. Open `session_security_interface.html` in your browser
2. Click "Create New Session"
3. Session will be created with 15-minute timeout

**Via API:**
```bash
curl -X POST http://127.0.0.1:8443 \
  -H "Content-Type: application/json" \
  -d '{"type": "create_session"}'
```

### 3. Executing Commands

**Via Web Interface:**
1. Enter command in the text field
2. Click "Execute Command" or press Enter
3. View results in the output area

**Via API:**
```bash
curl -X POST http://127.0.0.1:8443 \
  -H "Content-Type: application/json" \
  -d '{
    "type": "session_auth",
    "session_id": "your_session_id",
    "csrf_token": "your_csrf_token",
    "command": "squashplot --status"
  }'
```

### 4. Session Management

#### **Extending Sessions**
- Click "Extend Session" button in web interface
- Sessions can be extended before expiry
- Warning banner will disappear after extension

#### **Session Information**
- Click "Session Details" to view session info
- Shows session ID, CSRF token, and timeout settings
- Useful for debugging and troubleshooting

## 🔧 Configuration Options

### Security Settings

```python
class SecurityConfig:
    SESSION_TIMEOUT = 900  # 15 minutes (configurable)
    SESSION_WARNING_TIME = 300  # 5 minutes before timeout
    MAX_SESSIONS_PER_IP = 3  # Maximum sessions per IP
    MAX_REQUESTS_PER_MINUTE = 10  # Rate limiting
    CSRF_TOKEN_LENGTH = 32  # CSRF token length
```

### Customizing Timeouts

To change the default session timeout:

1. **Edit `secure_bridge_app.py`**:
```python
class SecurityConfig:
    SESSION_TIMEOUT = 1800  # 30 minutes
    SESSION_WARNING_TIME = 600  # 10 minutes before timeout
```

2. **Restart the bridge** for changes to take effect

## 🛡️ Security Benefits

### 1. **Prevents Direct Port Access**
- Hackers cannot send commands directly to port 8443
- All requests must include valid session credentials
- Session validation prevents unauthorized access

### 2. **Session Timeout Protection**
- Automatic session expiry prevents long-term unauthorized access
- Configurable timeout allows customization based on security needs
- Warning system gives users time to extend sessions

### 3. **CSRF Protection**
- Prevents cross-site request forgery attacks
- Each session has unique CSRF token
- Server validates tokens on every request

### 4. **Rate Limiting**
- Prevents DoS attacks and resource exhaustion
- Limits requests per IP address
- Automatic cleanup of old requests

### 5. **Command Sanitization**
- Blocks dangerous command patterns
- Whitelist-only command execution
- Prevents system compromise

## 📊 Monitoring and Logging

### Security Audit Logs

All security events are logged to `security_audit.log`:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event_type": "command_execution",
  "client_ip": "127.0.0.1",
  "session_id": "abc123...",
  "command": "squashplot --status",
  "success": true,
  "risk_level": "LOW"
}
```

### Event Types Logged

- **Session Creation**: New session created
- **Session Validation**: Session validation attempts
- **Command Execution**: Command execution attempts
- **Authentication Failures**: Failed authentication attempts
- **Rate Limiting**: Rate limit violations
- **Dangerous Commands**: Blocked dangerous commands

## 🔍 Troubleshooting

### Common Issues

#### **"Session validation failed"**
- Check if session has expired
- Verify CSRF token is correct
- Ensure IP address hasn't changed

#### **"Rate limit exceeded"**
- Wait for rate limit to reset (1 minute)
- Reduce request frequency
- Check for multiple clients from same IP

#### **"Command blocked"**
- Command not in whitelist
- Contains dangerous patterns
- Exceeds length limits

#### **"Cannot connect to bridge"**
- Ensure bridge is running on port 8443
- Check firewall settings
- Verify network connectivity

### Debug Mode

Enable debug logging by setting log level to DEBUG in the bridge configuration.

## 🚨 Security Recommendations

### 1. **Production Deployment**
- Change default session timeout based on security requirements
- Enable HTTPS for web interface
- Use strong, unique CSRF tokens
- Regular security audit log review

### 2. **Network Security**
- Run bridge on localhost only (127.0.0.1)
- Use firewall rules to restrict access
- Consider VPN for remote access
- Monitor network traffic

### 3. **System Security**
- Run bridge with minimal privileges
- Regular security updates
- Monitor system resources
- Backup security logs

## 📈 Future Enhancements

### Planned Security Features

1. **Multi-Factor Authentication**
   - SMS/Email verification
   - Hardware token support
   - Biometric authentication

2. **Advanced Session Management**
   - Session fingerprinting
   - Device binding
   - Geographic restrictions

3. **Enhanced Monitoring**
   - Real-time security dashboard
   - Automated threat detection
   - Security alert system

4. **Compliance Features**
   - GDPR compliance
   - SOC 2 compliance
   - Audit trail improvements

## 📞 Support

For security-related issues or questions:

1. Check the security audit logs
2. Review this documentation
3. Test with the web interface
4. Contact the development team

---

**⚠️ Security Warning**: This system executes commands on your local machine. Ensure you understand the security implications before deploying in production environments.
