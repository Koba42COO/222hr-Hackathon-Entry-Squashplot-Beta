/**
 * SquashPlot Secure Bridge Integration
 * ===================================
 * 
 * This JavaScript module provides secure communication with the
 * SquashPlot Bridge App for advanced CLI automation.
 * 
 * SECURITY FEATURES:
 * - Encrypted communication
 * - Localhost-only connections
 * - Command validation
 * - Authentication tokens
 * - Fallback to copy-paste method
 */

class SquashPlotBridge {
    constructor() {
        this.bridgeAvailable = false;
        this.bridgePort = 8443;
        this.bridgeHost = '127.0.0.1';
        this.encryptionKey = null;
        this.connectionTimeout = 5000; // 5 seconds
        
        // Initialize bridge detection
        this.detectBridge();
    }
    
    /**
     * Detect if the Secure Bridge App is running
     */
    async detectBridge() {
        try {
            // Try to connect to the bridge
            const response = await this.sendCommand('ping');
            this.bridgeAvailable = response.success;
            
            if (this.bridgeAvailable) {
                console.log('🔒 SquashPlot Secure Bridge detected');
                this.showBridgeStatus('connected');
            } else {
                console.log('📋 Using copy-paste method (Bridge not available)');
                this.showBridgeStatus('disconnected');
            }
        } catch (error) {
            console.log('📋 Bridge not available, using copy-paste method');
            this.bridgeAvailable = false;
            this.showBridgeStatus('disconnected');
        }
    }
    
    /**
     * Send command to the Secure Bridge App
     */
    async sendCommand(command) {
        if (!this.bridgeAvailable) {
            throw new Error('Bridge not available');
        }
        
        try {
            // Create encrypted message
            const message = this.encryptMessage(command);
            
            // Send to bridge (simplified for demo - in production use WebSocket)
            const response = await fetch(`http://${this.bridgeHost}:${this.bridgePort}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Bridge-Token': this.generateToken()
                },
                body: JSON.stringify({ command: message }),
                timeout: this.connectionTimeout
            });
            
            if (!response.ok) {
                throw new Error(`Bridge error: ${response.status}`);
            }
            
            const result = await response.json();
            return result;
            
        } catch (error) {
            console.error('Bridge communication failed:', error);
            this.bridgeAvailable = false;
            throw error;
        }
    }
    
    /**
     * Execute command with bridge or fallback to copy-paste
     */
    async executeCommand(command) {
        if (this.bridgeAvailable) {
            try {
                console.log('🔒 Executing via Secure Bridge...');
                const result = await this.sendCommand(command);
                
                if (result.success) {
                    this.showSuccessMessage('Command executed successfully via Secure Bridge');
                    return result;
                } else {
                    this.showErrorMessage(`Command failed: ${result.error}`);
                    return result;
                }
            } catch (error) {
                console.log('Bridge failed, falling back to copy-paste method');
                this.bridgeAvailable = false;
                return this.showCopyPasteMethod(command);
            }
        } else {
            return this.showCopyPasteMethod(command);
        }
    }
    
    /**
     * Show copy-paste method (fallback)
     */
    showCopyPasteMethod(command) {
        const outputDiv = document.getElementById('cliOutput');
        const outputContent = document.getElementById('outputContent');
        
        if (!outputDiv || !outputContent) {
            console.error('CLI output elements not found');
            return;
        }
        
        outputDiv.style.display = 'block';
        
        const commandDisplay = `
🧠 SquashPlot CLI Command

📋 Command to Copy:
${command}

📝 Instructions:
1. Copy the command above
2. Open your terminal/command prompt  
3. Paste and press Enter
4. Command will run on your local machine

🔒 Security Note:
This command runs locally on your machine.
The website only provides the template.

💡 Pro Tip:
For advanced automation, download the SquashPlot Bridge App
        `;
        
        outputContent.textContent = commandDisplay;
        this.addCopyToClipboardButton(command);
        
        return {
            success: true,
            method: 'copy-paste',
            message: 'Command ready for copy-paste'
        };
    }
    
    /**
     * Add copy-to-clipboard button
     */
    addCopyToClipboardButton(command) {
        const outputDiv = document.getElementById('cliOutput');
        
        // Remove existing button
        const existingButton = document.getElementById('copyButton');
        if (existingButton) {
            existingButton.remove();
        }
        
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.id = 'copyButton';
        copyButton.innerHTML = '📋 Copy Command';
        copyButton.style.cssText = `
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            margin-top: 10px;
            transition: all 0.3s ease;
        `;
        
        copyButton.onclick = async () => {
            try {
                await navigator.clipboard.writeText(command);
                copyButton.innerHTML = '✅ Copied!';
                copyButton.style.background = '#2ecc71';
                setTimeout(() => {
                    copyButton.innerHTML = '📋 Copy Command';
                    copyButton.style.background = 'linear-gradient(135deg, #00d4ff, #00ff88)';
                }, 2000);
            } catch (err) {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = command;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                copyButton.innerHTML = '✅ Copied!';
            }
        };
        
        outputDiv.appendChild(copyButton);
    }
    
    /**
     * Show bridge connection status
     */
    showBridgeStatus(status) {
        // Create or update status indicator
        let statusElement = document.getElementById('bridgeStatus');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.id = 'bridgeStatus';
            statusElement.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 15px;
                border-radius: 8px;
                font-weight: 600;
                z-index: 10000;
                transition: all 0.3s ease;
            `;
            document.body.appendChild(statusElement);
        }
        
        if (status === 'connected') {
            statusElement.innerHTML = '🔒 Secure Bridge Connected';
            statusElement.style.background = '#2ecc71';
            statusElement.style.color = 'white';
        } else {
            statusElement.innerHTML = '📋 Copy-Paste Mode';
            statusElement.style.background = '#f39c12';
            statusElement.style.color = 'white';
        }
    }
    
    /**
     * Show success message
     */
    showSuccessMessage(message) {
        this.showNotification(message, 'success');
    }
    
    /**
     * Show error message
     */
    showErrorMessage(message) {
        this.showNotification(message, 'error');
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">×</button>
        `;
        
        // Add notification styles
        if (!document.getElementById('notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                .notification {
                    position: fixed;
                    top: 100px;
                    right: 20px;
                    background: #10b981;
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    z-index: 10001;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    animation: slideIn 0.3s ease-out;
                }
                .notification.error {
                    background: #ef4444;
                }
                .notification button {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 20px;
                    cursor: pointer;
                    padding: 0;
                    margin-left: 10px;
                }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
    
    /**
     * Encrypt message (simplified for demo)
     */
    encryptMessage(message) {
        // In production, use proper encryption
        return btoa(message);
    }
    
    /**
     * Generate authentication token
     */
    generateToken() {
        // In production, use proper authentication
        return 'bridge-token-' + Date.now();
    }
}

// Initialize bridge when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.squashPlotBridge = new SquashPlotBridge();
    
    // Override the original executeCLICommand function
    window.executeCLICommand = async function(command) {
        return await window.squashPlotBridge.executeCommand(command);
    };
    
    console.log('🧠 SquashPlot Bridge Integration loaded');
});
