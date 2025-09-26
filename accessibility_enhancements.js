
// Accessibility Enhancements for SquashPlot Dashboard

class AccessibilityManager {
    constructor() {
        this.init();
    }
    
    init() {
        this.addSkipLinks();
        this.enhanceKeyboardNavigation();
        this.addARIALabels();
        this.implementFocusManagement();
        this.addScreenReaderSupport();
    }
    
    addSkipLinks() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-link';
        skipLink.textContent = 'Skip to main content';
        document.body.insertBefore(skipLink, document.body.firstChild);
    }
    
    enhanceKeyboardNavigation() {
        // Add keyboard navigation for all interactive elements
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-nav');
            }
        });
        
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-nav');
        });
    }
    
    addARIALabels() {
        // Add ARIA labels to interactive elements
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            if (!button.getAttribute('aria-label')) {
                button.setAttribute('aria-label', button.textContent || 'Button');
            }
        });
        
        const inputs = document.querySelectorAll('input');
        inputs.forEach(input => {
            if (!input.getAttribute('aria-label')) {
                const label = document.querySelector(`label[for="${input.id}"]`);
                if (label) {
                    input.setAttribute('aria-label', label.textContent);
                }
            }
        });
    }
    
    implementFocusManagement() {
        // Manage focus for modals and dialogs
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modal = document.querySelector('.modal:not([style*="display: none"])');
                if (modal) {
                    modal.style.display = 'none';
                    const focusable = modal.querySelector('button, input, select, textarea, a[href]');
                    if (focusable) {
                        focusable.focus();
                    }
                }
            }
        });
    }
    
    addScreenReaderSupport() {
        // Add live regions for dynamic content
        const liveRegion = document.createElement('div');
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        liveRegion.id = 'live-region';
        document.body.appendChild(liveRegion);
        
        // Announce status changes
        this.announce = (message) => {
            const liveRegion = document.getElementById('live-region');
            if (liveRegion) {
                liveRegion.textContent = message;
            }
        };
    }
}

// Initialize accessibility manager
document.addEventListener('DOMContentLoaded', () => {
    new AccessibilityManager();
});
