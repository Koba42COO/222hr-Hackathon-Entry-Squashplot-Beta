
# Enhanced Web Scraper with Better Redirect Handling
import requests
from urllib.parse import urljoin, urlparse
import time

class EnhancedWebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.max_redirects = 10
        self.timeout = 30
        
    def scrape_with_redirect_handling(self, url):
        """Scrape URL with comprehensive redirect handling"""
        try:
            # First, resolve all redirects
            final_url = self._resolve_redirects(url)
            
            # Then scrape the final URL
            response = self.session.get(final_url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            return {
                'success': True,
                'url': final_url,
                'content': response.text,
                'status_code': response.status_code
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _resolve_redirects(self, url):
        """Resolve all redirects to get final URL"""
        current_url = url
        redirect_count = 0
        
        while redirect_count < self.max_redirects:
            try:
                response = self.session.head(current_url, allow_redirects=False, timeout=10)
                
                if response.status_code in [301, 302, 303, 307, 308]:
                    redirect_url = response.headers.get('Location')
                    if redirect_url:
                        current_url = urljoin(current_url, redirect_url)
                        redirect_count += 1
                    else:
                        break
                else:
                    break
                    
            except Exception:
                break
                
        return current_url
