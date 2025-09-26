#!/usr/bin/env python3
"""
🌐 Enhanced Web Scraper with Redirect Handling
============================================
Improved web scraper that can handle redirect URLs and find actual article content.
"""

import requests
import time
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedWebScraper:
    """Enhanced web scraper with redirect handling and content extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.max_redirects = 10
        
    def follow_redirects(self, url):
        """Follow redirects to find the final destination URL"""
        logger.info(f"🔍 Following redirects for: {url}")
        
        redirects = []
        current_url = url
        
        for i in range(self.max_redirects):
            try:
                response = self.session.head(current_url, allow_redirects=False, timeout=10)
                
                if response.status_code in [301, 302, 303, 307, 308]:
                    redirect_url = response.headers.get('Location')
                    if redirect_url:
                        redirect_url = urljoin(current_url, redirect_url)
                        redirects.append((current_url, redirect_url, response.status_code))
                        current_url = redirect_url
                        logger.info(f"   Redirect {i+1}: {current_url}")
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                logger.error(f"   Error following redirect: {e}")
                break
                
        logger.info(f"✅ Final URL after {len(redirects)} redirects: {current_url}")
        return current_url, redirects
    
    def extract_content_from_url(self, url):
        """Extract content from URL with redirect handling"""
        try:
            # First, follow redirects to get the final URL
            final_url, redirects = self.follow_redirects(url)
            
            # Now get the actual content
            logger.info(f"📄 Extracting content from final URL: {final_url}")
            response = self.session.get(final_url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, final_url)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, response, final_url)
            metadata['redirects'] = redirects
            metadata['original_url'] = url
            metadata['final_url'] = final_url
            
            return {
                'url': final_url,
                'original_url': url,
                'title': title,
                'content': content,
                'metadata': metadata,
                'redirects_count': len(redirects),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract content from {url}: {e}")
            return {
                'url': url,
                'title': 'Failed to extract',
                'content': '',
                'metadata': {'error': str(e)},
                'redirects_count': 0,
                'success': False
            }
    
    def _extract_title(self, soup, url):
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try Open Graph title
        og_title = soup.find('meta', property='og:title')
        if og_title:
            return og_title.get('content', '').strip()
            
        # Try h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
            
        # Fallback to URL
        return urlparse(url).path.split('/')[-1] or urlparse(url).netloc
    
    def _extract_main_content(self, soup):
        """Extract main content from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '.main-content', 
            '.post-content', '.entry-content', '#content',
            '.article-content', '.story-content', '.text-content'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Get text content
            text = main_content.get_text()
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        return ""
    
    def _extract_metadata(self, soup, response, url):
        """Extract metadata from page"""
        metadata = {
            'url': url,
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(response.content),
            'last_modified': response.headers.get('last-modified', ''),
            'language': response.headers.get('content-language', ''),
        }
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[f'meta_{name}'] = content
        
        # Extract Open Graph data
        og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
        for og in og_tags:
            property_name = og.get('property', '').replace('og:', '')
            content = og.get('content')
            if property_name and content:
                metadata[f'og_{property_name}'] = content
        
        return metadata

def test_redirect_urls():
    """Test the redirect URLs to find actual article content"""
    
    scraper = EnhancedWebScraper()
    
    # URLs that failed to extract content
    failed_urls = [
        "https://search.app/9wg9F",  # GitHub Spec Kit
        "https://search.app/V2h1k",  # Energy Reporters cosmic magnetic fields
        "https://search.app/Q8Bxf",  # ScienceDaily Schwinger effect
        "https://search.app/kJnPz",  # Space.com Effort.jl emulator
        "https://search.app/QxBct",  # Popular Mechanics multiplication algorithm
    ]
    
    print("🔍 Testing Redirect URLs")
    print("=" * 60)
    
    results = []
    for url in failed_urls:
        print(f"\n📄 Testing: {url}")
        result = scraper.extract_content_from_url(url)
        results.append(result)
        
        if result['success']:
            print(f"✅ Success!")
            print(f"   Final URL: {result.get('url', 'Unknown')}")
            print(f"   Title: {result['title']}")
            print(f"   Content Length: {len(result['content'])} chars")
            print(f"   Redirects: {result['redirects_count']}")
        else:
            print(f"❌ Failed: {result['metadata'].get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    test_redirect_urls()
