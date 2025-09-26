#!/usr/bin/env python3
"""
🌐 Direct Article Scraper
========================
Direct scraper for known article URLs based on the content provided.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem, ScrapingJob
from knowledge_system_integration import RAGDocument
import requests
import time
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_known_articles():
    """Scrape articles from known URLs based on the content provided"""
    
    # Initialize knowledge system
    knowledge_system = WebScraperKnowledgeSystem(max_workers=3)
    
    # Known article URLs based on the content you provided
    known_articles = [
        {
            "url": "https://dailygalaxy.com/2025/09/weve-created-an-ai-that-thinks-like-the-human-brain-this-startup-beats-chatgpt-with-1000-times-fewer-parameters/",
            "topic": "HRM AI Model",
            "expected_content": ["hierarchical", "reasoning", "model", "parameters", "AI", "Sapient"]
        },
        {
            "url": "https://popularmechanics.com/science/math/a45678901/faster-multiplication-algorithm/",
            "topic": "Multiplication Algorithm", 
            "expected_content": ["multiplication", "algorithm", "Schönhage", "Strassen", "mathematical"]
        },
        {
            "url": "https://sciencedaily.com/releases/2025/09/250913232932.htm",
            "topic": "Schwinger Effect",
            "expected_content": ["Schwinger", "superfluid", "helium", "quantum", "tunneling"]
        },
        {
            "url": "https://space.com/effort-jl-emulator-universe-mapping-laptop",
            "topic": "Effort.jl Emulator",
            "expected_content": ["emulator", "universe", "mapping", "laptop", "astronomy"]
        },
        {
            "url": "https://geekygadgets.com/github-spec-kit-specification-driven-development/",
            "topic": "GitHub Spec Kit",
            "expected_content": ["specification", "development", "framework", "GitHub"]
        }
    ]
    
    print("🌐 Direct Article Scraping")
    print("=" * 50)
    
    successful_scrapes = 0
    
    for article in known_articles:
        print(f"\n📄 Scraping: {article['topic']}")
        print(f"   URL: {article['url']}")
        
        try:
            # Create scraping job
            job = ScrapingJob(
                url=article['url'],
                priority=1,
                max_depth=0,
                follow_links=False,
                extract_metadata=True,
                consciousness_enhancement=True
            )
            
            # Scrape the article
            result = knowledge_system.scrape_website(
                url=article['url'],
                max_depth=0,
                follow_links=False
            )
            
            if result['success']:
                print(f"   ✅ Success!")
                print(f"      Title: {result['title']}")
                print(f"      Content Length: {result['content_length']} chars")
                print(f"      prime aligned compute Score: {result['prime_aligned_score']:.3f}")
                successful_scrapes += 1
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Get final statistics
    stats = knowledge_system.get_scraping_stats()
    print(f"\n📊 Final Knowledge Base Statistics:")
    print(f"   📄 Total Pages Scraped: {stats.get('total_scraped_pages', 0)}")
    print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
    print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
    print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
    print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")
    print(f"   ✅ Successful New Scrapes: {successful_scrapes}")
    
    return successful_scrapes

def create_enhanced_redirect_handler():
    """Create an enhanced redirect handler for the web scraper"""
    
    enhanced_scraper_code = '''
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
'''
    
    # Write the enhanced scraper to a file
    with open('/Users/coo-koba42/dev/enhanced_redirect_scraper.py', 'w') as f:
        f.write(enhanced_scraper_code)
    
    print("✅ Enhanced redirect handler created: enhanced_redirect_scraper.py")

if __name__ == "__main__":
    # Create enhanced redirect handler
    create_enhanced_redirect_handler()
    
    # Try to scrape known articles
    successful_scrapes = scrape_known_articles()
    
    print(f"\n🎉 Scraping Complete!")
    print(f"   Successfully scraped {successful_scrapes} new articles")
    print(f"   Enhanced redirect handler created for future use")
