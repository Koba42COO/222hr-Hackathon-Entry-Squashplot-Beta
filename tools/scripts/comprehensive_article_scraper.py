#!/usr/bin/env python3
"""
🌐 Comprehensive Article Scraper
===============================
Scraper to find and extract content from the actual articles referenced in the search results.
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

class ComprehensiveArticleScraper:
    """Comprehensive scraper for finding and extracting article content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=3)
        
    def search_and_scrape_articles(self):
        """Search for and scrape the missing articles"""
        
        # Based on the web search results, let's try to find the actual articles
        article_searches = [
            {
                "topic": "GitHub Spec Kit",
                "search_terms": ["GitHub Spec Kit", "specification driven development", "better plans better code"],
                "expected_content": ["specification", "development", "framework", "GitHub"]
            },
            {
                "topic": "Effort.jl Emulator",
                "search_terms": ["Effort.jl", "universe mapping", "laptop", "computational astronomy"],
                "expected_content": ["emulator", "universe", "mapping", "laptop", "astronomy"]
            },
            {
                "topic": "Schwinger Effect",
                "search_terms": ["Schwinger effect", "superfluid helium", "something from nothing", "quantum tunneling"],
                "expected_content": ["Schwinger", "superfluid", "helium", "quantum", "tunneling"]
            },
            {
                "topic": "HRM AI Model",
                "search_terms": ["Hierarchical Reasoning Model", "HRM", "Sapient", "1000 times fewer parameters"],
                "expected_content": ["hierarchical", "reasoning", "model", "parameters", "AI"]
            },
            {
                "topic": "Multiplication Algorithm",
                "search_terms": ["faster multiplication", "Schönhage-Strassen", "algorithm", "mathematical breakthrough"],
                "expected_content": ["multiplication", "algorithm", "Schönhage", "Strassen", "mathematical"]
            }
        ]
        
        print("🔍 Comprehensive Article Search and Scrape")
        print("=" * 60)
        
        for article_info in article_searches:
            print(f"\n📄 Searching for: {article_info['topic']}")
            self._search_and_scrape_article(article_info)
            
    def _search_and_scrape_article(self, article_info):
        """Search for and scrape a specific article"""
        
        # Try common article URLs based on the search terms
        potential_urls = self._generate_potential_urls(article_info)
        
        for url in potential_urls:
            try:
                print(f"   🔍 Trying: {url}")
                result = self._extract_content_from_url(url)
                
                if result['success'] and self._validate_content(result['content'], article_info['expected_content']):
                    print(f"   ✅ Found valid content!")
                    print(f"      Title: {result['title']}")
                    print(f"      Content Length: {len(result['content'])} chars")
                    
                    # Store in knowledge system
                    self._store_article_in_knowledge_system(result, article_info['topic'])
                    return True
                else:
                    print(f"   ❌ No valid content found")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
        print(f"   ⚠️  No valid content found for {article_info['topic']}")
        return False
    
    def _generate_potential_urls(self, article_info):
        """Generate potential URLs based on search terms"""
        
        # Common domains that might have these articles
        domains = [
            "geekygadgets.com",
            "energy-reporters.com", 
            "sciencedaily.com",
            "space.com",
            "popularmechanics.com",
            "dailygalaxy.com",
            "infoq.com",
            "arxiv.org"
        ]
        
        urls = []
        
        # Try to construct URLs based on common patterns
        for domain in domains:
            for term in article_info['search_terms'][:2]:  # Use first 2 terms
                # Convert to URL-friendly format
                url_term = term.lower().replace(' ', '-').replace('"', '').replace("'", '')
                urls.append(f"https://{domain}/{url_term}")
                urls.append(f"https://{domain}/articles/{url_term}")
                urls.append(f"https://{domain}/news/{url_term}")
        
        return urls[:10]  # Limit to first 10 URLs
    
    def _extract_content_from_url(self, url):
        """Extract content from a URL"""
        try:
            response = self.session.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'success': len(content) > 500  # Must have substantial content
            }
            
        except Exception as e:
            return {
                'url': url,
                'title': 'Failed',
                'content': '',
                'success': False,
                'error': str(e)
            }
    
    def _extract_title(self, soup, url):
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
            
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
            text = main_content.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        return ""
    
    def _validate_content(self, content, expected_terms):
        """Validate that content contains expected terms"""
        content_lower = content.lower()
        matches = sum(1 for term in expected_terms if term.lower() in content_lower)
        return matches >= len(expected_terms) // 2  # At least half the terms must match
    
    def _store_article_in_knowledge_system(self, result, topic):
        """Store article in the knowledge system"""
        try:
            # Create scraping job
            job = ScrapingJob(
                url=result['url'],
                priority=1,
                max_depth=0,
                follow_links=False,
                extract_metadata=True,
                consciousness_enhancement=True
            )
            
            # Scrape and store
            scrape_result = self.knowledge_system.scrape_website(
                url=result['url'],
                max_depth=0,
                follow_links=False
            )
            
            if scrape_result['success']:
                print(f"      💾 Stored in knowledge system successfully!")
            else:
                print(f"      ❌ Failed to store in knowledge system")
                
        except Exception as e:
            print(f"      ❌ Error storing in knowledge system: {e}")

def main():
    """Main function to run comprehensive article scraping"""
    scraper = ComprehensiveArticleScraper()
    scraper.search_and_scrape_articles()
    
    # Get final statistics
    stats = scraper.knowledge_system.get_scraping_stats()
    print(f"\n📊 Final Knowledge Base Statistics:")
    print(f"   📄 Total Pages Scraped: {stats.get('total_scraped_pages', 0)}")
    print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
    print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
    print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
    print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")

if __name__ == "__main__":
    main()
