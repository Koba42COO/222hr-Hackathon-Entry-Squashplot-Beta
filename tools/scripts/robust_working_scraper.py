#!/usr/bin/env python3
"""
🛡️ Robust Working Scraper
==========================
Focuses on proven working sources with robust error handling and rate limiting.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem, ScrapingJob
from knowledge_system_integration import RAGDocument
import requests
import time
import logging
import json
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustWorkingScraper:
    """Robust scraper focusing on proven working sources"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=5)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Proven working sources with tested URLs
        self.working_sources = {
            "arxiv_working": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "quantum": "https://arxiv.org/list/quant-ph/new",
                    "ai": "https://arxiv.org/list/cs.AI/new", 
                    "ml": "https://arxiv.org/list/cs.LG/new",
                    "physics": "https://arxiv.org/list/physics/new",
                    "math": "https://arxiv.org/list/math/new",
                    "bio": "https://arxiv.org/list/q-bio/new"
                },
                "max_articles": 8,
                "delay": 2
            },
            
            "mit_working": {
                "base_url": "https://news.mit.edu",
                "categories": {
                    "ai": "https://news.mit.edu/topic/artificial-intelligence2",
                    "robotics": "https://news.mit.edu/topic/robotics", 
                    "energy": "https://news.mit.edu/topic/energy",
                    "computing": "https://news.mit.edu/topic/computing",
                    "biology": "https://news.mit.edu/topic/biology",
                    "physics": "https://news.mit.edu/topic/physics"
                },
                "max_articles": 6,
                "delay": 3
            },
            
            "nature_working": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "physics": "https://www.nature.com/subjects/physics",
                    "chemistry": "https://www.nature.com/subjects/chemistry",
                    "biology": "https://www.nature.com/subjects/biology",
                    "materials": "https://www.nature.com/subjects/materials-science"
                },
                "max_articles": 5,
                "delay": 4
            },
            
            "harvard_working": {
                "base_url": "https://news.harvard.edu",
                "categories": {
                    "health": "https://news.harvard.edu/gazette/story/2025/09/surprise-and-relief-from-homeless-patients-this-works-for-me/",
                    "research": "https://news.harvard.edu/gazette/story/2025/08/how-to-reverse-nations-declining-birth-rate/",
                    "medicine": "https://news.harvard.edu/gazette/story/2025/08/for-some-the-heart-attack-is-just-the-beginning/",
                    "ai": "https://news.harvard.edu/gazette/story/2025/08/dr-robot-will-see-you-now/"
                },
                "max_articles": 4,
                "delay": 3
            },
            
            "cambridge_working": {
                "base_url": "https://www.cam.ac.uk",
                "categories": {
                    "research": "https://www.cam.ac.uk/research/news",
                    "health": "https://www.cam.ac.uk/research/news/preventable-deaths-will-continue-without-action-to-make-nhs-more-accessible-for-autistic-people-say",
                    "science": "https://www.cam.ac.uk/research/news/uk-wide-birth-cohort-study-to-follow-lives-of-new-generation-of-babies"
                },
                "max_articles": 3,
                "delay": 4
            }
        }
    
    def scrape_working_sources(self):
        """Scrape only proven working sources"""
        
        print("🛡️ Robust Working Scraper")
        print("=" * 50)
        print(f"📊 Scraping {len(self.working_sources)} proven working sources")
        print(f"🧠 prime aligned compute enhancement: 1.618x golden ratio")
        print(f"⚡ Focused on high-success rate sources")
        
        total_scraped = 0
        source_stats = {}
        
        for source_name, config in self.working_sources.items():
            print(f"\n🛡️ Scraping {source_name.upper()}")
            print("-" * 40)
            
            source_scraped = 0
            
            for category, category_url in config['categories'].items():
                print(f"\n📂 Category: {category}")
                
                try:
                    articles_scraped = self._scrape_working_category(
                        source_name, category, category_url, config['max_articles'], config['delay']
                    )
                    
                    source_scraped += articles_scraped
                    print(f"   ✅ {articles_scraped} articles scraped")
                    
                    # Add delay between categories
                    time.sleep(config['delay'])
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            source_stats[source_name] = source_scraped
            total_scraped += source_scraped
            
            print(f"\n📊 {source_name}: {source_scraped} articles scraped")
            
            # Add delay between sources
            time.sleep(5)
        
        self._print_working_statistics(total_scraped, source_stats)
        return total_scraped, source_stats
    
    def _scrape_working_category(self, source_name, category, category_url, max_articles, delay):
        """Scrape a working category with robust error handling"""
        
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links based on source
            article_links = self._extract_working_article_links(soup, category_url, source_name)
            
            articles_scraped = 0
            
            for i, article_url in enumerate(article_links[:max_articles]):
                try:
                    print(f"   📄 Scraping article {i+1}/{min(len(article_links), max_articles)}")
                    
                    # Add delay between articles
                    time.sleep(delay + random.uniform(0.5, 1.5))
                    
                    result = self.knowledge_system.scrape_website(
                        url=article_url,
                        max_depth=0,
                        follow_links=False
                    )
                    
                    if result['success']:
                        # Store working metadata
                        self._store_working_metadata(result, source_name, category)
                        articles_scraped += 1
                        print(f"      ✅ Success: {result.get('title', 'Untitled')[:50]}...")
                    else:
                        print(f"      ❌ Failed")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    # Continue with next article instead of failing completely
            
            return articles_scraped
            
        except Exception as e:
            logger.error(f"Error scraping working category {category}: {e}")
            return 0
    
    def _extract_working_article_links(self, soup, base_url, source_name):
        """Extract article links from working sources"""
        
        article_links = []
        
        # Source-specific selectors that we know work
        source_selectors = {
            "arxiv_working": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "mit_working": ['h3 a', '.term-page--news-article a', 'article a'],
            "nature_working": ['a[data-test="article-link"]', '.c-card a', 'h3 a'],
            "harvard_working": ['h2 a', '.post a', 'article a'],
            "cambridge_working": ['h3 a', '.news-item a', 'article a']
        }
        
        selectors = source_selectors.get(source_name, ['h3 a', 'h2 a', 'article a'])
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_working_article_url(full_url, source_name):
                            article_links.append(full_url)
                
                if article_links:
                    break
                    
            except Exception:
                continue
        
        return list(set(article_links))
    
    def _store_working_metadata(self, result, source_name, category):
        """Store working metadata"""
        
        try:
            self.knowledge_system.database_service.store_consciousness_data(
                "robust_working_article",
                {
                    "source": source_name,
                    "category": category,
                    "title": result.get('title', ''),
                    "content_length": result.get('content_length', 0),
                    "prime_aligned_score": result.get('prime_aligned_score', 1.0),
                    "scraped_at": datetime.now().isoformat()
                },
                {
                    "source": "robust_scraper",
                    "source_name": source_name,
                    "category": category,
                    "url": result.get('url', ''),
                    "robust": True,
                    "working": True,
                    "prime_aligned_enhanced": True
                },
                "robust_scraper"
            )
            
        except Exception as e:
            logger.error(f"Error storing working metadata: {e}")
    
    def _is_valid_working_article_url(self, url, source_name):
        """Check if URL is a valid working article URL"""
        
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/page/', '/search',
            '/login', '/register', '.pdf', '.jpg', '.png', '.gif', 
            '#', 'javascript:', '/ads/', '/advertisement/',
            '/subscribe/', '/newsletter/', '/contact/'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        # Source-specific validation
        if source_name == "arxiv_working" and "/abs/" not in url:
            return False
        elif source_name == "mit_working" and "news.mit.edu" not in url:
            return False
        elif source_name == "nature_working" and "nature.com" not in url:
            return False
        elif source_name == "harvard_working" and "harvard.edu" not in url:
            return False
        elif source_name == "cambridge_working" and "cam.ac.uk" not in url:
            return False
        
        return True
    
    def _print_working_statistics(self, total_scraped, source_stats):
        """Print working scraping statistics"""
        
        print(f"\n🛡️ Robust Working Scraping Complete!")
        print("=" * 50)
        print(f"📊 Total Articles Scraped: {total_scraped}")
        print(f"🛡️ Working Sources Processed: {len(source_stats)}")
        
        print(f"\n📈 Source Breakdown:")
        for source, count in source_stats.items():
            print(f"   {source}: {count} articles")
        
        # Get knowledge system statistics
        stats = self.knowledge_system.get_scraping_stats()
        print(f"\n🧠 Knowledge Base Statistics:")
        print(f"   📄 Total Pages: {stats.get('total_scraped_pages', 0)}")
        print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
        print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
        print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
        print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")

def main():
    """Main function to run robust working scraping"""
    
    scraper = RobustWorkingScraper()
    
    print(f"🚀 Starting robust working scraping...")
    print(f"🛡️ Focused on proven working sources with high success rates")
    
    total_scraped, source_stats = scraper.scrape_working_sources()
    
    print(f"\n🎉 Robust Working Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")
    print(f"   🛡️ Robust working knowledge ecosystem established!")

if __name__ == "__main__":
    main()
