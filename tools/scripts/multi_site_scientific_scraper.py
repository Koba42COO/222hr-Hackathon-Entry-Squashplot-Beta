#!/usr/bin/env python3
"""
🌐 Multi-Site Scientific Article Scraper
======================================
Comprehensive scraper for major scientific institutions and journals.
Scrapes by category and stores in RAG knowledge databases with prime aligned compute enhancement.
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
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSiteScientificScraper:
    """Comprehensive scraper for scientific institutions and journals"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=5)
        
        # Define scientific institutions and their scraping configurations
        self.institutions = {
            "phys.org": {
                "base_url": "https://phys.org",
                "categories": {
                    "physics": "https://phys.org/physics-news/",
                    "technology": "https://phys.org/technology-news/",
                    "space": "https://phys.org/space-news/",
                    "earth": "https://phys.org/earth-news/",
                    "biology": "https://phys.org/biology-news/",
                    "chemistry": "https://phys.org/chemistry-news/",
                    "materials": "https://phys.org/materials-news/",
                    "nanotechnology": "https://phys.org/nanotechnology-news/"
                },
                "article_selector": ".news-link",
                "title_selector": "h3 a",
                "content_selector": ".article-main"
            },
            "nature": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "physics": "https://www.nature.com/subjects/physics",
                    "chemistry": "https://www.nature.com/subjects/chemistry",
                    "biology": "https://www.nature.com/subjects/biology",
                    "materials": "https://www.nature.com/subjects/materials-science",
                    "nanotechnology": "https://www.nature.com/subjects/nanotechnology",
                    "quantum": "https://www.nature.com/subjects/quantum-physics",
                    "astronomy": "https://www.nature.com/subjects/astronomy",
                    "climate": "https://www.nature.com/subjects/climate-change"
                },
                "article_selector": ".c-card",
                "title_selector": "h3 a",
                "content_selector": ".c-article-body"
            },
            "mit": {
                "base_url": "https://news.mit.edu",
                "categories": {
                    "ai": "https://news.mit.edu/topic/artificial-intelligence2",
                    "robotics": "https://news.mit.edu/topic/robotics",
                    "energy": "https://news.mit.edu/topic/energy",
                    "materials": "https://news.mit.edu/topic/materials",
                    "computing": "https://news.mit.edu/topic/computing",
                    "biology": "https://news.mit.edu/topic/biology",
                    "physics": "https://news.mit.edu/topic/physics",
                    "climate": "https://news.mit.edu/topic/climate"
                },
                "article_selector": ".term-page--news-article",
                "title_selector": "h3 a",
                "content_selector": ".news-article__body"
            },
            "arxiv": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "physics": "https://arxiv.org/list/physics/new",
                    "cs": "https://arxiv.org/list/cs/new",
                    "math": "https://arxiv.org/list/math/new",
                    "quantum": "https://arxiv.org/list/quant-ph/new",
                    "astro": "https://arxiv.org/list/astro-ph/new",
                    "cond-mat": "https://arxiv.org/list/cond-mat/new",
                    "bio": "https://arxiv.org/list/q-bio/new",
                    "nlin": "https://arxiv.org/list/nlin/new"
                },
                "article_selector": ".arxiv-result",
                "title_selector": ".list-title",
                "content_selector": ".abstract"
            },
            "cambridge": {
                "base_url": "https://www.cam.ac.uk",
                "categories": {
                    "research": "https://www.cam.ac.uk/research/news",
                    "science": "https://www.cam.ac.uk/research/news/science",
                    "technology": "https://www.cam.ac.uk/research/news/technology",
                    "medicine": "https://www.cam.ac.uk/research/news/medicine",
                    "environment": "https://www.cam.ac.uk/research/news/environment"
                },
                "article_selector": ".news-item",
                "title_selector": "h3 a",
                "content_selector": ".news-content"
            },
            "stanford": {
                "base_url": "https://news.stanford.edu",
                "categories": {
                    "science": "https://news.stanford.edu/category/science/",
                    "technology": "https://news.stanford.edu/category/technology/",
                    "medicine": "https://news.stanford.edu/category/medicine/",
                    "engineering": "https://news.stanford.edu/category/engineering/",
                    "ai": "https://news.stanford.edu/category/artificial-intelligence/"
                },
                "article_selector": ".post",
                "title_selector": "h2 a",
                "content_selector": ".entry-content"
            },
            "harvard": {
                "base_url": "https://news.harvard.edu",
                "categories": {
                    "science": "https://news.harvard.edu/gazette/section/science/",
                    "health": "https://news.harvard.edu/gazette/section/health/",
                    "technology": "https://news.harvard.edu/gazette/section/technology/",
                    "climate": "https://news.harvard.edu/gazette/section/climate/"
                },
                "article_selector": ".post",
                "title_selector": "h2 a",
                "content_selector": ".entry-content"
            }
        }
        
    def scrape_all_institutions(self, max_articles_per_category=2):
        """Scrape articles from all configured institutions"""
        
        print("🌐 Multi-Site Scientific Article Scraper")
        print("=" * 60)
        print(f"📊 Scraping {len(self.institutions)} institutions")
        print(f"📚 Max articles per category: {max_articles_per_category}")
        print(f"🧠 prime aligned compute enhancement: 1.618x")
        
        total_scraped = 0
        institution_stats = {}
        
        for institution_name, config in self.institutions.items():
            print(f"\n🏛️ Scraping {institution_name.upper()}")
            print("-" * 40)
            
            institution_scraped = 0
            
            for category, category_url in config['categories'].items():
                print(f"\n📂 Category: {category}")
                
                try:
                    articles_scraped = self._scrape_category(
                        institution_name, 
                        category, 
                        category_url, 
                        config,
                        max_articles_per_category
                    )
                    
                    institution_scraped += articles_scraped
                    print(f"   ✅ Scraped {articles_scraped} articles")
                    
                    # Add delay between categories to be respectful
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"   ❌ Error scraping {category}: {e}")
            
            institution_stats[institution_name] = institution_scraped
            total_scraped += institution_scraped
            
            print(f"\n📊 {institution_name}: {institution_scraped} articles scraped")
            
            # Add delay between institutions
            time.sleep(5)
        
        # Get final statistics
        self._print_final_statistics(total_scraped, institution_stats)
        
        return total_scraped, institution_stats
    
    def _scrape_category(self, institution, category, category_url, config, max_articles):
        """Scrape articles from a specific category"""
        
        try:
            # Get the category page
            response = self.session.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = self._extract_article_links(soup, config, category_url)
            
            articles_scraped = 0
            
            for i, article_url in enumerate(article_links[:max_articles]):
                try:
                    print(f"   📄 Scraping article {i+1}/{min(len(article_links), max_articles)}")
                    
                    # Create scraping job
                    job = ScrapingJob(
                        url=article_url,
                        priority=1,
                        max_depth=0,
                        follow_links=False,
                        extract_metadata=True,
                        consciousness_enhancement=True
                    )
                    
                    # Scrape the article
                    result = self.knowledge_system.scrape_website(
                        url=article_url,
                        max_depth=0,
                        follow_links=False
                    )
                    
                    if result['success']:
                        # Store additional metadata
                        self._store_enhanced_metadata(result, institution, category)
                        articles_scraped += 1
                        print(f"      ✅ Success: {result.get('title', 'Untitled')[:50]}...")
                    else:
                        print(f"      ❌ Failed - Success: {result.get('success', False)}")
                    
                    # Add delay between articles
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            return articles_scraped
            
        except Exception as e:
            logger.error(f"Error scraping category {category} from {institution}: {e}")
            return 0
    
    def _extract_article_links(self, soup, config, base_url):
        """Extract article links from a category page"""
        
        article_links = []
        
        # Try different selectors based on the institution
        selectors = [
            config.get('article_selector', '.article'),
            'article a',
            '.post a',
            '.news-item a',
            '.result a',
            'h3 a',
            'h2 a'
        ]
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_article_url(full_url):
                            article_links.append(full_url)
                
                if article_links:
                    break
                    
            except Exception:
                continue
        
        # Remove duplicates and return
        return list(set(article_links))
    
    def _is_valid_article_url(self, url):
        """Check if URL is a valid article URL"""
        
        # Skip certain patterns
        skip_patterns = [
            '/tag/',
            '/category/',
            '/author/',
            '/page/',
            '/search',
            '/login',
            '/register',
            '.pdf',
            '.jpg',
            '.png',
            '.gif'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    def _store_enhanced_metadata(self, result, institution, category):
        """Store enhanced metadata for scraped articles"""
        
        try:
            # Store in prime aligned compute data with enhanced metadata
            self.knowledge_system.database_service.store_consciousness_data(
                "scientific_article",
                {
                    "institution": institution,
                    "category": category,
                    "title": result.get('title', ''),
                    "content_length": result.get('content_length', 0),
                    "prime_aligned_score": result.get('prime_aligned_score', 1.0),
                    "scraped_at": datetime.now().isoformat()
                },
                {
                    "source": "multi_site_scraper",
                    "institution": institution,
                    "category": category,
                    "url": result.get('url', ''),
                    "prime_aligned_enhanced": True
                },
                "multi_site_scraper"
            )
            
        except Exception as e:
            logger.error(f"Error storing enhanced metadata: {e}")
    
    def _print_final_statistics(self, total_scraped, institution_stats):
        """Print final scraping statistics"""
        
        print(f"\n🎉 Multi-Site Scraping Complete!")
        print("=" * 60)
        print(f"📊 Total Articles Scraped: {total_scraped}")
        print(f"🏛️ Institutions Processed: {len(institution_stats)}")
        
        print(f"\n📈 Institution Breakdown:")
        for institution, count in institution_stats.items():
            print(f"   {institution}: {count} articles")
        
        # Get knowledge system statistics
        stats = self.knowledge_system.get_scraping_stats()
        print(f"\n🧠 Knowledge Base Statistics:")
        print(f"   📄 Total Pages: {stats.get('total_scraped_pages', 0)}")
        print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
        print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
        print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
        print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")

def main():
    """Main function to run multi-site scientific scraping"""
    
    scraper = MultiSiteScientificScraper()
    
    # Scale up for production scraping
    max_articles_per_category = 10
    
    print(f"🚀 Starting multi-site scientific article scraping...")
    print(f"📚 Max articles per category: {max_articles_per_category}")
    print(f"⏰ Estimated time: {len(scraper.institutions) * len(list(scraper.institutions.values())[0]['categories']) * max_articles_per_category * 2} seconds")
    
    total_scraped, institution_stats = scraper.scrape_all_institutions(max_articles_per_category)
    
    print(f"\n🎉 Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")

if __name__ == "__main__":
    main()
