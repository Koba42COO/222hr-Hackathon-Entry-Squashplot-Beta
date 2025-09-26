#!/usr/bin/env python3
"""
🚀 Massive Scientific Scraping System
=====================================
Comprehensive scaling system for large-scale scientific knowledge collection.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem, ScrapingJob
from knowledge_system_integration import RAGDocument
import asyncio
import concurrent.futures
import time
import logging
import json
from datetime import datetime
import threading
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MassiveScientificScraper:
    """Massive scaling system for scientific knowledge collection"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=10)
        
        # Massive list of scientific institutions and categories
        self.massive_sites = {
            "arxiv": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "quantum_physics": "https://arxiv.org/list/quant-ph/new",
                    "condensed_matter": "https://arxiv.org/list/cond-mat/new",
                    "high_energy_physics": "https://arxiv.org/list/hep-th/new",
                    "astrophysics": "https://arxiv.org/list/astro-ph/new",
                    "computer_science": "https://arxiv.org/list/cs/new",
                    "machine_learning": "https://arxiv.org/list/cs.LG/new",
                    "artificial_intelligence": "https://arxiv.org/list/cs.AI/new",
                    "mathematics": "https://arxiv.org/list/math/new",
                    "statistics": "https://arxiv.org/list/stat/new",
                    "biology": "https://arxiv.org/list/q-bio/new",
                    "neural_networks": "https://arxiv.org/list/cs.NE/new",
                    "computer_vision": "https://arxiv.org/list/cs.CV/new",
                    "natural_language": "https://arxiv.org/list/cs.CL/new",
                    "robotics": "https://arxiv.org/list/cs.RO/new",
                    "cryptography": "https://arxiv.org/list/cs.CR/new"
                }
            },
            "phys_org": {
                "base_url": "https://phys.org",
                "categories": {
                    "physics": "https://phys.org/physics-news/",
                    "technology": "https://phys.org/technology-news/",
                    "space": "https://phys.org/space-news/",
                    "earth": "https://phys.org/earth-news/",
                    "biology": "https://phys.org/biology-news/",
                    "chemistry": "https://phys.org/chemistry-news/"
                }
            },
            "mit": {
                "base_url": "https://news.mit.edu",
                "categories": {
                    "ai": "https://news.mit.edu/topic/artificial-intelligence2",
                    "robotics": "https://news.mit.edu/topic/robotics",
                    "energy": "https://news.mit.edu/topic/energy",
                    "computing": "https://news.mit.edu/topic/computing",
                    "biology": "https://news.mit.edu/topic/biology",
                    "physics": "https://news.mit.edu/topic/physics"
                }
            },
            "nature": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "physics": "https://www.nature.com/subjects/physics",
                    "chemistry": "https://www.nature.com/subjects/chemistry",
                    "biology": "https://www.nature.com/subjects/biology",
                    "materials": "https://www.nature.com/subjects/materials-science"
                }
            }
        }
        
        # Scaling parameters
        self.scaling_config = {
            "arxiv": {"max_papers": 20, "delay": 1},
            "phys_org": {"max_articles": 15, "delay": 2},
            "mit": {"max_articles": 12, "delay": 2},
            "nature": {"max_articles": 10, "delay": 3}
        }
    
    def run_massive_scraping(self):
        """Run massive scaling scraping across all sites"""
        
        print("🚀 Massive Scientific Scraping System")
        print("=" * 60)
        print(f"📊 Scraping {len(self.massive_sites)} major scientific sites")
        print(f"🧠 prime aligned compute enhancement: 1.618x golden ratio")
        print(f"⚡ Parallel processing enabled")
        
        total_scraped = 0
        site_stats = {}
        
        # Use ThreadPoolExecutor for parallel scraping
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all scraping tasks
            future_to_site = {}
            
            for site_name, config in self.massive_sites.items():
                scaling_config = self.scaling_config.get(site_name, {"max_articles": 10, "delay": 2})
                future = executor.submit(
                    self._scrape_site_parallel, 
                    site_name, 
                    config, 
                    scaling_config
                )
                future_to_site[future] = site_name
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_site):
                site_name = future_to_site[future]
                try:
                    site_scraped = future.result()
                    site_stats[site_name] = site_scraped
                    total_scraped += site_scraped
                    print(f"✅ {site_name.upper()}: {site_scraped} articles scraped")
                except Exception as e:
                    print(f"❌ {site_name.upper()}: Error - {e}")
                    site_stats[site_name] = 0
        
        self._print_massive_statistics(total_scraped, site_stats)
        return total_scraped, site_stats
    
    def _scrape_site_parallel(self, site_name, config, scaling_config):
        """Scrape a single site in parallel"""
        
        print(f"\n🏛️ Starting {site_name.upper()} scraping...")
        site_scraped = 0
        
        for category, category_url in config['categories'].items():
            try:
                print(f"   📂 Category: {category}")
                
                if site_name == "arxiv":
                    articles_scraped = self._scrape_arxiv_category_parallel(
                        category, category_url, scaling_config["max_papers"]
                    )
                else:
                    articles_scraped = self._scrape_general_category_parallel(
                        category, category_url, scaling_config["max_articles"]
                    )
                
                site_scraped += articles_scraped
                print(f"      ✅ {articles_scraped} articles scraped")
                
                # Add delay between categories
                time.sleep(scaling_config["delay"])
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        return site_scraped
    
    def _scrape_arxiv_category_parallel(self, category, category_url, max_papers):
        """Scrape arXiv category in parallel"""
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract paper links
            paper_links = []
            selectors = ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]']
            
            for selector in selectors:
                try:
                    links = soup.select(selector)
                    for link in links:
                        href = link.get('href')
                        if href and '/abs/' in href:
                            full_url = urljoin(category_url, href)
                            paper_links.append(full_url)
                    if paper_links:
                        break
                except Exception:
                    continue
            
            # Scrape papers in parallel
            articles_scraped = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {}
                
                for paper_url in paper_links[:max_papers]:
                    future = executor.submit(self._scrape_single_article, paper_url)
                    future_to_url[future] = paper_url
                
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        result = future.result()
                        if result:
                            articles_scraped += 1
                    except Exception as e:
                        logger.error(f"Error scraping paper: {e}")
            
            return articles_scraped
            
        except Exception as e:
            logger.error(f"Error scraping arXiv category {category}: {e}")
            return 0
    
    def _scrape_general_category_parallel(self, category, category_url, max_articles):
        """Scrape general category in parallel"""
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links
            article_links = []
            selectors = [
                'a[data-test="article-link"]',
                '.c-card a',
                'h3 a',
                '.article-title a',
                '.result-title a',
                'article a',
                '.post a'
            ]
            
            for selector in selectors:
                try:
                    links = soup.select(selector)
                    for link in links:
                        href = link.get('href')
                        if href:
                            full_url = urljoin(category_url, href)
                            if self._is_valid_article_url(full_url):
                                article_links.append(full_url)
                    if article_links:
                        break
                except Exception:
                    continue
            
            # Scrape articles in parallel
            articles_scraped = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {}
                
                for article_url in article_links[:max_articles]:
                    future = executor.submit(self._scrape_single_article, article_url)
                    future_to_url[future] = article_url
                
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        result = future.result()
                        if result:
                            articles_scraped += 1
                    except Exception as e:
                        logger.error(f"Error scraping article: {e}")
            
            return articles_scraped
            
        except Exception as e:
            logger.error(f"Error scraping category {category}: {e}")
            return 0
    
    def _scrape_single_article(self, url):
        """Scrape a single article"""
        
        try:
            result = self.knowledge_system.scrape_website(
                url=url,
                max_depth=0,
                follow_links=False
            )
            
            if result['success']:
                # Store enhanced metadata
                self.knowledge_system.database_service.store_consciousness_data(
                    "massive_scraped_article",
                    {
                        "title": result.get('title', ''),
                        "content_length": result.get('content_length', 0),
                        "prime_aligned_score": result.get('prime_aligned_score', 1.0),
                        "scraped_at": datetime.now().isoformat()
                    },
                    {
                        "source": "massive_scraper",
                        "url": url,
                        "prime_aligned_enhanced": True
                    },
                    "massive_scraper"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return False
    
    def _is_valid_article_url(self, url):
        """Check if URL is a valid article URL"""
        
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/page/', '/search',
            '/login', '/register', '.pdf', '.jpg', '.png', '.gif', '#', 'javascript:'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    def _print_massive_statistics(self, total_scraped, site_stats):
        """Print massive scraping statistics"""
        
        print(f"\n🚀 Massive Scraping Complete!")
        print("=" * 60)
        print(f"📊 Total Articles Scraped: {total_scraped}")
        print(f"🏛️ Sites Processed: {len(site_stats)}")
        
        print(f"\n📈 Site Breakdown:")
        for site, count in site_stats.items():
            print(f"   {site}: {count} articles")
        
        # Get knowledge system statistics
        stats = self.knowledge_system.get_scraping_stats()
        print(f"\n🧠 Knowledge Base Statistics:")
        print(f"   📄 Total Pages: {stats.get('total_scraped_pages', 0)}")
        print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
        print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
        print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
        print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")

def main():
    """Main function to run massive scientific scraping"""
    
    scraper = MassiveScientificScraper()
    
    print(f"🚀 Starting massive scientific article scraping...")
    print(f"⚡ Parallel processing enabled for maximum efficiency")
    
    total_scraped, site_stats = scraper.run_massive_scraping()
    
    print(f"\n🎉 Massive Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")

if __name__ == "__main__":
    main()
