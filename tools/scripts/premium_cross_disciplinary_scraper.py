#!/usr/bin/env python3
"""
🌟 Premium Cross-Disciplinary Scraper
======================================
High-priority scraper for the most important cross-disciplinary sources.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PremiumCrossDisciplinaryScraper:
    """Premium scraper for top cross-disciplinary sources"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=10)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Premium cross-disciplinary sources
        self.premium_sources = {
            "nature_news": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "ai": "https://www.nature.com/subjects/artificial-intelligence",
                    "quantum": "https://www.nature.com/subjects/quantum-physics",
                    "biotech": "https://www.nature.com/subjects/biotechnology",
                    "climate": "https://www.nature.com/subjects/climate-change",
                    "space": "https://www.nature.com/subjects/astronomy",
                    "energy": "https://www.nature.com/subjects/energy",
                    "medicine": "https://www.nature.com/subjects/medicine",
                    "materials": "https://www.nature.com/subjects/materials-science"
                },
                "max_articles": 20
            },
            
            "science_magazine": {
                "base_url": "https://www.science.org",
                "categories": {
                    "ai": "https://www.science.org/topic/category/artificial-intelligence",
                    "biology": "https://www.science.org/topic/category/biology",
                    "chemistry": "https://www.science.org/topic/category/chemistry",
                    "physics": "https://www.science.org/topic/category/physics",
                    "earth": "https://www.science.org/topic/category/earth-sciences",
                    "medicine": "https://www.science.org/topic/category/medicine",
                    "technology": "https://www.science.org/topic/category/technology",
                    "environment": "https://www.science.org/topic/category/environment"
                },
                "max_articles": 15
            },
            
            "cell_press": {
                "base_url": "https://www.cell.com",
                "categories": {
                    "biology": "https://www.cell.com/cell-biology",
                    "medicine": "https://www.cell.com/medicine",
                    "genetics": "https://www.cell.com/genetics",
                    "neuroscience": "https://www.cell.com/neuroscience",
                    "immunology": "https://www.cell.com/immunology",
                    "cancer": "https://www.cell.com/cancer-cell",
                    "stem_cells": "https://www.cell.com/stem-cell",
                    "metabolism": "https://www.cell.com/metabolism"
                },
                "max_articles": 12
            },
            
            "ieee_spectrum": {
                "base_url": "https://spectrum.ieee.org",
                "categories": {
                    "ai": "https://spectrum.ieee.org/topic/artificial-intelligence/",
                    "robotics": "https://spectrum.ieee.org/topic/robotics/",
                    "energy": "https://spectrum.ieee.org/topic/energy/",
                    "biomedical": "https://spectrum.ieee.org/topic/biomedical/",
                    "computing": "https://spectrum.ieee.org/topic/computing/",
                    "communications": "https://spectrum.ieee.org/topic/communications/",
                    "semiconductors": "https://spectrum.ieee.org/topic/semiconductors/",
                    "transportation": "https://spectrum.ieee.org/topic/transportation/"
                },
                "max_articles": 15
            },
            
            "mit_technology_review": {
                "base_url": "https://www.technologyreview.com",
                "categories": {
                    "ai": "https://www.technologyreview.com/topic/artificial-intelligence/",
                    "biotech": "https://www.technologyreview.com/topic/biotechnology/",
                    "climate": "https://www.technologyreview.com/topic/climate/",
                    "energy": "https://www.technologyreview.com/topic/energy/",
                    "space": "https://www.technologyreview.com/topic/space/",
                    "computing": "https://www.technologyreview.com/topic/computing/",
                    "robotics": "https://www.technologyreview.com/topic/robotics/",
                    "health": "https://www.technologyreview.com/topic/health/"
                },
                "max_articles": 18
            },
            
            "new_scientist": {
                "base_url": "https://www.newscientist.com",
                "categories": {
                    "technology": "https://www.newscientist.com/subject/technology/",
                    "space": "https://www.newscientist.com/subject/space/",
                    "physics": "https://www.newscientist.com/subject/physics/",
                    "biology": "https://www.newscientist.com/subject/biology/",
                    "chemistry": "https://www.newscientist.com/subject/chemistry/",
                    "earth": "https://www.newscientist.com/subject/earth/",
                    "health": "https://www.newscientist.com/subject/health/",
                    "climate": "https://www.newscientist.com/subject/climate/"
                },
                "max_articles": 20
            }
        }
    
    def scrape_premium_sources(self):
        """Scrape all premium cross-disciplinary sources"""
        
        print("🌟 Premium Cross-Disciplinary Scraper")
        print("=" * 60)
        print(f"📊 Scraping {len(self.premium_sources)} premium sources")
        print(f"🧠 prime aligned compute enhancement: 1.618x golden ratio")
        print(f"⚡ High-priority cross-disciplinary knowledge collection")
        
        total_scraped = 0
        source_stats = {}
        
        for source_name, config in self.premium_sources.items():
            print(f"\n🌟 Scraping {source_name.upper()}")
            print("-" * 50)
            
            source_scraped = 0
            
            for category, category_url in config['categories'].items():
                print(f"\n📂 Category: {category}")
                
                try:
                    articles_scraped = self._scrape_premium_category(
                        source_name, category, category_url, config['max_articles']
                    )
                    
                    source_scraped += articles_scraped
                    print(f"   ✅ {articles_scraped} articles scraped")
                    
                    # Add delay between categories
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            source_stats[source_name] = source_scraped
            total_scraped += source_scraped
            
            print(f"\n📊 {source_name}: {source_scraped} articles scraped")
            
            # Add delay between sources
            time.sleep(5)
        
        self._print_premium_statistics(total_scraped, source_stats)
        return total_scraped, source_stats
    
    def _scrape_premium_category(self, source_name, category, category_url, max_articles):
        """Scrape a premium category"""
        
        try:
            response = self.session.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links
            article_links = self._extract_premium_article_links(soup, category_url, source_name)
            
            articles_scraped = 0
            
            for i, article_url in enumerate(article_links[:max_articles]):
                try:
                    print(f"   📄 Scraping article {i+1}/{min(len(article_links), max_articles)}")
                    
                    result = self.knowledge_system.scrape_website(
                        url=article_url,
                        max_depth=0,
                        follow_links=False
                    )
                    
                    if result['success']:
                        # Store premium metadata
                        self._store_premium_metadata(result, source_name, category)
                        articles_scraped += 1
                        print(f"      ✅ Success: {result.get('title', 'Untitled')[:50]}...")
                    else:
                        print(f"      ❌ Failed")
                    
                    # Add delay between articles
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            return articles_scraped
            
        except Exception as e:
            logger.error(f"Error scraping premium category {category}: {e}")
            return 0
    
    def _extract_premium_article_links(self, soup, base_url, source_name):
        """Extract article links from premium sources"""
        
        article_links = []
        
        # Source-specific selectors
        source_selectors = {
            "nature_news": ['a[data-test="article-link"]', '.c-card a', 'h3 a'],
            "science_magazine": ['.article-title a', 'h3 a', '.result-title a'],
            "cell_press": ['.article-title a', 'h2 a', '.result-title a'],
            "ieee_spectrum": ['h3 a', '.post-title a', '.article-title a'],
            "mit_technology_review": ['h3 a', '.post-title a', '.article-title a'],
            "new_scientist": ['h3 a', '.post-title a', '.article-title a']
        }
        
        selectors = source_selectors.get(source_name, ['h3 a', 'h2 a', 'article a'])
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_premium_article_url(full_url, source_name):
                            article_links.append(full_url)
                
                if article_links:
                    break
                    
            except Exception:
                continue
        
        return list(set(article_links))
    
    def _store_premium_metadata(self, result, source_name, category):
        """Store premium metadata"""
        
        try:
            self.knowledge_system.database_service.store_consciousness_data(
                "premium_cross_disciplinary",
                {
                    "source": source_name,
                    "category": category,
                    "title": result.get('title', ''),
                    "content_length": result.get('content_length', 0),
                    "prime_aligned_score": result.get('prime_aligned_score', 1.0),
                    "scraped_at": datetime.now().isoformat()
                },
                {
                    "source": "premium_scraper",
                    "source_name": source_name,
                    "category": category,
                    "url": result.get('url', ''),
                    "premium": True,
                    "cross_disciplinary": True,
                    "prime_aligned_enhanced": True
                },
                "premium_scraper"
            )
            
        except Exception as e:
            logger.error(f"Error storing premium metadata: {e}")
    
    def _is_valid_premium_article_url(self, url, source_name):
        """Check if URL is a valid premium article URL"""
        
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/page/', '/search',
            '/login', '/register', '.pdf', '.jpg', '.png', '.gif', 
            '#', 'javascript:', '/ads/', '/advertisement/',
            '/subscribe/', '/newsletter/', '/contact/'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    def _print_premium_statistics(self, total_scraped, source_stats):
        """Print premium scraping statistics"""
        
        print(f"\n🌟 Premium Cross-Disciplinary Scraping Complete!")
        print("=" * 60)
        print(f"📊 Total Articles Scraped: {total_scraped}")
        print(f"🌟 Premium Sources Processed: {len(source_stats)}")
        
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
    """Main function to run premium cross-disciplinary scraping"""
    
    scraper = PremiumCrossDisciplinaryScraper()
    
    print(f"🚀 Starting premium cross-disciplinary scraping...")
    print(f"🌟 High-priority sources for maximum knowledge impact")
    
    total_scraped, source_stats = scraper.scrape_premium_sources()
    
    print(f"\n🎉 Premium Cross-Disciplinary Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")
    print(f"   🌟 Premium cross-disciplinary knowledge ecosystem established!")

if __name__ == "__main__":
    main()
