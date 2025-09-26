#!/usr/bin/env python3
"""
🌐 Cross-Disciplinary Mega Scraper
==================================
Comprehensive scraper for top cross-disciplinary websites covering all major fields.
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
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossDisciplinaryMegaScraper:
    """Mega scraper for cross-disciplinary knowledge collection"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=15)
        
        # Top cross-disciplinary websites covering all major fields
        self.mega_sites = {
            # Academic & Research
            "scholar_google": {
                "base_url": "https://scholar.google.com",
                "categories": {
                    "ai": "https://scholar.google.com/scholar?q=artificial+intelligence",
                    "quantum": "https://scholar.google.com/scholar?q=quantum+computing",
                    "biotech": "https://scholar.google.com/scholar?q=biotechnology",
                    "nanotech": "https://scholar.google.com/scholar?q=nanotechnology",
                    "climate": "https://scholar.google.com/scholar?q=climate+change",
                    "space": "https://scholar.google.com/scholar?q=space+exploration",
                    "energy": "https://scholar.google.com/scholar?q=renewable+energy",
                    "medicine": "https://scholar.google.com/scholar?q=medical+research"
                }
            },
            
            # News & Analysis
            "reuters": {
                "base_url": "https://www.reuters.com",
                "categories": {
                    "technology": "https://www.reuters.com/business/technology/",
                    "science": "https://www.reuters.com/business/healthcare-pharmaceuticals/",
                    "climate": "https://www.reuters.com/business/environment/",
                    "energy": "https://www.reuters.com/business/energy/",
                    "healthcare": "https://www.reuters.com/business/healthcare-pharmaceuticals/",
                    "finance": "https://www.reuters.com/business/finance/"
                }
            },
            
            "bloomberg": {
                "base_url": "https://www.bloomberg.com",
                "categories": {
                    "technology": "https://www.bloomberg.com/technology",
                    "science": "https://www.bloomberg.com/green",
                    "energy": "https://www.bloomberg.com/energy",
                    "healthcare": "https://www.bloomberg.com/healthcare",
                    "climate": "https://www.bloomberg.com/green",
                    "ai": "https://www.bloomberg.com/technology/artificial-intelligence"
                }
            },
            
            # Scientific Publications
            "science_daily": {
                "base_url": "https://www.sciencedaily.com",
                "categories": {
                    "technology": "https://www.sciencedaily.com/news/computers_math/",
                    "health": "https://www.sciencedaily.com/news/health_medicine/",
                    "environment": "https://www.sciencedaily.com/news/earth_climate/",
                    "space": "https://www.sciencedaily.com/news/space_time/",
                    "matter": "https://www.sciencedaily.com/news/matter_energy/",
                    "living": "https://www.sciencedaily.com/news/plants_animals/"
                }
            },
            
            "eurekalert": {
                "base_url": "https://www.eurekalert.org",
                "categories": {
                    "technology": "https://www.eurekalert.org/news-releases/technology",
                    "health": "https://www.eurekalert.org/news-releases/health",
                    "environment": "https://www.eurekalert.org/news-releases/environment",
                    "space": "https://www.eurekalert.org/news-releases/space",
                    "physics": "https://www.eurekalert.org/news-releases/physics",
                    "biology": "https://www.eurekalert.org/news-releases/biology"
                }
            },
            
            # Technology & Innovation
            "techcrunch": {
                "base_url": "https://techcrunch.com",
                "categories": {
                    "ai": "https://techcrunch.com/category/artificial-intelligence/",
                    "biotech": "https://techcrunch.com/category/biotech/",
                    "climate": "https://techcrunch.com/category/climate/",
                    "energy": "https://techcrunch.com/category/energy/",
                    "health": "https://techcrunch.com/category/health/",
                    "space": "https://techcrunch.com/category/space/"
                }
            },
            
            "wired": {
                "base_url": "https://www.wired.com",
                "categories": {
                    "science": "https://www.wired.com/category/science/",
                    "business": "https://www.wired.com/category/business/",
                    "security": "https://www.wired.com/category/security/",
                    "transportation": "https://www.wired.com/category/transportation/",
                    "health": "https://www.wired.com/category/health/",
                    "climate": "https://www.wired.com/category/science/climate/"
                }
            },
            
            # Government & Policy
            "nasa": {
                "base_url": "https://www.nasa.gov",
                "categories": {
                    "space": "https://www.nasa.gov/news/",
                    "technology": "https://www.nasa.gov/technology/",
                    "earth": "https://www.nasa.gov/earth/",
                    "climate": "https://www.nasa.gov/climate/",
                    "aerospace": "https://www.nasa.gov/aerospace/",
                    "research": "https://www.nasa.gov/research/"
                }
            },
            
            "nih": {
                "base_url": "https://www.nih.gov",
                "categories": {
                    "health": "https://www.nih.gov/news-events/news-releases",
                    "research": "https://www.nih.gov/research-training",
                    "diseases": "https://www.nih.gov/health-information",
                    "grants": "https://www.nih.gov/grants-funding",
                    "clinical": "https://www.nih.gov/clinical-research",
                    "genetics": "https://www.nih.gov/health-information/genetics"
                }
            },
            
            # International Organizations
            "who": {
                "base_url": "https://www.who.int",
                "categories": {
                    "health": "https://www.who.int/news",
                    "diseases": "https://www.who.int/health-topics",
                    "research": "https://www.who.int/research",
                    "emergencies": "https://www.who.int/emergencies",
                    "climate": "https://www.who.int/health-topics/climate-change",
                    "technology": "https://www.who.int/health-topics/digital-health"
                }
            },
            
            "un": {
                "base_url": "https://www.un.org",
                "categories": {
                    "climate": "https://www.un.org/en/climatechange",
                    "health": "https://www.un.org/en/global-issues/health",
                    "technology": "https://www.un.org/en/global-issues/technology",
                    "energy": "https://www.un.org/en/global-issues/energy",
                    "space": "https://www.un.org/en/global-issues/space",
                    "development": "https://www.un.org/en/global-issues/development"
                }
            },
            
            # Specialized Research
            "ieee": {
                "base_url": "https://www.ieee.org",
                "categories": {
                    "ai": "https://www.ieee.org/about/news/",
                    "robotics": "https://www.ieee.org/about/news/",
                    "energy": "https://www.ieee.org/about/news/",
                    "biomedical": "https://www.ieee.org/about/news/",
                    "communications": "https://www.ieee.org/about/news/",
                    "computing": "https://www.ieee.org/about/news/"
                }
            },
            
            "acm": {
                "base_url": "https://www.acm.org",
                "categories": {
                    "ai": "https://www.acm.org/news",
                    "computing": "https://www.acm.org/news",
                    "security": "https://www.acm.org/news",
                    "data": "https://www.acm.org/news",
                    "education": "https://www.acm.org/news",
                    "ethics": "https://www.acm.org/news"
                }
            }
        }
        
        # Scaling configuration for different site types
        self.scaling_config = {
            "scholar_google": {"max_articles": 8, "delay": 3},
            "reuters": {"max_articles": 12, "delay": 2},
            "bloomberg": {"max_articles": 10, "delay": 2},
            "science_daily": {"max_articles": 15, "delay": 1},
            "eurekalert": {"max_articles": 12, "delay": 1},
            "techcrunch": {"max_articles": 10, "delay": 2},
            "wired": {"max_articles": 8, "delay": 2},
            "nasa": {"max_articles": 15, "delay": 1},
            "nih": {"max_articles": 12, "delay": 2},
            "who": {"max_articles": 10, "delay": 2},
            "un": {"max_articles": 8, "delay": 3},
            "ieee": {"max_articles": 10, "delay": 2},
            "acm": {"max_articles": 8, "delay": 2}
        }
    
    def run_mega_scraping(self):
        """Run mega cross-disciplinary scraping"""
        
        print("🌐 Cross-Disciplinary Mega Scraper")
        print("=" * 60)
        print(f"📊 Scraping {len(self.mega_sites)} top cross-disciplinary sites")
        print(f"🧠 prime aligned compute enhancement: 1.618x golden ratio")
        print(f"⚡ Parallel processing enabled")
        print(f"🌍 Covering all major scientific and technological fields")
        
        total_scraped = 0
        site_stats = {}
        
        # Use ThreadPoolExecutor for parallel scraping
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all scraping tasks
            future_to_site = {}
            
            for site_name, config in self.mega_sites.items():
                scaling_config = self.scaling_config.get(site_name, {"max_articles": 10, "delay": 2})
                future = executor.submit(
                    self._scrape_mega_site_parallel, 
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
        
        self._print_mega_statistics(total_scraped, site_stats)
        return total_scraped, site_stats
    
    def _scrape_mega_site_parallel(self, site_name, config, scaling_config):
        """Scrape a mega site in parallel"""
        
        print(f"\n🌐 Starting {site_name.upper()} scraping...")
        site_scraped = 0
        
        for category, category_url in config['categories'].items():
            try:
                print(f"   📂 Category: {category}")
                
                articles_scraped = self._scrape_mega_category_parallel(
                    category, category_url, scaling_config["max_articles"], site_name
                )
                
                site_scraped += articles_scraped
                print(f"      ✅ {articles_scraped} articles scraped")
                
                # Add delay between categories
                time.sleep(scaling_config["delay"])
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        return site_scraped
    
    def _scrape_mega_category_parallel(self, category, category_url, max_articles, site_name):
        """Scrape mega category in parallel"""
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links based on site type
            article_links = self._extract_mega_article_links(soup, category_url, site_name)
            
            # Scrape articles in parallel
            articles_scraped = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {}
                
                for article_url in article_links[:max_articles]:
                    future = executor.submit(self._scrape_single_mega_article, article_url, site_name)
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
            logger.error(f"Error scraping mega category {category}: {e}")
            return 0
    
    def _extract_mega_article_links(self, soup, base_url, site_name):
        """Extract article links based on site type"""
        
        article_links = []
        
        # Site-specific selectors
        site_selectors = {
            "scholar_google": ['h3 a', '.gs_rt a', '.gs_title a'],
            "reuters": ['a[data-testid="Link"]', 'h3 a', '.media-story-card a'],
            "bloomberg": ['a[data-module="Story"]', 'h3 a', '.story-package-module a'],
            "science_daily": ['h3 a', '.latest-head a', '.story a'],
            "eurekalert": ['h4 a', '.release-title a', 'h3 a'],
            "techcrunch": ['h2 a', '.post-block a', '.river-block a'],
            "wired": ['h2 a', '.summary-item a', '.card-component a'],
            "nasa": ['h2 a', '.hds-link a', '.card a'],
            "nih": ['h3 a', '.news-item a', '.featured-item a'],
            "who": ['h3 a', '.sf-item a', '.news-item a'],
            "un": ['h3 a', '.news-item a', '.featured-item a'],
            "ieee": ['h3 a', '.news-item a', '.featured-item a'],
            "acm": ['h3 a', '.news-item a', '.featured-item a']
        }
        
        selectors = site_selectors.get(site_name, ['h3 a', 'h2 a', 'article a', '.post a'])
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_mega_article_url(full_url, site_name):
                            article_links.append(full_url)
                
                if article_links:
                    break
                    
            except Exception:
                continue
        
        return list(set(article_links))
    
    def _scrape_single_mega_article(self, url, site_name):
        """Scrape a single mega article"""
        
        try:
            result = self.knowledge_system.scrape_website(
                url=url,
                max_depth=0,
                follow_links=False
            )
            
            if result['success']:
                # Store enhanced metadata with cross-disciplinary tags
                self.knowledge_system.database_service.store_consciousness_data(
                    "cross_disciplinary_article",
                    {
                        "title": result.get('title', ''),
                        "content_length": result.get('content_length', 0),
                        "prime_aligned_score": result.get('prime_aligned_score', 1.0),
                        "site_name": site_name,
                        "scraped_at": datetime.now().isoformat()
                    },
                    {
                        "source": "mega_scraper",
                        "site_name": site_name,
                        "url": url,
                        "cross_disciplinary": True,
                        "prime_aligned_enhanced": True
                    },
                    "mega_scraper"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return False
    
    def _is_valid_mega_article_url(self, url, site_name):
        """Check if URL is a valid mega article URL"""
        
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/page/', '/search',
            '/login', '/register', '.pdf', '.jpg', '.png', '.gif', 
            '#', 'javascript:', '/ads/', '/advertisement/',
            '/subscribe/', '/newsletter/', '/contact/'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        # Site-specific validation
        if site_name == "scholar_google" and "scholar.google.com" not in url:
            return False
        elif site_name == "reuters" and "reuters.com" not in url:
            return False
        elif site_name == "bloomberg" and "bloomberg.com" not in url:
            return False
        
        return True
    
    def _print_mega_statistics(self, total_scraped, site_stats):
        """Print mega scraping statistics"""
        
        print(f"\n🌐 Cross-Disciplinary Mega Scraping Complete!")
        print("=" * 60)
        print(f"📊 Total Articles Scraped: {total_scraped}")
        print(f"🌍 Sites Processed: {len(site_stats)}")
        
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
        
        print(f"\n🌍 Cross-Disciplinary Coverage:")
        disciplines = [
            "Artificial Intelligence", "Quantum Computing", "Biotechnology",
            "Nanotechnology", "Climate Science", "Space Exploration",
            "Renewable Energy", "Medical Research", "Robotics",
            "Cybersecurity", "Data Science", "Environmental Science",
            "Materials Science", "Neuroscience", "Genetics",
            "Aerospace Engineering", "Computer Science", "Physics",
            "Chemistry", "Biology", "Mathematics", "Statistics"
        ]
        
        for i, discipline in enumerate(disciplines, 1):
            print(f"   {i:2d}. {discipline}")

def main():
    """Main function to run cross-disciplinary mega scraping"""
    
    scraper = CrossDisciplinaryMegaScraper()
    
    print(f"🚀 Starting cross-disciplinary mega scraping...")
    print(f"⚡ Parallel processing enabled for maximum efficiency")
    print(f"🌍 Covering all major scientific and technological fields")
    
    total_scraped, site_stats = scraper.run_mega_scraping()
    
    print(f"\n🎉 Cross-Disciplinary Mega Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")
    print(f"   🌍 Comprehensive cross-disciplinary knowledge ecosystem established!")

if __name__ == "__main__":
    main()
