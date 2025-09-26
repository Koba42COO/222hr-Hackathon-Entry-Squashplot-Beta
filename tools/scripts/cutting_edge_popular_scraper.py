#!/usr/bin/env python3
"""
🔥 Cutting-Edge & Popular Topics Scraper
=========================================
Focuses on the most popular and cutting-edge topics across all fields.
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

class CuttingEdgePopularScraper:
    """Scraper for cutting-edge and popular topics"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=6)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Cutting-edge and popular topic sources
        self.cutting_edge_sources = {
            # AI & Machine Learning (Most Popular)
            "ai_ml": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "large_language_models": "https://arxiv.org/list/cs.CL/new",
                    "computer_vision": "https://arxiv.org/list/cs.CV/new",
                    "reinforcement_learning": "https://arxiv.org/list/cs.LG/new",
                    "neural_networks": "https://arxiv.org/list/cs.NE/new",
                    "deep_learning": "https://arxiv.org/list/cs.LG/new",
                    "generative_ai": "https://arxiv.org/list/cs.AI/new",
                    "robotics": "https://arxiv.org/list/cs.RO/new",
                    "natural_language": "https://arxiv.org/list/cs.CL/new"
                },
                "max_articles": 8,
                "delay": 2
            },
            
            # Quantum Computing & Physics (Cutting-Edge)
            "quantum": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "quantum_computing": "https://arxiv.org/list/quant-ph/new",
                    "quantum_algorithms": "https://arxiv.org/list/quant-ph/new",
                    "quantum_materials": "https://arxiv.org/list/cond-mat/new",
                    "quantum_optics": "https://arxiv.org/list/quant-ph/new",
                    "quantum_information": "https://arxiv.org/list/quant-ph/new",
                    "quantum_simulation": "https://arxiv.org/list/quant-ph/new"
                },
                "max_articles": 6,
                "delay": 2
            },
            
            # Biotechnology & Genomics (Popular)
            "biotech": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "genomics": "https://arxiv.org/list/q-bio.GN/new",
                    "bioinformatics": "https://arxiv.org/list/q-bio.BM/new",
                    "synthetic_biology": "https://arxiv.org/list/q-bio.SC/new",
                    "molecular_biology": "https://arxiv.org/list/q-bio.MN/new",
                    "systems_biology": "https://arxiv.org/list/q-bio.SC/new",
                    "neuroscience": "https://arxiv.org/list/q-bio.NC/new"
                },
                "max_articles": 6,
                "delay": 2
            },
            
            # Climate & Energy (Critical Topics)
            "climate_energy": {
                "base_url": "https://news.mit.edu",
                "categories": {
                    "renewable_energy": "https://news.mit.edu/topic/energy",
                    "climate_science": "https://news.mit.edu/topic/climate",
                    "sustainability": "https://news.mit.edu/topic/sustainability",
                    "carbon_capture": "https://news.mit.edu/topic/energy",
                    "battery_technology": "https://news.mit.edu/topic/energy",
                    "solar_technology": "https://news.mit.edu/topic/energy"
                },
                "max_articles": 5,
                "delay": 3
            },
            
            # Space & Astrophysics (Popular)
            "space": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "exoplanets": "https://arxiv.org/list/astro-ph.EP/new",
                    "black_holes": "https://arxiv.org/list/astro-ph.HE/new",
                    "cosmology": "https://arxiv.org/list/astro-ph.CO/new",
                    "galaxy_formation": "https://arxiv.org/list/astro-ph.GA/new",
                    "space_exploration": "https://arxiv.org/list/astro-ph.IM/new",
                    "gravitational_waves": "https://arxiv.org/list/gr-qc/new"
                },
                "max_articles": 6,
                "delay": 2
            },
            
            # Materials Science & Nanotechnology (Cutting-Edge)
            "materials": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "nanotechnology": "https://www.nature.com/subjects/nanotechnology",
                    "graphene": "https://www.nature.com/subjects/graphene",
                    "superconductors": "https://www.nature.com/subjects/superconductivity",
                    "metamaterials": "https://www.nature.com/subjects/metamaterials",
                    "2d_materials": "https://www.nature.com/subjects/two-dimensional-materials",
                    "smart_materials": "https://www.nature.com/subjects/smart-materials"
                },
                "max_articles": 5,
                "delay": 3
            },
            
            # Cryptography & Security (Critical)
            "crypto_security": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "cryptography": "https://arxiv.org/list/cs.CR/new",
                    "blockchain": "https://arxiv.org/list/cs.CR/new",
                    "cybersecurity": "https://arxiv.org/list/cs.CR/new",
                    "privacy": "https://arxiv.org/list/cs.CR/new",
                    "quantum_crypto": "https://arxiv.org/list/quant-ph/new",
                    "post_quantum": "https://arxiv.org/list/cs.CR/new"
                },
                "max_articles": 5,
                "delay": 2
            },
            
            # Philosophy of Science & Technology (Popular)
            "philosophy_tech": {
                "base_url": "https://plato.stanford.edu",
                "categories": {
                    "ai_ethics": "https://plato.stanford.edu/entries/ethics-ai/",
                    "prime aligned compute": "https://plato.stanford.edu/entries/prime aligned compute/",
                    "free_will": "https://plato.stanford.edu/entries/freewill/",
                    "scientific_method": "https://plato.stanford.edu/entries/scientific-method/",
                    "technology_ethics": "https://plato.stanford.edu/entries/ethics-technology/",
                    "information_philosophy": "https://plato.stanford.edu/entries/information/"
                },
                "max_articles": 4,
                "delay": 4
            },
            
            # Popular Mathematics (Accessible)
            "popular_math": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "number_theory": "https://arxiv.org/list/math.NT/new",
                    "topology": "https://arxiv.org/list/math.AT/new",
                    "geometry": "https://arxiv.org/list/math.DG/new",
                    "algebra": "https://arxiv.org/list/math.AG/new",
                    "analysis": "https://arxiv.org/list/math.AP/new",
                    "combinatorics": "https://arxiv.org/list/math.CO/new"
                },
                "max_articles": 5,
                "delay": 2
            },
            
            # Engineering Innovations (Popular)
            "engineering_innovations": {
                "base_url": "https://news.mit.edu",
                "categories": {
                    "biomedical": "https://news.mit.edu/topic/biomedical-engineering",
                    "aerospace": "https://news.mit.edu/topic/aerospace-engineering",
                    "materials": "https://news.mit.edu/topic/materials-engineering",
                    "robotics": "https://news.mit.edu/topic/robotics",
                    "nuclear": "https://news.mit.edu/topic/nuclear-engineering",
                    "civil": "https://news.mit.edu/topic/civil-engineering"
                },
                "max_articles": 5,
                "delay": 3
            }
        }
    
    def scrape_cutting_edge_popular(self):
        """Scrape cutting-edge and popular topics"""
        
        print("🔥 Cutting-Edge & Popular Topics Scraper")
        print("=" * 60)
        print(f"📊 Scraping {len(self.cutting_edge_sources)} cutting-edge domains")
        print(f"🧠 prime aligned compute enhancement: 1.618x golden ratio")
        print(f"⚡ Focus on most popular and cutting-edge topics")
        
        total_scraped = 0
        domain_stats = {}
        
        for domain_name, config in self.cutting_edge_sources.items():
            print(f"\n🔥 Scraping {domain_name.upper()}")
            print("-" * 50)
            
            domain_scraped = 0
            
            for category, category_url in config['categories'].items():
                print(f"\n📂 Category: {category}")
                
                try:
                    articles_scraped = self._scrape_cutting_edge_category(
                        domain_name, category, category_url, config['max_articles'], config['delay']
                    )
                    
                    domain_scraped += articles_scraped
                    print(f"   ✅ {articles_scraped} articles scraped")
                    
                    # Add delay between categories
                    time.sleep(config['delay'])
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            domain_stats[domain_name] = domain_scraped
            total_scraped += domain_scraped
            
            print(f"\n📊 {domain_name}: {domain_scraped} articles scraped")
            
            # Add delay between domains
            time.sleep(5)
        
        self._print_cutting_edge_statistics(total_scraped, domain_stats)
        return total_scraped, domain_stats
    
    def _scrape_cutting_edge_category(self, domain_name, category, category_url, max_articles, delay):
        """Scrape a cutting-edge category"""
        
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links based on domain
            article_links = self._extract_cutting_edge_article_links(soup, category_url, domain_name)
            
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
                        # Store cutting-edge metadata
                        self._store_cutting_edge_metadata(result, domain_name, category)
                        articles_scraped += 1
                        print(f"      ✅ Success: {result.get('title', 'Untitled')[:50]}...")
                    else:
                        print(f"      ❌ Failed")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    # Continue with next article instead of failing completely
            
            return articles_scraped
            
        except Exception as e:
            logger.error(f"Error scraping cutting-edge category {category}: {e}")
            return 0
    
    def _extract_cutting_edge_article_links(self, soup, base_url, domain_name):
        """Extract article links from cutting-edge sources"""
        
        article_links = []
        
        # Domain-specific selectors
        domain_selectors = {
            "ai_ml": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "quantum": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "biotech": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "climate_energy": ['h3 a', '.term-page--news-article a', 'article a'],
            "space": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "materials": ['a[data-test="article-link"]', '.c-card a', 'h3 a'],
            "crypto_security": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "philosophy_tech": ['h2 a', '.entry-title a', 'article a'],
            "popular_math": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "engineering_innovations": ['h3 a', '.term-page--news-article a', 'article a']
        }
        
        selectors = domain_selectors.get(domain_name, ['h3 a', 'h2 a', 'article a'])
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_cutting_edge_article_url(full_url, domain_name):
                            article_links.append(full_url)
                
                if article_links:
                    break
                    
            except Exception:
                continue
        
        return list(set(article_links))
    
    def _store_cutting_edge_metadata(self, result, domain_name, category):
        """Store cutting-edge metadata"""
        
        try:
            self.knowledge_system.database_service.store_consciousness_data(
                "cutting_edge_popular",
                {
                    "domain": domain_name,
                    "category": category,
                    "title": result.get('title', ''),
                    "content_length": result.get('content_length', 0),
                    "prime_aligned_score": result.get('prime_aligned_score', 1.0),
                    "scraped_at": datetime.now().isoformat()
                },
                {
                    "source": "cutting_edge_scraper",
                    "domain": domain_name,
                    "category": category,
                    "url": result.get('url', ''),
                    "cutting_edge": True,
                    "popular": True,
                    "prime_aligned_enhanced": True
                },
                "cutting_edge_scraper"
            )
            
        except Exception as e:
            logger.error(f"Error storing cutting-edge metadata: {e}")
    
    def _is_valid_cutting_edge_article_url(self, url, domain_name):
        """Check if URL is a valid cutting-edge article URL"""
        
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/page/', '/search',
            '/login', '/register', '.pdf', '.jpg', '.png', '.gif', 
            '#', 'javascript:', '/ads/', '/advertisement/',
            '/subscribe/', '/newsletter/', '/contact/'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        # Domain-specific validation
        if domain_name in ["ai_ml", "quantum", "biotech", "space", "crypto_security", "popular_math"] and "/abs/" not in url:
            return False
        elif domain_name == "materials" and "nature.com" not in url:
            return False
        elif domain_name in ["climate_energy", "engineering_innovations"] and "news.mit.edu" not in url:
            return False
        elif domain_name == "philosophy_tech" and "plato.stanford.edu" not in url:
            return False
        
        return True
    
    def _print_cutting_edge_statistics(self, total_scraped, domain_stats):
        """Print cutting-edge scraping statistics"""
        
        print(f"\n🔥 Cutting-Edge & Popular Scraping Complete!")
        print("=" * 60)
        print(f"📊 Total Articles Scraped: {total_scraped}")
        print(f"🔥 Cutting-Edge Domains Processed: {len(domain_stats)}")
        
        print(f"\n📈 Domain Breakdown:")
        for domain, count in domain_stats.items():
            print(f"   {domain}: {count} articles")
        
        # Get knowledge system statistics
        stats = self.knowledge_system.get_scraping_stats()
        print(f"\n🧠 Knowledge Base Statistics:")
        print(f"   📄 Total Pages: {stats.get('total_scraped_pages', 0)}")
        print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
        print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
        print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
        print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")
        
        print(f"\n🔥 Cutting-Edge Topics Covered:")
        topics = [
            "AI & Machine Learning", "Quantum Computing", "Biotechnology", 
            "Climate & Energy", "Space & Astrophysics", "Materials Science",
            "Cryptography & Security", "Philosophy of Technology", 
            "Popular Mathematics", "Engineering Innovations"
        ]
        
        for i, topic in enumerate(topics, 1):
            print(f"   {i:2d}. {topic}")

def main():
    """Main function to run cutting-edge popular scraping"""
    
    scraper = CuttingEdgePopularScraper()
    
    print(f"🚀 Starting cutting-edge & popular topics scraping...")
    print(f"🔥 Focus on most popular and cutting-edge topics")
    print(f"🧠 AI, Quantum, Biotech, Climate, Space, and more!")
    
    total_scraped, domain_stats = scraper.scrape_cutting_edge_popular()
    
    print(f"\n🎉 Cutting-Edge & Popular Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")
    print(f"   🔥 Cutting-edge knowledge ecosystem established!")

if __name__ == "__main__":
    main()
