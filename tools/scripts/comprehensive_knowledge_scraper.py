#!/usr/bin/env python3
"""
🧠 Comprehensive Knowledge Scraper
==================================
Covers all major fields: math, science, history, mechanics, engineering, 
computer science, coding, philosophy, biology, chemistry, and cutting-edge topics.
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

class ComprehensiveKnowledgeScraper:
    """Comprehensive scraper covering all major knowledge domains"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=8)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Comprehensive knowledge sources covering all major fields
        self.comprehensive_sources = {
            # Mathematics & Pure Sciences
            "mathematics": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "algebra": "https://arxiv.org/list/math.AG/new",
                    "analysis": "https://arxiv.org/list/math.AP/new",
                    "topology": "https://arxiv.org/list/math.AT/new",
                    "geometry": "https://arxiv.org/list/math.DG/new",
                    "number_theory": "https://arxiv.org/list/math.NT/new",
                    "statistics": "https://arxiv.org/list/stat/new",
                    "probability": "https://arxiv.org/list/math.PR/new",
                    "combinatorics": "https://arxiv.org/list/math.CO/new"
                },
                "max_articles": 6,
                "delay": 2
            },
            
            "physics": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "quantum": "https://arxiv.org/list/quant-ph/new",
                    "condensed_matter": "https://arxiv.org/list/cond-mat/new",
                    "high_energy": "https://arxiv.org/list/hep-th/new",
                    "astrophysics": "https://arxiv.org/list/astro-ph/new",
                    "nuclear": "https://arxiv.org/list/nucl-th/new",
                    "plasma": "https://arxiv.org/list/physics.plasm-ph/new",
                    "optics": "https://arxiv.org/list/physics.optics/new",
                    "fluid_dynamics": "https://arxiv.org/list/physics.flu-dyn/new"
                },
                "max_articles": 6,
                "delay": 2
            },
            
            # Biology & Chemistry
            "biology": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "quantitative_bio": "https://arxiv.org/list/q-bio/new",
                    "bioinformatics": "https://arxiv.org/list/q-bio.BM/new",
                    "cell_biology": "https://arxiv.org/list/q-bio.CB/new",
                    "genomics": "https://arxiv.org/list/q-bio.GN/new",
                    "molecular_bio": "https://arxiv.org/list/q-bio.MN/new",
                    "neuroscience": "https://arxiv.org/list/q-bio.NC/new",
                    "populations": "https://arxiv.org/list/q-bio.PE/new",
                    "systems_bio": "https://arxiv.org/list/q-bio.SC/new"
                },
                "max_articles": 5,
                "delay": 2
            },
            
            "chemistry": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "organic": "https://www.nature.com/subjects/organic-chemistry",
                    "inorganic": "https://www.nature.com/subjects/inorganic-chemistry",
                    "physical": "https://www.nature.com/subjects/physical-chemistry",
                    "analytical": "https://www.nature.com/subjects/analytical-chemistry",
                    "materials": "https://www.nature.com/subjects/materials-chemistry",
                    "catalysis": "https://www.nature.com/subjects/catalysis",
                    "medicinal": "https://www.nature.com/subjects/medicinal-chemistry",
                    "environmental": "https://www.nature.com/subjects/environmental-chemistry"
                },
                "max_articles": 4,
                "delay": 3
            },
            
            # Engineering & Mechanics
            "engineering": {
                "base_url": "https://news.mit.edu",
                "categories": {
                    "mechanical": "https://news.mit.edu/topic/mechanical-engineering",
                    "electrical": "https://news.mit.edu/topic/electrical-engineering",
                    "civil": "https://news.mit.edu/topic/civil-engineering",
                    "aerospace": "https://news.mit.edu/topic/aerospace-engineering",
                    "chemical": "https://news.mit.edu/topic/chemical-engineering",
                    "materials": "https://news.mit.edu/topic/materials-engineering",
                    "nuclear": "https://news.mit.edu/topic/nuclear-engineering",
                    "biomedical": "https://news.mit.edu/topic/biomedical-engineering"
                },
                "max_articles": 5,
                "delay": 3
            },
            
            # Computer Science & Coding
            "computer_science": {
                "base_url": "https://arxiv.org",
                "categories": {
                    "algorithms": "https://arxiv.org/list/cs.DS/new",
                    "artificial_intelligence": "https://arxiv.org/list/cs.AI/new",
                    "machine_learning": "https://arxiv.org/list/cs.LG/new",
                    "computer_vision": "https://arxiv.org/list/cs.CV/new",
                    "natural_language": "https://arxiv.org/list/cs.CL/new",
                    "robotics": "https://arxiv.org/list/cs.RO/new",
                    "cryptography": "https://arxiv.org/list/cs.CR/new",
                    "networking": "https://arxiv.org/list/cs.NI/new",
                    "databases": "https://arxiv.org/list/cs.DB/new",
                    "programming": "https://arxiv.org/list/cs.PL/new",
                    "software_engineering": "https://arxiv.org/list/cs.SE/new",
                    "human_computer": "https://arxiv.org/list/cs.HC/new"
                },
                "max_articles": 6,
                "delay": 2
            },
            
            # Philosophy & History
            "philosophy": {
                "base_url": "https://plato.stanford.edu",
                "categories": {
                    "ethics": "https://plato.stanford.edu/entries/ethics/",
                    "metaphysics": "https://plato.stanford.edu/entries/metaphysics/",
                    "epistemology": "https://plato.stanford.edu/entries/epistemology/",
                    "logic": "https://plato.stanford.edu/entries/logic/",
                    "philosophy_of_science": "https://plato.stanford.edu/entries/scientific-method/",
                    "philosophy_of_mind": "https://plato.stanford.edu/entries/prime aligned compute/",
                    "political_philosophy": "https://plato.stanford.edu/entries/political-philosophy/",
                    "aesthetics": "https://plato.stanford.edu/entries/aesthetics/"
                },
                "max_articles": 4,
                "delay": 4
            },
            
            "history": {
                "base_url": "https://www.history.com",
                "categories": {
                    "ancient": "https://www.history.com/topics/ancient-history",
                    "medieval": "https://www.history.com/topics/middle-ages",
                    "renaissance": "https://www.history.com/topics/renaissance",
                    "modern": "https://www.history.com/topics/modern-history",
                    "world_wars": "https://www.history.com/topics/world-war-i",
                    "science_history": "https://www.history.com/topics/inventions",
                    "technology_history": "https://www.history.com/topics/inventions",
                    "cultural": "https://www.history.com/topics/culture"
                },
                "max_articles": 4,
                "delay": 3
            },
            
            # Cutting-Edge & Emerging Fields
            "cutting_edge": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "nanotechnology": "https://www.nature.com/subjects/nanotechnology",
                    "quantum_computing": "https://www.nature.com/subjects/quantum-computing",
                    "biotechnology": "https://www.nature.com/subjects/biotechnology",
                    "artificial_intelligence": "https://www.nature.com/subjects/artificial-intelligence",
                    "climate_science": "https://www.nature.com/subjects/climate-change",
                    "space_exploration": "https://www.nature.com/subjects/space-exploration",
                    "renewable_energy": "https://www.nature.com/subjects/renewable-energy",
                    "genomics": "https://www.nature.com/subjects/genomics"
                },
                "max_articles": 5,
                "delay": 3
            },
            
            # Interdisciplinary & Applied Sciences
            "interdisciplinary": {
                "base_url": "https://news.mit.edu",
                "categories": {
                    "cognitive_science": "https://news.mit.edu/topic/cognitive-science",
                    "data_science": "https://news.mit.edu/topic/data-science",
                    "environmental": "https://news.mit.edu/topic/environment",
                    "healthcare": "https://news.mit.edu/topic/healthcare",
                    "sustainability": "https://news.mit.edu/topic/sustainability",
                    "urban_planning": "https://news.mit.edu/topic/urban-planning",
                    "economics": "https://news.mit.edu/topic/economics",
                    "education": "https://news.mit.edu/topic/education"
                },
                "max_articles": 4,
                "delay": 3
            }
        }
    
    def scrape_comprehensive_knowledge(self):
        """Scrape comprehensive knowledge across all major fields"""
        
        print("🧠 Comprehensive Knowledge Scraper")
        print("=" * 60)
        print(f"📊 Scraping {len(self.comprehensive_sources)} major knowledge domains")
        print(f"🧠 prime aligned compute enhancement: 1.618x golden ratio")
        print(f"⚡ Covering all major fields of human knowledge")
        
        total_scraped = 0
        domain_stats = {}
        
        for domain_name, config in self.comprehensive_sources.items():
            print(f"\n🧠 Scraping {domain_name.upper()}")
            print("-" * 50)
            
            domain_scraped = 0
            
            for category, category_url in config['categories'].items():
                print(f"\n📂 Category: {category}")
                
                try:
                    articles_scraped = self._scrape_comprehensive_category(
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
        
        self._print_comprehensive_statistics(total_scraped, domain_stats)
        return total_scraped, domain_stats
    
    def _scrape_comprehensive_category(self, domain_name, category, category_url, max_articles, delay):
        """Scrape a comprehensive category with robust error handling"""
        
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links based on domain
            article_links = self._extract_comprehensive_article_links(soup, category_url, domain_name)
            
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
                        # Store comprehensive metadata
                        self._store_comprehensive_metadata(result, domain_name, category)
                        articles_scraped += 1
                        print(f"      ✅ Success: {result.get('title', 'Untitled')[:50]}...")
                    else:
                        print(f"      ❌ Failed")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    # Continue with next article instead of failing completely
            
            return articles_scraped
            
        except Exception as e:
            logger.error(f"Error scraping comprehensive category {category}: {e}")
            return 0
    
    def _extract_comprehensive_article_links(self, soup, base_url, domain_name):
        """Extract article links from comprehensive sources"""
        
        article_links = []
        
        # Domain-specific selectors
        domain_selectors = {
            "mathematics": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "physics": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "biology": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "chemistry": ['a[data-test="article-link"]', '.c-card a', 'h3 a'],
            "engineering": ['h3 a', '.term-page--news-article a', 'article a'],
            "computer_science": ['dt a[href*="/abs/"]', '.list-dateline a[href*="/abs/"]', 'a[href*="/abs/"]'],
            "philosophy": ['h2 a', '.entry-title a', 'article a'],
            "history": ['h3 a', '.post-title a', 'article a'],
            "cutting_edge": ['a[data-test="article-link"]', '.c-card a', 'h3 a'],
            "interdisciplinary": ['h3 a', '.term-page--news-article a', 'article a']
        }
        
        selectors = domain_selectors.get(domain_name, ['h3 a', 'h2 a', 'article a'])
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_comprehensive_article_url(full_url, domain_name):
                            article_links.append(full_url)
                
                if article_links:
                    break
                    
            except Exception:
                continue
        
        return list(set(article_links))
    
    def _store_comprehensive_metadata(self, result, domain_name, category):
        """Store comprehensive metadata"""
        
        try:
            self.knowledge_system.database_service.store_consciousness_data(
                "comprehensive_knowledge",
                {
                    "domain": domain_name,
                    "category": category,
                    "title": result.get('title', ''),
                    "content_length": result.get('content_length', 0),
                    "prime_aligned_score": result.get('prime_aligned_score', 1.0),
                    "scraped_at": datetime.now().isoformat()
                },
                {
                    "source": "comprehensive_scraper",
                    "domain": domain_name,
                    "category": category,
                    "url": result.get('url', ''),
                    "comprehensive": True,
                    "prime_aligned_enhanced": True
                },
                "comprehensive_scraper"
            )
            
        except Exception as e:
            logger.error(f"Error storing comprehensive metadata: {e}")
    
    def _is_valid_comprehensive_article_url(self, url, domain_name):
        """Check if URL is a valid comprehensive article URL"""
        
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
        if domain_name in ["mathematics", "physics", "biology", "computer_science"] and "/abs/" not in url:
            return False
        elif domain_name == "chemistry" and "nature.com" not in url:
            return False
        elif domain_name == "engineering" and "news.mit.edu" not in url:
            return False
        elif domain_name == "philosophy" and "plato.stanford.edu" not in url:
            return False
        elif domain_name == "history" and "history.com" not in url:
            return False
        
        return True
    
    def _print_comprehensive_statistics(self, total_scraped, domain_stats):
        """Print comprehensive scraping statistics"""
        
        print(f"\n🧠 Comprehensive Knowledge Scraping Complete!")
        print("=" * 60)
        print(f"📊 Total Articles Scraped: {total_scraped}")
        print(f"🧠 Knowledge Domains Processed: {len(domain_stats)}")
        
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
        
        print(f"\n🧠 Knowledge Domains Covered:")
        domains = [
            "Mathematics", "Physics", "Biology", "Chemistry",
            "Engineering", "Computer Science", "Philosophy", "History",
            "Cutting-Edge Technology", "Interdisciplinary Studies"
        ]
        
        for i, domain in enumerate(domains, 1):
            print(f"   {i:2d}. {domain}")

def main():
    """Main function to run comprehensive knowledge scraping"""
    
    scraper = ComprehensiveKnowledgeScraper()
    
    print(f"🚀 Starting comprehensive knowledge scraping...")
    print(f"🧠 Covering all major fields of human knowledge")
    print(f"📚 Mathematics, Science, History, Engineering, Philosophy, and more!")
    
    total_scraped, domain_stats = scraper.scrape_comprehensive_knowledge()
    
    print(f"\n🎉 Comprehensive Knowledge Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")
    print(f"   🧠 Comprehensive knowledge ecosystem established!")

if __name__ == "__main__":
    main()
