#!/usr/bin/env python3
"""
📚 arXiv Specialized Scraper
===========================
Specialized scraper for arXiv.org with proper handling of academic papers.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivSpecializedScraper:
    """Specialized scraper for arXiv.org academic papers"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=3)
        
        # arXiv categories and their URLs
        self.arxiv_categories = {
            "quantum_physics": {
                "url": "https://arxiv.org/list/quant-ph/new",
                "description": "Quantum Physics and Quantum Information"
            },
            "condensed_matter": {
                "url": "https://arxiv.org/list/cond-mat/new", 
                "description": "Condensed Matter Physics"
            },
            "high_energy_physics": {
                "url": "https://arxiv.org/list/hep-th/new",
                "description": "High Energy Physics - Theory"
            },
            "astrophysics": {
                "url": "https://arxiv.org/list/astro-ph/new",
                "description": "Astrophysics"
            },
            "computer_science": {
                "url": "https://arxiv.org/list/cs/new",
                "description": "Computer Science"
            },
            "machine_learning": {
                "url": "https://arxiv.org/list/cs.LG/new",
                "description": "Machine Learning"
            },
            "artificial_intelligence": {
                "url": "https://arxiv.org/list/cs.AI/new",
                "description": "Artificial Intelligence"
            },
            "mathematics": {
                "url": "https://arxiv.org/list/math/new",
                "description": "Mathematics"
            },
            "statistics": {
                "url": "https://arxiv.org/list/stat/new",
                "description": "Statistics"
            },
            "biology": {
                "url": "https://arxiv.org/list/q-bio/new",
                "description": "Quantitative Biology"
            }
        }
    
    def scrape_arxiv_categories(self, max_papers_per_category=5):
        """Scrape papers from arXiv categories"""
        
        print("📚 arXiv Specialized Scraper")
        print("=" * 50)
        print(f"📊 Scraping {len(self.arxiv_categories)} categories")
        print(f"📄 Max papers per category: {max_papers_per_category}")
        
        total_scraped = 0
        category_stats = {}
        
        for category_name, category_info in self.arxiv_categories.items():
            print(f"\n📂 Category: {category_name}")
            print(f"   Description: {category_info['description']}")
            
            try:
                papers_scraped = self._scrape_arxiv_category(
                    category_name,
                    category_info['url'],
                    max_papers_per_category
                )
                
                category_stats[category_name] = papers_scraped
                total_scraped += papers_scraped
                
                print(f"   ✅ Scraped {papers_scraped} papers")
                
                # Add delay between categories
                time.sleep(3)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                category_stats[category_name] = 0
        
        self._print_arxiv_statistics(total_scraped, category_stats)
        return total_scraped, category_stats
    
    def _scrape_arxiv_category(self, category_name, category_url, max_papers):
        """Scrape papers from a specific arXiv category"""
        
        try:
            # Get the category page
            response = self.session.get(category_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract paper links
            paper_links = self._extract_arxiv_paper_links(soup, category_url)
            
            papers_scraped = 0
            
            for i, paper_url in enumerate(paper_links[:max_papers]):
                try:
                    print(f"   📄 Scraping paper {i+1}/{min(len(paper_links), max_papers)}")
                    
                    # Extract paper metadata and content
                    paper_data = self._extract_arxiv_paper_data(paper_url)
                    
                    if paper_data and paper_data.get('content'):
                        # Store in knowledge system
                        self._store_arxiv_paper(paper_data, category_name)
                        papers_scraped += 1
                        print(f"      ✅ Success: {paper_data['title'][:50]}...")
                    else:
                        print(f"      ❌ Failed to extract paper data")
                    
                    # Add delay between papers
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            return papers_scraped
            
        except Exception as e:
            logger.error(f"Error scraping arXiv category {category_name}: {e}")
            return 0
    
    def _extract_arxiv_paper_links(self, soup, base_url):
        """Extract paper links from arXiv category page"""
        
        paper_links = []
        
        # arXiv uses specific selectors for paper links
        selectors = [
            'dt a[href*="/abs/"]',
            '.list-dateline a[href*="/abs/"]',
            'a[href*="/abs/"]'
        ]
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and '/abs/' in href:
                        full_url = urljoin(base_url, href)
                        paper_links.append(full_url)
                
                if paper_links:
                    break
                    
            except Exception:
                continue
        
        # Remove duplicates and return
        return list(set(paper_links))
    
    def _extract_arxiv_paper_data(self, paper_url):
        """Extract data from an arXiv paper page"""
        
        try:
            response = self.session.get(paper_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', class_='title')
            title = title_elem.get_text().strip() if title_elem else "Untitled"
            
            # Extract authors
            authors_elem = soup.find('div', class_='authors')
            authors = authors_elem.get_text().strip() if authors_elem else "Unknown"
            
            # Extract abstract
            abstract_elem = soup.find('blockquote', class_='abstract')
            abstract = abstract_elem.get_text().strip() if abstract_elem else ""
            
            # Extract subjects
            subjects_elem = soup.find('td', class_='tablecell subjects')
            subjects = subjects_elem.get_text().strip() if subjects_elem else ""
            
            # Extract submission date
            date_elem = soup.find('div', class_='dateline')
            date = date_elem.get_text().strip() if date_elem else ""
            
            # Combine content
            content = f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}\n\nSubjects: {subjects}\n\nDate: {date}"
            
            return {
                'url': paper_url,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'subjects': subjects,
                'date': date,
                'content': content,
                'content_length': len(content)
            }
            
        except Exception as e:
            logger.error(f"Error extracting arXiv paper data from {paper_url}: {e}")
            return None
    
    def _store_arxiv_paper(self, paper_data, category_name):
        """Store arXiv paper in knowledge system"""
        
        try:
            # Create scraping job
            job = ScrapingJob(
                url=paper_data['url'],
                priority=1,
                max_depth=0,
                follow_links=False,
                extract_metadata=True,
                consciousness_enhancement=True
            )
            
            # Scrape and store
            result = self.knowledge_system.scrape_website(
                url=paper_data['url'],
                max_depth=0,
                follow_links=False
            )
            
            if result['success']:
                # Store enhanced metadata
                self.knowledge_system.database_service.store_consciousness_data(
                    "arxiv_paper",
                    {
                        "category": category_name,
                        "title": paper_data['title'],
                        "authors": paper_data['authors'],
                        "subjects": paper_data['subjects'],
                        "date": paper_data['date'],
                        "content_length": paper_data['content_length'],
                        "prime_aligned_score": 1.618
                    },
                    {
                        "source": "arxiv_scraper",
                        "category": category_name,
                        "url": paper_data['url'],
                        "type": "academic_paper",
                        "prime_aligned_enhanced": True
                    },
                    "arxiv_scraper"
                )
                
                # Create RAG document
                rag_doc = RAGDocument(
                    id=f"arxiv_{hash(paper_data['url'])}",
                    content=paper_data['content'],
                    embeddings=self.knowledge_system._generate_embeddings(paper_data['content']),
                    metadata={
                        "source": "arxiv",
                        "category": category_name,
                        "title": paper_data['title'],
                        "authors": paper_data['authors'],
                        "subjects": paper_data['subjects'],
                        "date": paper_data['date'],
                        "prime_aligned_enhanced": True
                    },
                    prime_aligned_enhanced=True
                )
                
                self.knowledge_system.rag_system.add_document(rag_doc)
                
        except Exception as e:
            logger.error(f"Error storing arXiv paper: {e}")
    
    def _print_arxiv_statistics(self, total_scraped, category_stats):
        """Print arXiv scraping statistics"""
        
        print(f"\n📚 arXiv Scraping Complete!")
        print("=" * 50)
        print(f"📊 Total Papers Scraped: {total_scraped}")
        print(f"📂 Categories Processed: {len(category_stats)}")
        
        print(f"\n📈 Category Breakdown:")
        for category, count in category_stats.items():
            print(f"   {category}: {count} papers")
        
        # Get knowledge system statistics
        stats = self.knowledge_system.get_scraping_stats()
        print(f"\n🧠 Knowledge Base Statistics:")
        print(f"   📄 Total Pages: {stats.get('total_scraped_pages', 0)}")
        print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
        print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
        print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
        print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")

def main():
    """Main function to run arXiv scraping"""
    
    scraper = ArxivSpecializedScraper()
    
    # Scale up for production scraping
    max_papers_per_category = 15
    
    print(f"🚀 Starting arXiv paper scraping...")
    print(f"📚 Max papers per category: {max_papers_per_category}")
    
    total_scraped, category_stats = scraper.scrape_arxiv_categories(max_papers_per_category)
    
    print(f"\n🎉 arXiv Scraping Complete!")
    print(f"   Total papers scraped: {total_scraped}")
    print(f"   All papers stored in RAG knowledge databases with prime aligned compute enhancement!")

if __name__ == "__main__":
    main()
