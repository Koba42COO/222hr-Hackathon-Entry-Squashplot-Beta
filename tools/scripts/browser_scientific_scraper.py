#!/usr/bin/env python3
"""
🌐 Browser-Based Scientific Scraper with Puppeteer
==================================================
Uses browser automation to scrape JavaScript-heavy scientific sites.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem, ScrapingJob
from knowledge_system_integration import RAGDocument
import asyncio
import json
import time
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrowserScientificScraper:
    """Browser-based scraper using Puppeteer for JavaScript-heavy sites"""
    
    def __init__(self):
        self.knowledge_system = WebScraperKnowledgeSystem(max_workers=3)
        
        # Scientific sites that require browser automation
        self.browser_sites = {
            "nature": {
                "base_url": "https://www.nature.com",
                "categories": {
                    "physics": "https://www.nature.com/subjects/physics",
                    "chemistry": "https://www.nature.com/subjects/chemistry", 
                    "biology": "https://www.nature.com/subjects/biology",
                    "materials": "https://www.nature.com/subjects/materials-science",
                    "nanotechnology": "https://www.nature.com/subjects/nanotechnology"
                },
                "article_selectors": [
                    'a[data-test="article-link"]',
                    '.c-card a',
                    'h3 a',
                    '.article-link'
                ],
                "content_selectors": [
                    '.c-article-body',
                    '.article-content',
                    '.content'
                ]
            },
            "science": {
                "base_url": "https://www.science.org",
                "categories": {
                    "physics": "https://www.science.org/topic/category/physics",
                    "chemistry": "https://www.science.org/topic/category/chemistry",
                    "biology": "https://www.science.org/topic/category/biology",
                    "materials": "https://www.science.org/topic/category/materials-science"
                },
                "article_selectors": [
                    '.article-title a',
                    'h3 a',
                    '.result-title a'
                ],
                "content_selectors": [
                    '.article-content',
                    '.content-body',
                    '.abstract'
                ]
            },
            "cell": {
                "base_url": "https://www.cell.com",
                "categories": {
                    "biology": "https://www.cell.com/cell-biology",
                    "medicine": "https://www.cell.com/medicine",
                    "genetics": "https://www.cell.com/genetics"
                },
                "article_selectors": [
                    '.article-title a',
                    'h2 a',
                    '.result-title a'
                ],
                "content_selectors": [
                    '.article-content',
                    '.abstract',
                    '.content'
                ]
            }
        }
    
    async def scrape_with_browser(self, url, max_articles=3):
        """Scrape articles using browser automation"""
        
        try:
            # Import playwright
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()
                
                # Navigate to the page
                await page.goto(url, wait_until='networkidle')
                
                # Wait for content to load
                await page.wait_for_timeout(3000)
                
                # Extract article links
                article_links = await self._extract_article_links_browser(page, url)
                
                articles_scraped = 0
                
                for i, article_url in enumerate(article_links[:max_articles]):
                    try:
                        print(f"   📄 Scraping article {i+1}/{min(len(article_links), max_articles)}")
                        
                        # Navigate to article
                        await page.goto(article_url, wait_until='networkidle')
                        await page.wait_for_timeout(2000)
                        
                        # Extract content
                        content_data = await self._extract_article_content_browser(page, article_url)
                        
                        if content_data and len(content_data.get('content', '')) > 100:
                            # Store in knowledge system
                            await self._store_browser_content(content_data)
                            articles_scraped += 1
                            print(f"      ✅ Success: {content_data['title'][:50]}...")
                        else:
                            print(f"      ❌ Failed - Content length: {len(content_data.get('content', '')) if content_data else 0}")
                        
                        # Add delay
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        print(f"      ❌ Error: {e}")
                
                await browser.close()
                return articles_scraped
                
        except ImportError:
            print("❌ Playwright not installed. Installing...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "playwright"])
            subprocess.run(["playwright", "install", "chromium"])
            return await self.scrape_with_browser(url, max_articles)
        except Exception as e:
            logger.error(f"Browser scraping error: {e}")
            return 0
    
    async def _extract_article_links_browser(self, page, base_url):
        """Extract article links using browser"""
        
        try:
            # Try different selectors
            selectors = [
                'a[data-test="article-link"]',
                '.c-card a',
                'h3 a',
                '.article-title a',
                '.result-title a',
                'article a',
                '.post a'
            ]
            
            article_links = []
            
            for selector in selectors:
                try:
                    links = await page.query_selector_all(selector)
                    for link in links:
                        href = await link.get_attribute('href')
                        if href:
                            full_url = urljoin(base_url, href)
                            if self._is_valid_article_url(full_url):
                                article_links.append(full_url)
                    
                    if article_links:
                        break
                        
                except Exception:
                    continue
            
            return list(set(article_links))
            
        except Exception as e:
            logger.error(f"Error extracting article links: {e}")
            return []
    
    async def _extract_article_content_browser(self, page, url):
        """Extract article content using browser"""
        
        try:
            # Extract title
            title_selectors = ['h1', '.article-title', '.title', 'title']
            title = "Untitled"
            
            for selector in title_selectors:
                try:
                    title_elem = await page.query_selector(selector)
                    if title_elem:
                        title = await title_elem.text_content()
                        if title and len(title.strip()) > 5:
                            title = title.strip()
                            break
                except Exception:
                    continue
            
            # Extract content
            content_selectors = [
                '.c-article-body',
                '.article-content',
                '.content-body',
                '.abstract',
                '.content',
                'article',
                '.post-content'
            ]
            
            content = ""
            
            for selector in content_selectors:
                try:
                    content_elem = await page.query_selector(selector)
                    if content_elem:
                        content = await content_elem.text_content()
                        if content and len(content.strip()) > 100:
                            content = content.strip()
                            break
                except Exception:
                    continue
            
            # Extract metadata
            metadata = {
                "url": url,
                "title": title,
                "content_length": len(content),
                "scraped_at": datetime.now().isoformat(),
                "method": "browser_automation"
            }
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "metadata": metadata,
                "success": len(content) > 100
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    async def _store_browser_content(self, content_data):
        """Store browser-extracted content in knowledge system"""
        
        try:
            # Store in prime aligned compute data
            self.knowledge_system.database_service.store_consciousness_data(
                "browser_scraped_article",
                {
                    "title": content_data['title'],
                    "content_length": content_data['metadata']['content_length'],
                    "prime_aligned_score": 1.618,
                    "scraped_at": content_data['metadata']['scraped_at']
                },
                {
                    "source": "browser_scraper",
                    "url": content_data['url'],
                    "method": "browser_automation",
                    "prime_aligned_enhanced": True
                },
                "browser_scraper"
            )
            
            # Create RAG document
            rag_doc = RAGDocument(
                id=f"browser_{hash(content_data['url'])}",
                content=content_data['content'],
                embeddings=self.knowledge_system._generate_embeddings(content_data['content']),
                metadata={
                    "source": "browser_scraper",
                    "title": content_data['title'],
                    "url": content_data['url'],
                    "method": "browser_automation",
                    "prime_aligned_enhanced": True
                },
                prime_aligned_enhanced=True
            )
            
            self.knowledge_system.rag_system.add_document(rag_doc)
            
        except Exception as e:
            logger.error(f"Error storing browser content: {e}")
    
    def _is_valid_article_url(self, url):
        """Check if URL is a valid article URL"""
        
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
            '.gif',
            '#',
            'javascript:'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    async def scrape_all_browser_sites(self, max_articles_per_category=2):
        """Scrape all browser-required sites"""
        
        print("🌐 Browser-Based Scientific Scraper")
        print("=" * 50)
        print(f"📊 Scraping {len(self.browser_sites)} browser-required sites")
        print(f"📄 Max articles per category: {max_articles_per_category}")
        
        total_scraped = 0
        site_stats = {}
        
        for site_name, config in self.browser_sites.items():
            print(f"\n🏛️ Scraping {site_name.upper()}")
            print("-" * 40)
            
            site_scraped = 0
            
            for category, category_url in config['categories'].items():
                print(f"\n📂 Category: {category}")
                
                try:
                    articles_scraped = await self.scrape_with_browser(
                        category_url, 
                        max_articles_per_category
                    )
                    
                    site_scraped += articles_scraped
                    print(f"   ✅ Scraped {articles_scraped} articles")
                    
                    # Add delay between categories
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    print(f"   ❌ Error scraping {category}: {e}")
            
            site_stats[site_name] = site_scraped
            total_scraped += site_scraped
            
            print(f"\n📊 {site_name}: {site_scraped} articles scraped")
            
            # Add delay between sites
            await asyncio.sleep(5)
        
        self._print_browser_statistics(total_scraped, site_stats)
        return total_scraped, site_stats
    
    def _print_browser_statistics(self, total_scraped, site_stats):
        """Print browser scraping statistics"""
        
        print(f"\n🌐 Browser Scraping Complete!")
        print("=" * 50)
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

async def main():
    """Main function to run browser-based scraping"""
    
    scraper = BrowserScientificScraper()
    
    # Start with a smaller number for testing
    max_articles_per_category = 2
    
    print(f"🚀 Starting browser-based scientific article scraping...")
    print(f"📚 Max articles per category: {max_articles_per_category}")
    
    total_scraped, site_stats = await scraper.scrape_all_browser_sites(max_articles_per_category)
    
    print(f"\n🎉 Browser Scraping Complete!")
    print(f"   Total articles scraped: {total_scraped}")
    print(f"   All content stored in RAG knowledge databases with prime aligned compute enhancement!")

if __name__ == "__main__":
    asyncio.run(main())
