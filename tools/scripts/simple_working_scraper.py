#!/usr/bin/env python3
"""
Simple Working Web Scraper
=========================
A straightforward, working web scraper that actually adds content to the knowledge base.
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import random
from urllib.parse import urljoin, urlparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWorkingScraper:
    def __init__(self):
        self.db_path = "web_knowledge.db"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Actually accessible, working sources
        self.working_sources = {
            'wikipedia_ai': {
                'base_url': 'https://en.wikipedia.org/wiki/Artificial_intelligence',
                'pages': [
                    '/wiki/Artificial_intelligence',
                    '/wiki/Machine_learning',
                    '/wiki/Deep_learning',
                    '/wiki/Neural_network',
                    '/wiki/Natural_language_processing',
                    '/wiki/Computer_vision',
                    '/wiki/Reinforcement_learning',
                    '/wiki/Expert_system'
                ]
            },
            'wikipedia_science': {
                'base_url': 'https://en.wikipedia.org',
                'pages': [
                    '/wiki/Quantum_computing',
                    '/wiki/Quantum_mechanics',
                    '/wiki/General_relativity',
                    '/wiki/String_theory',
                    '/wiki/Particle_physics',
                    '/wiki/Neuroscience',
                    '/wiki/prime aligned compute',
                    '/wiki/Biology'
                ]
            },
            'wikipedia_tech': {
                'base_url': 'https://en.wikipedia.org',
                'pages': [
                    '/wiki/Blockchain',
                    '/wiki/Cryptography',
                    '/wiki/Computer_security',
                    '/wiki/Distributed_computing',
                    '/wiki/Parallel_computing',
                    '/wiki/Algorithm',
                    '/wiki/Data_structure'
                ]
            },
            'arxiv_abstracts': {
                'base_url': 'https://arxiv.org',
                'pages': [
                    '/abs/2001.08361',  # GPT-3 paper
                    '/abs/1810.04805',  # BERT paper
                    '/abs/1706.03762',  # Attention is all you need
                    '/abs/1607.06450',  # WaveNet
                    '/abs/1409.3215'    # Generative Adversarial Networks
                ]
            }
        }

    def scrape_working_sources(self):
        """Scrape from actually working sources"""

        print("🔍 Starting simple, working web scraper...")

        total_scraped = 0
        total_errors = 0

        for source_name, source_config in self.working_sources.items():
            print(f"\n📂 Processing {source_name}...")

            for page in source_config['pages']:
                url = urljoin(source_config['base_url'], page)

                try:
                    print(f"  🌐 Scraping: {url}")

                    # Add random delay to be respectful
                    time.sleep(random.uniform(1, 3))

                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()

                    # Extract content
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Extract title
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "No Title"

                    # Extract main content
                    content = self._extract_content(soup, url)

                    if content and len(content) > 200:  # Only store substantial content
                        self._store_content(url, title_text, content)
                        print(f"    ✅ Stored: {title_text[:50]}...")
                        total_scraped += 1
                    else:
                        print(f"    ⚠️ Insufficient content")
                        total_errors += 1

                except Exception as e:
                    print(f"    ❌ Error: {str(e)}")
                    total_errors += 1

        print(f"\n📊 Scraping complete:")
        print(f"  ✅ Successful: {total_scraped}")
        print(f"  ❌ Errors: {total_errors}")

        return total_scraped

    def _extract_content(self, soup, url):
        """Extract meaningful content from the page"""

        content = ""

        # Try different content selectors based on site
        if 'wikipedia.org' in url:
            # Wikipedia content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                # Get text from paragraphs
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs[:10]])  # First 10 paragraphs

        elif 'arxiv.org' in url:
            # ArXiv abstracts
            abstract_div = soup.find('blockquote', {'class': 'abstract'})
            if abstract_div:
                content = abstract_div.get_text()
            else:
                # Fallback to meta description or any blockquote
                meta_desc = soup.find('meta', {'name': 'description'})
                if meta_desc:
                    content = meta_desc.get('content', '')
                else:
                    blockquotes = soup.find_all('blockquote')
                    if blockquotes:
                        content = blockquotes[0].get_text()

        else:
            # Generic content extraction
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': 'content'})
            if main_content:
                paragraphs = main_content.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs[:5]])
            else:
                # Fallback: get all paragraph text
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs[:5]])

        # Clean up the content
        content = content.strip()
        # Remove extra whitespace
        content = ' '.join(content.split())
        # Limit length
        if len(content) > 5000:
            content = content[:5000] + "..."

        return content

    def _store_content(self, url, title, content):
        """Store content in the database"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate content hash for deduplication
            content_hash = hash(content)

            # Check if URL already exists
            cursor.execute("SELECT id FROM web_content WHERE url = ?", (url,))
            existing = cursor.fetchone()

            if existing:
                # Update existing
                cursor.execute("""
                    UPDATE web_content
                    SET title = ?, content = ?, content_hash = ?, scraped_at = ?
                    WHERE url = ?
                """, (title, content, str(content_hash), time.time(), url))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO web_content (url, title, content, content_hash, scraped_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (url, title, content, str(content_hash), time.time()))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Database error: {e}")

def main():
    scraper = SimpleWorkingScraper()
    scraped_count = scraper.scrape_working_sources()

    print("\n🎉 Simple scraper completed!")
    print(f"📊 Documents added/updated: {scraped_count}")

    # Show final count
    try:
        conn = sqlite3.connect("web_knowledge.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM web_content")
        final_count = cursor.fetchone()[0]
        conn.close()
        print(f"📄 Total documents in database: {final_count}")
    except Exception as e:
        print(f"Error checking final count: {e}")

if __name__ == "__main__":
    main()
