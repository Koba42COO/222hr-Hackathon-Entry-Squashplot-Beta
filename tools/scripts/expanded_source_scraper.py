#!/usr/bin/env python3
"""
Expanded Source Scraper
======================
Expands the educational ecosystem to include additional high-quality learning sources.
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import random
from urllib.parse import urljoin, urlparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedSourceScraper:
    def __init__(self):
        self.db_path = "web_knowledge.db"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Expanded sources for comprehensive education
        self.expanded_sources = {
            'coursera_courses': {
                'base_url': 'https://www.coursera.org',
                'categories': ['machine-learning', 'data-science', 'artificial-intelligence', 'computer-science'],
                'content_type': 'courses'
            },
            'edx_courses': {
                'base_url': 'https://www.edx.org',
                'categories': ['computer-science', 'data-analysis-statistics', 'engineering'],
                'content_type': 'courses'
            },
            'khan_academy': {
                'base_url': 'https://www.khanacademy.org',
                'categories': ['math', 'science', 'computer-programming', 'economics-finance-domain'],
                'content_type': 'lessons'
            },
            'mit_opencourseware': {
                'base_url': 'https://ocw.mit.edu',
                'categories': ['electrical-engineering-computer-science', 'mathematics', 'physics'],
                'content_type': 'courses'
            },
            'stanford_online': {
                'base_url': 'https://online.stanford.edu',
                'categories': ['artificial-intelligence', 'computer-science', 'data-science'],
                'content_type': 'courses'
            },
            'harvard_online': {
                'base_url': 'https://online-learning.harvard.edu',
                'categories': ['data-science', 'computer-science', 'business'],
                'content_type': 'courses'
            },
            'nature_articles': {
                'base_url': 'https://www.nature.com',
                'categories': ['articles', 'news'],
                'content_type': 'scientific_articles'
            },
            'science_magazine': {
                'base_url': 'https://www.sciencemag.org',
                'categories': ['news', 'research-articles'],
                'content_type': 'scientific_articles'
            },
            'plos_biology': {
                'base_url': 'https://journals.plos.org/plosbiology',
                'categories': ['research-article'],
                'content_type': 'scientific_articles'
            },
            'scientific_american': {
                'base_url': 'https://www.scientificamerican.com',
                'categories': ['article'],
                'content_type': 'popular_science'
            }
        }

    def scrape_expanded_sources(self, max_per_source=5):
        """Scrape from expanded educational sources"""

        print("🚀 Starting expanded source scraping...")
        print(f"🎯 Targeting {len(self.expanded_sources)} new sources")

        total_scraped = 0
        source_stats = {}

        for source_name, source_config in self.expanded_sources.items():
            print(f"\n📂 Processing {source_name}...")

            try:
                source_count = self._scrape_source(source_name, source_config, max_per_source)
                source_stats[source_name] = source_count
                total_scraped += source_count

                print(f"   ✅ {source_name}: {source_count} items")

                # Respectful delay between sources
                time.sleep(random.uniform(2, 5))

            except Exception as e:
                print(f"   ❌ {source_name}: Error - {str(e)}")
                source_stats[source_name] = 0

        print(f"\n📊 Expanded Scraping Complete:")
        print(f"   🌐 Sources processed: {len(self.expanded_sources)}")
        print(f"   📄 Total new content: {total_scraped}")
        print(f"   📈 Average per source: {total_scraped/len(self.expanded_sources):.1f}")

        return total_scraped, source_stats

    def _scrape_source(self, source_name, source_config, max_items):
        """Scrape content from a specific source"""

        scraped_count = 0
        base_url = source_config['base_url']

        for category in source_config['categories']:
            if scraped_count >= max_items:
                break

            try:
                # Construct URL based on source type
                if source_name in ['coursera_courses', 'edx_courses']:
                    url = f"{base_url}/browse/{category}"
                elif source_name == 'khan_academy':
                    url = f"{base_url}/{category}"
                elif source_name == 'mit_opencourseware':
                    url = f"{base_url}/courses/{category}/"
                elif source_name in ['stanford_online', 'harvard_online']:
                    url = f"{base_url}/courses/{category}"
                elif source_name in ['nature_articles', 'science_magazine']:
                    url = f"{base_url}/{category}"
                elif source_name == 'plos_biology':
                    url = f"{base_url}/browse/{category}"
                elif source_name == 'scientific_american':
                    url = f"{base_url}/{category}"
                else:
                    url = base_url

                # Add delay between requests
                time.sleep(random.uniform(1, 3))

                print(f"     🌐 Scraping: {url}")

                response = self.session.get(url, timeout=15)
                response.raise_for_status()

                # Extract content based on source type
                content_items = self._extract_source_content(response.content, source_name, source_config)

                # Store content
                for item in content_items:
                    if scraped_count >= max_items:
                        break

                    try:
                        self._store_content(item['url'], item['title'], item['content'], source_name)
                        scraped_count += 1
                        print(f"       📄 Stored: {item['title'][:40]}...")
                    except Exception as e:
                        logger.error(f"Storage error for {item['title']}: {e}")

            except Exception as e:
                logger.error(f"Error scraping {source_name} category {category}: {e}")

        return scraped_count

    def _extract_source_content(self, html_content, source_name, source_config):
        """Extract content from source-specific HTML structure"""

        soup = BeautifulSoup(html_content, 'html.parser')
        content_items = []

        try:
            if source_name in ['coursera_courses', 'edx_courses']:
                # Course platforms
                course_cards = soup.find_all(['div', 'article'], class_=lambda x: x and ('card' in x.lower() or 'course' in x.lower()))
                for card in course_cards[:10]:  # Limit to avoid overwhelming
                    title_elem = card.find(['h3', 'h2', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        link_elem = card.find('a')
                        url = urljoin(source_config['base_url'], link_elem['href']) if link_elem else source_config['base_url']

                        # Extract description
                        desc_elem = card.find(['p', 'div'], class_=lambda x: x and ('description' in x.lower() or 'summary' in x.lower()))
                        content = desc_elem.get_text().strip() if desc_elem else f"Course: {title}"

                        content_items.append({
                            'url': url,
                            'title': title,
                            'content': content
                        })

            elif source_name == 'khan_academy':
                # Khan Academy lessons
                lesson_links = soup.find_all('a', href=lambda x: x and '/math/' in x or '/science/' in x or '/computer-programming/' in x)
                for link in lesson_links[:15]:
                    title = link.get_text().strip()
                    url = urljoin(source_config['base_url'], link['href'])
                    content = f"Khan Academy Lesson: {title}"

                    content_items.append({
                        'url': url,
                        'title': title,
                        'content': content
                    })

            elif source_name == 'mit_opencourseware':
                # MIT OCW courses
                course_links = soup.find_all('a', href=lambda x: x and 'courses/' in x and len(x) > 20)
                for link in course_links[:10]:
                    title = link.get_text().strip()
                    if title and len(title) > 10:  # Filter meaningful titles
                        url = urljoin(source_config['base_url'], link['href'])
                        content = f"MIT OpenCourseWare: {title}"

                        content_items.append({
                            'url': url,
                            'title': title,
                            'content': content
                        })

            elif source_name in ['nature_articles', 'science_magazine', 'plos_biology']:
                # Scientific articles
                article_links = soup.find_all(['h3', 'h2', 'a'], class_=lambda x: x and ('title' in x.lower() or 'article' in x.lower()))
                for elem in article_links[:12]:
                    if elem.name in ['h3', 'h2']:
                        link_elem = elem.find('a')
                        if link_elem:
                            title = elem.get_text().strip()
                            url = urljoin(source_config['base_url'], link_elem['href'])
                        else:
                            continue
                    else:  # Direct link
                        title = elem.get_text().strip()
                        url = urljoin(source_config['base_url'], elem['href'])

                    content = f"Scientific Article: {title}"

                    content_items.append({
                        'url': url,
                        'title': title,
                        'content': content
                    })

            elif source_name == 'scientific_american':
                # Popular science articles
                article_cards = soup.find_all(['article', 'div'], class_=lambda x: x and ('article' in x.lower() or 'card' in x.lower()))
                for card in article_cards[:8]:
                    title_elem = card.find(['h2', 'h3', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        link_elem = card.find('a')
                        url = urljoin(source_config['base_url'], link_elem['href']) if link_elem else source_config['base_url']

                        content = f"Scientific American: {title}"

                        content_items.append({
                            'url': url,
                            'title': title,
                            'content': content
                        })

            else:
                # Generic fallback
                links = soup.find_all('a', href=True)
                for link in links[:5]:
                    title = link.get_text().strip()
                    if title and len(title) > 5:
                        url = urljoin(source_config['base_url'], link['href'])
                        content = f"Content from {source_name}: {title}"

                        content_items.append({
                            'url': url,
                            'title': title,
                            'content': content
                        })

        except Exception as e:
            logger.error(f"Content extraction error for {source_name}: {e}")

        return content_items

    def _store_content(self, url, title, content, source):
        """Store content in the database"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate content hash for deduplication
            content_hash = str(hash(content + url))

            # Check if URL already exists
            cursor.execute("SELECT id FROM web_content WHERE url = ?", (url,))
            existing = cursor.fetchone()

            if existing:
                # Update existing
                cursor.execute("""
                    UPDATE web_content
                    SET title = ?, content = ?, content_hash = ?, scraped_at = ?
                    WHERE url = ?
                """, (title, content, content_hash, time.time(), url))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO web_content (url, title, content, content_hash, scraped_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (url, title, content, content_hash, time.time()))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Database error storing {title}: {e}")

    def get_expanded_stats(self):
        """Get statistics on the expanded content"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total content
            cursor.execute("SELECT COUNT(*) FROM web_content")
            total_content = cursor.fetchone()[0]

            # Content by source type (inferred from URL patterns)
            source_patterns = {
                'Wikipedia': '%wikipedia.org%',
                'arXiv': '%arxiv.org%',
                'Coursera': '%coursera.org%',
                'edX': '%edx.org%',
                'Khan Academy': '%khanacademy.org%',
                'MIT OCW': '%ocw.mit.edu%',
                'Stanford': '%stanford.edu%',
                'Harvard': '%harvard.edu%',
                'Nature': '%nature.com%',
                'Science': '%sciencemag.org%',
                'PLOS': '%plos.org%',
                'Scientific American': '%scientificamerican.com%'
            }

            source_counts = {}
            for source_name, pattern in source_patterns.items():
                cursor.execute(f"SELECT COUNT(*) FROM web_content WHERE url LIKE '{pattern}'")
                count = cursor.fetchone()[0]
                if count > 0:
                    source_counts[source_name] = count

            # Content quality metrics
            cursor.execute("SELECT AVG(LENGTH(content)) FROM web_content")
            avg_length = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM web_content WHERE LENGTH(content) > 1000")
            substantial_content = cursor.fetchone()[0]

            conn.close()

            return {
                'total_content': total_content,
                'source_distribution': source_counts,
                'average_content_length': round(avg_length, 0),
                'substantial_content_count': substantial_content,
                'sources_count': len([c for c in source_counts.values() if c > 0])
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}

def main():
    """Main function to run expanded source scraping"""

    print("🚀 Starting Expanded Source Educational Scraper")
    print("🎯 Expanding educational ecosystem with additional high-quality sources...")

    scraper = ExpandedSourceScraper()

    # Scrape expanded sources
    total_scraped, source_stats = scraper.scrape_expanded_sources(max_per_source=5)

    print(f"\n📊 Scraping Results:")
    print(f"   🌐 Sources attempted: {len(scraper.expanded_sources)}")
    print(f"   📄 Content items added: {total_scraped}")

    print(f"\n📈 Source Performance:")
    successful_sources = [(name, count) for name, count in source_stats.items() if count > 0]
    failed_sources = [(name, count) for name, count in source_stats.items() if count == 0]

    if successful_sources:
        print("   ✅ Successful sources:")
        for name, count in successful_sources:
            print(f"      • {name}: {count} items")

    if failed_sources:
        print("   ❌ Failed sources:")
        for name, count in failed_sources:
            print(f"      • {name}: {count} items")

    # Get final statistics
    print(f"\n📊 Final Knowledge Base Status:")
    stats = scraper.get_expanded_stats()

    if stats:
        print(f"   📚 Total documents: {stats['total_content']}")
        print(f"   🌐 Educational sources: {stats['sources_count']}")
        print(f"   📏 Average content length: {stats['average_content_length']:,} chars")
        print(f"   ⭐ Substantial content: {stats['substantial_content_count']} items")

        print(f"\n📊 Source Distribution:")
        for source, count in stats['source_distribution'].items():
            percentage = (count / stats['total_content']) * 100
            print(f"   • {source}: {count} items ({percentage:.1f}%)")

    print(f"\n🎉 Expanded Educational Ecosystem Complete!")
    print(f"📚 Enhanced knowledge base with {len(scraper.expanded_sources)} additional sources!")
    print(f"🎓 More comprehensive educational content available!")

if __name__ == "__main__":
    main()
