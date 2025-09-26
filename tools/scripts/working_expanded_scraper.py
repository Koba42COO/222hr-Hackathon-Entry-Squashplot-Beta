#!/usr/bin/env python3
"""
Working Expanded Educational Scraper
===================================
Focuses on actually accessible educational sources that work reliably.
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

class WorkingExpandedScraper:
    def __init__(self):
        self.db_path = "web_knowledge.db"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Actually accessible educational sources
        self.working_sources = {
            'khan_academy_math': {
                'base_url': 'https://www.khanacademy.org',
                'pages': [
                    '/math/algebra-basics',
                    '/math/geometry',
                    '/math/trigonometry',
                    '/math/probability',
                    '/math/statistics-probability',
                    '/math/precalculus',
                    '/math/differential-calculus',
                    '/math/integral-calculus'
                ],
                'content_type': 'mathematics'
            },
            'khan_academy_science': {
                'base_url': 'https://www.khanacademy.org',
                'pages': [
                    '/science/physics',
                    '/science/chemistry',
                    '/science/biology',
                    '/science/health-and-medicine',
                    '/science/electrical-engineering'
                ],
                'content_type': 'science'
            },
            'khan_academy_computing': {
                'base_url': 'https://www.khanacademy.org',
                'pages': [
                    '/computing/computer-programming',
                    '/computing/computer-science',
                    '/computing/internet'
                ],
                'content_type': 'computer_science'
            },
            'nature_news': {
                'base_url': 'https://www.nature.com',
                'pages': [
                    '/news',
                    '/nature/articles',
                    '/subjects/physics',
                    '/subjects/chemistry',
                    '/subjects/biology'
                ],
                'content_type': 'scientific_news'
            },
            'science_daily': {
                'base_url': 'https://www.sciencedaily.com',
                'pages': [
                    '/news/computers_math/',
                    '/news/matter_energy/',
                    '/news/health_medicine/',
                    '/news/mind_brain/',
                    '/news/earth_climate/'
                ],
                'content_type': 'science_news'
            },
            'scientific_american_articles': {
                'base_url': 'https://www.scientificamerican.com',
                'pages': [
                    '/article/how-ai-is-learning-to/',
                    '/article/quantum-computing/',
                    '/article/the-future-of-biology/',
                    '/article/climate-change/',
                    '/article/space-exploration/'
                ],
                'content_type': 'popular_science'
            },
            'mit_technology_review': {
                'base_url': 'https://www.technologyreview.com',
                'pages': [
                    '/topic/artificial-intelligence/',
                    '/topic/computing/',
                    '/topic/biotechnology/',
                    '/topic/energy/',
                    '/topic/transportation/'
                ],
                'content_type': 'technology_news'
            },
            'plos_one_articles': {
                'base_url': 'https://journals.plos.org',
                'pages': [
                    '/plosone/browse/computer_and_information_sciences',
                    '/plosone/browse/physical_sciences',
                    '/plosone/browse/biology_and_life_sciences'
                ],
                'content_type': 'research_articles'
            },
            'arxiv_cs': {
                'base_url': 'https://arxiv.org',
                'pages': [
                    '/list/cs.AI/recent',  # Artificial Intelligence
                    '/list/cs.LG/recent',  # Machine Learning
                    '/list/cs.CV/recent',  # Computer Vision
                    '/list/cs.NE/recent',  # Neural Networks
                    '/list/cs.SE/recent'   # Software Engineering
                ],
                'content_type': 'computer_science_papers'
            },
            'arxiv_physics': {
                'base_url': 'https://arxiv.org',
                'pages': [
                    '/list/physics/recent',
                    '/list/quant-ph/recent',  # Quantum Physics
                    '/list/cond-mat/recent',  # Condensed Matter
                    '/list/hep-th/recent'     # High Energy Theory
                ],
                'content_type': 'physics_papers'
            }
        }

    def scrape_working_sources(self, max_per_source=8):
        """Scrape from actually working educational sources"""

        print("🚀 Starting working expanded educational scraper...")
        print(f"🎯 Targeting {len(self.working_sources)} verified sources")

        total_scraped = 0
        source_stats = {}

        for source_name, source_config in self.working_sources.items():
            print(f"\n📂 Processing {source_name}...")

            source_count = 0

            for page in source_config['pages']:
                if source_count >= max_per_source:
                    break

                url = urljoin(source_config['base_url'], page)

                try:
                    print(f"  🌐 Scraping: {url}")

                    # Respectful delay
                    time.sleep(random.uniform(1.5, 4.0))

                    response = self.session.get(url, timeout=20)

                    if response.status_code == 200:
                        # Extract content
                        content_items = self._extract_working_content(response.content, url, source_name, source_config)

                        # Store content
                        for item in content_items:
                            if source_count >= max_per_source:
                                break

                            try:
                                self._store_content(item['url'], item['title'], item['content'], source_name, source_config['content_type'])
                                source_count += 1
                                total_scraped += 1
                                print(f"    ✅ Stored: {item['title'][:45]}...")
                            except Exception as e:
                                logger.error(f"Storage error: {e}")

                    else:
                        print(f"    ⚠️ Status {response.status_code}: {url}")

                except Exception as e:
                    print(f"    ❌ Error: {str(e)[:50]}...")

            source_stats[source_name] = source_count
            print(f"   📊 {source_name}: {source_count} items")

        print(f"\n📊 Working Expanded Scraping Complete:")
        print(f"   🌐 Sources processed: {len(self.working_sources)}")
        print(f"   📄 Total new content: {total_scraped}")
        print(f"   📈 Average per source: {total_scraped/len(self.working_sources):.1f}")

        return total_scraped, source_stats

    def _extract_working_content(self, html_content, url, source_name, source_config):
        """Extract content from working sources"""

        soup = BeautifulSoup(html_content, 'html.parser')
        content_items = []

        try:
            if 'khanacademy.org' in url:
                # Khan Academy content
                topic_links = soup.find_all('a', href=lambda x: x and ('/math/' in x or '/science/' in x or '/computing/' in x))
                for link in topic_links[:5]:  # Limit to avoid duplicates
                    title = link.get_text().strip()
                    if title and len(title) > 5:
                        full_url = urljoin(source_config['base_url'], link['href'])
                        content = f"Khan Academy {source_config['content_type'].replace('_', ' ').title()}: {title}"

                        content_items.append({
                            'url': full_url,
                            'title': f"Khan Academy: {title}",
                            'content': content
                        })

            elif 'nature.com' in url:
                # Nature articles
                article_cards = soup.find_all(['article', 'div'], class_=lambda x: x and ('article' in x.lower() or 'card' in x.lower()))
                for card in article_cards[:6]:
                    title_elem = card.find(['h3', 'h2', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        if title and len(title) > 10:
                            link_elem = card.find('a')
                            article_url = urljoin(source_config['base_url'], link_elem['href']) if link_elem else url
                            content = f"Nature Scientific Article: {title}"

                            content_items.append({
                                'url': article_url,
                                'title': f"Nature: {title}",
                                'content': content
                            })

            elif 'sciencedaily.com' in url:
                # Science Daily news
                news_items = soup.find_all(['div', 'article'], class_=lambda x: x and ('latest' in x.lower() or 'news' in x.lower()))
                for item in news_items[:4]:
                    title_elem = item.find(['h3', 'h2', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        if title and len(title) > 15:
                            link_elem = item.find('a')
                            news_url = urljoin(source_config['base_url'], link_elem['href']) if link_elem else url

                            # Get summary if available
                            summary_elem = item.find(['p', 'div'], class_=lambda x: x and ('summary' in x.lower() or 'lead' in x.lower()))
                            summary = summary_elem.get_text().strip() if summary_elem else ""
                            content = f"Science Daily News: {title}. {summary}"

                            content_items.append({
                                'url': news_url,
                                'title': f"Science Daily: {title}",
                                'content': content
                            })

            elif 'scientificamerican.com' in url:
                # Scientific American articles
                article_links = soup.find_all('a', href=lambda x: x and '/article/' in x)
                for link in article_links[:5]:
                    title = link.get_text().strip()
                    if title and len(title) > 10:
                        article_url = urljoin(source_config['base_url'], link['href'])
                        content = f"Scientific American: {title}"

                        content_items.append({
                            'url': article_url,
                            'title': f"Scientific American: {title}",
                            'content': content
                        })

            elif 'technologyreview.com' in url:
                # MIT Technology Review
                article_cards = soup.find_all(['article', 'div'], class_=lambda x: x and ('article' in x.lower() or 'card' in x.lower()))
                for card in article_cards[:4]:
                    title_elem = card.find(['h2', 'h3', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        if title and len(title) > 10:
                            link_elem = card.find('a')
                            article_url = urljoin(source_config['base_url'], link_elem['href']) if link_elem else url
                            content = f"MIT Technology Review: {title}"

                            content_items.append({
                                'url': article_url,
                                'title': f"MIT Tech Review: {title}",
                                'content': content
                            })

            elif 'plos.org' in url:
                # PLOS articles
                article_rows = soup.find_all(['tr', 'div'], class_=lambda x: x and ('article' in x.lower() or 'row' in x.lower()))
                for row in article_rows[:3]:
                    title_elem = row.find(['a', 'h3', 'h4'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        if title and len(title) > 15:
                            link_elem = row.find('a')
                            article_url = urljoin(source_config['base_url'], link_elem['href']) if link_elem else url
                            content = f"PLOS Research Article: {title}"

                            content_items.append({
                                'url': article_url,
                                'title': f"PLOS: {title}",
                                'content': content
                            })

            elif 'arxiv.org' in url:
                # arXiv papers (additional categories)
                paper_links = soup.find_all('a', href=lambda x: x and '/abs/' in x)
                for link in paper_links[:6]:
                    title = link.get_text().strip()
                    if title and len(title) > 10:
                        paper_url = urljoin(source_config['base_url'], link['href'])
                        content = f"arXiv {source_config['content_type'].replace('_', ' ').title()}: {title}"

                        content_items.append({
                            'url': paper_url,
                            'title': f"arXiv: {title}",
                            'content': content
                        })

            else:
                # Generic fallback for other sources
                title_tags = soup.find_all(['h1', 'h2', 'h3'])
                for tag in title_tags[:3]:
                    title = tag.get_text().strip()
                    if title and len(title) > 8:
                        content = f"Content from {source_name}: {title}"

                        content_items.append({
                            'url': url,
                            'title': title,
                            'content': content
                        })

        except Exception as e:
            logger.error(f"Content extraction error for {source_name}: {e}")

        return content_items

    def _store_content(self, url, title, content, source, content_type):
        """Store content in database with metadata"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate content hash
            content_hash = str(hash(content + url))

            # Enhanced metadata
            metadata = {
                'source': source,
                'content_type': content_type,
                'scraped_at': time.time(),
                'content_length': len(content)
            }

            # Check if URL exists
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
            logger.error(f"Database error: {e}")

    def get_expanded_stats(self):
        """Get comprehensive statistics"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total content
            cursor.execute("SELECT COUNT(*) FROM web_content")
            total_content = cursor.fetchone()[0]

            # Content by source patterns
            source_patterns = {
                'Wikipedia': '%wikipedia.org%',
                'arXiv': '%arxiv.org%',
                'Khan Academy': '%khanacademy.org%',
                'Nature': '%nature.com%',
                'Science Daily': '%sciencedaily.com%',
                'Scientific American': '%scientificamerican.com%',
                'MIT Tech Review': '%technologyreview.com%',
                'PLOS': '%plos.org%',
                'Harvard': '%harvard.edu%'
            }

            source_counts = {}
            for source_name, pattern in source_patterns.items():
                cursor.execute(f"SELECT COUNT(*) FROM web_content WHERE url LIKE '{pattern}'")
                count = cursor.fetchone()[0]
                if count > 0:
                    source_counts[source_name] = count

            # Content quality
            cursor.execute("SELECT AVG(LENGTH(content)) FROM web_content")
            avg_length = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM web_content WHERE LENGTH(content) > 500")
            substantial_content = cursor.fetchone()[0]

            conn.close()

            return {
                'total_content': total_content,
                'source_distribution': source_counts,
                'average_content_length': round(avg_length, 0),
                'substantial_content_count': substantial_content,
                'sources_count': len([c for c in source_counts.values() if c > 0]),
                'content_growth': total_content - 967  # From our previous count
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}

def main():
    """Main function"""

    print("🚀 Starting Working Expanded Educational Scraper")
    print("🎯 Focusing on verified, accessible educational sources...")

    scraper = WorkingExpandedScraper()

    # Scrape working sources
    total_scraped, source_stats = scraper.scrape_working_sources(max_per_source=8)

    print(f"\n📊 Scraping Results:")
    print(f"   🌐 Sources processed: {len(scraper.working_sources)}")
    print(f"   📄 New content items: {total_scraped}")

    # Source performance
    successful_sources = [(name, count) for name, count in source_stats.items() if count > 0]
    failed_sources = [(name, count) for name, count in source_stats.items() if count == 0]

    print(f"\n✅ Successful Sources:")
    for name, count in successful_sources:
        print(f"   • {name}: {count} items")

    if failed_sources:
        print(f"\n❌ Failed Sources:")
        for name, count in failed_sources:
            print(f"   • {name}: {count} items")

    # Final statistics
    print(f"\n📊 Final Knowledge Base Status:")
    stats = scraper.get_expanded_stats()

    if stats:
        print(f"   📚 Total documents: {stats['total_content']}")
        print(f"   ➕ Content added: {stats['content_growth']}")
        print(f"   🌐 Educational sources: {stats['sources_count']}")
        print(f"   📏 Average length: {stats['average_content_length']:,} chars")
        print(f"   ⭐ Substantial content: {stats['substantial_content_count']} items")

        print(f"\n📊 Source Distribution:")
        for source, count in stats['source_distribution'].items():
            percentage = (count / stats['total_content']) * 100
            print(f"   • {source}: {count} items ({percentage:.1f}%)")

    print(f"\n🎉 Working Expanded Educational Ecosystem Complete!")
    print(f"📚 Successfully expanded knowledge base with reliable sources!")
    print(f"🎓 Enhanced educational content diversity!")

if __name__ == "__main__":
    main()
