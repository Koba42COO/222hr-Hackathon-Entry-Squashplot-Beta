#!/usr/bin/env python3
"""
KOBA42 PHYS.ORG ARTICLE SCRAPER
===============================
Comprehensive Web Scraper for Phys.org Articles by Field
=======================================================

Features:
1. Multi-field Article Scraping (Physics, Chemistry, Technology, etc.)
2. Quantum Physics Research Categorization
3. Article Content Extraction and Analysis
4. Research Trend Analysis
5. Automated Article Import and Processing
6. Enhanced KOBA42 Research Integration
"""

import requests
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin, urlparse
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PhysOrgArticle:
    """Phys.org article data structure."""
    title: str
    url: str
    field: str  # physics, chemistry, technology, etc.
    subfield: str  # quantum_physics, materials_science, etc.
    publication_date: str
    authors: List[str]
    summary: str
    content: str
    tags: List[str]
    research_impact: float  # 0-10 scale
    quantum_relevance: float  # 0-10 scale
    article_id: str

@dataclass
class ResearchField:
    """Research field configuration."""
    field_name: str
    subfields: List[str]
    keywords: List[str]
    quantum_relevance_keywords: List[str]
    priority_score: float  # 0-10 scale

class PhysOrgArticleScraper:
    """Comprehensive Phys.org article scraper with field categorization."""
    
    def __init__(self):
        self.base_url = "https://phys.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.research_fields = self._define_research_fields()
        self.scraped_articles = []
        self.article_cache = {}
        
        logger.info("Phys.org Article Scraper initialized")
    
    def _define_research_fields(self) -> Dict[str, ResearchField]:
        """Define research fields and their characteristics."""
        return {
            'physics': ResearchField(
                field_name='Physics',
                subfields=['quantum_physics', 'particle_physics', 'condensed_matter', 'optics', 'plasma_physics'],
                keywords=['quantum', 'particle', 'matter', 'energy', 'force', 'wave', 'field', 'atom', 'molecule'],
                quantum_relevance_keywords=['quantum', 'entanglement', 'superposition', 'qubit', 'quantum_computing', 'quantum_mechanics'],
                priority_score=9.5
            ),
            'chemistry': ResearchField(
                field_name='Chemistry',
                subfields=['quantum_chemistry', 'materials_chemistry', 'biochemistry', 'physical_chemistry'],
                keywords=['molecule', 'reaction', 'catalyst', 'synthesis', 'compound', 'element', 'bond'],
                quantum_relevance_keywords=['quantum_chemistry', 'molecular_orbital', 'electronic_structure', 'quantum_tunneling'],
                priority_score=8.0
            ),
            'technology': ResearchField(
                field_name='Technology',
                subfields=['quantum_computing', 'nanotechnology', 'electronics', 'artificial_intelligence'],
                keywords=['computer', 'algorithm', 'device', 'sensor', 'processor', 'memory', 'network'],
                quantum_relevance_keywords=['quantum_computer', 'quantum_algorithm', 'quantum_sensor', 'quantum_network'],
                priority_score=9.0
            ),
            'materials_science': ResearchField(
                field_name='Materials Science',
                subfields=['quantum_materials', 'nanomaterials', 'semiconductors', 'superconductors'],
                keywords=['material', 'crystal', 'structure', 'property', 'conductivity', 'magnetic'],
                quantum_relevance_keywords=['quantum_material', 'topological_insulator', 'quantum_dot', 'quantum_well'],
                priority_score=8.5
            ),
            'astronomy': ResearchField(
                field_name='Astronomy',
                subfields=['quantum_astronomy', 'cosmology', 'astrophysics', 'planetary_science'],
                keywords=['star', 'galaxy', 'planet', 'universe', 'cosmic', 'space', 'telescope'],
                quantum_relevance_keywords=['quantum_gravity', 'quantum_cosmology', 'quantum_black_hole'],
                priority_score=7.5
            ),
            'biology': ResearchField(
                field_name='Biology',
                subfields=['quantum_biology', 'biophysics', 'molecular_biology', 'neuroscience'],
                keywords=['cell', 'protein', 'gene', 'organism', 'evolution', 'neural', 'brain'],
                quantum_relevance_keywords=['quantum_biology', 'quantum_coherence', 'quantum_tunneling_biological'],
                priority_score=7.0
            )
        }
    
    def scrape_phys_org_articles(self, max_articles: int = 50, field_filter: str = None) -> List[PhysOrgArticle]:
        """Scrape articles from Phys.org with field categorization."""
        logger.info(f"🔍 Scraping Phys.org articles (max: {max_articles})")
        
        articles = []
        
        # Scrape main page for recent articles
        main_articles = self._scrape_main_page(max_articles // 2)
        articles.extend(main_articles)
        
        # Scrape physics section specifically
        physics_articles = self._scrape_physics_section(max_articles // 2)
        articles.extend(physics_articles)
        
        # Filter by field if specified
        if field_filter:
            articles = [article for article in articles if article.field == field_filter]
        
        # Remove duplicates
        unique_articles = self._remove_duplicates(articles)
        
        # Sort by quantum relevance
        unique_articles.sort(key=lambda x: x.quantum_relevance, reverse=True)
        
        logger.info(f"✅ Scraped {len(unique_articles)} unique articles")
        return unique_articles[:max_articles]
    
    def _scrape_main_page(self, max_articles: int) -> List[PhysOrgArticle]:
        """Scrape articles from the main Phys.org page."""
        try:
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find article links
            article_links = soup.find_all('a', href=re.compile(r'/news/\d{4}-\d{2}-\d{2}-'))
            
            for link in article_links[:max_articles]:
                try:
                    article_url = urljoin(self.base_url, link['href'])
                    article = self._scrape_single_article(article_url)
                    if article:
                        articles.append(article)
                    
                    # Be respectful with requests
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape article {link['href']}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to scrape main page: {e}")
            return []
    
    def _scrape_physics_section(self, max_articles: int) -> List[PhysOrgArticle]:
        """Scrape articles from the physics section."""
        try:
            physics_url = f"{self.base_url}/physics-news/"
            response = self.session.get(physics_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find physics article links
            article_links = soup.find_all('a', href=re.compile(r'/news/\d{4}-\d{2}-\d{2}-'))
            
            for link in article_links[:max_articles]:
                try:
                    article_url = urljoin(self.base_url, link['href'])
                    article = self._scrape_single_article(article_url)
                    if article and article.field == 'physics':
                        articles.append(article)
                    
                    # Be respectful with requests
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape physics article {link['href']}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to scrape physics section: {e}")
            return []
    
    def _scrape_single_article(self, url: str) -> Optional[PhysOrgArticle]:
        """Scrape a single article from its URL."""
        # Check cache first
        if url in self.article_cache:
            return self.article_cache[url]
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data
            title = self._extract_title(soup)
            if not title:
                return None
            
            publication_date = self._extract_publication_date(soup)
            authors = self._extract_authors(soup)
            summary = self._extract_summary(soup)
            content = self._extract_content(soup)
            tags = self._extract_tags(soup)
            
            # Categorize article
            field, subfield = self._categorize_article(title, summary, content, tags)
            
            # Calculate relevance scores
            research_impact = self._calculate_research_impact(title, summary, content)
            quantum_relevance = self._calculate_quantum_relevance(title, summary, content, tags)
            
            # Generate article ID
            article_id = self._generate_article_id(url, title)
            
            article = PhysOrgArticle(
                title=title,
                url=url,
                field=field,
                subfield=subfield,
                publication_date=publication_date,
                authors=authors,
                summary=summary,
                content=content,
                tags=tags,
                research_impact=research_impact,
                quantum_relevance=quantum_relevance,
                article_id=article_id
            )
            
            # Cache the article
            self.article_cache[url] = article
            
            return article
            
        except Exception as e:
            logger.warning(f"Failed to scrape article {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title."""
        title_elem = soup.find('h1') or soup.find('title')
        if title_elem:
            return title_elem.get_text().strip()
        return ""
    
    def _extract_publication_date(self, soup: BeautifulSoup) -> str:
        """Extract publication date."""
        # Look for date in various formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}\s+\w+\s+\d{4}',
            r'\w+\s+\d{1,2},\s+\d{4}'
        ]
        
        text = soup.get_text()
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        
        return datetime.now().strftime("%Y-%m-%d")
    
    def _extract_authors(self, soup: BeautifulSoup) -> List[str]:
        """Extract article authors."""
        authors = []
        
        # Look for author information
        author_elements = soup.find_all(['span', 'div', 'p'], class_=re.compile(r'author|byline'))
        
        for elem in author_elements:
            text = elem.get_text().strip()
            if 'by' in text.lower():
                author_text = text.split('by')[-1].strip()
                authors.extend([a.strip() for a in author_text.split(',')])
        
        return authors if authors else ["Unknown Author"]
    
    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract article summary."""
        # Look for summary/abstract
        summary_elem = soup.find(['div', 'p'], class_=re.compile(r'summary|abstract|lead'))
        if summary_elem:
            return summary_elem.get_text().strip()
        
        # Fallback to first paragraph
        first_p = soup.find('p')
        if first_p:
            return first_p.get_text().strip()
        
        return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content."""
        # Look for main content area
        content_elem = soup.find(['div', 'article'], class_=re.compile(r'content|article|body'))
        if content_elem:
            # Remove script and style elements
            for script in content_elem(["script", "style"]):
                script.decompose()
            
            return content_elem.get_text().strip()
        
        return ""
    
    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract article tags."""
        tags = []
        
        # Look for tag elements
        tag_elements = soup.find_all(['a', 'span'], class_=re.compile(r'tag|category'))
        
        for elem in tag_elements:
            tag_text = elem.get_text().strip()
            if tag_text:
                tags.append(tag_text)
        
        return tags
    
    def _categorize_article(self, title: str, summary: str, content: str, tags: List[str]) -> Tuple[str, str]:
        """Categorize article by field and subfield."""
        text = f"{title} {summary} {' '.join(tags)}".lower()
        
        best_field = "general"
        best_subfield = "general"
        best_score = 0
        
        for field_name, field_config in self.research_fields.items():
            # Calculate field match score
            field_score = 0
            for keyword in field_config.keywords:
                if keyword.lower() in text:
                    field_score += 1
            
            # Calculate quantum relevance score
            quantum_score = 0
            for keyword in field_config.quantum_relevance_keywords:
                if keyword.lower() in text:
                    quantum_score += 2  # Higher weight for quantum keywords
            
            total_score = field_score + quantum_score
            
            if total_score > best_score:
                best_score = total_score
                best_field = field_name
                
                # Determine subfield
                for subfield in field_config.subfields:
                    if subfield.replace('_', ' ') in text:
                        best_subfield = subfield
                        break
                else:
                    best_subfield = field_config.subfields[0] if field_config.subfields else "general"
        
        return best_field, best_subfield
    
    def _calculate_research_impact(self, title: str, summary: str, content: str) -> float:
        """Calculate research impact score (0-10)."""
        text = f"{title} {summary} {content}".lower()
        
        impact_score = 0
        
        # Keywords indicating high impact
        high_impact_keywords = [
            'breakthrough', 'discovery', 'first', 'novel', 'revolutionary',
            'groundbreaking', 'milestone', 'advance', 'innovation', 'new'
        ]
        
        for keyword in high_impact_keywords:
            if keyword in text:
                impact_score += 1
        
        # Normalize to 0-10 scale
        return min(impact_score, 10.0)
    
    def _calculate_quantum_relevance(self, title: str, summary: str, content: str, tags: List[str]) -> float:
        """Calculate quantum relevance score (0-10)."""
        text = f"{title} {summary} {' '.join(tags)}".lower()
        
        quantum_score = 0
        
        # Quantum-specific keywords with weights
        quantum_keywords = {
            'quantum': 3,
            'qubit': 3,
            'entanglement': 3,
            'superposition': 3,
            'quantum_computing': 4,
            'quantum_mechanics': 3,
            'quantum_physics': 3,
            'quantum_chemistry': 3,
            'quantum_material': 3,
            'quantum_algorithm': 3,
            'quantum_sensor': 3,
            'quantum_network': 3,
            'quantum_internet': 4,
            'quantum_hall': 3,
            'quantum_spin': 3,
            'quantum_optics': 3,
            'quantum_information': 3,
            'quantum_cryptography': 3,
            'quantum_simulation': 3,
            'quantum_advantage': 4
        }
        
        for keyword, weight in quantum_keywords.items():
            if keyword.replace('_', ' ') in text:
                quantum_score += weight
        
        # Normalize to 0-10 scale
        return min(quantum_score, 10.0)
    
    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID."""
        content = f"{url}{title}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _remove_duplicates(self, articles: List[PhysOrgArticle]) -> List[PhysOrgArticle]:
        """Remove duplicate articles based on URL and title."""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls and article.title not in seen_titles:
                seen_urls.add(article.url)
                seen_titles.add(article.title)
                unique_articles.append(article)
        
        return unique_articles
    
    def analyze_research_trends(self, articles: List[PhysOrgArticle]) -> Dict[str, Any]:
        """Analyze research trends from scraped articles."""
        logger.info("📊 Analyzing research trends")
        
        trends = {
            'total_articles': len(articles),
            'field_distribution': {},
            'quantum_relevance_distribution': {},
            'research_impact_distribution': {},
            'top_keywords': {},
            'recent_breakthroughs': [],
            'quantum_focus_areas': {}
        }
        
        # Field distribution
        for article in articles:
            field = article.field
            trends['field_distribution'][field] = trends['field_distribution'].get(field, 0) + 1
        
        # Quantum relevance distribution
        for article in articles:
            relevance_level = int(article.quantum_relevance)
            trends['quantum_relevance_distribution'][relevance_level] = trends['quantum_relevance_distribution'].get(relevance_level, 0) + 1
        
        # Research impact distribution
        for article in articles:
            impact_level = int(article.research_impact)
            trends['research_impact_distribution'][impact_level] = trends['research_impact_distribution'].get(impact_level, 0) + 1
        
        # Top keywords
        all_text = " ".join([f"{a.title} {a.summary} {' '.join(a.tags)}" for a in articles]).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 20 keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        trends['top_keywords'] = dict(top_keywords)
        
        # Recent breakthroughs (high impact articles)
        breakthroughs = [a for a in articles if a.research_impact >= 7.0]
        trends['recent_breakthroughs'] = [
            {
                'title': a.title,
                'url': a.url,
                'impact_score': a.research_impact,
                'quantum_relevance': a.quantum_relevance
            }
            for a in breakthroughs[:10]
        ]
        
        # Quantum focus areas
        quantum_articles = [a for a in articles if a.quantum_relevance >= 5.0]
        for article in quantum_articles:
            subfield = article.subfield
            trends['quantum_focus_areas'][subfield] = trends['quantum_focus_areas'].get(subfield, 0) + 1
        
        return trends
    
    def generate_research_report(self, articles: List[PhysOrgArticle], trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'scraping_summary': {
                'total_articles_scraped': len(articles),
                'scraping_date': datetime.now().strftime("%Y-%m-%d"),
                'source': 'Phys.org'
            },
            'articles': [],
            'trends_analysis': trends,
            'quantum_research_highlights': [],
            'recommendations': []
        }
        
        # Add articles
        for article in articles:
            report['articles'].append({
                'title': article.title,
                'url': article.url,
                'field': article.field,
                'subfield': article.subfield,
                'publication_date': article.publication_date,
                'authors': article.authors,
                'summary': article.summary,
                'tags': article.tags,
                'research_impact': article.research_impact,
                'quantum_relevance': article.quantum_relevance,
                'article_id': article.article_id
            })
        
        # Quantum research highlights
        quantum_articles = [a for a in articles if a.quantum_relevance >= 7.0]
        for article in quantum_articles[:10]:
            report['quantum_research_highlights'].append({
                'title': article.title,
                'url': article.url,
                'quantum_relevance': article.quantum_relevance,
                'research_impact': article.research_impact,
                'field': article.field,
                'subfield': article.subfield
            })
        
        # Generate recommendations
        report['recommendations'] = [
            "Focus on high quantum relevance articles for KOBA42 integration",
            "Monitor breakthrough research in quantum computing and materials",
            "Track emerging quantum technologies and applications",
            "Analyze quantum research trends for optimization opportunities",
            "Integrate quantum physics discoveries into optimization algorithms",
            "Stay updated on quantum internet and communication developments",
            "Follow quantum chemistry and materials science advances"
        ]
        
        return report

def demonstrate_phys_org_scraping():
    """Demonstrate Phys.org article scraping and analysis."""
    logger.info("🚀 KOBA42 Phys.org Article Scraper")
    logger.info("=" * 50)
    
    # Initialize scraper
    scraper = PhysOrgArticleScraper()
    
    # Scrape articles
    print("\n🔍 Scraping Phys.org articles...")
    articles = scraper.scrape_phys_org_articles(max_articles=30)
    
    if not articles:
        print("❌ No articles scraped successfully")
        return
    
    print(f"✅ Successfully scraped {len(articles)} articles")
    
    # Analyze trends
    print("\n📊 Analyzing research trends...")
    trends = scraper.analyze_research_trends(articles)
    
    # Generate report
    print("\n📄 Generating research report...")
    report = scraper.generate_research_report(articles, trends)
    
    # Display summary
    print("\n📋 SCRAPING SUMMARY")
    print("=" * 50)
    print(f"Total Articles: {len(articles)}")
    print(f"Field Distribution: {trends['field_distribution']}")
    print(f"Quantum Relevance (≥7): {sum(1 for a in articles if a.quantum_relevance >= 7)}")
    print(f"High Impact (≥7): {sum(1 for a in articles if a.research_impact >= 7)}")
    
    # Display top quantum articles
    print("\n🔬 TOP QUANTUM RESEARCH ARTICLES")
    print("=" * 50)
    quantum_articles = [a for a in articles if a.quantum_relevance >= 5.0]
    quantum_articles.sort(key=lambda x: x.quantum_relevance, reverse=True)
    
    for i, article in enumerate(quantum_articles[:5], 1):
        print(f"\n{i}. {article.title}")
        print(f"   Field: {article.field} ({article.subfield})")
        print(f"   Quantum Relevance: {article.quantum_relevance:.1f}/10")
        print(f"   Research Impact: {article.research_impact:.1f}/10")
        print(f"   URL: {article.url}")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'phys_org_research_report_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Research report saved to {report_file}")
    
    return articles, trends, report_file

if __name__ == "__main__":
    # Run Phys.org scraping demonstration
    articles, trends, report_file = demonstrate_phys_org_scraping()
    
    print(f"\n🎉 Phys.org article scraping demonstration completed!")
    print(f"📊 Report saved to: {report_file}")
    print(f"🔬 Scraped {len(articles)} articles from Phys.org")
    print(f"📈 Analyzed research trends and quantum relevance")
