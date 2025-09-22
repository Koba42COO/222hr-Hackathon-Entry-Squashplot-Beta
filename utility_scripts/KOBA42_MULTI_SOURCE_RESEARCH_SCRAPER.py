#!/usr/bin/env python3
"""
KOBA42 MULTI-SOURCE RESEARCH SCRAPER
====================================
Comprehensive Multi-Source Research Scraper (Phys.org, Nature, InfoQ)
===================================================================

Features:
1. Multi-Source Article Scraping (Phys.org, Nature, InfoQ)
2. Cross-Domain Research Categorization
3. Quantum Physics Research Integration
4. Technology and Software Development Trends
5. Research Impact Analysis Across Sources
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
class ResearchArticle:
    """Multi-source research article data structure."""
    title: str
    url: str
    source: str  # phys_org, nature, infoq
    field: str  # physics, chemistry, technology, software, etc.
    subfield: str  # quantum_physics, materials_science, etc.
    publication_date: str
    authors: List[str]
    summary: str
    content: str
    tags: List[str]
    research_impact: float  # 0-10 scale
    quantum_relevance: float  # 0-10 scale
    technology_relevance: float  # 0-10 scale
    article_id: str

@dataclass
class ResearchSource:
    """Research source configuration."""
    source_name: str
    base_url: str
    article_pattern: str
    title_selector: str
    content_selector: str
    author_selector: str
    date_selector: str
    tags_selector: str
    requires_auth: bool
    rate_limit: float  # seconds between requests

class MultiSourceResearchScraper:
    """Comprehensive multi-source research scraper."""
    
    def __init__(self):
        self.sources = self._define_research_sources()
        self.research_fields = self._define_research_fields()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.scraped_articles = []
        self.article_cache = {}
        
        logger.info("Multi-Source Research Scraper initialized")
    
    def _define_research_sources(self) -> Dict[str, ResearchSource]:
        """Define research sources and their configurations."""
        return {
            'phys_org': ResearchSource(
                source_name='Phys.org',
                base_url='https://phys.org',
                article_pattern=r'/news/\d{4}-\d{2}-\d{2}-',
                title_selector='h1, title',
                content_selector='div.article-content, article, div.content',
                author_selector='span.author, div.byline, p.author',
                date_selector='span.date, time, div.date',
                tags_selector='a.tag, span.category, div.tags',
                requires_auth=False,
                rate_limit=2.0
            ),
            'nature': ResearchSource(
                source_name='Nature',
                base_url='https://www.nature.com',
                article_pattern=r'/articles/',
                title_selector='h1[data-test="article-title"], h1.title',
                content_selector='div.article__body, div.content, article',
                author_selector='span.author, a.author, div.authors',
                date_selector='time, span.date, div.publication-date',
                tags_selector='a.tag, span.category, div.keywords',
                requires_auth=True,
                rate_limit=3.0
            ),
            'infoq': ResearchSource(
                source_name='InfoQ',
                base_url='https://www.infoq.com',
                article_pattern=r'/news/\d{4}/\d{2}/',
                title_selector='h1.title, h1.article-title',
                content_selector='div.article-content, div.content, article',
                author_selector='span.author, a.author, div.byline',
                date_selector='time, span.date, div.publication-date',
                tags_selector='a.tag, span.category, div.tags',
                requires_auth=False,
                rate_limit=2.5
            )
        }
    
    def _define_research_fields(self) -> Dict[str, Dict[str, Any]]:
        """Define research fields and their characteristics."""
        return {
            'physics': {
                'keywords': ['quantum', 'particle', 'matter', 'energy', 'force', 'wave', 'field', 'atom', 'molecule'],
                'quantum_keywords': ['quantum', 'entanglement', 'superposition', 'qubit', 'quantum_computing', 'quantum_mechanics'],
                'priority_score': 9.5
            },
            'chemistry': {
                'keywords': ['molecule', 'reaction', 'catalyst', 'synthesis', 'compound', 'element', 'bond'],
                'quantum_keywords': ['quantum_chemistry', 'molecular_orbital', 'electronic_structure', 'quantum_tunneling'],
                'priority_score': 8.0
            },
            'technology': {
                'keywords': ['computer', 'algorithm', 'device', 'sensor', 'processor', 'memory', 'network'],
                'quantum_keywords': ['quantum_computer', 'quantum_algorithm', 'quantum_sensor', 'quantum_network'],
                'priority_score': 9.0
            },
            'software': {
                'keywords': ['software', 'programming', 'code', 'application', 'system', 'framework', 'library'],
                'quantum_keywords': ['quantum_software', 'quantum_programming', 'quantum_framework'],
                'priority_score': 7.5
            },
            'materials_science': {
                'keywords': ['material', 'crystal', 'structure', 'property', 'conductivity', 'magnetic'],
                'quantum_keywords': ['quantum_material', 'topological_insulator', 'quantum_dot', 'quantum_well'],
                'priority_score': 8.5
            },
            'artificial_intelligence': {
                'keywords': ['ai', 'machine_learning', 'neural_network', 'algorithm', 'intelligence', 'automation'],
                'quantum_keywords': ['quantum_ai', 'quantum_machine_learning', 'quantum_neural_network'],
                'priority_score': 8.0
            }
        }
    
    def scrape_all_sources(self, max_articles_per_source: int = 20) -> List[ResearchArticle]:
        """Scrape articles from all configured sources."""
        logger.info(f"🔍 Scraping articles from all sources (max: {max_articles_per_source} per source)")
        
        all_articles = []
        
        for source_name, source_config in self.sources.items():
            try:
                logger.info(f"Scraping from {source_config.source_name}...")
                articles = self._scrape_source(source_name, source_config, max_articles_per_source)
                all_articles.extend(articles)
                
                # Rate limiting between sources
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                logger.error(f"Failed to scrape {source_name}: {e}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_articles = self._remove_duplicates(all_articles)
        unique_articles.sort(key=lambda x: (x.quantum_relevance + x.technology_relevance) / 2, reverse=True)
        
        logger.info(f"✅ Total articles scraped: {len(unique_articles)}")
        return unique_articles
    
    def _scrape_source(self, source_name: str, source_config: ResearchSource, max_articles: int) -> List[ResearchArticle]:
        """Scrape articles from a specific source."""
        articles = []
        
        try:
            # Get main page
            response = self.session.get(source_config.base_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=re.compile(source_config.article_pattern))
            
            for link in article_links[:max_articles]:
                try:
                    article_url = urljoin(source_config.base_url, link['href'])
                    article = self._scrape_single_article(article_url, source_name, source_config)
                    
                    if article:
                        articles.append(article)
                    
                    # Rate limiting
                    time.sleep(random.uniform(source_config.rate_limit, source_config.rate_limit + 1))
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape article from {source_name}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to scrape {source_name}: {e}")
            return []
    
    def _scrape_single_article(self, url: str, source_name: str, source_config: ResearchSource) -> Optional[ResearchArticle]:
        """Scrape a single article from its URL."""
        # Check cache first
        if url in self.article_cache:
            return self.article_cache[url]
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data using source-specific selectors
            title = self._extract_with_selectors(soup, source_config.title_selector)
            if not title:
                return None
            
            content = self._extract_with_selectors(soup, source_config.content_selector)
            authors = self._extract_authors_with_selectors(soup, source_config.author_selector)
            publication_date = self._extract_date_with_selectors(soup, source_config.date_selector)
            tags = self._extract_tags_with_selectors(soup, source_config.tags_selector)
            
            # Generate summary from content
            summary = self._generate_summary(content)
            
            # Categorize article
            field, subfield = self._categorize_article(title, summary, content, tags)
            
            # Calculate relevance scores
            research_impact = self._calculate_research_impact(title, summary, content)
            quantum_relevance = self._calculate_quantum_relevance(title, summary, content, tags)
            technology_relevance = self._calculate_technology_relevance(title, summary, content, tags)
            
            # Generate article ID
            article_id = self._generate_article_id(url, title)
            
            article = ResearchArticle(
                title=title,
                url=url,
                source=source_name,
                field=field,
                subfield=subfield,
                publication_date=publication_date,
                authors=authors,
                summary=summary,
                content=content,
                tags=tags,
                research_impact=research_impact,
                quantum_relevance=quantum_relevance,
                technology_relevance=technology_relevance,
                article_id=article_id
            )
            
            # Cache the article
            self.article_cache[url] = article
            
            return article
            
        except Exception as e:
            logger.warning(f"Failed to scrape article {url}: {e}")
            return None
    
    def _extract_with_selectors(self, soup: BeautifulSoup, selectors: str) -> str:
        """Extract text using multiple CSS selectors."""
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                if text:
                    return text
        return ""
    
    def _extract_authors_with_selectors(self, soup: BeautifulSoup, selectors: str) -> List[str]:
        """Extract authors using CSS selectors."""
        authors = []
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                if text and 'author' in text.lower():
                    # Clean up author text
                    author_text = re.sub(r'by\s+', '', text, flags=re.IGNORECASE)
                    authors.extend([a.strip() for a in author_text.split(',')])
        
        return authors if authors else ["Unknown Author"]
    
    def _extract_date_with_selectors(self, soup: BeautifulSoup, selectors: str) -> str:
        """Extract publication date using CSS selectors."""
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                # Look for date patterns
                date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4}', text)
                if date_match:
                    return date_match.group()
        
        return datetime.now().strftime("%Y-%m-%d")
    
    def _extract_tags_with_selectors(self, soup: BeautifulSoup, selectors: str) -> List[str]:
        """Extract tags using CSS selectors."""
        tags = []
        for selector in selectors.split(', '):
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text().strip()
                if text:
                    tags.append(text)
        
        return tags
    
    def _generate_summary(self, content: str) -> str:
        """Generate summary from content."""
        if not content:
            return ""
        
        # Take first 200 characters as summary
        summary = content[:200].strip()
        if len(content) > 200:
            summary += "..."
        
        return summary
    
    def _categorize_article(self, title: str, summary: str, content: str, tags: List[str]) -> Tuple[str, str]:
        """Categorize article by field and subfield."""
        text = f"{title} {summary} {' '.join(tags)}".lower()
        
        best_field = "general"
        best_subfield = "general"
        best_score = 0
        
        for field_name, field_config in self.research_fields.items():
            # Calculate field match score
            field_score = 0
            for keyword in field_config['keywords']:
                if keyword.lower() in text:
                    field_score += 1
            
            # Calculate quantum relevance score
            quantum_score = 0
            for keyword in field_config['quantum_keywords']:
                if keyword.lower() in text:
                    quantum_score += 2  # Higher weight for quantum keywords
            
            total_score = field_score + quantum_score
            
            if total_score > best_score:
                best_score = total_score
                best_field = field_name
                best_subfield = f"{field_name}_research"
        
        return best_field, best_subfield
    
    def _calculate_research_impact(self, title: str, summary: str, content: str) -> float:
        """Calculate research impact score (0-10)."""
        text = f"{title} {summary} {content}".lower()
        
        impact_score = 0
        
        # Keywords indicating high impact
        high_impact_keywords = [
            'breakthrough', 'discovery', 'first', 'novel', 'revolutionary',
            'groundbreaking', 'milestone', 'advance', 'innovation', 'new',
            'significant', 'important', 'major', 'key', 'critical'
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
    
    def _calculate_technology_relevance(self, title: str, summary: str, content: str, tags: List[str]) -> float:
        """Calculate technology relevance score (0-10)."""
        text = f"{title} {summary} {' '.join(tags)}".lower()
        
        tech_score = 0
        
        # Technology-specific keywords with weights
        tech_keywords = {
            'software': 2,
            'programming': 2,
            'algorithm': 3,
            'computer': 2,
            'technology': 2,
            'digital': 2,
            'electronic': 2,
            'automation': 3,
            'artificial_intelligence': 4,
            'machine_learning': 4,
            'neural_network': 3,
            'data_science': 3,
            'cloud_computing': 3,
            'blockchain': 3,
            'cybersecurity': 3,
            'internet_of_things': 3,
            'virtual_reality': 3,
            'augmented_reality': 3,
            'robotics': 3,
            'automation': 3
        }
        
        for keyword, weight in tech_keywords.items():
            if keyword.replace('_', ' ') in text:
                tech_score += weight
        
        # Normalize to 0-10 scale
        return min(tech_score, 10.0)
    
    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID."""
        content = f"{url}{title}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _remove_duplicates(self, articles: List[ResearchArticle]) -> List[ResearchArticle]:
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
    
    def analyze_cross_source_trends(self, articles: List[ResearchArticle]) -> Dict[str, Any]:
        """Analyze research trends across all sources."""
        logger.info("📊 Analyzing cross-source research trends")
        
        trends = {
            'total_articles': len(articles),
            'source_distribution': {},
            'field_distribution': {},
            'quantum_relevance_distribution': {},
            'technology_relevance_distribution': {},
            'research_impact_distribution': {},
            'top_keywords': {},
            'cross_source_breakthroughs': [],
            'quantum_focus_areas': {},
            'technology_focus_areas': {}
        }
        
        # Source distribution
        for article in articles:
            source = article.source
            trends['source_distribution'][source] = trends['source_distribution'].get(source, 0) + 1
        
        # Field distribution
        for article in articles:
            field = article.field
            trends['field_distribution'][field] = trends['field_distribution'].get(field, 0) + 1
        
        # Relevance distributions
        for article in articles:
            quantum_level = int(article.quantum_relevance)
            tech_level = int(article.technology_relevance)
            impact_level = int(article.research_impact)
            
            trends['quantum_relevance_distribution'][quantum_level] = trends['quantum_relevance_distribution'].get(quantum_level, 0) + 1
            trends['technology_relevance_distribution'][tech_level] = trends['technology_relevance_distribution'].get(tech_level, 0) + 1
            trends['research_impact_distribution'][impact_level] = trends['research_impact_distribution'].get(impact_level, 0) + 1
        
        # Top keywords
        all_text = " ".join([f"{a.title} {a.summary} {' '.join(a.tags)}" for a in articles]).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 25 keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:25]
        trends['top_keywords'] = dict(top_keywords)
        
        # Cross-source breakthroughs (high impact articles)
        breakthroughs = [a for a in articles if a.research_impact >= 7.0]
        trends['cross_source_breakthroughs'] = [
            {
                'title': a.title,
                'url': a.url,
                'source': a.source,
                'impact_score': a.research_impact,
                'quantum_relevance': a.quantum_relevance,
                'technology_relevance': a.technology_relevance
            }
            for a in breakthroughs[:15]
        ]
        
        # Focus areas
        quantum_articles = [a for a in articles if a.quantum_relevance >= 5.0]
        tech_articles = [a for a in articles if a.technology_relevance >= 5.0]
        
        for article in quantum_articles:
            subfield = article.subfield
            trends['quantum_focus_areas'][subfield] = trends['quantum_focus_areas'].get(subfield, 0) + 1
        
        for article in tech_articles:
            subfield = article.subfield
            trends['technology_focus_areas'][subfield] = trends['technology_focus_areas'].get(subfield, 0) + 1
        
        return trends
    
    def generate_comprehensive_report(self, articles: List[ResearchArticle], trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive multi-source research report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'scraping_summary': {
                'total_articles_scraped': len(articles),
                'scraping_date': datetime.now().strftime("%Y-%m-%d"),
                'sources': list(self.sources.keys())
            },
            'articles': [],
            'trends_analysis': trends,
            'quantum_research_highlights': [],
            'technology_research_highlights': [],
            'cross_source_insights': [],
            'recommendations': []
        }
        
        # Add articles
        for article in articles:
            report['articles'].append({
                'title': article.title,
                'url': article.url,
                'source': article.source,
                'field': article.field,
                'subfield': article.subfield,
                'publication_date': article.publication_date,
                'authors': article.authors,
                'summary': article.summary,
                'tags': article.tags,
                'research_impact': article.research_impact,
                'quantum_relevance': article.quantum_relevance,
                'technology_relevance': article.technology_relevance,
                'article_id': article.article_id
            })
        
        # Quantum research highlights
        quantum_articles = [a for a in articles if a.quantum_relevance >= 7.0]
        for article in quantum_articles[:10]:
            report['quantum_research_highlights'].append({
                'title': article.title,
                'url': article.url,
                'source': article.source,
                'quantum_relevance': article.quantum_relevance,
                'research_impact': article.research_impact,
                'field': article.field,
                'subfield': article.subfield
            })
        
        # Technology research highlights
        tech_articles = [a for a in articles if a.technology_relevance >= 7.0]
        for article in tech_articles[:10]:
            report['technology_research_highlights'].append({
                'title': article.title,
                'url': article.url,
                'source': article.source,
                'technology_relevance': article.technology_relevance,
                'research_impact': article.research_impact,
                'field': article.field,
                'subfield': article.subfield
            })
        
        # Cross-source insights
        report['cross_source_insights'] = [
            f"Total articles from {len(self.sources)} sources: {len(articles)}",
            f"Quantum-focused articles: {len(quantum_articles)}",
            f"Technology-focused articles: {len(tech_articles)}",
            f"High-impact research (≥7): {len([a for a in articles if a.research_impact >= 7.0])}",
            f"Cross-source breakthroughs: {len(trends['cross_source_breakthroughs'])}"
        ]
        
        # Generate recommendations
        report['recommendations'] = [
            "Focus on high quantum relevance articles for KOBA42 integration",
            "Monitor breakthrough research across all sources",
            "Track emerging quantum technologies and applications",
            "Analyze cross-source research trends for optimization opportunities",
            "Integrate quantum physics discoveries into optimization algorithms",
            "Stay updated on quantum internet and communication developments",
            "Follow quantum chemistry and materials science advances",
            "Monitor software and technology trends for quantum integration",
            "Track artificial intelligence and machine learning developments",
            "Analyze cross-disciplinary research for innovation opportunities"
        ]
        
        return report

def demonstrate_multi_source_scraping():
    """Demonstrate multi-source research scraping and analysis."""
    logger.info("🚀 KOBA42 Multi-Source Research Scraper")
    logger.info("=" * 50)
    
    # Initialize scraper
    scraper = MultiSourceResearchScraper()
    
    # Scrape articles from all sources
    print("\n🔍 Scraping articles from all sources...")
    articles = scraper.scrape_all_sources(max_articles_per_source=15)
    
    if not articles:
        print("❌ No articles scraped successfully")
        return
    
    print(f"✅ Successfully scraped {len(articles)} articles from all sources")
    
    # Analyze trends
    print("\n📊 Analyzing cross-source research trends...")
    trends = scraper.analyze_cross_source_trends(articles)
    
    # Generate report
    print("\n📄 Generating comprehensive research report...")
    report = scraper.generate_comprehensive_report(articles, trends)
    
    # Display summary
    print("\n📋 MULTI-SOURCE SCRAPING SUMMARY")
    print("=" * 50)
    print(f"Total Articles: {len(articles)}")
    print(f"Sources: {list(trends['source_distribution'].keys())}")
    print(f"Field Distribution: {trends['field_distribution']}")
    print(f"Quantum Relevance (≥7): {sum(1 for a in articles if a.quantum_relevance >= 7)}")
    print(f"Technology Relevance (≥7): {sum(1 for a in articles if a.technology_relevance >= 7)}")
    print(f"High Impact (≥7): {sum(1 for a in articles if a.research_impact >= 7)}")
    
    # Display top quantum articles
    print("\n🔬 TOP QUANTUM RESEARCH ARTICLES")
    print("=" * 50)
    quantum_articles = [a for a in articles if a.quantum_relevance >= 5.0]
    quantum_articles.sort(key=lambda x: x.quantum_relevance, reverse=True)
    
    for i, article in enumerate(quantum_articles[:5], 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Field: {article.field} ({article.subfield})")
        print(f"   Quantum Relevance: {article.quantum_relevance:.1f}/10")
        print(f"   Research Impact: {article.research_impact:.1f}/10")
        print(f"   URL: {article.url}")
    
    # Display top technology articles
    print("\n💻 TOP TECHNOLOGY RESEARCH ARTICLES")
    print("=" * 50)
    tech_articles = [a for a in articles if a.technology_relevance >= 5.0]
    tech_articles.sort(key=lambda x: x.technology_relevance, reverse=True)
    
    for i, article in enumerate(tech_articles[:5], 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Field: {article.field} ({article.subfield})")
        print(f"   Technology Relevance: {article.technology_relevance:.1f}/10")
        print(f"   Research Impact: {article.research_impact:.1f}/10")
        print(f"   URL: {article.url}")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'multi_source_research_report_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Comprehensive research report saved to {report_file}")
    
    return articles, trends, report_file

if __name__ == "__main__":
    # Run multi-source scraping demonstration
    articles, trends, report_file = demonstrate_multi_source_scraping()
    
    print(f"\n🎉 Multi-source research scraping demonstration completed!")
    print(f"📊 Report saved to: {report_file}")
    print(f"🔬 Scraped {len(articles)} articles from multiple sources")
    print(f"📈 Analyzed cross-source research trends and relevance")
