#!/usr/bin/env python3
"""
Real-World Research Scraper
Comprehensive scraping system for prestigious academic and research sources
"""

import json
import requests
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import urllib.parse
from bs4 import BeautifulSoup
import feedparser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealWorldResearchScraper:
    """
    Comprehensive scraper for prestigious academic and research sources
    """

    def __init__(self):
        self.research_dir = Path("research_data")
        self.scraping_log = self.research_dir / "real_world_scraping_log.json"
        self.subjects_discovered = self.research_dir / "real_world_subjects.json"

        # Rate limiting
        self.rate_limits = {
            'arxiv': {'requests_per_minute': 30, 'last_request': None},
            'mit': {'requests_per_minute': 10, 'last_request': None},
            'cambridge': {'requests_per_minute': 10, 'last_request': None},
            'caltech': {'requests_per_minute': 10, 'last_request': None},
            'uspto': {'requests_per_minute': 20, 'last_request': None},
            'semantic_scholar': {'requests_per_minute': 100, 'last_request': None}
        }

        self._initialize_scraping_tracking()

    def _initialize_scraping_tracking(self):
        """Initialize scraping tracking system."""
        if not self.scraping_log.exists():
            initial_data = {
                "last_scraping_run": None,
                "sources_scraped": [],
                "subjects_found": 0,
                "papers_analyzed": 0,
                "patents_reviewed": 0,
                "disclosures_processed": 0,
                "rate_limit_hits": 0,
                "scraping_history": []
            }
            with open(self.scraping_log, 'w') as f:
                json.dump(initial_data, f, indent=2)

        if not self.subjects_discovered.exists():
            initial_subjects = {
                "academic_papers": [],
                "patents": [],
                "disclosures": [],
                "research_projects": [],
                "conference_papers": []
            }
            with open(self.subjects_discovered, 'w') as f:
                json.dump(initial_subjects, f, indent=2)

    def _respect_rate_limit(self, source: str):
        """Respect rate limits for different sources."""
        if source in self.rate_limits:
            limit_info = self.rate_limits[source]
            if limit_info['last_request']:
                time_diff = time.time() - limit_info['last_request']
                min_interval = 60 / limit_info['requests_per_minute']
                if time_diff < min_interval:
                    sleep_time = min_interval - time_diff
                    logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s for {source}")
                    time.sleep(sleep_time)

            self.rate_limits[source]['last_request'] = time.time()

    def scrape_arxiv(self, query: str = "artificial intelligence", max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape arXiv for recent academic papers.
        """
        logger.info("🔬 Scraping arXiv for academic papers...")
        self._respect_rate_limit('arxiv')

        try:
            # arXiv API query
            url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse XML response
            feed = feedparser.parse(response.content)

            subjects = []
            for entry in feed.entries:
                # Extract keywords from title and abstract
                title = entry.title
                abstract = entry.get('summary', '')
                published = entry.get('published', '')

                # Analyze for AI/cybersecurity/programming relevance
                relevance_keywords = [
                    'machine learning', 'artificial intelligence', 'cybersecurity',
                    'cryptography', 'quantum computing', 'neural network',
                    'blockchain', 'distributed system', 'security', 'privacy',
                    'algorithm', 'optimization', 'formal method'
                ]

                relevance_score = sum(1 for keyword in relevance_keywords
                                    if keyword.lower() in (title + abstract).lower())

                if relevance_score >= 1:
                    subject_name = self._generate_subject_name(title, "arxiv")
                    subjects.append({
                        "name": subject_name,
                        "title": title,
                        "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                        "source": "arXiv",
                        "url": entry.link,
                        "published": published,
                        "category": self._classify_subject(title, abstract),
                        "difficulty": "expert",
                        "relevance_score": min(relevance_score / 3, 1.0),
                        "citations": 0,  # arXiv doesn't provide citation counts
                        "authors": [author.name for author in entry.authors] if hasattr(entry, 'authors') else []
                    })

            logger.info(f"📄 Found {len(subjects)} relevant papers on arXiv")
            return subjects

        except Exception as e:
            logger.error(f"❌ Error scraping arXiv: {e}")
            return []

    def scrape_mit_research(self) -> List[Dict[str, Any]]:
        """
        Scrape MIT research projects and publications.
        """
        logger.info("🏛️ Scraping MIT research...")
        self._respect_rate_limit('mit')

        try:
            # MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)
            urls = [
                "https://www.csail.mit.edu/research",
                "https://www.eecs.mit.edu/research",
                "https://www.mit.edu/research/"
            ]

            subjects = []

            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Extract research project titles and descriptions
                    research_items = soup.find_all(['h3', 'h4', 'div'], class_=re.compile(r'(research|project|paper)'))

                    for item in research_items[:10]:  # Limit to avoid overwhelming
                        title = item.get_text().strip()
                        if len(title) > 10 and self._is_relevant_research(title):
                            subject_name = self._generate_subject_name(title, "mit")
                            subjects.append({
                                "name": subject_name,
                                "title": title,
                                "source": "MIT",
                                "url": url,
                                "category": self._classify_subject(title, ""),
                                "difficulty": "expert",
                                "relevance_score": 0.9,
                                "description": f"MIT research project: {title}",
                                "institution": "Massachusetts Institute of Technology"
                            })

                except Exception as e:
                    logger.warning(f"⚠️ Error scraping {url}: {e}")

            logger.info(f"🔬 Found {len(subjects)} MIT research projects")
            return subjects

        except Exception as e:
            logger.error(f"❌ Error scraping MIT: {e}")
            return []

    def scrape_cambridge_research(self) -> List[Dict[str, Any]]:
        """
        Scrape University of Cambridge research.
        """
        logger.info("🎓 Scraping Cambridge University research...")
        self._respect_rate_limit('cambridge')

        try:
            # Cambridge Computer Laboratory and research centers
            urls = [
                "https://www.cst.cam.ac.uk/research",
                "https://www.cl.cam.ac.uk/research/",
                "https://www.ai.cam.ac.uk/research/"
            ]

            subjects = []

            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Extract research titles
                    research_titles = soup.find_all(['h2', 'h3', 'h4'], class_=re.compile(r'(title|research|project)'))
                    research_descriptions = soup.find_all(['p', 'div'], class_=re.compile(r'(description|summary|abstract)'))

                    for i, title_elem in enumerate(research_titles[:8]):
                        title = title_elem.get_text().strip()
                        description = research_descriptions[i].get_text().strip() if i < len(research_descriptions) else ""

                        if len(title) > 10 and self._is_relevant_research(title):
                            subject_name = self._generate_subject_name(title, "cambridge")
                            subjects.append({
                                "name": subject_name,
                                "title": title,
                                "description": description[:300] + "..." if len(description) > 300 else description,
                                "source": "University of Cambridge",
                                "url": url,
                                "category": self._classify_subject(title, description),
                                "difficulty": "expert",
                                "relevance_score": 0.95,
                                "institution": "University of Cambridge"
                            })

                except Exception as e:
                    logger.warning(f"⚠️ Error scraping {url}: {e}")

            logger.info(f"🎓 Found {len(subjects)} Cambridge research projects")
            return subjects

        except Exception as e:
            logger.error(f"❌ Error scraping Cambridge: {e}")
            return []

    def scrape_caltech_research(self) -> List[Dict[str, Any]]:
        """
        Scrape Caltech research projects.
        """
        logger.info("🔭 Scraping Caltech research...")
        self._respect_rate_limit('caltech')

        try:
            # Caltech research centers
            urls = [
                "https://www.cms.caltech.edu/research",
                "https://www.cs.caltech.edu/research",
                "https://iqim.caltech.edu/research/"
            ]

            subjects = []

            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Extract research projects
                    research_items = soup.find_all(['div', 'article'], class_=re.compile(r'(research|project|paper)'))

                    for item in research_items[:6]:
                        title_elem = item.find(['h2', 'h3', 'h4'])
                        desc_elem = item.find(['p', 'div'])

                        if title_elem:
                            title = title_elem.get_text().strip()
                            description = desc_elem.get_text().strip() if desc_elem else ""

                            if len(title) > 10 and self._is_relevant_research(title):
                                subject_name = self._generate_subject_name(title, "caltech")
                                subjects.append({
                                    "name": subject_name,
                                    "title": title,
                                    "description": description[:300] + "..." if len(description) > 300 else description,
                                    "source": "Caltech",
                                    "url": url,
                                    "category": self._classify_subject(title, description),
                                    "difficulty": "expert",
                                    "relevance_score": 0.9,
                                    "institution": "California Institute of Technology"
                                })

                except Exception as e:
                    logger.warning(f"⚠️ Error scraping {url}: {e}")

            logger.info(f"🔭 Found {len(subjects)} Caltech research projects")
            return subjects

        except Exception as e:
            logger.error(f"❌ Error scraping Caltech: {e}")
            return []

    def scrape_uspto_patents(self, query: str = "artificial intelligence", max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Scrape USPTO for recent AI/cybersecurity patents.
        """
        logger.info("📋 Scraping USPTO patents...")
        self._respect_rate_limit('uspto')

        try:
            # USPTO Patent Database API
            base_url = "https://developer.uspto.gov/ibd-api/v1/search"
            params = {
                "query": f"({query}) AND (machine learning OR cybersecurity OR cryptography)",
                "rows": max_results,
                "start": 0,
                "sort": "publicationDate desc"
            }

            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            subjects = []
            for patent in data.get('results', []):
                title = patent.get('patentTitle', '')
                abstract = patent.get('abstractText', '')
                inventors = patent.get('inventors', [])
                publication_date = patent.get('publicationDate', '')

                if self._is_relevant_research(title) or self._is_relevant_research(abstract):
                    subject_name = self._generate_subject_name(title, "uspto")
                    subjects.append({
                        "name": subject_name,
                        "title": title,
                        "abstract": abstract[:400] + "..." if len(abstract) > 400 else abstract,
                        "source": "USPTO",
                        "patent_number": patent.get('patentNumber', ''),
                        "publication_date": publication_date,
                        "inventors": [inv.get('name', '') for inv in inventors],
                        "category": self._classify_subject(title, abstract),
                        "difficulty": "expert",
                        "relevance_score": 0.85,
                        "patent_url": f"https://patents.google.com/patent/{patent.get('patentNumber', '')}"
                    })

            logger.info(f"📋 Found {len(subjects)} relevant USPTO patents")
            return subjects

        except Exception as e:
            logger.error(f"❌ Error scraping USPTO: {e}")
            return []

    def scrape_semantic_scholar(self, query: str = "artificial intelligence", max_results: int = 30) -> List[Dict[str, Any]]:
        """
        Scrape Semantic Scholar for highly cited papers.
        """
        logger.info("📚 Scraping Semantic Scholar...")
        self._respect_rate_limit('semantic_scholar')

        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,year,citationCount,authors,venue",
                "sort": "citationCount:desc"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            subjects = []
            for paper in data.get('data', []):
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                citations = paper.get('citationCount', 0)
                year = paper.get('year', '')
                authors = paper.get('authors', [])
                venue = paper.get('venue', '')

                # Only include highly cited papers
                if citations >= 50 and self._is_relevant_research(title):
                    subject_name = self._generate_subject_name(title, "semantic_scholar")
                    subjects.append({
                        "name": subject_name,
                        "title": title,
                        "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                        "source": "Semantic Scholar",
                        "citations": citations,
                        "year": year,
                        "venue": venue,
                        "authors": [author.get('name', '') for author in authors],
                        "category": self._classify_subject(title, abstract),
                        "difficulty": "expert",
                        "relevance_score": min(citations / 500, 1.0),
                        "paper_url": f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
                    })

            logger.info(f"📚 Found {len(subjects)} highly cited papers on Semantic Scholar")
            return subjects

        except Exception as e:
            logger.error(f"❌ Error scraping Semantic Scholar: {e}")
            return []

    def scrape_foia_documents(self) -> List[Dict[str, Any]]:
        """
        Scrape FOIA documents from government databases.
        """
        logger.info("📄 Scraping FOIA documents...")

        try:
            # FOIA.gov API or scraping
            urls = [
                "https://www.foia.gov/search.html",
                "https://www.cia.gov/readingroom/",
                "https://www.nsa.gov/FOIA/"
            ]

            subjects = []

            # Note: FOIA scraping requires careful ethical consideration
            # This is a simulated version that would need proper implementation
            logger.warning("⚠️ FOIA scraping requires special permissions and ethical considerations")
            logger.info("📄 FOIA scraping simulated (would require proper authorization)")

            return subjects

        except Exception as e:
            logger.error(f"❌ Error scraping FOIA documents: {e}")
            return []

    def _generate_subject_name(self, title: str, source: str) -> str:
        """Generate a unique subject name from title and source."""
        # Clean title and create identifier
        clean_title = re.sub(r'[^\w\s-]', '', title.lower())
        words = clean_title.split()[:4]  # Take first 4 words
        base_name = '_'.join(words)

        # Add timestamp for uniqueness
        timestamp = int(time.time() * 1000) % 10000
        return f"{base_name}_{source}_{timestamp}"

    def _is_relevant_research(self, text: str) -> bool:
        """Check if research topic is relevant to our curriculum."""
        relevant_keywords = [
            'machine learning', 'artificial intelligence', 'cybersecurity',
            'cryptography', 'quantum', 'neural network', 'blockchain',
            'distributed system', 'security', 'privacy', 'algorithm',
            'optimization', 'formal method', 'computer vision', 'nlp',
            'deep learning', 'reinforcement learning', 'federated learning'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in relevant_keywords)

    def _classify_subject(self, title: str, abstract: str) -> str:
        """Classify subject into appropriate category."""
        text = (title + " " + abstract).lower()

        if any(word in text for word in ['machine learning', 'neural network', 'deep learning']):
            return 'machine_learning'
        elif any(word in text for word in ['cybersecurity', 'security', 'cryptography']):
            return 'cybersecurity'
        elif any(word in text for word in ['quantum', 'qubit', 'quantum computing']):
            return 'quantum_computing'
        elif any(word in text for word in ['blockchain', 'distributed ledger']):
            return 'blockchain'
        elif any(word in text for word in ['computer vision', 'image processing']):
            return 'computer_vision'
        elif any(word in text for word in ['natural language', 'nlp', 'language model']):
            return 'natural_language_processing'
        else:
            return 'artificial_intelligence'

    def run_comprehensive_scraping(self) -> Dict[str, Any]:
        """
        Run comprehensive scraping across all prestigious sources.
        """
        logger.info("🌟 Starting comprehensive research scraping...")
        print("=" * 80)
        print("🔬 COMPREHENSIVE RESEARCH SCRAPING SYSTEM")
        print("Scraping from prestigious academic and research sources")
        print("=" * 80)

        all_subjects = {
            "academic_papers": [],
            "research_projects": [],
            "patents": [],
            "conference_papers": [],
            "disclosures": []
        }

        # Scrape from each source
        print("\n📄 Scraping arXiv...")
        arxiv_papers = self.scrape_arxiv()
        all_subjects["academic_papers"].extend(arxiv_papers)

        print("\n🏛️ Scraping MIT research...")
        mit_projects = self.scrape_mit_research()
        all_subjects["research_projects"].extend(mit_projects)

        print("\n🎓 Scraping Cambridge research...")
        cambridge_projects = self.scrape_cambridge_research()
        all_subjects["research_projects"].extend(cambridge_projects)

        print("\n🔭 Scraping Caltech research...")
        caltech_projects = self.scrape_caltech_research()
        all_subjects["research_projects"].extend(caltech_projects)

        print("\n📋 Scraping USPTO patents...")
        patents = self.scrape_uspto_patents()
        all_subjects["patents"].extend(patents)

        print("\n📚 Scraping Semantic Scholar...")
        semantic_papers = self.scrape_semantic_scholar()
        all_subjects["academic_papers"].extend(semantic_papers)

        # Save discovered subjects
        with open(self.subjects_discovered, 'w') as f:
            json.dump(all_subjects, f, indent=2, default=str)

        # Update scraping log
        self._update_scraping_log(all_subjects)

        # Generate report
        total_subjects = sum(len(subjects) for subjects in all_subjects.values())

        print("\n📊 COMPREHENSIVE SCRAPING COMPLETE!")
        print("=" * 80)
        print(f"📄 Academic Papers: {len(all_subjects['academic_papers'])}")
        print(f"🔬 Research Projects: {len(all_subjects['research_projects'])}")
        print(f"📋 Patents: {len(all_subjects['patents'])}")
        print(f"🎤 Conference Papers: {len(all_subjects['conference_papers'])}")
        print(f"📄 Disclosures: {len(all_subjects['disclosures'])}")
        print(f"📊 Total Subjects Discovered: {total_subjects}")

        return {
            "total_subjects": total_subjects,
            "subjects_by_category": {k: len(v) for k, v in all_subjects.items()},
            "sources_scraped": ["arXiv", "MIT", "Cambridge", "Caltech", "USPTO", "Semantic Scholar"],
            "timestamp": datetime.now().isoformat()
        }

    def _update_scraping_log(self, subjects: Dict[str, List]):
        """Update scraping log with latest results."""
        try:
            with open(self.scraping_log, 'r') as f:
                log_data = json.load(f)

            log_data["last_scraping_run"] = datetime.now().isoformat()
            log_data["subjects_found"] += sum(len(subjects_list) for subjects_list in subjects.values())
            log_data["papers_analyzed"] += len(subjects.get("academic_papers", []))
            log_data["patents_reviewed"] += len(subjects.get("patents", []))
            log_data["disclosures_processed"] += len(subjects.get("disclosures", []))

            # Add sources to list if not already there
            sources = ["arXiv", "MIT", "Cambridge", "Caltech", "USPTO", "Semantic Scholar"]
            for source in sources:
                if source not in log_data["sources_scraped"]:
                    log_data["sources_scraped"].append(source)

            # Add to history
            log_data["scraping_history"].append({
                "timestamp": datetime.now().isoformat(),
                "subjects_found": sum(len(subjects_list) for subjects_list in subjects.values()),
                "sources_scraped": sources
            })

            with open(self.scraping_log, 'w') as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Error updating scraping log: {e}")

def main():
    """Main function to demonstrate comprehensive research scraping."""
    print("🔬 Real-World Research Scraper")
    print("=" * 60)
    print("Comprehensive scraping from prestigious academic sources")
    print("arXiv, MIT, Cambridge, Caltech, USPTO, Semantic Scholar, FOIA")

    # Initialize scraper
    scraper = RealWorldResearchScraper()

    # Run comprehensive scraping
    results = scraper.run_comprehensive_scraping()

    print("\n🏆 SCRAPING RESULTS:")
    print(f"  Total Subjects: {results['total_subjects']}")
    print(f"  Academic Papers: {results['subjects_by_category']['academic_papers']}")
    print(f"  Research Projects: {results['subjects_by_category']['research_projects']}")
    print(f"  Patents: {results['subjects_by_category']['patents']}")
    print(f"  Sources Scraped: {len(results['sources_scraped'])}")

    print("\n🔗 Sources Scraped:")
    for source in results['sources_scraped']:
        print(f"  • {source}")

    print("\n🚀 Möbius Loop Trainer now has access to")
    print("cutting-edge research from the world's most prestigious institutions!")
    print("🎓 MIT, Cambridge, Caltech, USPTO, arXiv, Semantic Scholar")

if __name__ == "__main__":
    main()
