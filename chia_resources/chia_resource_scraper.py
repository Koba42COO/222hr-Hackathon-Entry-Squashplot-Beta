#!/usr/bin/env python3
"""
Chia Resources Scraper and Database
Comprehensive scraper for Chia Network documentation, APIs, and resources

Scrapes and stores information from:
- Official Chia documentation
- API references and documentation
- GitHub repositories
- Developer resources
- Community resources

Stores data in structured JSON format for easy querying and integration.
"""

import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from pathlib import Path

class ChiaResourceScraper:
    """
    Comprehensive Chia resource scraper and database
    """

    def __init__(self):
        self.base_dir = Path("/Users/coo-koba42/dev/SquashPlot_Complete_Package/chia_resources")
        self.database_file = self.base_dir / "chia_resources_database.json"
        self.resources_index = self.base_dir / "resources_index.json"

        # Chia resource URLs
        self.chia_resources = {
            "docs": {
                "chia_main_docs": "https://docs.chia.net/",
                "chialisp_docs": "https://chialisp.com/",
                "dev_guides": "https://docs.chia.net/dev-guides-home/",
                "rpc_reference": "https://docs.chia.net/reference-client/rpc-reference/rpc/",
                "cli_reference": "https://docs.chia.net/reference-client/cli-reference/cli/"
            },
            "apis": {
                "spacescan_api": "https://docs.spacescan.io/api/address/xch_balance/",
                "mintgarden_api": "https://api.mintgarden.io/docs",
                "dexie_api": "https://dexie.space/api",
                "xch_network_dev": "https://xch.network/developers/"
            },
            "github": {
                "chia_network_org": "https://github.com/chia-network"
            },
            "community": {
                "discord": "https://discord.com/invite/chia",
                "spacescan": "https://www.spacescan.io/"
            }
        }

        # Database structure
        self.database = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_resources": 0,
                "categories": list(self.chia_resources.keys())
            },
            "resources": {},
            "apis": {},
            "github_repos": {},
            "documentation": {},
            "community": {},
            "errors": []
        }

        # Load existing database if it exists
        self.load_database()

    def load_database(self):
        """Load existing database if available"""
        if self.database_file.exists():
            try:
                with open(self.database_file, 'r', encoding='utf-8') as f:
                    self.database = json.load(f)
                print(f"✅ Loaded existing database with {self.database['metadata']['total_resources']} resources")
            except Exception as e:
                print(f"⚠️ Could not load existing database: {e}")

    def save_database(self):
        """Save database to file"""
        self.database["metadata"]["last_updated"] = datetime.now().isoformat()

        with open(self.database_file, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)

        with open(self.resources_index, 'w', encoding='utf-8') as f:
            # Create a simplified index for quick lookup
            index = {
                "categories": list(self.database.keys()),
                "total_resources": self.database["metadata"]["total_resources"],
                "last_updated": self.database["metadata"]["last_updated"],
                "resource_summary": self._create_resource_summary()
            }
            json.dump(index, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved database with {self.database['metadata']['total_resources']} resources")

    def scrape_all_resources(self):
        """Scrape all Chia resources"""
        print("🚀 Starting comprehensive Chia resource scraping...")
        print("=" * 60)

        total_scraped = 0

        for category, resources in self.chia_resources.items():
            print(f"\n📂 Scraping {category.upper()} resources...")
            print("-" * 40)

            for name, url in resources.items():
                print(f"   🔍 Scraping {name}...")
                try:
                    if category == "docs":
                        data = self.scrape_documentation(url, name)
                    elif category == "apis":
                        data = self.scrape_api_documentation(url, name)
                    elif category == "github":
                        data = self.scrape_github_org(url, name)
                    elif category == "community":
                        data = self.scrape_community_resource(url, name)
                    else:
                        data = self.scrape_generic_resource(url, name)

                    if data:
                        self.store_resource(category, name, data)
                        total_scraped += 1
                        print(f"      ✅ Successfully scraped {name}")
                    else:
                        print(f"      ⚠️ No data extracted from {name}")

                except Exception as e:
                    error_info = {
                        "resource": name,
                        "url": url,
                        "category": category,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.database["errors"].append(error_info)
                    print(f"      ❌ Error scraping {name}: {e}")

                # Rate limiting
                time.sleep(1)

        self.database["metadata"]["total_resources"] = total_scraped
        self.save_database()

        print("\n✅ Chia resource scraping complete!")
        print(f"   Total resources scraped: {total_scraped}")
        print(f"   Categories processed: {len(self.chia_resources)}")
        print(f"   Database saved to: {self.database_file}")

    def scrape_documentation(self, url: str, name: str) -> Dict[str, Any]:
        """Scrape documentation sites"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract documentation structure
            data = {
                "url": url,
                "type": "documentation",
                "title": self._extract_title(soup),
                "description": self._extract_description(soup),
                "sections": self._extract_sections(soup),
                "navigation": self._extract_navigation(soup),
                "code_examples": self._extract_code_examples(soup),
                "api_endpoints": self._extract_api_endpoints(soup),
                "last_updated": datetime.now().isoformat(),
                "scraped_content": {
                    "headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
                    "paragraphs": [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()],
                    "links": [{"text": a.get_text().strip(), "href": a.get('href')} for a in soup.find_all('a', href=True)]
                }
            }

            return data

        except Exception as e:
            print(f"      Error scraping documentation {url}: {e}")
            return None

    def scrape_api_documentation(self, url: str, name: str) -> Dict[str, Any]:
        """Scrape API documentation"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            data = {
                "url": url,
                "type": "api_documentation",
                "title": self._extract_title(soup),
                "description": self._extract_description(soup),
                "endpoints": self._extract_api_endpoints(soup),
                "parameters": self._extract_api_parameters(soup),
                "examples": self._extract_api_examples(soup),
                "authentication": self._extract_authentication_info(soup),
                "rate_limits": self._extract_rate_limits(soup),
                "last_updated": datetime.now().isoformat()
            }

            return data

        except Exception as e:
            print(f"      Error scraping API docs {url}: {e}")
            return None

    def scrape_github_org(self, url: str, name: str) -> Dict[str, Any]:
        """Scrape GitHub organization"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            data = {
                "url": url,
                "type": "github_organization",
                "name": "chia-network",
                "description": self._extract_github_description(soup),
                "repositories": self._extract_github_repos(soup),
                "members": self._extract_github_members(soup),
                "stats": self._extract_github_stats(soup),
                "last_updated": datetime.now().isoformat()
            }

            return data

        except Exception as e:
            print(f"      Error scraping GitHub {url}: {e}")
            return None

    def scrape_community_resource(self, url: str, name: str) -> Dict[str, Any]:
        """Scrape community resources like Discord, forums"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            data = {
                "url": url,
                "type": "community_resource",
                "title": self._extract_title(soup),
                "description": self._extract_description(soup),
                "features": self._extract_features(soup),
                "links": self._extract_external_links(soup),
                "last_updated": datetime.now().isoformat()
            }

            return data

        except Exception as e:
            print(f"      Error scraping community resource {url}: {e}")
            return None

    def scrape_generic_resource(self, url: str, name: str) -> Dict[str, Any]:
        """Generic resource scraper"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            data = {
                "url": url,
                "type": "generic_resource",
                "title": self._extract_title(soup),
                "description": self._extract_description(soup),
                "content": self._extract_main_content(soup),
                "last_updated": datetime.now().isoformat()
            }

            return data

        except Exception as e:
            print(f"      Error scraping resource {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()

        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()

        return "Unknown Title"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description"""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()

        # Try first paragraph
        first_p = soup.find('p')
        if first_p:
            return first_p.get_text().strip()[:200]

        return "No description available"

    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract documentation sections"""
        sections = []

        for header in soup.find_all(['h1', 'h2', 'h3']):
            section = {
                "level": int(header.name[1]),  # h1 -> 1, h2 -> 2, etc.
                "title": header.get_text().strip(),
                "id": header.get('id', ''),
                "content": self._extract_section_content(header)
            }
            sections.append(section)

        return sections

    def _extract_section_content(self, header) -> str:
        """Extract content following a header"""
        content = []
        current = header.find_next_sibling()

        while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if current.name in ['p', 'ul', 'ol', 'pre', 'code']:
                content.append(current.get_text().strip())
            current = current.find_next_sibling()

        return ' '.join(content)

    def _extract_navigation(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract navigation links"""
        nav_links = []

        # Look for navigation elements
        nav_elements = soup.find_all(['nav', 'ul'], class_=re.compile(r'nav|menu|sidebar'))

        for nav in nav_elements:
            for link in nav.find_all('a', href=True):
                nav_links.append({
                    "text": link.get_text().strip(),
                    "href": link['href'],
                    "full_url": urljoin("https://docs.chia.net", link['href'])
                })

        return nav_links

    def _extract_code_examples(self, soup: BeautifulSoup) -> List[str]:
        """Extract code examples"""
        code_examples = []

        for code_block in soup.find_all(['pre', 'code']):
            code_text = code_block.get_text().strip()
            if code_text and len(code_text) > 10:  # Filter out short code snippets
                code_examples.append(code_text)

        return code_examples

    def _extract_api_endpoints(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract API endpoints"""
        endpoints = []

        # Look for endpoint patterns
        endpoint_patterns = [
            r'/api/[^\'"\s]+',
            r'/v\d+/[^\'"\s]+',
            r'https?://[^\'"\s]+/api/[^\'"\s]+'
        ]

        text_content = soup.get_text()
        found_endpoints = set()

        for pattern in endpoint_patterns:
            matches = re.findall(pattern, text_content)
            found_endpoints.update(matches)

        for endpoint in found_endpoints:
            endpoints.append({
                "url": endpoint,
                "method": "GET",  # Default assumption
                "description": f"API endpoint: {endpoint}"
            })

        return endpoints

    def _extract_api_parameters(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract API parameters"""
        parameters = []

        # Look for parameter tables or lists
        param_tables = soup.find_all('table')
        for table in param_tables:
            headers = [th.get_text().strip().lower() for th in table.find_all('th')]
            if any('parameter' in h or 'name' in h for h in headers):
                for row in table.find_all('tr')[1:]:  # Skip header
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        param = {
                            "name": cells[0].get_text().strip(),
                            "type": cells[1].get_text().strip() if len(cells) > 1 else "string",
                            "required": "required" in cells[0].get_text().lower() if len(cells) > 2 else False,
                            "description": cells[2].get_text().strip() if len(cells) > 2 else ""
                        }
                        parameters.append(param)

        return parameters

    def _extract_api_examples(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract API examples"""
        examples = []

        # Look for example code blocks
        for pre in soup.find_all('pre'):
            code = pre.get_text().strip()
            if any(keyword in code.lower() for keyword in ['curl', 'http', 'api', 'request']):
                examples.append({
                    "type": "http_request",
                    "content": code,
                    "language": "bash" if "curl" in code else "json"
                })

        return examples

    def _extract_authentication_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract authentication information"""
        auth_info = {
            "methods": [],
            "requirements": [],
            "examples": []
        }

        # Look for authentication-related content
        auth_keywords = ['auth', 'authentication', 'token', 'api key', 'bearer']

        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p']):
            text = element.get_text().lower()
            if any(keyword in text for keyword in auth_keywords):
                if element.name in ['h1', 'h2', 'h3', 'h4']:
                    auth_info["methods"].append(element.get_text().strip())
                else:
                    auth_info["requirements"].append(element.get_text().strip())

        return auth_info

    def _extract_rate_limits(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract rate limiting information"""
        rate_limits = {
            "limits": [],
            "policies": []
        }

        # Look for rate limit content
        rate_keywords = ['rate limit', 'throttle', 'quota', 'limit']

        for element in soup.find_all(['p', 'li', 'td']):
            text = element.get_text().lower()
            if any(keyword in text for keyword in rate_keywords):
                rate_limits["policies"].append(element.get_text().strip())

        return rate_limits

    def _extract_github_description(self, soup: BeautifulSoup) -> str:
        """Extract GitHub organization description"""
        desc_element = soup.find('div', class_=re.compile(r'description|bio'))
        if desc_element:
            return desc_element.get_text().strip()

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '')

        return "Chia Network GitHub Organization"

    def _extract_github_repos(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract GitHub repositories"""
        repos = []

        # Look for repository listings
        repo_elements = soup.find_all('a', href=re.compile(r'/chia-network/[^/]+/?$'))

        for repo_link in repo_elements:
            href = repo_link['href']
            if '/chia-network/' in href and href.count('/') <= 3:  # Repository link
                repo_name = href.split('/')[-1]
                repos.append({
                    "name": repo_name,
                    "url": f"https://github.com{href}",
                    "description": repo_link.find_next('p', class_=re.compile(r'description|summary'))
                })

        return repos

    def _extract_github_members(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract GitHub organization members"""
        members = []

        # Look for member listings
        member_elements = soup.find_all('a', href=re.compile(r'/[^/]+/?$'))

        for member_link in member_elements:
            href = member_link['href']
            if href and not href.startswith('/') and href != '/':
                continue

            username = href.strip('/')
            if username and username != 'chia-network':
                members.append({
                    "username": username,
                    "profile_url": f"https://github.com{href}"
                })

        return members[:20]  # Limit to first 20 members

    def _extract_github_stats(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract GitHub organization statistics"""
        stats = {
            "repositories": 0,
            "members": 0,
            "location": "Worldwide",
            "website": "https://chia.net"
        }

        # Try to extract stats from the page
        stat_elements = soup.find_all('span', class_=re.compile(r'Counter|count'))

        for stat in stat_elements:
            text = stat.get_text().strip()
            if text.isdigit():
                if 'repositories' in str(stat.parent).lower() or 'repos' in str(stat.parent).lower():
                    stats["repositories"] = int(text)
                elif 'members' in str(stat.parent).lower() or 'people' in str(stat.parent).lower():
                    stats["members"] = int(text)

        return stats

    def _extract_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract features from community resources"""
        features = []

        # Look for feature lists
        feature_elements = soup.find_all(['li', 'p'], class_=re.compile(r'feature|benefit'))

        for element in feature_elements:
            text = element.get_text().strip()
            if len(text) > 20 and len(text) < 200:  # Reasonable length
                features.append(text)

        return features

    def _extract_external_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract external links"""
        links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') and 'chia' not in href.lower():
                links.append({
                    "text": link.get_text().strip(),
                    "url": href,
                    "type": "external_link"
                })

        return links[:10]  # Limit to 10 links

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from generic pages"""
        # Try to find main content area
        main_content = soup.find(['main', 'article', 'div'], class_=re.compile(r'content|main'))

        if main_content:
            return main_content.get_text().strip()

        # Fallback to body text
        body = soup.find('body')
        if body:
            return body.get_text().strip()

        return "No content extracted"

    def store_resource(self, category: str, name: str, data: Dict[str, Any]):
        """Store resource data in database"""
        if category not in self.database:
            self.database[category] = {}

        self.database[category][name] = data

    def _create_resource_summary(self) -> Dict[str, Any]:
        """Create a summary of all resources"""
        summary = {}

        for category in self.chia_resources.keys():
            if category in self.database:
                summary[category] = {
                    "count": len(self.database[category]),
                    "resources": list(self.database[category].keys())
                }

        return summary

    def query_resources(self, category: str = None, resource_type: str = None,
                       search_term: str = None) -> List[Dict[str, Any]]:
        """Query resources from database"""
        results = []

        categories_to_search = [category] if category else list(self.chia_resources.keys())

        for cat in categories_to_search:
            if cat in self.database:
                for name, resource in self.database[cat].items():
                    if resource_type and resource.get('type') != resource_type:
                        continue

                    if search_term:
                        searchable_text = json.dumps(resource).lower()
                        if search_term.lower() not in searchable_text:
                            continue

                    results.append({
                        "category": cat,
                        "name": name,
                        "resource": resource
                    })

        return results

    def generate_resource_report(self) -> str:
        """Generate a comprehensive report of all resources"""
        report = []
        report.append("🌱 CHIA NETWORK RESOURCES DATABASE REPORT")
        report.append("=" * 60)
        report.append("")

        report.append(f"📊 Database Overview:")
        report.append(f"   Created: {self.database['metadata']['created']}")
        report.append(f"   Last Updated: {self.database['metadata']['last_updated']}")
        report.append(f"   Total Resources: {self.database['metadata']['total_resources']}")
        report.append("")

        for category in self.chia_resources.keys():
            if category in self.database and self.database[category]:
                report.append(f"📂 {category.upper()} Resources ({len(self.database[category])}):")
                for name, resource in self.database[category].items():
                    report.append(f"   • {name}")
                    if 'title' in resource:
                        report.append(f"     └─ {resource['title'][:60]}...")
                    if 'description' in resource:
                        report.append(f"     └─ {resource['description'][:60]}...")
                    if 'url' in resource:
                        report.append(f"     └─ URL: {resource['url']}")
                report.append("")

        if self.database.get('errors'):
            report.append(f"⚠️ Scraping Errors ({len(self.database['errors'])}):")
            for error in self.database['errors'][:5]:  # Show first 5 errors
                report.append(f"   • {error['resource']}: {error['error'][:50]}...")
            report.append("")

        report.append("🎯 Query Examples:")
        report.append("   • Find all documentation: query_resources(category='docs')")
        report.append("   • Find API resources: query_resources(resource_type='api_documentation')")
        report.append("   • Search for 'wallet': query_resources(search_term='wallet')")
        report.append("")

        return "\n".join(report)

def main():
    """Main function to run the Chia resource scraper"""
    print("🌱 Chia Resource Scraper and Database")
    print("=" * 50)

    scraper = ChiaResourceScraper()

    # Scrape all resources
    scraper.scrape_all_resources()

    # Generate and display report
    report = scraper.generate_resource_report()
    print(report)

    # Save report to file
    report_file = scraper.base_dir / "chia_resources_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"📄 Report saved to: {report_file}")
    print("\n✅ Chia resource scraping and database creation complete!")

if __name__ == "__main__":
    main()
