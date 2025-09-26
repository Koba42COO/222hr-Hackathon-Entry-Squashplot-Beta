#!/usr/bin/env python3
"""
GitHub AI & Programming Repository Crawler
Discovers and analyzes interesting open source repositories

Features:
- Multi-criteria repository discovery
- Code pattern analysis
- Technology trend detection
- Repository scoring and ranking
- Automated code extraction
"""

import requests
import json
import time
import random
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import re
from collections import defaultdict
import pandas as pd

class GitHubAICrawler:
    """Advanced GitHub crawler for AI and programming repositories"""
    
    def __init__(self, output_dir: str = "~/dev/github_crawl"):
        self.output_dir = Path(os.path.expanduser(output_dir))
        self.repos_dir = self.output_dir / "repositories"
        self.analysis_dir = self.output_dir / "analysis"
        self.code_dir = self.output_dir / "code_samples"
        
        # Create directories
        for dir_path in [self.repos_dir, self.analysis_dir, self.code_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # GitHub API configuration
        self.api_base = "https://api.github.com"
        self.headers = {
            'User-Agent': 'GitHub-AI-Crawler/1.0',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Add GitHub token if available
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
        
        # AI and programming keywords
        self.ai_keywords = [
            'artificial-intelligence', 'machine-learning', 'deep-learning',
            'neural-network', 'ai', 'ml', 'nlp', 'computer-vision',
            'reinforcement-learning', 'transformer', 'gpt', 'llm',
            'quantum-computing', 'quantum-ai', 'consciousness',
            'post-quantum-cryptography', 'zero-knowledge-proofs',
            'blockchain', 'web3', 'metaverse', 'augmented-reality',
            'virtual-reality', 'robotics', 'autonomous-systems'
        ]
        
        self.programming_keywords = [
            'python', 'javascript', 'typescript', 'rust', 'go',
            'c++', 'java', 'scala', 'kotlin', 'swift',
            'react', 'vue', 'angular', 'nextjs', 'tauri',
            'docker', 'kubernetes', 'terraform', 'ansible',
            'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'apache-kafka', 'rabbitmq', 'grpc', 'graphql',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas',
            'numpy', 'matplotlib', 'seaborn', 'plotly'
        ]
        
        # Repository discovery patterns
        self.search_patterns = [
            # AI/ML specific
            'language:python topic:artificial-intelligence',
            'language:python topic:machine-learning',
            'language:python topic:deep-learning',
            'language:python topic:neural-networks',
            'language:python topic:quantum-computing',
            'language:python topic:consciousness',
            'language:python topic:zero-knowledge-proofs',
            'language:rust topic:ai',
            'language:rust topic:quantum',
            'language:go topic:ai',
            'language:javascript topic:ai',
            'language:typescript topic:ai',
            
            # Programming frameworks
            'language:python topic:web-framework',
            'language:javascript topic:framework',
            'language:typescript topic:framework',
            'language:rust topic:web',
            'language:go topic:web',
            
            # Emerging technologies
            'topic:web3',
            'topic:blockchain',
            'topic:metaverse',
            'topic:ar-vr',
            'topic:robotics',
            'topic:autonomous-vehicles',
            
            # High-level searches
            'stars:>1000 language:python',
            'stars:>500 language:rust',
            'stars:>500 language:go',
            'created:>2024-01-01 language:python',
            'pushed:>2024-01-01 language:python'
        ]
        
        # Repository scoring criteria
        self.scoring_criteria = {
            'stars': 0.3,
            'forks': 0.2,
            'recent_activity': 0.2,
            'code_quality': 0.15,
            'documentation': 0.1,
            'license': 0.05
        }
        
        # Discovered repositories
        self.discovered_repos = []
        self.analyzed_repos = []
        self.code_patterns = defaultdict(int)
        self.technology_trends = defaultdict(int)
    
    def search_repositories(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search GitHub repositories using the API
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of repository data
        """
        print(f"🔍 Searching: {query}")
        
        repos = []
        page = 1
        per_page = min(100, max_results)
        
        while len(repos) < max_results:
            url = f"{self.api_base}/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'page': page,
                'per_page': per_page
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    
                    if not items:
                        break
                    
                    for repo in items:
                        if len(repos) >= max_results:
                            break
                        
                        repo_data = {
                            'id': repo['id'],
                            'name': repo['name'],
                            'full_name': repo['full_name'],
                            'description': repo.get('description', ''),
                            'language': repo.get('language', ''),
                            'stars': repo['stargazers_count'],
                            'forks': repo['forks_count'],
                            'watchers': repo['watchers_count'],
                            'open_issues': repo['open_issues_count'],
                            'created_at': repo['created_at'],
                            'updated_at': repo['updated_at'],
                            'pushed_at': repo['pushed_at'],
                            'size': repo['size'],
                            'license': repo.get('license', {}).get('name', ''),
                            'topics': repo.get('topics', []),
                            'url': repo['html_url'],
                            'api_url': repo['url'],
                            'clone_url': repo['clone_url'],
                            'search_query': query
                        }
                        
                        repos.append(repo_data)
                    
                    page += 1
                    
                    # Rate limiting
                    if 'X-RateLimit-Remaining' in response.headers:
                        remaining = int(response.headers['X-RateLimit-Remaining'])
                        if remaining < 10:
                            print(f"⚠️ Rate limit warning: {remaining} requests remaining")
                            time.sleep(60)  # Wait 1 minute
                    else:
                        time.sleep(random.uniform(1, 3))  # Random delay
                
                else:
                    print(f"❌ API request failed: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"❌ Error searching repositories: {e}")
                break
        
        print(f"✅ Found {len(repos)} repositories for query: {query}")
        return repos
    
    def discover_repositories(self, max_per_pattern: int = 50) -> List[Dict]:
        """
        Discover repositories using multiple search patterns
        
        Args:
            max_per_pattern: Maximum repositories per search pattern
            
        Returns:
            List of discovered repositories
        """
        print("🚀 Starting comprehensive repository discovery...")
        
        all_repos = []
        seen_repos = set()
        
        for pattern in self.search_patterns:
            print(f"\n🔍 Pattern: {pattern}")
            repos = self.search_repositories(pattern, max_per_pattern)
            
            for repo in repos:
                repo_id = repo['id']
                if repo_id not in seen_repos:
                    seen_repos.add(repo_id)
                    all_repos.append(repo)
        
        # Remove duplicates and sort by stars
        unique_repos = list({repo['id']: repo for repo in all_repos}.values())
        unique_repos.sort(key=lambda x: x['stars'], reverse=True)
        
        self.discovered_repos = unique_repos
        
        print(f"\n🎉 Discovery complete! Found {len(unique_repos)} unique repositories")
        
        # Save discovered repositories
        self.save_discovered_repos()
        
        return unique_repos
    
    def analyze_repository(self, repo: Dict) -> Dict:
        """
        Analyze a single repository for code patterns and quality
        
        Args:
            repo: Repository data dictionary
            
        Returns:
            Analysis results
        """
        print(f"🔬 Analyzing: {repo['full_name']}")
        
        analysis = {
            'repo_id': repo['id'],
            'full_name': repo['full_name'],
            'analysis_timestamp': datetime.now().isoformat(),
            'score': 0,
            'code_patterns': [],
            'technologies': [],
            'file_analysis': {},
            'readme_analysis': {},
            'license_analysis': {},
            'activity_analysis': {},
            'recommendations': []
        }
        
        try:
            # Get repository details
            repo_details = self.get_repository_details(repo['api_url'])
            if repo_details:
                analysis.update(repo_details)
            
            # Analyze README
            readme_analysis = self.analyze_readme(repo['full_name'])
            analysis['readme_analysis'] = readme_analysis
            
            # Analyze code patterns
            code_analysis = self.analyze_code_patterns(repo['full_name'])
            analysis['code_patterns'] = code_analysis['patterns']
            analysis['technologies'] = code_analysis['technologies']
            analysis['file_analysis'] = code_analysis['files']
            
            # Calculate repository score
            analysis['score'] = self.calculate_repository_score(repo, analysis)
            
            # Generate recommendations
            analysis['recommendations'] = self.generate_recommendations(repo, analysis)
            
        except Exception as e:
            print(f"❌ Error analyzing {repo['full_name']}: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def get_repository_details(self, api_url: str) -> Optional[Dict]:
        """Get detailed repository information"""
        try:
            response = requests.get(api_url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ Error getting repository details: {e}")
        return None
    
    def analyze_readme(self, full_name: str) -> Dict:
        """Analyze repository README file"""
        readme_analysis = {
            'has_readme': False,
            'readme_size': 0,
            'readme_content': '',
            'sections': [],
            'code_blocks': 0,
            'links': 0,
            'images': 0
        }
        
        try:
            # Try to get README content
            readme_url = f"{self.api_base}/repos/{full_name}/readme"
            response = requests.get(readme_url, headers=self.headers)
            
            if response.status_code == 200:
                readme_data = response.json()
                readme_content = readme_data.get('content', '')
                
                if readme_content:
                    readme_analysis['has_readme'] = True
                    readme_analysis['readme_size'] = len(readme_content)
                    readme_analysis['readme_content'] = readme_content
                    
                    # Analyze README content
                    readme_analysis['sections'] = self.extract_readme_sections(readme_content)
                    readme_analysis['code_blocks'] = readme_content.count('```')
                    readme_analysis['links'] = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', readme_content))
                    readme_analysis['images'] = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', readme_content))
        
        except Exception as e:
            print(f"❌ Error analyzing README: {e}")
        
        return readme_analysis
    
    def extract_readme_sections(self, content: str) -> List[str]:
        """Extract section headers from README"""
        sections = []
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('#'):
                sections.append(line.strip())
        
        return sections
    
    def analyze_code_patterns(self, full_name: str) -> Dict:
        """Analyze code patterns in the repository"""
        code_analysis = {
            'patterns': [],
            'technologies': [],
            'files': {},
            'languages': {},
            'complexity': 0
        }
        
        try:
            # Get repository contents
            contents_url = f"{self.api_base}/repos/{full_name}/contents"
            response = requests.get(contents_url, headers=self.headers)
            
            if response.status_code == 200:
                contents = response.json()
                
                for item in contents:
                    if item['type'] == 'file':
                        file_analysis = self.analyze_file(full_name, item['path'])
                        code_analysis['files'][item['path']] = file_analysis
                        
                        # Extract patterns and technologies
                        if file_analysis.get('content'):
                            patterns = self.extract_code_patterns(file_analysis['content'])
                            code_analysis['patterns'].extend(patterns)
                            
                            technologies = self.extract_technologies(file_analysis['content'])
                            code_analysis['technologies'].extend(technologies)
                
                # Count languages
                for file_info in code_analysis['files'].values():
                    lang = file_info.get('language', 'unknown')
                    code_analysis['languages'][lang] = code_analysis['languages'].get(lang, 0) + 1
        
        except Exception as e:
            print(f"❌ Error analyzing code patterns: {e}")
        
        return code_analysis
    
    def analyze_file(self, full_name: str, file_path: str) -> Dict:
        """Analyze a single file"""
        file_analysis = {
            'path': file_path,
            'size': 0,
            'language': '',
            'content': '',
            'lines': 0,
            'complexity': 0
        }
        
        try:
            file_url = f"{self.api_base}/repos/{full_name}/contents/{file_path}"
            response = requests.get(file_url, headers=self.headers)
            
            if response.status_code == 200:
                file_data = response.json()
                file_analysis['size'] = file_data.get('size', 0)
                file_analysis['language'] = self.detect_language(file_path)
                
                # Get file content for small files
                if file_analysis['size'] < 100000:  # 100KB limit
                    content = file_data.get('content', '')
                    if content:
                        file_analysis['content'] = content
                        file_analysis['lines'] = len(content.split('\n'))
                        file_analysis['complexity'] = self.calculate_complexity(content)
        
        except Exception as e:
            print(f"❌ Error analyzing file {file_path}: {e}")
        
        return file_analysis
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.rs': 'Rust',
            '.go': 'Go',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'MATLAB',
            '.sh': 'Shell',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.conf': 'Configuration'
        }
        
        return language_map.get(ext, 'Unknown')
    
    def extract_code_patterns(self, content: str) -> List[str]:
        """Extract interesting code patterns"""
        patterns = []
        
        # AI/ML patterns
        if 'import tensorflow' in content or 'import torch' in content:
            patterns.append('deep_learning_framework')
        
        if 'import sklearn' in content or 'from sklearn' in content:
            patterns.append('machine_learning')
        
        if 'import numpy' in content or 'import pandas' in content:
            patterns.append('data_science')
        
        if 'quantum' in content.lower():
            patterns.append('quantum_computing')
        
        if 'consciousness' in content.lower():
            patterns.append('consciousness_research')
        
        # Web development patterns
        if 'import react' in content or 'from react' in content:
            patterns.append('react_framework')
        
        if 'import vue' in content or 'from vue' in content:
            patterns.append('vue_framework')
        
        if 'express' in content.lower():
            patterns.append('express_server')
        
        if 'flask' in content.lower() or 'django' in content.lower():
            patterns.append('python_web_framework')
        
        # Database patterns
        if 'postgresql' in content.lower() or 'postgres' in content.lower():
            patterns.append('postgresql_database')
        
        if 'mongodb' in content.lower():
            patterns.append('mongodb_database')
        
        if 'redis' in content.lower():
            patterns.append('redis_cache')
        
        # Cloud/DevOps patterns
        if 'docker' in content.lower():
            patterns.append('docker_containerization')
        
        if 'kubernetes' in content.lower() or 'k8s' in content.lower():
            patterns.append('kubernetes_orchestration')
        
        if 'terraform' in content.lower():
            patterns.append('terraform_infrastructure')
        
        return patterns
    
    def extract_technologies(self, content: str) -> List[str]:
        """Extract technologies mentioned in code"""
        technologies = []
        
        # Common technology keywords
        tech_keywords = [
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'react', 'vue', 'angular', 'nextjs', 'nuxtjs',
            'express', 'fastapi', 'django', 'flask', 'spring',
            'postgresql', 'mongodb', 'redis', 'mysql', 'sqlite',
            'docker', 'kubernetes', 'terraform', 'ansible',
            'aws', 'azure', 'gcp', 'heroku', 'vercel',
            'graphql', 'grpc', 'rest', 'websocket',
            'jwt', 'oauth', 'oauth2', 'openid',
            'blockchain', 'ethereum', 'bitcoin', 'web3',
            'quantum', 'consciousness', 'ai', 'ml', 'nlp'
        ]
        
        content_lower = content.lower()
        for tech in tech_keywords:
            if tech in content_lower:
                technologies.append(tech)
        
        return list(set(technologies))  # Remove duplicates
    
    def calculate_complexity(self, content: str) -> int:
        """Calculate code complexity score"""
        complexity = 0
        
        # Count various complexity indicators
        complexity += content.count('if ') * 1
        complexity += content.count('for ') * 2
        complexity += content.count('while ') * 2
        complexity += content.count('def ') * 1
        complexity += content.count('class ') * 2
        complexity += content.count('import ') * 0.5
        complexity += content.count('from ') * 0.5
        
        return int(complexity)
    
    def calculate_repository_score(self, repo: Dict, analysis: Dict) -> float:
        """Calculate overall repository score"""
        score = 0
        
        # Stars weight
        stars_score = min(repo['stars'] / 1000, 1.0)  # Normalize to 0-1
        score += stars_score * self.scoring_criteria['stars']
        
        # Forks weight
        forks_score = min(repo['forks'] / 100, 1.0)
        score += forks_score * self.scoring_criteria['forks']
        
        # Recent activity weight
        pushed_at = datetime.fromisoformat(repo['pushed_at'].replace('Z', '+00:00'))
        days_since_push = (datetime.now(pushed_at.tzinfo) - pushed_at).days
        activity_score = max(0, 1 - (days_since_push / 365))  # Higher score for recent activity
        score += activity_score * self.scoring_criteria['recent_activity']
        
        # Code quality weight
        code_quality_score = 0
        if analysis.get('readme_analysis', {}).get('has_readme'):
            code_quality_score += 0.5
        if analysis.get('code_patterns'):
            code_quality_score += 0.3
        if analysis.get('technologies'):
            code_quality_score += 0.2
        score += code_quality_score * self.scoring_criteria['code_quality']
        
        # Documentation weight
        readme_analysis = analysis.get('readme_analysis', {})
        doc_score = 0
        if readme_analysis.get('has_readme'):
            doc_score += 0.4
        if readme_analysis.get('sections'):
            doc_score += 0.3
        if readme_analysis.get('code_blocks', 0) > 0:
            doc_score += 0.3
        score += doc_score * self.scoring_criteria['documentation']
        
        # License weight
        if repo.get('license'):
            score += self.scoring_criteria['license']
        
        return min(score, 1.0)  # Normalize to 0-1
    
    def generate_recommendations(self, repo: Dict, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # High-scoring repositories
        if analysis['score'] > 0.8:
            recommendations.append("⭐ High-quality repository with excellent code and documentation")
        
        # AI/ML repositories
        if any(pattern in analysis.get('code_patterns', []) for pattern in ['deep_learning_framework', 'machine_learning']):
            recommendations.append("🤖 AI/ML repository with potential for quantum integration")
        
        # Quantum computing repositories
        if 'quantum_computing' in analysis.get('code_patterns', []):
            recommendations.append("⚛️ Quantum computing repository - highly relevant for our research")
        
        # Consciousness research
        if 'consciousness_research' in analysis.get('code_patterns', []):
            recommendations.append("🧠 Consciousness research repository - unique and valuable")
        
        # Recent activity
        pushed_at = datetime.fromisoformat(repo['pushed_at'].replace('Z', '+00:00'))
        days_since_push = (datetime.now(pushed_at.tzinfo) - pushed_at).days
        if days_since_push < 30:
            recommendations.append("🔄 Recently updated - actively maintained")
        
        # Good documentation
        if analysis.get('readme_analysis', {}).get('has_readme'):
            recommendations.append("📚 Well-documented repository")
        
        return recommendations
    
    def save_discovered_repos(self):
        """Save discovered repositories to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = self.repos_dir / f"discovered_repos_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.discovered_repos, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.discovered_repos)} discovered repositories to {file_path}")
    
    def save_analysis_results(self):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = self.analysis_dir / f"analysis_results_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.analyzed_repos, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved analysis results to {file_path}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.analyzed_repos:
            print("❌ No analyzed repositories to report on")
            return
        
        print("\n📊 Generating comprehensive summary report...")
        
        # Calculate statistics
        total_repos = len(self.analyzed_repos)
        avg_score = sum(repo['score'] for repo in self.analyzed_repos) / total_repos
        high_quality_repos = [repo for repo in self.analyzed_repos if repo['score'] > 0.8]
        
        # Technology trends
        all_technologies = []
        all_patterns = []
        for repo in self.analyzed_repos:
            all_technologies.extend(repo.get('technologies', []))
            all_patterns.extend(repo.get('code_patterns', []))
        
        tech_counts = pd.Series(all_technologies).value_counts()
        pattern_counts = pd.Series(all_patterns).value_counts()
        
        # Generate report
        report = {
            'summary': {
                'total_repositories': total_repos,
                'average_score': round(avg_score, 3),
                'high_quality_repos': len(high_quality_repos),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'top_repositories': sorted(self.analyzed_repos, key=lambda x: x['score'], reverse=True)[:10],
            'technology_trends': tech_counts.head(20).to_dict(),
            'code_patterns': pattern_counts.head(20).to_dict(),
            'recommendations': self.generate_global_recommendations()
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.analysis_dir / f"summary_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Total repositories analyzed: {total_repos}")
        print(f"⭐ Average score: {avg_score:.3f}")
        print(f"🏆 High-quality repositories: {len(high_quality_repos)}")
        print(f"📈 Top technologies: {list(tech_counts.head(5).index)}")
        print(f"🔍 Top patterns: {list(pattern_counts.head(5).index)}")
        print(f"📄 Report saved to: {report_path}")
        
        return report
    
    def generate_global_recommendations(self) -> List[str]:
        """Generate global recommendations based on all analyzed repositories"""
        recommendations = []
        
        # Count patterns across all repositories
        all_patterns = []
        for repo in self.analyzed_repos:
            all_patterns.extend(repo.get('code_patterns', []))
        
        pattern_counts = pd.Series(all_patterns).value_counts()
        
        # AI/ML recommendations
        if 'deep_learning_framework' in pattern_counts:
            recommendations.append("🤖 Consider integrating TensorFlow/PyTorch patterns into quantum systems")
        
        if 'quantum_computing' in pattern_counts:
            recommendations.append("⚛️ Found quantum computing repositories - valuable for research")
        
        if 'consciousness_research' in pattern_counts:
            recommendations.append("🧠 Consciousness research repositories discovered - unique opportunity")
        
        # Technology recommendations
        if 'react_framework' in pattern_counts:
            recommendations.append("⚛️ React patterns could enhance quantum UI components")
        
        if 'docker_containerization' in pattern_counts:
            recommendations.append("🐳 Docker patterns useful for quantum system deployment")
        
        if 'kubernetes_orchestration' in pattern_counts:
            recommendations.append("☸️ Kubernetes patterns for quantum infrastructure scaling")
        
        return recommendations
    
    def run_comprehensive_crawl(self, max_repos_per_pattern: int = 30, analyze_top: int = 50):
        """
        Run comprehensive GitHub crawling and analysis
        
        Args:
            max_repos_per_pattern: Maximum repositories per search pattern
            analyze_top: Number of top repositories to analyze in detail
        """
        print("🚀 Starting comprehensive GitHub AI repository crawl...")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Step 1: Discover repositories
        discovered_repos = self.discover_repositories(max_repos_per_pattern)
        
        if not discovered_repos:
            print("❌ No repositories discovered")
            return
        
        # Step 2: Analyze top repositories
        print(f"\n🔬 Analyzing top {analyze_top} repositories...")
        
        for i, repo in enumerate(discovered_repos[:analyze_top]):
            print(f"\n📊 Progress: {i+1}/{min(analyze_top, len(discovered_repos))}")
            
            analysis = self.analyze_repository(repo)
            self.analyzed_repos.append(analysis)
            
            # Rate limiting
            time.sleep(random.uniform(1, 3))
        
        # Step 3: Save results
        self.save_analysis_results()
        
        # Step 4: Generate summary report
        report = self.generate_summary_report()
        
        print(f"\n🎉 Comprehensive crawl completed successfully!")
        print(f"📁 Check the following directories:")
        print(f"   Repositories: {self.repos_dir}")
        print(f"   Analysis: {self.analysis_dir}")
        print(f"   Code Samples: {self.code_dir}")

def main():
    """Main function"""
    print("🔍 GitHub AI Repository Crawler")
    print("=" * 50)
    print("Discovers and analyzes interesting AI and programming repositories")
    print("=" * 50)
    
    # Initialize crawler
    crawler = GitHubAICrawler()
    
    # Show options
    print("\n📋 Available options:")
    print("1. Run comprehensive crawl (discover + analyze)")
    print("2. Discover repositories only")
    print("3. Analyze specific repository")
    print("4. Generate summary report from existing data")
    print("5. View discovered repositories")
    
    choice = input("\n🎯 Choose an option (1-5): ").strip()
    
    if choice == "1":
        max_per_pattern = int(input("📊 Max repos per pattern (default 30): ") or "30")
        analyze_top = int(input("🔬 Analyze top N repos (default 50): ") or "50")
        crawler.run_comprehensive_crawl(max_per_pattern, analyze_top)
    
    elif choice == "2":
        max_per_pattern = int(input("📊 Max repos per pattern (default 30): ") or "30")
        crawler.discover_repositories(max_per_pattern)
    
    elif choice == "3":
        repo_name = input("📁 Repository name (e.g., 'username/repo'): ").strip()
        if repo_name:
            # Create mock repo data for analysis
            mock_repo = {
                'id': 12345,
                'name': repo_name.split('/')[-1],
                'full_name': repo_name,
                'api_url': f"https://api.github.com/repos/{repo_name}",
                'stars': 100,
                'forks': 50,
                'pushed_at': datetime.now().isoformat()
            }
            analysis = crawler.analyze_repository(mock_repo)
            print(f"\n📊 Analysis results for {repo_name}:")
            print(json.dumps(analysis, indent=2))
    
    elif choice == "4":
        crawler.generate_summary_report()
    
    elif choice == "5":
        if crawler.discovered_repos:
            print(f"\n📊 Discovered {len(crawler.discovered_repos)} repositories:")
            for i, repo in enumerate(crawler.discovered_repos[:10]):
                print(f"{i+1}. {repo['full_name']} (⭐{repo['stars']}) - {repo['description'][:100]}...")
        else:
            print("❌ No discovered repositories. Run discovery first.")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
