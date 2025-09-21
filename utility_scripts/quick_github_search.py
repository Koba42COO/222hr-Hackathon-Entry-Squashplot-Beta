#!/usr/bin/env python3
"""
Quick GitHub AI Repository Search
Fast discovery of interesting AI and programming repositories
"""

import requests
import json
import time
from datetime import datetime

def search_github_repos(query, max_results=20):
    """Search GitHub repositories"""
    print(f"🔍 Searching: {query}")
    
    url = "https://api.github.com/search/repositories"
    headers = {
        'User-Agent': 'GitHub-AI-Search/1.0',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    params = {
        'q': query,
        'sort': 'stars',
        'order': 'desc',
        'per_page': min(max_results, 30)
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            repos = []
            
            for repo in data.get('items', []):
                repo_info = {
                    'name': repo['full_name'],
                    'description': repo.get('description', ''),
                    'language': repo.get('language', ''),
                    'stars': repo['stargazers_count'],
                    'forks': repo['forks_count'],
                    'topics': repo.get('topics', []),
                    'url': repo['html_url'],
                    'created_at': repo['created_at'],
                    'pushed_at': repo['pushed_at']
                }
                repos.append(repo_info)
            
            print(f"✅ Found {len(repos)} repositories")
            return repos
        else:
            print(f"❌ API request failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    """Main function"""
    print("🔍 Quick GitHub AI Repository Search")
    print("=" * 50)
    
    # Search patterns for interesting repositories
    search_patterns = [
        'language:python topic:artificial-intelligence',
        'language:python topic:machine-learning',
        'language:python topic:quantum-computing',
        'language:python topic:consciousness',
        'language:rust topic:ai',
        'language:rust topic:quantum',
        'language:go topic:ai',
        'topic:web3',
        'topic:blockchain',
        'stars:>1000 language:python created:>2024-01-01'
    ]
    
    all_repos = []
    seen_repos = set()
    
    print("🚀 Starting quick search...")
    
    for pattern in search_patterns:
        repos = search_github_repos(pattern, 15)
        
        for repo in repos:
            if repo['name'] not in seen_repos:
                seen_repos.add(repo['name'])
                all_repos.append(repo)
        
        time.sleep(2)  # Rate limiting
    
    # Sort by stars
    all_repos.sort(key=lambda x: x['stars'], reverse=True)
    
    print(f"\n🎉 Search complete! Found {len(all_repos)} unique repositories")
    
    # Display top results
    print("\n🏆 Top 20 Most Interesting Repositories:")
    print("=" * 80)
    
    for i, repo in enumerate(all_repos[:20], 1):
        print(f"{i:2d}. {repo['name']}")
        print(f"    ⭐ {repo['stars']:,} stars | 🍴 {repo['forks']:,} forks | {repo['language']}")
        print(f"    📝 {repo['description'][:100] if repo['description'] else 'No description'}...")
        print(f"    🏷️  Topics: {', '.join(repo['topics'][:5])}")
        print(f"    🔗 {repo['url']}")
        print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"github_search_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_repos, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filename}")
    
    # Generate summary
    print("\n📊 Summary:")
    print(f"   Total repositories: {len(all_repos)}")
    print(f"   Languages found: {set(repo['language'] for repo in all_repos if repo['language'])}")
    
    # Count topics
    all_topics = []
    for repo in all_repos:
        all_topics.extend(repo['topics'])
    
    topic_counts = {}
    for topic in all_topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   Top topics: {[topic for topic, count in top_topics[:5]]}")

if __name__ == "__main__":
    main()
