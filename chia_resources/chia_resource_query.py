#!/usr/bin/env python3
"""
Chia Resources Query Interface
Interactive tool to query and explore the Chia resources database
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse

class ChiaResourceQuery:
    """
    Query interface for the Chia resources database
    """

    def __init__(self, database_path: str = None):
        if database_path is None:
            self.database_path = Path("/Users/coo-koba42/dev/SquashPlot_Complete_Package/chia_resources/chia_resources_database.json")
        else:
            self.database_path = Path(database_path)

        self.database = {}
        self.load_database()

    def load_database(self):
        """Load the database"""
        if self.database_path.exists():
            with open(self.database_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            print(f"✅ Loaded database with {self.database['metadata']['total_resources']} resources")
        else:
            print(f"❌ Database not found at {self.database_path}")
            self.database = {"resources": {}, "apis": {}, "documentation": {}}

    def query_resources(self, category: str = None, resource_type: str = None,
                       search_term: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Query resources from database"""
        results = []

        # Determine which categories to search
        if category:
            categories_to_search = [category]
        else:
            categories_to_search = [cat for cat in self.database.keys()
                                  if cat not in ['metadata', 'errors']]

        for cat in categories_to_search:
            if cat in self.database and isinstance(self.database[cat], dict):
                for name, resource in self.database[cat].items():
                    if isinstance(resource, dict):
                        # Apply filters
                        if resource_type and resource.get('type') != resource_type:
                            continue

                        if search_term:
                            # Search in all text fields
                            searchable_text = json.dumps(resource, default=str).lower()
                            if search_term.lower() not in searchable_text:
                                continue

                        results.append({
                            "category": cat,
                            "name": name,
                            "resource": resource
                        })

        # Sort by relevance (if search term provided)
        if search_term:
            results.sort(key=lambda x: self._calculate_relevance(x, search_term), reverse=True)

        return results[:limit]

    def _calculate_relevance(self, result: Dict, search_term: str) -> float:
        """Calculate relevance score for search results"""
        resource = result['resource']
        score = 0.0

        # Title matches get highest score
        if 'title' in resource and search_term.lower() in resource['title'].lower():
            score += 1.0

        # Description matches get medium score
        if 'description' in resource and search_term.lower() in resource['description'].lower():
            score += 0.7

        # URL matches get medium score
        if 'url' in resource and search_term.lower() in resource['url'].lower():
            score += 0.6

        # Content matches get lower score
        if 'scraped_content' in resource:
            content_text = json.dumps(resource['scraped_content']).lower()
            if search_term.lower() in content_text:
                score += 0.4

        return score

    def get_resource_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a specific resource"""
        for category in self.database:
            if category in ['metadata', 'errors']:
                continue

            if name in self.database[category]:
                resource = self.database[category][name]
                return {
                    "category": category,
                    "name": name,
                    "info": resource
                }

        return {"error": f"Resource '{name}' not found"}

    def list_categories(self) -> List[str]:
        """List all available categories"""
        return [cat for cat in self.database.keys()
               if cat not in ['metadata', 'errors'] and isinstance(self.database[cat], dict)]

    def list_resources_in_category(self, category: str) -> List[str]:
        """List all resources in a category"""
        if category in self.database and isinstance(self.database[category], dict):
            return list(self.database[category].keys())
        return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            "total_resources": self.database.get('metadata', {}).get('total_resources', 0),
            "categories": len(self.list_categories()),
            "errors": len(self.database.get('errors', [])),
            "last_updated": self.database.get('metadata', {}).get('last_updated', 'Unknown')
        }

        # Count resources per category
        category_counts = {}
        for category in self.list_categories():
            if category in self.database and isinstance(self.database[category], dict):
                category_counts[category] = len(self.database[category])

        stats["category_breakdown"] = category_counts

        return stats

    def search_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Search for resources related to a specific topic"""
        # Define topic keywords
        topic_keywords = {
            "wallet": ["wallet", "keys", "address", "balance", "transaction"],
            "farming": ["farming", "plot", "harvester", "farmer", "reward"],
            "development": ["api", "rpc", "sdk", "library", "development"],
            "trading": ["dex", "offers", "trading", "market", "exchange"],
            "nft": ["nft", "token", "collection", "metadata", "royalty"],
            "staking": ["staking", "pool", "delegate", "reward"],
            "mining": ["mining", "proof", "space", "plotting", "challenge"]
        }

        if topic not in topic_keywords:
            return self.query_resources(search_term=topic, limit=20)

        # Search for all keywords related to the topic
        all_results = []
        for keyword in topic_keywords[topic]:
            results = self.query_resources(search_term=keyword, limit=10)
            all_results.extend(results)

        # Remove duplicates and sort by relevance
        seen = set()
        unique_results = []
        for result in all_results:
            key = f"{result['category']}_{result['name']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        return unique_results[:15]

    def get_api_endpoints(self) -> List[Dict[str, Any]]:
        """Get all API endpoints from the database"""
        apis = []

        # Search API category
        api_results = self.query_resources(category='apis', limit=50)

        for result in api_results:
            resource = result['resource']

            if 'endpoints' in resource and resource['endpoints']:
                for endpoint in resource['endpoints']:
                    apis.append({
                        "service": result['name'],
                        "endpoint": endpoint,
                        "url": resource.get('url', ''),
                        "description": endpoint.get('description', '')
                    })

        return apis

    def get_github_repositories(self) -> List[Dict[str, Any]]:
        """Get GitHub repository information"""
        repos = []

        github_results = self.query_resources(category='github', limit=20)

        for result in github_results:
            resource = result['resource']

            if 'repositories' in resource:
                for repo in resource['repositories']:
                    repos.append({
                        "name": repo.get('name', ''),
                        "url": repo.get('url', ''),
                        "description": repo.get('description', ''),
                        "organization": "chia-network"
                    })

        return repos

    def export_resource_data(self, resource_name: str, format: str = 'json') -> str:
        """Export resource data in specified format"""
        resource_info = self.get_resource_info(resource_name)

        if 'error' in resource_info:
            return f"Error: {resource_info['error']}"

        if format.lower() == 'json':
            return json.dumps(resource_info, indent=2, ensure_ascii=False)
        elif format.lower() == 'text':
            return self._format_resource_as_text(resource_info)
        else:
            return "Unsupported format. Use 'json' or 'text'."

    def _format_resource_as_text(self, resource_info: Dict) -> str:
        """Format resource information as readable text"""
        lines = []
        lines.append(f"Resource: {resource_info['name']}")
        lines.append(f"Category: {resource_info['category']}")
        lines.append("")

        resource = resource_info['info']

        # Basic info
        if 'title' in resource:
            lines.append(f"Title: {resource['title']}")
        if 'description' in resource:
            lines.append(f"Description: {resource['description']}")
        if 'url' in resource:
            lines.append(f"URL: {resource['url']}")
        if 'type' in resource:
            lines.append(f"Type: {resource['type']}")

        # Additional structured data
        if 'sections' in resource and resource['sections']:
            lines.append("")
            lines.append("Sections:")
            for section in resource['sections'][:5]:  # First 5 sections
                lines.append(f"  • {section['title']}")

        if 'endpoints' in resource and resource['endpoints']:
            lines.append("")
            lines.append("API Endpoints:")
            for endpoint in resource['endpoints'][:5]:  # First 5 endpoints
                lines.append(f"  • {endpoint.get('url', 'N/A')}")

        if 'repositories' in resource and resource['repositories']:
            lines.append("")
            lines.append("Repositories:")
            for repo in resource['repositories'][:10]:  # First 10 repos
                lines.append(f"  • {repo.get('name', 'N/A')}: {repo.get('url', 'N/A')}")

        return "\n".join(lines)

def interactive_query():
    """Interactive query interface"""
    query = ChiaResourceQuery()

    print("🌱 Chia Resources Query Interface")
    print("=" * 50)
    print("Available commands:")
    print("  list - List all categories")
    print("  category <name> - List resources in category")
    print("  info <resource> - Get detailed info about resource")
    print("  search <term> - Search for resources")
    print("  topic <topic> - Search by topic (wallet, farming, development, etc.)")
    print("  apis - List all API endpoints")
    print("  repos - List GitHub repositories")
    print("  stats - Show database statistics")
    print("  help - Show this help")
    print("  quit - Exit")
    print()

    while True:
        try:
            cmd = input("chia-query> ").strip()

            if not cmd:
                continue

            parts = cmd.split()
            command = parts[0].lower()

            if command == 'quit':
                break
            elif command == 'help':
                print("Available commands:")
                print("  list - List all categories")
                print("  category <name> - List resources in category")
                print("  info <resource> - Get detailed info about resource")
                print("  search <term> - Search for resources")
                print("  topic <topic> - Search by topic")
                print("  apis - List all API endpoints")
                print("  repos - List GitHub repositories")
                print("  stats - Show database statistics")
                print("  export <resource> [format] - Export resource data")
                print("  help - Show this help")
                print("  quit - Exit")

            elif command == 'list':
                categories = query.list_categories()
                print(f"Available categories ({len(categories)}):")
                for cat in categories:
                    count = len(query.list_resources_in_category(cat))
                    print(f"  • {cat}: {count} resources")

            elif command == 'category' and len(parts) > 1:
                category = parts[1]
                resources = query.list_resources_in_category(category)
                if resources:
                    print(f"Resources in '{category}' ({len(resources)}):")
                    for resource in resources:
                        print(f"  • {resource}")
                else:
                    print(f"No resources found in category '{category}'")

            elif command == 'info' and len(parts) > 1:
                resource_name = parts[1]
                info = query.get_resource_info(resource_name)
                if 'error' in info:
                    print(f"❌ {info['error']}")
                else:
                    print(query.export_resource_data(resource_name, 'text'))

            elif command == 'search' and len(parts) > 1:
                search_term = ' '.join(parts[1:])
                results = query.query_resources(search_term=search_term, limit=10)
                if results:
                    print(f"Search results for '{search_term}' ({len(results)}):")
                    for i, result in enumerate(results, 1):
                        resource = result['resource']
                        title = resource.get('title', result['name'])
                        print(f"  {i}. {title}")
                        print(f"     Category: {result['category']}")
                        if 'url' in resource:
                            print(f"     URL: {resource['url']}")
                        print()
                else:
                    print(f"No results found for '{search_term}'")

            elif command == 'topic' and len(parts) > 1:
                topic = parts[1]
                results = query.search_by_topic(topic)
                if results:
                    print(f"Resources related to '{topic}' ({len(results)}):")
                    for i, result in enumerate(results, 1):
                        resource = result['resource']
                        title = resource.get('title', result['name'])
                        print(f"  {i}. {title}")
                        print(f"     Category: {result['category']}")
                        if 'url' in resource:
                            print(f"     URL: {resource['url']}")
                        print()
                else:
                    print(f"No resources found for topic '{topic}'")

            elif command == 'apis':
                apis = query.get_api_endpoints()
                if apis:
                    print(f"API Endpoints ({len(apis)}):")
                    for i, api in enumerate(apis, 1):
                        print(f"  {i}. {api['service']}")
                        print(f"     Endpoint: {api['endpoint']}")
                        if api['description']:
                            print(f"     Description: {api['description']}")
                        print()
                else:
                    print("No API endpoints found")

            elif command == 'repos':
                repos = query.get_github_repositories()
                if repos:
                    print(f"GitHub Repositories ({len(repos)}):")
                    for i, repo in enumerate(repos, 1):
                        print(f"  {i}. {repo['name']}")
                        if repo['description']:
                            print(f"     Description: {repo['description']}")
                        print(f"     URL: {repo['url']}")
                        print()
                else:
                    print("No GitHub repositories found")

            elif command == 'stats':
                stats = query.get_database_stats()
                print("Database Statistics:")
                print(f"  Total Resources: {stats['total_resources']}")
                print(f"  Categories: {stats['categories']}")
                print(f"  Errors: {stats['errors']}")
                print(f"  Last Updated: {stats['last_updated']}")
                print("  Category Breakdown:")
                for cat, count in stats['category_breakdown'].items():
                    print(f"    • {cat}: {count} resources")

            elif command == 'export' and len(parts) >= 2:
                resource_name = parts[1]
                format_type = parts[2] if len(parts) > 2 else 'json'
                result = query.export_resource_data(resource_name, format_type)
                print(result)

            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Chia Resources Query Interface')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive query mode')
    parser.add_argument('--search', '-s', type=str,
                       help='Search for resources containing term')
    parser.add_argument('--category', '-c', type=str,
                       help='List resources in category')
    parser.add_argument('--info', type=str,
                       help='Get detailed info about specific resource')
    parser.add_argument('--topic', type=str,
                       help='Search by topic (wallet, farming, development, etc.)')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')

    args = parser.parse_args()

    query = ChiaResourceQuery()

    if args.interactive:
        interactive_query()
    elif args.search:
        results = query.query_resources(search_term=args.search)
        print(f"Search results for '{args.search}':")
        for result in results:
            print(f"  • {result['name']} ({result['category']})")
    elif args.category:
        resources = query.list_resources_in_category(args.category)
        print(f"Resources in '{args.category}':")
        for resource in resources:
            print(f"  • {resource}")
    elif args.info:
        info = query.get_resource_info(args.info)
        if 'error' in info:
            print(f"❌ {info['error']}")
        else:
            print(query.export_resource_data(args.info, 'text'))
    elif args.topic:
        results = query.search_by_topic(args.topic)
        print(f"Resources related to '{args.topic}':")
        for result in results:
            print(f"  • {result['name']} ({result['category']})")
    elif args.stats:
        stats = query.get_database_stats()
        print("Database Statistics:")
        print(f"  Total Resources: {stats['total_resources']}")
        print(f"  Categories: {stats['categories']}")
        print(f"  Category Breakdown: {stats['category_breakdown']}")
    else:
        print("Chia Resources Query Interface")
        print("Use --interactive for interactive mode, or see --help for options")

if __name__ == "__main__":
    main()
