#!/usr/bin/env python3
"""
🧪 Quick Test Scraper
=====================
Simple test to verify the scraping system is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem

def test_scraper():
    """Test the scraper with a simple URL"""
    
    print("🧪 Quick Test Scraper")
    print("=" * 30)
    
    # Initialize knowledge system
    knowledge_system = WebScraperKnowledgeSystem(max_workers=1)
    
    # Test URLs
    test_urls = [
        "https://arxiv.org/abs/2509.12761",  # Quantum physics paper
        "https://news.mit.edu/2025/3-questions-caroline-uhler-biology-medicine-data-revolution-0902"  # MIT article
    ]
    
    total_scraped = 0
    
    for i, url in enumerate(test_urls):
        print(f"\n📄 Testing URL {i+1}/{len(test_urls)}")
        print(f"URL: {url}")
        
        try:
            result = knowledge_system.scrape_website(
                url=url,
                max_depth=0,
                follow_links=False
            )
            
            if result['success']:
                total_scraped += 1
                print(f"✅ Success: {result.get('title', 'Untitled')[:50]}...")
                print(f"   Content length: {result.get('content_length', 0)}")
                print(f"   prime aligned compute score: {result.get('prime_aligned_score', 0)}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Get final statistics
    stats = knowledge_system.get_scraping_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   📄 Total Pages: {stats.get('total_scraped_pages', 0)}")
    print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
    print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
    print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
    print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")
    
    print(f"\n🎉 Test Complete!")
    print(f"   Successfully scraped: {total_scraped}/{len(test_urls)} URLs")

if __name__ == "__main__":
    test_scraper()
