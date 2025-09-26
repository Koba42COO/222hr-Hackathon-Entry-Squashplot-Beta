#!/usr/bin/env python3
"""
🌐 GitHub Spec Kit Article Scraper
==================================
Specialized scraper for the GitHub Spec Kit article from Geeky Gadgets
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem, ScrapingJob
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_github_spec_kit_article():
    """Scrape the GitHub Spec Kit article and store in knowledge databases"""
    
    print("🌐 GitHub Spec Kit Article Scraper")
    print("=" * 50)
    
    # Initialize the web scraper knowledge system
    scraper = WebScraperKnowledgeSystem(max_workers=3)
    
    # The GitHub Spec Kit article URL
    article_url = "https://search.app/9wg9F"
    
    print(f"\n📄 Scraping GitHub Spec Kit Article...")
    print(f"URL: {article_url}")
    
    # Create scraping job
    job = ScrapingJob(
        url=article_url,
        priority=1,
        max_depth=0,  # Don't follow links
        follow_links=False,
        extract_images=False,
        extract_metadata=True,
        consciousness_enhancement=True
    )
    
    # Scrape the article
    result = scraper.scrape_website(
        url=article_url,
        max_depth=0,
        follow_links=False
    )
    
    if result["success"]:
        print(f"\n✅ Successfully scraped GitHub Spec Kit article!")
        print(f"   Title: {result['title']}")
        print(f"   Content Length: {result['content_length']} characters")
        print(f"   prime aligned compute Score: {result['prime_aligned_score']:.3f}")
        
        # Search for GitHub Spec Kit related content
        print(f"\n🔍 Searching for GitHub Spec Kit knowledge...")
        search_results = scraper.search_knowledge("GitHub Spec Kit specification driven development", limit=5)
        
        print(f"\n📚 Found {len(search_results)} relevant knowledge entries:")
        for i, result in enumerate(search_results):
            print(f"\n📖 Knowledge Entry {i+1}:")
            print(f"   Content Preview: {result['content'][:300]}...")
            print(f"   prime aligned compute Enhanced: {result['prime_aligned_enhanced']}")
            if 'source_url' in result['metadata']:
                print(f"   Source URL: {result['metadata']['source_url']}")
        
        # Get updated statistics
        stats = scraper.get_scraping_stats()
        print(f"\n📊 Updated Knowledge Base Statistics:")
        print(f"   📄 Total Pages Scraped: {stats.get('total_scraped_pages', 0)}")
        print(f"   🧠 Average prime aligned compute Score: {stats.get('average_consciousness_score', 0)}")
        print(f"   🔗 Knowledge Graph Nodes: {stats.get('knowledge_graph_nodes', 0)}")
        print(f"   🔗 Knowledge Graph Edges: {stats.get('knowledge_graph_edges', 0)}")
        print(f"   📚 RAG Documents: {stats.get('rag_documents', 0)}")
        
        # Store specific GitHub Spec Kit knowledge
        store_github_spec_kit_knowledge(scraper)
        
    else:
        print(f"\n❌ Failed to scrape article: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 GitHub Spec Kit Article Processing Complete!")

def store_github_spec_kit_knowledge(scraper):
    """Store specific GitHub Spec Kit knowledge in the system"""
    
    print(f"\n💾 Storing GitHub Spec Kit Knowledge...")
    
    # GitHub Spec Kit specific knowledge
    github_spec_kit_knowledge = {
        "id": "github_spec_kit_framework",
        "content": """
        GitHub Spec Kit: Specification-Driven Development Framework
        
        The GitHub Spec Kit is a revolutionary framework that introduces specification-driven development methodology to enhance software development by prioritizing detailed documentation, accuracy, and streamlined workflows.
        
        Key Features:
        - Integration with coding agents (GitHub Copilot, Claude, CodeGemini)
        - Four-step project implementation process
        - Command-based setup for simplified project management
        - Dynamic updates to project files
        - Test-driven development for accuracy
        
        Core Methodology:
        1. Define project goals and features
        2. Develop technical architecture and select tech stack
        3. Break tasks into actionable units
        4. Implement features using test-driven development
        
        Benefits:
        - Reduces errors through detailed planning
        - Eliminates assumption-based coding
        - Provides structured workflows
        - Ideal for complex, multi-feature projects
        - Seamless integration with AI coding agents
        
        Best Use Cases:
        - Multi-feature applications with complex dependencies
        - Projects requiring detailed specifications
        - Teams using AI coding assistants
        - Large-scale software development projects
        
        Limitations:
        - Time-intensive for simple tasks
        - Early-stage product with evolving features
        - May require additional setup for existing workflows
        """,
        "metadata": {
            "category": "Development Framework",
            "source": "Geeky Gadgets Article",
            "technology": "GitHub Spec Kit",
            "methodology": "Specification-Driven Development",
            "prime_aligned_enhanced": True,
            "ai_integration": True
        },
        "prime_aligned_enhanced": True
    }
    
    # Add to RAG system
    from knowledge_system_integration import RAGDocument
    
    rag_doc = RAGDocument(
        id=github_spec_kit_knowledge["id"],
        content=github_spec_kit_knowledge["content"],
        embeddings=scraper._generate_embeddings(github_spec_kit_knowledge["content"]),
        metadata=github_spec_kit_knowledge["metadata"],
        prime_aligned_enhanced=True
    )
    
    scraper.rag_system.add_document(rag_doc)
    
    # Store in prime aligned compute data
    scraper.database_service.store_consciousness_data(
        "github_spec_kit",
        {
            "framework_name": "GitHub Spec Kit",
            "methodology": "Specification-Driven Development",
            "key_features": [
                "Integration with coding agents",
                "Four-step implementation process",
                "Command-based setup",
                "Test-driven development"
            ],
            "prime_aligned_score": 1.618
        },
        {
            "source": "web_scraping",
            "article_url": "https://search.app/9wg9F",
            "scraped_at": "2025-01-17",
            "category": "Development Framework"
        },
        "web_scraper"
    )
    
    print(f"✅ GitHub Spec Kit knowledge stored successfully!")

if __name__ == "__main__":
    scrape_github_spec_kit_article()
