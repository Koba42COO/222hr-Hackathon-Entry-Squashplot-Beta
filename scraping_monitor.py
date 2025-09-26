#!/usr/bin/env python3
"""
📊 Scraping Monitor
===================
Monitor the progress of running scrapers and display real-time statistics.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from web_scraper_knowledge_system import WebScraperKnowledgeSystem
import time
import psutil
import subprocess

def monitor_scraping():
    """Monitor running scraping processes"""
    
    print("📊 Scientific Scraping Monitor")
    print("=" * 40)
    
    # Initialize knowledge system to get stats
    knowledge_system = WebScraperKnowledgeSystem(max_workers=1)
    
    # Get initial stats
    initial_stats = knowledge_system.get_scraping_stats()
    initial_pages = initial_stats.get('total_scraped_pages', 0)
    
    print(f"📄 Initial Pages: {initial_pages}")
    print(f"🧠 Initial prime aligned compute Score: {initial_stats.get('average_consciousness_score', 0)}")
    print(f"🔗 Initial Knowledge Graph Nodes: {initial_stats.get('knowledge_graph_nodes', 0)}")
    print(f"📚 Initial RAG Documents: {initial_stats.get('rag_documents', 0)}")
    
    print(f"\n⏰ Monitoring for 60 seconds...")
    print(f"Press Ctrl+C to stop monitoring")
    
    try:
        for i in range(60):
            time.sleep(1)
            
            # Get current stats
            current_stats = knowledge_system.get_scraping_stats()
            current_pages = current_stats.get('total_scraped_pages', 0)
            
            # Calculate progress
            pages_added = current_pages - initial_pages
            
            # Display progress
            if i % 10 == 0:  # Update every 10 seconds
                print(f"\n⏰ {i}s: +{pages_added} pages | Total: {current_pages}")
                print(f"   🧠 prime aligned compute: {current_stats.get('average_consciousness_score', 0)}")
                print(f"   🔗 Nodes: {current_stats.get('knowledge_graph_nodes', 0)}")
                print(f"   📚 RAG Docs: {current_stats.get('rag_documents', 0)}")
            
            # Check for Python processes
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] == 'python3' and 'scraping' in ' '.join(proc.info['cmdline']):
                        python_processes.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if i % 30 == 0 and i > 0:  # Every 30 seconds
                print(f"   🔄 Active scraping processes: {len(python_processes)}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoring stopped by user")
    
    # Final stats
    final_stats = knowledge_system.get_scraping_stats()
    final_pages = final_stats.get('total_scraped_pages', 0)
    total_added = final_pages - initial_pages
    
    print(f"\n📊 Final Results:")
    print(f"   📄 Pages Added: {total_added}")
    print(f"   📄 Total Pages: {final_pages}")
    print(f"   🧠 Final prime aligned compute Score: {final_stats.get('average_consciousness_score', 0)}")
    print(f"   🔗 Final Knowledge Graph Nodes: {final_stats.get('knowledge_graph_nodes', 0)}")
    print(f"   📚 Final RAG Documents: {final_stats.get('rag_documents', 0)}")

if __name__ == "__main__":
    monitor_scraping()
