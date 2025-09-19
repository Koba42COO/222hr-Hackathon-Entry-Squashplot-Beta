#!/usr/bin/env python3
"""
KOBA42 SCRAPED DATA INTEGRATION STATUS
======================================
Comprehensive Status Report of All Scraped Data Integration
=========================================================

This script provides a complete overview of:
1. What data we have scraped and stored
2. What has been processed and analyzed
3. What integration opportunities exist
4. What might be missing or need updating
"""

import sqlite3
import json
from datetime import datetime, timedelta

def generate_scraped_data_integration_status():
    """Generate comprehensive status report of scraped data integration."""
    
    print("🔍 KOBA42 SCRAPED DATA INTEGRATION STATUS REPORT")
    print("=" * 70)
    print("Comprehensive Analysis of All Scraped Data and Integration")
    print("=" * 70)
    
    # Database paths
    research_db = "research_data/research_articles.db"
    explorations_db = "research_data/agentic_explorations.db"
    ledger_db = "research_data/digital_ledger.db"
    
    # 1. RESEARCH ARTICLES STATUS
    print("\n📊 RESEARCH ARTICLES DATABASE STATUS")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect(research_db)
        cursor = conn.cursor()
        
        # Total articles
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        print(f"Total Articles Stored: {total_articles}")
        
        # Articles by source
        cursor.execute("SELECT source, COUNT(*) as count FROM articles GROUP BY source ORDER BY count DESC")
        sources = cursor.fetchall()
        print("\nArticles by Source:")
        for source, count in sources:
            print(f"  {source}: {count} articles")
        
        # Articles by field
        cursor.execute("SELECT field, COUNT(*) as count FROM articles GROUP BY field ORDER BY count DESC LIMIT 10")
        fields = cursor.fetchall()
        print("\nTop 10 Fields:")
        for field, count in fields:
            print(f"  {field}: {count} articles")
        
        # Last scraping timestamp
        cursor.execute("SELECT MAX(scraped_timestamp) FROM articles")
        last_scraped = cursor.fetchone()[0]
        print(f"\nLast Scraping Activity: {last_scraped}")
        
        # Articles with high integration potential
        cursor.execute("SELECT COUNT(*) FROM articles WHERE koba42_integration_potential > 0.7")
        high_potential = cursor.fetchone()[0]
        print(f"Articles with High Integration Potential (>0.7): {high_potential}")
        
        # Articles with quantum relevance
        cursor.execute("SELECT COUNT(*) FROM articles WHERE quantum_relevance > 0.8")
        quantum_relevant = cursor.fetchone()[0]
        print(f"Articles with High Quantum Relevance (>0.8): {quantum_relevant}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error accessing research articles database: {e}")
    
    # 2. AGENTIC EXPLORATIONS STATUS
    print("\n🤖 AGENTIC EXPLORATIONS STATUS")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect(explorations_db)
        cursor = conn.cursor()
        
        # Total explorations
        cursor.execute("SELECT COUNT(*) FROM agentic_explorations")
        total_explorations = cursor.fetchone()[0]
        print(f"Total Agentic Explorations: {total_explorations}")
        
        # Explorations with analysis
        cursor.execute("SELECT COUNT(*) FROM agentic_explorations WHERE f2_optimization_analysis IS NOT NULL OR ml_improvement_analysis IS NOT NULL OR cpu_enhancement_analysis IS NOT NULL OR weighting_analysis IS NOT NULL")
        analyzed_explorations = cursor.fetchone()[0]
        print(f"Explorations with AI Analysis: {analyzed_explorations}")
        
        # Explorations by field
        cursor.execute("SELECT field, COUNT(*) as count FROM agentic_explorations GROUP BY field ORDER BY count DESC")
        exploration_fields = cursor.fetchall()
        print("\nExplorations by Field:")
        for field, count in exploration_fields:
            print(f"  {field}: {count} explorations")
        
        # High priority opportunities
        cursor.execute("SELECT COUNT(*) FROM agentic_explorations WHERE implementation_priority = 'high'")
        high_priority = cursor.fetchone()[0]
        print(f"High Priority Implementation Opportunities: {high_priority}")
        
        # High impact opportunities
        cursor.execute("SELECT COUNT(*) FROM agentic_explorations WHERE potential_impact = 'high'")
        high_impact = cursor.fetchone()[0]
        print(f"High Impact Opportunities: {high_impact}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error accessing agentic explorations database: {e}")
    
    # 3. DIGITAL LEDGER INTEGRATION STATUS
    print("\n📜 DIGITAL LEDGER INTEGRATION STATUS")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect(ledger_db)
        cursor = conn.cursor()
        
        # Total ledger entries
        cursor.execute("SELECT COUNT(*) FROM ledger_entries")
        total_entries = cursor.fetchone()[0]
        print(f"Total Ledger Entries: {total_entries}")
        
        # Entries by contribution type
        cursor.execute("SELECT contribution_type, COUNT(*) as count FROM ledger_entries GROUP BY contribution_type ORDER BY count DESC")
        contribution_types = cursor.fetchall()
        print("\nLedger Entries by Contribution Type:")
        for contrib_type, count in contribution_types:
            print(f"  {contrib_type}: {count} entries")
        
        # Total credits distributed
        cursor.execute("SELECT SUM(credit_amount) FROM ledger_entries")
        total_credits = cursor.fetchone()[0] or 0
        print(f"Total Credits Distributed: {total_credits:.2f}")
        
        # Attribution chains
        cursor.execute("SELECT COUNT(*) FROM attribution_chains")
        total_chains = cursor.fetchone()[0]
        print(f"Total Attribution Chains: {total_chains}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error accessing digital ledger database: {e}")
    
    # 4. INTEGRATION GAPS ANALYSIS
    print("\n🔍 INTEGRATION GAPS ANALYSIS")
    print("-" * 50)
    
    # Check for articles without explorations
    try:
        conn_research = sqlite3.connect(research_db)
        conn_explorations = sqlite3.connect(explorations_db)
        
        cursor_research = conn_research.cursor()
        cursor_explorations = conn_explorations.cursor()
        
        # Get all article IDs
        cursor_research.execute("SELECT article_id FROM articles")
        article_ids = [row[0] for row in cursor_research.fetchall()]
        
        # Get explored article IDs
        cursor_explorations.execute("SELECT paper_id FROM agentic_explorations")
        explored_ids = [row[0] for row in cursor_explorations.fetchall()]
        
        # Find unexplored articles
        unexplored = set(article_ids) - set(explored_ids)
        print(f"Articles Without Agentic Exploration: {len(unexplored)}")
        
        if len(unexplored) > 0:
            print("Sample Unexplored Articles:")
            cursor_research.execute("SELECT title, field FROM articles WHERE article_id IN (?) LIMIT 5", (list(unexplored)[:5],))
            for title, field in cursor_research.fetchall():
                print(f"  {title[:50]}... ({field})")
        
        conn_research.close()
        conn_explorations.close()
        
    except Exception as e:
        print(f"Error analyzing integration gaps: {e}")
    
    # 5. RECENT ACTIVITY ANALYSIS
    print("\n⏰ RECENT ACTIVITY ANALYSIS")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect(research_db)
        cursor = conn.cursor()
        
        # Articles from last 24 hours
        cursor.execute("SELECT COUNT(*) FROM articles WHERE scraped_timestamp > datetime('now', '-1 day')")
        recent_24h = cursor.fetchone()[0]
        print(f"Articles Scraped in Last 24 Hours: {recent_24h}")
        
        # Articles from last week
        cursor.execute("SELECT COUNT(*) FROM articles WHERE scraped_timestamp > datetime('now', '-7 days')")
        recent_week = cursor.fetchone()[0]
        print(f"Articles Scraped in Last Week: {recent_week}")
        
        # Articles from last month
        cursor.execute("SELECT COUNT(*) FROM articles WHERE scraped_timestamp > datetime('now', '-30 days')")
        recent_month = cursor.fetchone()[0]
        print(f"Articles Scraped in Last Month: {recent_month}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error analyzing recent activity: {e}")
    
    # 6. INTEGRATION OPPORTUNITIES
    print("\n🎯 INTEGRATION OPPORTUNITIES")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect(research_db)
        cursor = conn.cursor()
        
        # High potential articles by field
        cursor.execute("""
            SELECT field, COUNT(*) as count, AVG(koba42_integration_potential) as avg_potential 
            FROM articles 
            WHERE koba42_integration_potential > 0.7 
            GROUP BY field 
            ORDER BY avg_potential DESC 
            LIMIT 5
        """)
        high_potential_fields = cursor.fetchall()
        print("High Integration Potential by Field:")
        for field, count, avg_potential in high_potential_fields:
            print(f"  {field}: {count} articles (avg potential: {avg_potential:.3f})")
        
        # Quantum-relevant articles
        cursor.execute("""
            SELECT title, quantum_relevance, koba42_integration_potential 
            FROM articles 
            WHERE quantum_relevance > 0.8 
            ORDER BY koba42_integration_potential DESC 
            LIMIT 5
        """)
        quantum_articles = cursor.fetchall()
        print("\nTop Quantum-Relevant Articles:")
        for title, quantum_rel, integration_pot in quantum_articles:
            print(f"  {title[:60]}... (Q: {quantum_rel:.2f}, I: {integration_pot:.2f})")
        
        conn.close()
        
    except Exception as e:
        print(f"Error analyzing integration opportunities: {e}")
    
    # 7. RECOMMENDATIONS
    print("\n💡 INTEGRATION RECOMMENDATIONS")
    print("-" * 50)
    
    recommendations = [
        "1. Run fresh scraping for latest research (last scraping: 2025-08-29)",
        "2. Process unexplored articles with agentic exploration",
        "3. Integrate high-potential articles into digital ledger",
        "4. Focus on quantum-relevant articles for immediate integration",
        "5. Expand scraping to include more recent sources",
        "6. Implement automated integration pipeline for new articles",
        "7. Create summary reports for high-impact discoveries",
        "8. Prioritize articles with high KOBA42 integration potential"
    ]
    
    for recommendation in recommendations:
        print(recommendation)
    
    # 8. SUMMARY STATISTICS
    print("\n📈 SUMMARY STATISTICS")
    print("-" * 50)
    
    try:
        conn_research = sqlite3.connect(research_db)
        conn_explorations = sqlite3.connect(explorations_db)
        conn_ledger = sqlite3.connect(ledger_db)
        
        # Research articles
        cursor = conn_research.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles WHERE koba42_integration_potential > 0.7")
        high_potential = cursor.fetchone()[0]
        
        # Explorations
        cursor = conn_explorations.cursor()
        cursor.execute("SELECT COUNT(*) FROM agentic_explorations")
        total_explorations = cursor.fetchone()[0]
        
        # Ledger
        cursor = conn_ledger.cursor()
        cursor.execute("SELECT COUNT(*) FROM ledger_entries")
        total_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(credit_amount) FROM ledger_entries")
        total_credits = cursor.fetchone()[0] or 0
        
        print(f"Total Research Articles: {total_articles}")
        print(f"High Integration Potential: {high_potential}")
        print(f"Agentic Explorations: {total_explorations}")
        print(f"Digital Ledger Entries: {total_entries}")
        print(f"Total Credits Distributed: {total_credits:.2f}")
        print(f"Integration Coverage: {(total_explorations/total_articles*100):.1f}%")
        
        conn_research.close()
        conn_explorations.close()
        conn_ledger.close()
        
    except Exception as e:
        print(f"Error generating summary statistics: {e}")
    
    print("\n🎉 SCRAPED DATA INTEGRATION STATUS REPORT COMPLETE")
    print("=" * 70)
    print("The system has comprehensive data but could benefit from:")
    print("• Fresh scraping for latest research")
    print("• Processing of unexplored articles")
    print("• Enhanced integration of high-potential discoveries")
    print("• Automated pipeline for continuous integration")
    print("=" * 70)

if __name__ == "__main__":
    generate_scraped_data_integration_status()
