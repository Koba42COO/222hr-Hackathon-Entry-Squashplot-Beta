#!/usr/bin/env python3
"""
KOBA42 DATA EXPLORER
====================
Data Exploration and Analysis Tool for Research Scraping Results
===============================================================

Features:
1. Database Analysis and Statistics
2. Filtering Process Analysis
3. Research Trends Discovery
4. KOBA42 Integration Insights
5. Data Quality Assessment
6. Future Development Recommendations
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KOBA42DataExplorer:
    """Data exploration and analysis tool for research scraping results."""
    
    def __init__(self, db_path: str = "research_data/research_articles.db"):
        self.db_path = db_path
        self.conn = None
        self.connect_database()
    
    def connect_database(self):
        """Connect to the research database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"✅ Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database summary."""
        summary = {
            'database_info': {},
            'sessions': {},
            'batches': {},
            'articles': {},
            'filtering_analysis': {},
            'recommendations': []
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Database info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            summary['database_info']['tables'] = tables
            
            # Sessions analysis
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            summary['sessions']['total_sessions'] = total_sessions
            
            if total_sessions > 0:
                cursor.execute("""
                    SELECT session_id, start_time, end_time, total_articles_scraped, 
                           total_articles_stored, sources_processed, status
                    FROM sessions ORDER BY start_time DESC LIMIT 5
                """)
                recent_sessions = cursor.fetchall()
                summary['sessions']['recent_sessions'] = [
                    {
                        'session_id': row[0],
                        'start_time': row[1],
                        'end_time': row[2],
                        'articles_scraped': row[3],
                        'articles_stored': row[4],
                        'sources_processed': json.loads(row[5]) if row[5] else [],
                        'status': row[6]
                    }
                    for row in recent_sessions
                ]
            
            # Batches analysis
            cursor.execute("SELECT COUNT(*) FROM batches")
            total_batches = cursor.fetchone()[0]
            summary['batches']['total_batches'] = total_batches
            
            if total_batches > 0:
                cursor.execute("""
                    SELECT source, COUNT(*) as batch_count, 
                           SUM(articles_scraped) as total_scraped,
                           SUM(articles_stored) as total_stored
                    FROM batches GROUP BY source
                """)
                batch_stats = cursor.fetchall()
                summary['batches']['source_statistics'] = [
                    {
                        'source': row[0],
                        'batch_count': row[1],
                        'total_scraped': row[2],
                        'total_stored': row[3],
                        'storage_rate': (row[3] / row[2] * 100) if row[2] > 0 else 0
                    }
                    for row in batch_stats
                ]
            
            # Articles analysis
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            summary['articles']['total_articles'] = total_articles
            
            if total_articles > 0:
                # Source distribution
                cursor.execute("SELECT source, COUNT(*) FROM articles GROUP BY source")
                source_dist = dict(cursor.fetchall())
                summary['articles']['source_distribution'] = source_dist
                
                # Field distribution
                cursor.execute("SELECT field, COUNT(*) FROM articles GROUP BY field")
                field_dist = dict(cursor.fetchall())
                summary['articles']['field_distribution'] = field_dist
                
                # Relevance analysis
                cursor.execute("""
                    SELECT AVG(quantum_relevance) as avg_quantum,
                           AVG(technology_relevance) as avg_tech,
                           AVG(research_impact) as avg_impact,
                           AVG(relevance_score) as avg_relevance,
                           AVG(koba42_integration_potential) as avg_koba42
                    FROM articles
                """)
                relevance_stats = cursor.fetchone()
                summary['articles']['relevance_statistics'] = {
                    'average_quantum_relevance': relevance_stats[0],
                    'average_technology_relevance': relevance_stats[1],
                    'average_research_impact': relevance_stats[2],
                    'average_relevance_score': relevance_stats[3],
                    'average_koba42_potential': relevance_stats[4]
                }
                
                # High-value articles
                cursor.execute("""
                    SELECT COUNT(*) FROM articles 
                    WHERE relevance_score >= 7.0
                """)
                high_relevance = cursor.fetchone()[0]
                summary['articles']['high_relevance_count'] = high_relevance
                
                cursor.execute("""
                    SELECT COUNT(*) FROM articles 
                    WHERE koba42_integration_potential >= 7.0
                """)
                high_koba42 = cursor.fetchone()[0]
                summary['articles']['high_koba42_potential_count'] = high_koba42
            
            # Filtering analysis
            if total_batches > 0:
                cursor.execute("""
                    SELECT SUM(articles_scraped) as total_scraped,
                           SUM(articles_stored) as total_stored
                    FROM batches
                """)
                filtering_stats = cursor.fetchone()
                total_scraped = filtering_stats[0] or 0
                total_stored = filtering_stats[1] or 0
                
                summary['filtering_analysis'] = {
                    'total_articles_scraped': total_scraped,
                    'total_articles_stored': total_stored,
                    'filtering_rate': ((total_scraped - total_stored) / total_scraped * 100) if total_scraped > 0 else 0,
                    'storage_rate': (total_stored / total_scraped * 100) if total_scraped > 0 else 0
                }
            
            # Generate recommendations
            summary['recommendations'] = self._generate_recommendations(summary)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing database: {e}")
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on data analysis."""
        recommendations = []
        
        # Check if articles were scraped but not stored
        if summary.get('filtering_analysis', {}).get('total_articles_scraped', 0) > 0:
            storage_rate = summary['filtering_analysis']['storage_rate']
            
            if storage_rate == 0:
                recommendations.append("🔍 No articles met relevance criteria - consider lowering minimum relevance threshold")
                recommendations.append("📊 Review filtering criteria to ensure important research isn't being excluded")
                recommendations.append("🎯 Focus on specific sources or fields that align better with KOBA42 requirements")
            
            elif storage_rate < 20:
                recommendations.append("📈 Low storage rate detected - consider adjusting relevance scoring algorithm")
                recommendations.append("🔧 Review quantum and technology relevance keywords for better coverage")
                recommendations.append("📋 Analyze scraped but filtered articles to understand exclusion patterns")
        
        # Check session performance
        if summary.get('sessions', {}).get('total_sessions', 0) > 0:
            recent_session = summary['sessions']['recent_sessions'][0]
            if recent_session['articles_stored'] == 0:
                recommendations.append("🚀 Consider running scraping with more sources or different time periods")
                recommendations.append("⚡ Implement incremental scraping to build article database over time")
                recommendations.append("🎯 Focus on high-impact sources like Nature for better quality articles")
        
        # General recommendations
        recommendations.extend([
            "💾 Database structure is ready for future research data collection",
            "📊 Implement regular scraping sessions to build comprehensive research database",
            "🔬 Focus on quantum physics, materials science, and technology articles",
            "🎯 Prioritize articles with high KOBA42 integration potential",
            "📈 Monitor storage rates and adjust filtering criteria as needed"
        ])
        
        return recommendations
    
    def analyze_filtering_process(self) -> Dict[str, Any]:
        """Analyze the filtering process and identify potential improvements."""
        analysis = {
            'filtering_criteria': {},
            'potential_issues': [],
            'improvement_suggestions': [],
            'sample_filtered_articles': []
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Get batch-level filtering statistics
            cursor.execute("""
                SELECT batch_id, source, articles_scraped, articles_stored,
                       (articles_scraped - articles_stored) as filtered_out
                FROM batches 
                WHERE articles_scraped > 0
                ORDER BY articles_scraped DESC
            """)
            
            batch_analysis = cursor.fetchall()
            
            if batch_analysis:
                analysis['filtering_criteria'] = {
                    'minimum_relevance_score': 5.0,
                    'minimum_quantum_relevance': 5.0,
                    'minimum_technology_relevance': 5.0,
                    'relevance_calculation': 'Average of quantum, technology, and research impact scores'
                }
                
                # Analyze filtering patterns
                total_scraped = sum(row[2] for row in batch_analysis)
                total_stored = sum(row[3] for row in batch_analysis)
                total_filtered = sum(row[4] for row in batch_analysis)
                
                analysis['filtering_statistics'] = {
                    'total_scraped': total_scraped,
                    'total_stored': total_stored,
                    'total_filtered': total_filtered,
                    'filtering_rate': (total_filtered / total_scraped * 100) if total_scraped > 0 else 0,
                    'storage_rate': (total_stored / total_scraped * 100) if total_scraped > 0 else 0
                }
                
                # Identify potential issues
                if total_stored == 0:
                    analysis['potential_issues'].append("All scraped articles were filtered out")
                    analysis['potential_issues'].append("Filtering criteria may be too strict")
                    analysis['potential_issues'].append("Source content may not match expected patterns")
                
                # Generate improvement suggestions
                if total_stored == 0:
                    analysis['improvement_suggestions'].extend([
                        "Lower minimum relevance score from 5.0 to 3.0",
                        "Reduce quantum/technology relevance requirements",
                        "Expand keyword coverage for better categorization",
                        "Implement adaptive filtering based on source quality",
                        "Add manual review process for borderline articles"
                    ])
                
                analysis['improvement_suggestions'].extend([
                    "Implement source-specific relevance thresholds",
                    "Add trending topic detection for emerging research",
                    "Create whitelist for high-impact research institutions",
                    "Implement machine learning-based relevance scoring",
                    "Add cross-reference with existing KOBA42 research database"
                ])
            
        except Exception as e:
            logger.error(f"❌ Error analyzing filtering process: {e}")
        
        return analysis
    
    def get_research_trends(self) -> Dict[str, Any]:
        """Analyze research trends from stored data."""
        trends = {
            'source_performance': {},
            'field_distribution': {},
            'relevance_trends': {},
            'koba42_opportunities': []
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Check if we have any articles to analyze
            cursor.execute("SELECT COUNT(*) FROM articles")
            article_count = cursor.fetchone()[0]
            
            if article_count == 0:
                trends['status'] = "No articles available for trend analysis"
                trends['recommendations'] = [
                    "Run scraping with lower relevance thresholds",
                    "Focus on specific high-impact sources",
                    "Implement broader keyword coverage",
                    "Consider manual article curation for initial dataset"
                ]
                return trends
            
            # Source performance analysis
            cursor.execute("""
                SELECT source, COUNT(*) as article_count,
                       AVG(quantum_relevance) as avg_quantum,
                       AVG(technology_relevance) as avg_tech,
                       AVG(koba42_integration_potential) as avg_koba42
                FROM articles GROUP BY source
            """)
            
            source_performance = cursor.fetchall()
            trends['source_performance'] = [
                {
                    'source': row[0],
                    'article_count': row[1],
                    'avg_quantum_relevance': row[2],
                    'avg_technology_relevance': row[3],
                    'avg_koba42_potential': row[4]
                }
                for row in source_performance
            ]
            
            # Field distribution analysis
            cursor.execute("""
                SELECT field, COUNT(*) as count,
                       AVG(relevance_score) as avg_relevance,
                       AVG(koba42_integration_potential) as avg_koba42
                FROM articles GROUP BY field
            """)
            
            field_data = cursor.fetchall()
            trends['field_distribution'] = [
                {
                    'field': row[0],
                    'count': row[1],
                    'avg_relevance': row[2],
                    'avg_koba42_potential': row[3]
                }
                for row in field_data
            ]
            
            # KOBA42 opportunities
            cursor.execute("""
                SELECT title, source, field, koba42_integration_potential, key_insights
                FROM articles 
                WHERE koba42_integration_potential >= 7.0
                ORDER BY koba42_integration_potential DESC
                LIMIT 10
            """)
            
            high_potential = cursor.fetchall()
            trends['koba42_opportunities'] = [
                {
                    'title': row[0],
                    'source': row[1],
                    'field': row[2],
                    'koba42_potential': row[3],
                    'key_insights': json.loads(row[4]) if row[4] else []
                }
                for row in high_potential
            ]
            
        except Exception as e:
            logger.error(f"❌ Error analyzing research trends: {e}")
        
        return trends
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive data exploration report."""
        logger.info("📊 Generating comprehensive data exploration report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_summary': self.get_database_summary(),
            'filtering_analysis': self.analyze_filtering_process(),
            'research_trends': self.get_research_trends(),
            'data_quality_assessment': self._assess_data_quality(),
            'future_development_insights': self._generate_future_insights()
        }
        
        return report
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of collected data."""
        quality_assessment = {
            'completeness': {},
            'accuracy': {},
            'relevance': {},
            'overall_score': 0
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Check data completeness
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            
            if total_articles > 0:
                # Check for missing data
                cursor.execute("""
                    SELECT COUNT(*) FROM articles WHERE title IS NULL OR title = ''
                """)
                missing_titles = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM articles WHERE summary IS NULL OR summary = ''
                """)
                missing_summaries = cursor.fetchone()[0]
                
                quality_assessment['completeness'] = {
                    'total_articles': total_articles,
                    'missing_titles': missing_titles,
                    'missing_summaries': missing_summaries,
                    'completeness_rate': ((total_articles - missing_titles - missing_summaries) / total_articles * 100) if total_articles > 0 else 0
                }
                
                # Check relevance accuracy
                cursor.execute("""
                    SELECT AVG(relevance_score) FROM articles
                """)
                avg_relevance = cursor.fetchone()[0]
                
                quality_assessment['relevance'] = {
                    'average_relevance_score': avg_relevance,
                    'high_relevance_articles': 0,
                    'relevance_distribution': {}
                }
                
                if avg_relevance:
                    cursor.execute("""
                        SELECT COUNT(*) FROM articles WHERE relevance_score >= 7.0
                    """)
                    high_relevance = cursor.fetchone()[0]
                    quality_assessment['relevance']['high_relevance_articles'] = high_relevance
                
                # Calculate overall quality score
                completeness_score = quality_assessment['completeness']['completeness_rate']
                relevance_score = (avg_relevance / 10 * 100) if avg_relevance else 0
                
                quality_assessment['overall_score'] = (completeness_score + relevance_score) / 2
            else:
                quality_assessment['overall_score'] = 0
                quality_assessment['status'] = "No articles available for quality assessment"
        
        except Exception as e:
            logger.error(f"❌ Error assessing data quality: {e}")
        
        return quality_assessment
    
    def _generate_future_insights(self) -> Dict[str, Any]:
        """Generate insights for future development."""
        insights = {
            'scraping_strategy': [],
            'data_management': [],
            'koba42_integration': [],
            'research_focus_areas': []
        }
        
        # Scraping strategy insights
        insights['scraping_strategy'].extend([
            "Implement incremental scraping to build database over time",
            "Focus on high-impact sources (Nature, Science, Physical Review Letters)",
            "Add source-specific relevance thresholds",
            "Implement trending topic detection",
            "Create research institution whitelist for quality filtering"
        ])
        
        # Data management insights
        insights['data_management'].extend([
            "Implement data versioning for research article updates",
            "Add citation tracking and impact factor integration",
            "Create research collaboration network analysis",
            "Implement automated duplicate detection",
            "Add research funding source tracking"
        ])
        
        # KOBA42 integration insights
        insights['koba42_integration'].extend([
            "Focus on quantum computing and materials science articles",
            "Prioritize articles with algorithm and optimization content",
            "Track emerging quantum technologies and applications",
            "Monitor quantum internet and communication developments",
            "Integrate with existing KOBA42 optimization frameworks"
        ])
        
        # Research focus areas
        insights['research_focus_areas'].extend([
            "Quantum computing and quantum algorithms",
            "Quantum materials and topological insulators",
            "Quantum internet and quantum communication",
            "Quantum machine learning and AI",
            "Quantum sensors and quantum metrology",
            "Quantum chemistry and molecular simulations",
            "Quantum cryptography and quantum security"
        ])
        
        return insights
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("✅ Database connection closed")

def demonstrate_data_exploration():
    """Demonstrate comprehensive data exploration."""
    logger.info("🔍 KOBA42 Data Explorer")
    logger.info("=" * 50)
    
    # Initialize explorer
    explorer = KOBA42DataExplorer()
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive data exploration report...")
    report = explorer.generate_comprehensive_report()
    
    # Display key findings
    print("\n📋 DATABASE SUMMARY")
    print("=" * 50)
    
    db_summary = report['database_summary']
    print(f"Total Sessions: {db_summary['sessions']['total_sessions']}")
    print(f"Total Batches: {db_summary['batches']['total_batches']}")
    print(f"Total Articles: {db_summary['articles']['total_articles']}")
    
    if db_summary['sessions']['recent_sessions']:
        recent = db_summary['sessions']['recent_sessions'][0]
        print(f"Latest Session: {recent['session_id']}")
        print(f"Articles Scraped: {recent['articles_scraped']}")
        print(f"Articles Stored: {recent['articles_stored']}")
        print(f"Sources: {', '.join(recent['sources_processed'])}")
    
    # Display filtering analysis
    print("\n🔍 FILTERING ANALYSIS")
    print("=" * 50)
    
    filtering = report['filtering_analysis']
    if 'filtering_statistics' in filtering:
        stats = filtering['filtering_statistics']
        print(f"Total Scraped: {stats['total_scraped']}")
        print(f"Total Stored: {stats['total_stored']}")
        print(f"Filtering Rate: {stats['filtering_rate']:.1f}%")
        print(f"Storage Rate: {stats['storage_rate']:.1f}%")
    
    if filtering['potential_issues']:
        print("\n⚠️ Potential Issues:")
        for issue in filtering['potential_issues']:
            print(f"  • {issue}")
    
    if filtering['improvement_suggestions']:
        print("\n💡 Improvement Suggestions:")
        for suggestion in filtering['improvement_suggestions'][:5]:
            print(f"  • {suggestion}")
    
    # Display recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = db_summary['recommendations']
    for i, rec in enumerate(recommendations[:8], 1):
        print(f"{i}. {rec}")
    
    # Display data quality assessment
    print("\n📈 DATA QUALITY ASSESSMENT")
    print("=" * 50)
    
    quality = report['data_quality_assessment']
    print(f"Overall Quality Score: {quality['overall_score']:.1f}/100")
    
    if 'completeness' in quality and quality['completeness']:
        completeness = quality['completeness']
        if 'completeness_rate' in completeness:
            print(f"Completeness Rate: {completeness['completeness_rate']:.1f}%")
        else:
            print("Completeness Rate: N/A (no articles to assess)")
    else:
        print("Completeness Rate: N/A (no articles to assess)")
    
    # Display future insights
    print("\n🚀 FUTURE DEVELOPMENT INSIGHTS")
    print("=" * 50)
    
    insights = report['future_development_insights']
    print("Scraping Strategy:")
    for insight in insights['scraping_strategy'][:3]:
        print(f"  • {insight}")
    
    print("\nKOBA42 Integration:")
    for insight in insights['koba42_integration'][:3]:
        print(f"  • {insight}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'data_exploration_report_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Detailed report saved to {report_file}")
    
    # Close explorer
    explorer.close()
    
    return report

if __name__ == "__main__":
    # Run data exploration demonstration
    report = demonstrate_data_exploration()
    
    print(f"\n🎉 Data exploration completed!")
    print(f"📊 Comprehensive analysis of research scraping data")
    print(f"💡 Insights generated for future development")
    print(f"🔬 Ready for KOBA42 integration optimization")
