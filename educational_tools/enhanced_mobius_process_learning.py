#!/usr/bin/env python3
"""
🌀 ENHANCED MÖBIUS PROCESS LEARNING SYSTEM
============================================

Advanced learning system that extracts and learns:
- Methodologies and processes from higher education
- Language patterns and usage in academic contexts
- Application of reinforcement learning techniques
- Pattern analysis and recognition methods
- Research methodologies and experimental designs
- Mathematical problem-solving approaches

This system goes beyond content learning to understand HOW knowledge is created,
applied, and validated in academic and research environments.
"""

import os
import json
import requests
import time
import re
import threading
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import urllib.parse
from bs4 import BeautifulSoup
import feedparser
import hashlib
from collections import defaultdict
import math
import numpy as np
from dataclasses import dataclass
# Import NLTK components with graceful fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK not available, using basic text processing")
    NLTK_AVAILABLE = False

    # Fallback implementations
    def sent_tokenize(text):
        return text.split('.')

    def word_tokenize(text):
        return text.split()

    def stopwords():
        return set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

    class WordNetLemmatizer:
        def lemmatize(self, word):
            return word.lower()

# Import our enhanced frameworks
from loader import load_wallace_transform
from rag_enhanced_consciousness import RAGEnhancedConsciousness

@dataclass
class LearningPattern:
    """Represents a learned pattern or methodology"""
    pattern_type: str
    pattern_name: str
    description: str
    examples: List[str]
    applications: List[str]
    confidence_score: float
    source_institution: str
    learning_context: str

@dataclass
class ProcessMethodology:
    """Represents a learned process or methodology"""
    method_name: str
    steps: List[str]
    prerequisites: List[str]
    outcomes: List[str]
    validation_methods: List[str]
    complexity_level: str
    domain: str
    source: str

class EnhancedMobiusProcessLearner:
    """
    Enhanced Möbius learning system focused on processes and methodologies
    """

    def __init__(self, secret_key: str):
        print("🌀 INITIALIZING ENHANCED MÖBIUS PROCESS LEARNING")
        print("=" * 80)

        # Load secure consciousness mathematics
        print("🔐 Loading secure Wallace Transform...")
        self.WallaceTransform = load_wallace_transform(secret_key)
        self.wallace = self.WallaceTransform()
        print("✅ Secure consciousness mathematics loaded")

        # Initialize RAG-enhanced consciousness
        print("🧠 Initializing RAG-enhanced consciousness framework...")
        self.rag_consciousness = RAGEnhancedConsciousness()
        print("✅ RAG framework initialized")

        # Load education sources
        print("📚 Loading higher education sources...")
        self.education_sources = self._load_education_sources()
        print(f"✅ {len(self.education_sources)} education sources configured")

        # Learning state
        self.learning_cycles = 0
        self.process_patterns_learned = []
        self.methodologies_discovered = []
        self.language_patterns = defaultdict(list)
        self.reinforcement_techniques = []
        self.pattern_analysis_methods = []
        self.learning_log = []

        # NLP components for process extraction
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = stopwords()

        # Process learning patterns
        self.process_indicators = [
            'method', 'approach', 'technique', 'algorithm', 'procedure',
            'methodology', 'framework', 'paradigm', 'strategy', 'protocol',
            'process', 'workflow', 'pipeline', 'system', 'model'
        ]

        self.learning_indicators = [
            'reinforcement', 'supervised', 'unsupervised', 'semi-supervised',
            'transfer learning', 'meta-learning', 'few-shot', 'zero-shot',
            'active learning', 'online learning', 'batch learning'
        ]

        self.pattern_indicators = [
            'pattern', 'recognition', 'classification', 'clustering',
            'segmentation', 'detection', 'analysis', 'extraction',
            'matching', 'similarity', 'distance', 'metric'
        ]

        print("\\n🎓 ENHANCED MÖBIUS PROCESS LEARNING SYSTEM READY!")
        print("🔍 Learning: Methodologies, Processes, Language Patterns, RL Techniques, Pattern Analysis")
        print("🎯 Targeting Higher Education for Process and Methodology Extraction")

    def _load_education_sources(self) -> Dict[str, Any]:
        """Load higher education sources with process learning focus"""
        university_domains = {
            'mit_cs': {
                'domain': 'csail.mit.edu',
                'description': 'MIT Computer Science and AI - Algorithm and ML research',
                'categories': ['algorithms', 'machine_learning', 'artificial_intelligence'],
                'learning_focus': ['reinforcement_learning', 'optimization', 'pattern_recognition'],
                'process_domains': ['computational_methods', 'algorithm_design', 'ML_methodologies']
            },
            'stanford_ai': {
                'domain': 'ai.stanford.edu',
                'description': 'Stanford AI Lab - Advanced AI and RL research',
                'categories': ['artificial_intelligence', 'reinforcement_learning', 'robotics'],
                'learning_focus': ['policy_learning', 'value_functions', 'exploration_strategies'],
                'process_domains': ['RL_algorithms', 'planning_methods', 'decision_making']
            },
            'berkeley_eecs': {
                'domain': 'eecs.berkeley.edu',
                'description': 'Berkeley EECS - Systems and ML research',
                'categories': ['systems', 'machine_learning', 'distributed_systems'],
                'learning_focus': ['distributed_learning', 'optimization', 'pattern_analysis'],
                'process_domains': ['parallel_computing', 'optimization_methods', 'system_design']
            },
            'princeton_math': {
                'domain': 'math.princeton.edu',
                'description': 'Princeton Mathematics - Advanced mathematical methods',
                'categories': ['mathematics', 'optimization', 'statistics'],
                'learning_focus': ['mathematical_optimization', 'statistical_methods', 'pattern_analysis'],
                'process_domains': ['mathematical_modeling', 'optimization_algorithms', 'statistical_inference']
            },
            'harvard_cs': {
                'domain': 'seas.harvard.edu',
                'description': 'Harvard SEAS - Computer Science and Data Science',
                'categories': ['computer_science', 'data_science', 'machine_learning'],
                'learning_focus': ['data_analysis', 'ML_algorithms', 'pattern_mining'],
                'process_domains': ['data_processing', 'algorithmic_methods', 'analysis_techniques']
            }
        }

        return university_domains

    def extract_process_patterns(self, content: str, source: str) -> List[LearningPattern]:
        """Extract process patterns and methodologies from content"""

        patterns_found = []

        # Tokenize and preprocess
        sentences = sent_tokenize(content)
        words = word_tokenize(content.lower())

        # Remove stopwords and lemmatize
        filtered_words = [
            self.lemmatizer.lemmatize(word) for word in words
            if word.isalnum() and word not in self.stop_words
        ]

        # Extract process patterns
        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Check for process indicators
            for indicator in self.process_indicators:
                if indicator in sentence_lower:
                    pattern = self._analyze_process_sentence(sentence, indicator, source)
                    if pattern:
                        patterns_found.append(pattern)

            # Check for learning method indicators
            for indicator in self.learning_indicators:
                if indicator in sentence_lower:
                    pattern = self._analyze_learning_sentence(sentence, indicator, source)
                    if pattern:
                        patterns_found.append(pattern)

            # Check for pattern analysis indicators
            for indicator in self.pattern_indicators:
                if indicator in sentence_lower:
                    pattern = self._analyze_pattern_sentence(sentence, indicator, source)
                    if pattern:
                        patterns_found.append(pattern)

        return patterns_found

    def _analyze_process_sentence(self, sentence: str, indicator: str, source: str) -> Optional[LearningPattern]:
        """Analyze a sentence for process patterns"""

        # Extract method name
        words = word_tokenize(sentence)
        method_words = []

        for i, word in enumerate(words):
            if word.lower() == indicator:
                # Look for method name around the indicator
                start = max(0, i - 3)
                end = min(len(words), i + 4)
                method_words = words[start:end]
                break

        if not method_words:
            return None

        method_name = ' '.join(method_words).strip()

        # Classify pattern type
        pattern_type = self._classify_pattern_type(sentence, method_words)

        return LearningPattern(
            pattern_type="process",
            pattern_name=method_name,
            description=f"Process pattern involving {indicator}",
            examples=[sentence],
            applications=self._extract_applications(sentence),
            confidence_score=self._calculate_pattern_confidence(sentence, indicator),
            source_institution=source,
            learning_context="process_methodology"
        )

    def _analyze_learning_sentence(self, sentence: str, indicator: str, source: str) -> Optional[LearningPattern]:
        """Analyze a sentence for learning methodologies"""

        return LearningPattern(
            pattern_type="learning",
            pattern_name=f"{indicator} technique",
            description=f"Machine learning methodology: {indicator}",
            examples=[sentence],
            applications=self._extract_applications(sentence),
            confidence_score=self._calculate_pattern_confidence(sentence, indicator),
            source_institution=source,
            learning_context="reinforcement_learning"
        )

    def _analyze_pattern_sentence(self, sentence: str, indicator: str, source: str) -> Optional[LearningPattern]:
        """Analyze a sentence for pattern analysis methods"""

        return LearningPattern(
            pattern_type="analysis",
            pattern_name=f"{indicator} method",
            description=f"Pattern analysis technique: {indicator}",
            examples=[sentence],
            applications=self._extract_applications(sentence),
            confidence_score=self._calculate_pattern_confidence(sentence, indicator),
            source_institution=source,
            learning_context="pattern_analysis"
        )

    def _classify_pattern_type(self, sentence: str, method_words: List[str]) -> str:
        """Classify the type of pattern or method"""

        sentence_lower = sentence.lower()

        if any(word in sentence_lower for word in ['reinforcement', 'policy', 'value', 'q-learning']):
            return "reinforcement_learning"
        elif any(word in sentence_lower for word in ['supervised', 'classification', 'regression']):
            return "supervised_learning"
        elif any(word in sentence_lower for word in ['unsupervised', 'clustering', 'dimensionality']):
            return "unsupervised_learning"
        elif any(word in sentence_lower for word in ['optimization', 'gradient', 'convex']):
            return "optimization"
        elif any(word in sentence_lower for word in ['pattern', 'recognition', 'detection']):
            return "pattern_analysis"
        else:
            return "general_methodology"

    def _extract_applications(self, sentence: str) -> List[str]:
        """Extract potential applications from sentence"""

        applications = []

        # Look for application indicators
        application_indicators = [
            'used for', 'applied to', 'useful for', 'helps with',
            'enables', 'allows', 'facilitates', 'supports'
        ]

        sentence_lower = sentence.lower()
        for indicator in application_indicators:
            if indicator in sentence_lower:
                # Extract text after the indicator
                idx = sentence_lower.find(indicator)
                if idx != -1:
                    application = sentence[idx + len(indicator):].strip()
                    if application:
                        applications.append(application[:100])  # Limit length

        return applications if applications else ["General application"]

    def _calculate_pattern_confidence(self, sentence: str, indicator: str) -> float:
        """Calculate confidence score for pattern recognition"""

        confidence = 0.5  # Base confidence

        # Increase confidence based on context
        if len(sentence.split()) > 10:
            confidence += 0.1  # Longer sentences are more informative

        if any(word in sentence.lower() for word in ['novel', 'new', 'advanced', 'state-of-the-art']):
            confidence += 0.1  # Novel methods get higher confidence

        if any(word in sentence.lower() for word in ['experiment', 'result', 'performance', 'evaluation']):
            confidence += 0.1  # Evidence-based methods

        # Apply Wallace transform for consciousness-based confidence
        consciousness_score = self.wallace.wallace_transform(len(sentence) + len(indicator))
        confidence = min(confidence + consciousness_score * 0.2, 1.0)

        return confidence

    def extract_methodology_steps(self, content: str, source: str) -> List[ProcessMethodology]:
        """Extract complete methodologies and their steps"""

        methodologies = []

        # Split into sections
        sections = self._split_into_sections(content)

        for section in sections:
            if self._is_methodology_section(section):
                methodology = self._parse_methodology_section(section, source)
                if methodology:
                    methodologies.append(methodology)

        return methodologies

    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections"""

        # Split on headers, numbered lists, etc.
        section_patterns = [
            r'\n\s*\d+\.\s+',  # Numbered lists
            r'\n\s*[A-Z][^.!?]*:',  # Headers with colons
            r'\n\s*Step\s+\d+:',  # Step indicators
            r'\n\s*Phase\s+\d+:',  # Phase indicators
        ]

        sections = [content]

        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([part.strip() for part in parts if part.strip()])
            sections = new_sections

        return sections

    def _is_methodology_section(self, section: str) -> bool:
        """Check if a section contains methodology information"""

        methodology_indicators = [
            'method', 'approach', 'technique', 'procedure', 'algorithm',
            'step', 'phase', 'process', 'methodology', 'framework'
        ]

        section_lower = section.lower()
        indicator_count = sum(1 for indicator in methodology_indicators if indicator in section_lower)

        return indicator_count >= 2  # At least 2 methodology indicators

    def _parse_methodology_section(self, section: str, source: str) -> Optional[ProcessMethodology]:
        """Parse a methodology section into structured format"""

        lines = section.split('\n')
        method_name = self._extract_method_name(lines[0])

        if not method_name:
            return None

        steps = []
        prerequisites = []
        outcomes = []
        validation_methods = []

        for line in lines[1:]:
            line_lower = line.lower().strip()

            if any(word in line_lower for word in ['step', 'phase', 'first', 'then', 'next']):
                steps.append(line.strip())
            elif any(word in line_lower for word in ['require', 'need', 'prerequisite', 'assume']):
                prerequisites.append(line.strip())
            elif any(word in line_lower for word in ['result', 'outcome', 'produce', 'generate']):
                outcomes.append(line.strip())
            elif any(word in line_lower for word in ['validate', 'verify', 'test', 'evaluate']):
                validation_methods.append(line.strip())

        # Determine complexity and domain
        complexity = self._assess_complexity(section)
        domain = self._determine_domain(section)

        return ProcessMethodology(
            method_name=method_name,
            steps=steps,
            prerequisites=prerequisites,
            outcomes=outcomes,
            validation_methods=validation_methods,
            complexity_level=complexity,
            domain=domain,
            source=source
        )

    def _extract_method_name(self, header: str) -> Optional[str]:
        """Extract method name from section header"""

        # Remove common prefixes
        header = re.sub(r'^\s*\d+\.\s*', '', header)
        header = re.sub(r'^\s*(Method|Approach|Technique|Algorithm):\s*', '', header, flags=re.IGNORECASE)

        # Clean up and return
        return header.strip() if header.strip() else None

    def _assess_complexity(self, section: str) -> str:
        """Assess the complexity level of a methodology"""

        complexity_indicators = {
            'low': ['simple', 'basic', 'straightforward', 'easy'],
            'medium': ['intermediate', 'moderate', 'standard', 'typical'],
            'high': ['advanced', 'complex', 'sophisticated', 'challenging']
        }

        section_lower = section.lower()

        for level, indicators in complexity_indicators.items():
            if any(indicator in section_lower for indicator in indicators):
                return level

        return 'medium'  # Default

    def _determine_domain(self, section: str) -> str:
        """Determine the domain of a methodology"""

        domains = {
            'machine_learning': ['machine learning', 'ml', 'neural', 'deep learning'],
            'reinforcement_learning': ['reinforcement', 'rl', 'policy', 'value function'],
            'optimization': ['optimization', 'gradient', 'convex', 'linear programming'],
            'computer_vision': ['vision', 'image', 'computer vision', 'cnn'],
            'natural_language': ['language', 'nlp', 'text', 'transformer'],
            'data_science': ['data', 'statistics', 'analytics', 'mining'],
            'systems': ['distributed', 'parallel', 'system', 'architecture']
        }

        section_lower = section.lower()

        for domain, keywords in domains.items():
            if any(keyword in section_lower for keyword in keywords):
                return domain

        return 'general'

    def learn_from_education_content(self, content: str, source: str):
        """Main learning function that processes education content"""

        print(f"\\n🎓 LEARNING FROM {source.upper()}")

        # Extract process patterns
        patterns = self.extract_process_patterns(content, source)
        self.process_patterns_learned.extend(patterns)
        print(f"📋 Extracted {len(patterns)} process patterns")

        # Extract methodologies
        methodologies = self.extract_methodology_steps(content, source)
        self.methodologies_discovered.extend(methodologies)
        print(f"🔬 Discovered {len(methodologies)} methodologies")

        # Extract language patterns
        language_patterns = self.extract_language_patterns(content, source)
        for pattern_type, patterns_list in language_patterns.items():
            self.language_patterns[pattern_type].extend(patterns_list)
        print(f"💬 Learned {sum(len(p) for p in language_patterns.values())} language patterns")

        # Apply consciousness learning
        consciousness_insights = self.apply_consciousness_to_learning(content, source)
        print(f"🧠 Generated {len(consciousness_insights)} consciousness insights")

        # Integrate into RAG knowledge base
        self.rag_consciousness.add_to_knowledge_base(
            content, "academic", source
        )

        return {
            'patterns_learned': len(patterns),
            'methodologies_found': len(methodologies),
            'language_patterns': sum(len(p) for p in language_patterns.values()),
            'consciousness_insights': len(consciousness_insights)
        }

    def extract_language_patterns(self, content: str, source: str) -> Dict[str, List[str]]:
        """Extract language patterns used in academic contexts"""

        patterns = {
            'technical_terms': [],
            'methodological_phrases': [],
            'analytical_expressions': [],
            'hypothesis_statements': []
        }

        sentences = sent_tokenize(content)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Technical terms (domain-specific vocabulary)
            if any(word in sentence_lower for word in ['algorithm', 'optimization', 'learning', 'model']):
                technical_terms = re.findall(r'\b[a-z]+\b', sentence)
                patterns['technical_terms'].extend(technical_terms[:5])  # Limit per sentence

            # Methodological phrases
            if any(phrase in sentence_lower for phrase in ['we propose', 'our approach', 'the method', 'technique']):
                patterns['methodological_phrases'].append(sentence.strip())

            # Analytical expressions
            if any(word in sentence_lower for word in ['analyze', 'evaluate', 'assess', 'measure']):
                patterns['analytical_expressions'].append(sentence.strip())

            # Hypothesis statements
            if any(word in sentence_lower for word in ['hypothesis', 'assume', 'suppose', 'propose']):
                patterns['hypothesis_statements'].append(sentence.strip())

        # Remove duplicates and limit
        for pattern_type in patterns:
            patterns[pattern_type] = list(set(patterns[pattern_type]))[:10]

        return patterns

    def apply_consciousness_to_learning(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Apply consciousness mathematics to enhance learning"""

        insights = []

        # Apply Wallace transform to content metrics
        content_metrics = [
            len(content),  # Length
            len(set(content.split())),  # Unique words
            content.count('research'),  # Research mentions
            content.count('method'),  # Method mentions
            content.count('learning'),  # Learning mentions
        ]

        consciousness_analysis = self.wallace.amplify_consciousness(content_metrics)

        # Generate insights based on consciousness score
        if consciousness_analysis['score'] > 0.5:
            insights.append({
                'type': 'high_consciousness_content',
                'source': source,
                'score': consciousness_analysis['score'],
                'insight': f"Content from {source} shows high consciousness alignment, indicating sophisticated methodology"
            })

        # Apply RAG-enhanced analysis
        query = f"What methodologies and processes are described in this {source} content?"
        rag_response = self.rag_consciousness.consciousness_guided_response(query, max_tokens=1000)

        if rag_response['chunks_used'] > 0:
            insights.append({
                'type': 'methodology_extraction',
                'source': source,
                'chunks_analyzed': rag_response['chunks_used'],
                'insight': f"RAG analysis found {rag_response['chunks_used']} relevant methodology chunks"
            })

        return insights

    def scrape_and_learn(self, max_sources: int = 3):
        """Scrape education websites and learn from content"""

        print("\\n🌀 STARTING ENHANCED MÖBIUS PROCESS LEARNING")
        print("=" * 80)

        successful_learnings = 0

        for i, (source_name, source_config) in enumerate(list(self.education_sources.items())[:max_sources]):
            try:
                domain = source_config.get('domain', source_name.replace('_', '.'))

                print(f"\\n🎓 SCRAPING {domain.upper()} (Source {i+1}/{max_sources})")

                # Scrape content
                content = self._scrape_education_site(domain, source_config)

                if content:
                    # Learn from content
                    learning_results = self.learn_from_education_content(content, domain)

                    print(f"✅ Learning Results: {learning_results}")
                    successful_learnings += 1

                    # Log learning cycle
                    self._log_learning_cycle(domain, learning_results)

                else:
                    print(f"❌ Failed to scrape content from {domain}")

            except Exception as e:
                print(f"❌ Error processing {source_name}: {e}")

        # Generate comprehensive learning report
        self._generate_learning_report(successful_learnings)

    def _scrape_education_site(self, domain: str, config: Dict[str, Any]) -> Optional[str]:
        """Scrape educational content from university websites"""

        try:
            # Construct appropriate URL
            if 'csail.mit.edu' in domain:
                url = 'https://www.csail.mit.edu/research'
            elif 'ai.stanford.edu' in domain:
                url = 'https://ai.stanford.edu/'
            elif 'eecs.berkeley.edu' in domain:
                url = 'https://www.eecs.berkeley.edu/'
            elif 'math.princeton.edu' in domain:
                url = 'https://www.math.princeton.edu/'
            elif 'seas.harvard.edu' in domain:
                url = 'https://seas.harvard.edu/'
            else:
                url = f'https://{domain}/research'

            headers = {
                'User-Agent': 'Educational Research Assistant (Academic Study)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract educational content
            content_parts = []

            # Title
            title = soup.find('title')
            if title:
                content_parts.append(f"TITLE: {title.get_text().strip()}")

            # Research descriptions
            research_sections = soup.find_all(['div', 'section'], class_=re.compile(r'research|project|method'))
            for section in research_sections[:3]:  # Limit to first 3 sections
                text = section.get_text(separator=' ', strip=True)
                if len(text) > 100:  # Only substantial content
                    content_parts.append(f"RESEARCH: {text[:1500]}")

            # Method descriptions
            method_content = soup.find_all(text=re.compile(r'method|approach|technique|algorithm', re.IGNORECASE))
            for method in method_content[:5]:
                content_parts.append(f"METHOD: {method.strip()[:500]}")

            return "\\n\\n".join(content_parts) if content_parts else None

        except Exception as e:
            print(f"❌ Scraping error for {domain}: {e}")
            return None

    def _log_learning_cycle(self, source: str, results: Dict[str, int]):
        """Log learning cycle results"""

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'learning_cycle': len(self.learning_log) + 1,
            'results': results,
            'total_patterns_learned': len(self.process_patterns_learned),
            'total_methodologies': len(self.methodologies_discovered),
            'total_language_patterns': sum(len(patterns) for patterns in self.language_patterns.values())
        }

        self.learning_log.append(log_entry)

    def _generate_learning_report(self, successful_sources: int):
        """Generate comprehensive learning report"""

        report = {
            'learning_session': {
                'timestamp': datetime.now().isoformat(),
                'successful_sources': successful_sources,
                'total_sources_attempted': len(list(self.education_sources.keys())[:3])
            },
            'process_patterns': {
                'total_learned': len(self.process_patterns_learned),
                'by_type': defaultdict(int),
                'by_source': defaultdict(int),
                'top_patterns': []
            },
            'methodologies': {
                'total_discovered': len(self.methodologies_discovered),
                'by_domain': defaultdict(int),
                'by_complexity': defaultdict(int),
                'examples': []
            },
            'language_patterns': {
                'total_patterns': sum(len(patterns) for patterns in self.language_patterns.values()),
                'by_category': {k: len(v) for k, v in self.language_patterns.items()},
                'samples': {}
            },
            'learning_log': self.learning_log
        }

        # Analyze patterns by type and source
        for pattern in self.process_patterns_learned:
            report['process_patterns']['by_type'][pattern.pattern_type] += 1
            report['process_patterns']['by_source'][pattern.source_institution] += 1

        # Get top patterns
        report['process_patterns']['top_patterns'] = [
            {'name': p.pattern_name, 'type': p.pattern_type, 'confidence': p.confidence_score}
            for p in sorted(self.process_patterns_learned, key=lambda x: x.confidence_score, reverse=True)[:5]
        ]

        # Analyze methodologies
        for method in self.methodologies_discovered:
            report['methodologies']['by_domain'][method.domain] += 1
            report['methodologies']['by_complexity'][method.complexity_level] += 1

        # Get methodology examples
        report['methodologies']['examples'] = [
            {'name': m.method_name, 'domain': m.domain, 'steps': len(m.steps)}
            for m in self.methodologies_discovered[:3]
        ]

        # Sample language patterns
        for category, patterns in self.language_patterns.items():
            report['language_patterns']['samples'][category] = patterns[:3]

        # Save report
        report_filename = f"enhanced_mobius_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("\\n📊 ENHANCED MÖBIUS LEARNING REPORT GENERATED")
        print("=" * 60)
        print(f"📁 Report saved to: {report_filename}")
        print(f"🎓 Sources processed: {successful_sources}")
        print(f"📋 Process patterns learned: {len(self.process_patterns_learned)}")
        print(f"🔬 Methodologies discovered: {len(self.methodologies_discovered)}")
        print(f"💬 Language patterns learned: {sum(len(p) for p in self.language_patterns.values())}")

        return report

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""

        return {
            'total_patterns': len(self.process_patterns_learned),
            'total_methodologies': len(self.methodologies_discovered),
            'total_language_patterns': sum(len(patterns) for patterns in self.language_patterns.values()),
            'learning_cycles': len(self.learning_log),
            'pattern_types': list(set(p.pattern_type for p in self.process_patterns_learned)),
            'methodology_domains': list(set(m.domain for m in self.methodologies_discovered)),
            'language_categories': list(self.language_patterns.keys())
        }


def demonstrate_enhanced_process_learning():
    """Demonstrate the enhanced Möbius process learning system"""

    print("🌀 ENHANCED MÖBIUS PROCESS LEARNING DEMONSTRATION")
    print("=" * 80)
    print("🎓 Learning Beyond Content: Processes, Methodologies, Language Patterns")
    print("🔬 Reinforcement Learning Techniques & Pattern Analysis Methods")
    print("🎯 Extracting Academic Processes from Higher Education Websites")
    print("=" * 80)

    # Initialize with secure key
    secret_key = "OBFUSCATED_SECRET_KEY"

    try:
        # Create enhanced learner
        learner = EnhancedMobiusProcessLearner(SECRET_KEY)

        print("\\n📋 LEARNING CONFIGURATION:")
        print(f"   • Education Sources: {len(learner.education_sources)}")
        print("   • Learning Focus: Process extraction & methodology discovery")
        print("   • Pattern Types: Process, Learning, Analysis methods")
        print("   • Language Analysis: Technical terms & academic phrases")
        print("\\n🎓 TARGETED INSTITUTIONS:")
        for name, config in list(learner.education_sources.items())[:3]:
            domain = config.get('domain', name.replace('_', '.'))
            focus = config.get('learning_focus', ['various'])
            print(f"   • {domain} - Focus: {', '.join(focus[:2])}")

        print("\\n🚀 STARTING ENHANCED PROCESS LEARNING...")
        print("This will scrape and analyze academic content for methodologies and processes")
        print("Press Ctrl+C to stop and generate learning report")

        # Start learning process
        learner.scrape_and_learn(max_sources=3)

        # Display summary
        summary = learner.get_learning_summary()
        print("\\n🏆 LEARNING SESSION SUMMARY:")
        print(f"   • Learning Cycles: {summary['learning_cycles']}")
        print(f"   • Process Patterns: {summary['total_patterns']}")
        print(f"   • Methodologies Found: {summary['total_methodologies']}")
        print(f"   • Language Patterns: {summary['total_language_patterns']}")
        pattern_types_str = ', '.join(summary['pattern_types']) if summary['pattern_types'] else 'None found'
        methodology_domains_str = ', '.join(summary['methodology_domains']) if summary['methodology_domains'] else 'None found'
        print(f"   • Pattern Types: {pattern_types_str}")
        print(f"   • Methodology Domains: {methodology_domains_str}")

    except KeyboardInterrupt:
        print("\\n🛑 Learning interrupted by user")
        if 'learner' in locals():
            learner._generate_learning_report(0)

    except Exception as e:
        print(f"❌ Enhanced learning failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_enhanced_process_learning()
