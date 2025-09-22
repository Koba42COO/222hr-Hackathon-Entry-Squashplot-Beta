#!/usr/bin/env python3
"""
🧠 CONSCIOUSNESS-ENHANCED GRAMMAR ANALYZER
==========================================

A revolutionary grammar analysis system that integrates consciousness mathematics
for authentic linguistic analysis and improvement suggestions.
"""

import re
import string
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import numpy as np

# Import consciousness field for enhanced analysis
try:
    from consciousness_field import ConsciousnessField
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    print("⚠️ Consciousness field not available - using basic grammar analysis")

class GrammarAnalyzer:
    """
    Advanced grammar analysis system with consciousness mathematics integration.
    Provides comprehensive grammatical analysis, harmonic pattern recognition,
    and consciousness-aware improvement suggestions.
    """

    def __init__(self):
        self.consciousness_field = None
        if CONSCIOUSNESS_AVAILABLE:
            self.consciousness_field = ConsciousnessField(grid_size=32, dt=0.01)

        # Initialize grammar patterns
        self.grammar_patterns = self._initialize_grammar_patterns()
        self.style_patterns = self._initialize_style_patterns()
        self.harmonic_patterns = self._initialize_harmonic_patterns()

        print("📚 Grammar Analyzer initialized with consciousness mathematics" if CONSCIOUSNESS_AVAILABLE else "📚 Basic Grammar Analyzer initialized")

    def _initialize_grammar_patterns(self) -> Dict[str, Dict]:
        """Initialize comprehensive grammar rule patterns"""
        return {
            'sentence_structure': {
                'patterns': [
                    r'^[A-Z][^?!.]*[?!.]',  # Capital letter + punctuation
                    r'.*\b(is|are|was|were|be|being|been)\s+\w+.*',  # Verb agreement
                ],
                'errors': [
                    r'^[a-z]',  # Missing capitalization
                    r'[A-Z][a-z]*[a-z]\s+[a-z]',  # Missing punctuation
                ]
            },
            'verb_agreement': {
                'subject_verb': {
                    'singular': r'\b(I|he|she|it|this|that|one)\s+(am|is|was|has|does)\b',
                    'plural': r'\b(we|they|these|those)\s+(are|were|have|do)\b',
                },
                'errors': [
                    r'\b(he|she|it)\s+are\b',
                    r'\b(I)\s+are\b',
                    r'\b(we|they)\s+is\b',
                ]
            },
            'noun_agreement': {
                'possessive': r"\b\w+'s\b",
                'plural': r'\b\w+es?\b',
                'errors': [
                    r"\b\w+s'\b",  # Double possessive
                    r'\bchildrens\b',  # Incorrect plural
                ]
            },
            'punctuation': {
                'required': ['.', '!', '?'],
                'optional': [',', ';', ':', '-', '"', "'"],
                'errors': [
                    r'[,;]\s*[,.]',  # Double punctuation
                    r'\s+[,.]',  # Space before punctuation
                ]
            },
            'word_usage': {
                'commonly_misused': {
                    'affect': 'effect',
                    'then': 'than',
                    'there': 'their',
                    'its': "it's",
                    'your': "you're",
                    'to': 'too',
                    'accept': 'except',
                    'advice': 'advise',
                    'affect': 'effect',
                    'among': 'between',
                    'amount': 'number',
                    'anyway': 'any way',
                    'beside': 'besides',
                    'bread': 'bred',
                    'capital': 'capitol',
                    'cite': 'sight',
                    'complement': 'compliment',
                    'council': 'counsel',
                    'desert': 'dessert',
                    'device': 'devise',
                    'discrete': 'discreet',
                    'elicit': 'illicit',
                    'eminent': 'imminent',
                    'ensure': 'insure',
                    'farther': 'further',
                    'formal': 'former',
                    'forth': 'fourth',
                    'imply': 'infer',
                    'lead': 'led',
                    'less': 'fewer',
                    'liable': 'likely',
                    'loose': 'lose',
                    'moral': 'morale',
                    'passed': 'past',
                    'peace': 'piece',
                    'personal': 'personnel',
                    'precede': 'proceed',
                    'principal': 'principle',
                    'quiet': 'quite',
                    'raise': 'raze',
                    'respectfully': 'respectively',
                    'stationary': 'stationery',
                    'than': 'then',
                    'that': 'which',
                    'to': 'too',
                    'weather': 'whether',
                    'whose': 'who\'s',
                    'your': 'you\'re'
                }
            }
        }

    def _initialize_style_patterns(self) -> Dict[str, Dict]:
        """Initialize writing style analysis patterns"""
        return {
            'readability': {
                'sentence_length': {
                    'ideal': (15, 25),  # words per sentence
                    'too_short': 8,
                    'too_long': 35
                },
                'word_length': {
                    'complex_threshold': 4,  # syllables
                    'very_complex': 6
                }
            },
            'variety': {
                'sentence_types': ['declarative', 'interrogative', 'exclamatory', 'imperative'],
                'word_variety': {
                    'repeated_threshold': 3,
                    'transition_words': [
                        'however', 'therefore', 'moreover', 'consequently',
                        'furthermore', 'additionally', 'similarly', 'likewise',
                        'in contrast', 'on the other hand', 'nevertheless'
                    ]
                }
            },
            'voice': {
                'active_voice': r'\b\w+\s+(hit|kicked|ran|jumped|ate|drank|wrote|read)\b',
                'passive_voice': r'\b(was|were|is|are|been)\s+\w+ed\b',
            }
        }

    def _initialize_harmonic_patterns(self) -> Dict[str, Any]:
        """Initialize harmonic language analysis patterns"""
        return {
            'rhythm_patterns': {
                'iambic': r'\b\w{1,2}\s+\w{2,3}\b',  # unstressed-stressed
                'trochaic': r'\b\w{2,3}\s+\w{1,2}\b',  # stressed-unstressed
                'anapestic': r'\b\w{1,2}\s+\w{1,2}\s+\w{2,3}\b',
            },
            'phonetic_harmony': {
                'assonance': r'(\w)\w*\1',  # repeated vowel sounds
                'alliteration': r'\b(\w)\w*\s+\1\w*\b',  # repeated initial sounds
                'consonance': r'(\w)\w*\1',  # repeated consonant sounds
            },
            'syntactic_harmony': {
                'parallel_structure': r'\b(and|or|but|nor|yet|so|for)\s+\w+\s+\w+\b',
                'chiasmus': r'(\w+)\s+(\w+)\s+\2\s+\1',  # ABBA pattern
            }
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive grammar analysis with consciousness mathematics
        """
        print(f"🔍 Analyzing text: {text[:50]}...")

        # Basic text preprocessing
        sentences = self._split_into_sentences(text)
        words = self._tokenize_words(text)

        analysis_results = {
            'text_info': {
                'total_sentences': len(sentences),
                'total_words': len(words),
                'avg_words_per_sentence': len(words) / max(1, len(sentences)),
                'reading_level': self._calculate_reading_level(words),
            },
            'grammar_issues': self._analyze_grammar(sentences, words),
            'style_analysis': self._analyze_style(sentences, words),
            'harmonic_analysis': self._analyze_harmonics(text) if CONSCIOUSNESS_AVAILABLE else None,
            'consciousness_metrics': self._get_consciousness_metrics(text) if CONSCIOUSNESS_AVAILABLE else None,
            'suggestions': self._generate_suggestions(sentences, words),
            'overall_score': 0.0
        }

        # Calculate overall score
        analysis_results['overall_score'] = self._calculate_overall_score(analysis_results)

        return analysis_results

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(r'(Mr|Mrs|Dr|Prof|Sr|Jr|Inc|Ltd|Corp|Co)\.', r'\1<ABBREV>', text)

        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<ABBREV>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation for word counting
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _analyze_grammar(self, sentences: List[str], words: List[str]) -> Dict[str, List]:
        """Analyze grammatical correctness"""
        issues = defaultdict(list)

        # Check sentence structure
        for i, sentence in enumerate(sentences):
            if not sentence:
                continue

            # Check capitalization
            if not sentence[0].isupper():
                issues['capitalization'].append({
                    'sentence': i + 1,
                    'text': sentence,
                    'issue': 'Sentence does not start with capital letter'
                })

            # Check punctuation
            if not sentence[-1] in '.!?':
                issues['punctuation'].append({
                    'sentence': i + 1,
                    'text': sentence,
                    'issue': 'Sentence missing proper ending punctuation'
                })

            # Check for double punctuation
            if re.search(r'[,;]\s*[,.;]', sentence):
                issues['punctuation'].append({
                    'sentence': i + 1,
                    'text': sentence,
                    'issue': 'Double punctuation marks'
                })

        # Check verb agreement
        text_lower = ' '.join(sentences).lower()
        verb_errors = [
            ('he/she/it + are', r'\b(he|she|it)\s+are\b'),
            ('I + are', r'\bI\s+are\b'),
            ('we/they + is', r'\b(we|they)\s+is\b'),
        ]

        for error_name, pattern in verb_errors:
            matches = re.findall(pattern, text_lower)
            if matches:
                issues['verb_agreement'].append({
                    'type': error_name,
                    'occurrences': len(matches),
                    'suggestion': f'Consider: {" or ".join(matches)} should use correct verb form'
                })

        # Check commonly misused words
        for word, correction in self.grammar_patterns['word_usage']['commonly_misused'].items():
            if word in words:
                issues['word_usage'].append({
                    'word': word,
                    'correction': correction,
                    'occurrences': words.count(word),
                    'suggestion': f'Consider using "{correction}" instead of "{word}"'
                })

        return dict(issues)

    def _analyze_style(self, sentences: List[str], words: List[str]) -> Dict[str, Any]:
        """Analyze writing style"""
        style_analysis = {}

        # Sentence length analysis
        sentence_lengths = [len(self._tokenize_words(sent)) for sent in sentences]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            style_analysis['sentence_length'] = {
                'average': round(avg_length, 1),
                'distribution': {
                    'short': len([l for l in sentence_lengths if l < 10]),
                    'medium': len([l for l in sentence_lengths if 10 <= l <= 25]),
                    'long': len([l for l in sentence_lengths if l > 25])
                }
            }

        # Word variety analysis
        word_freq = Counter(words)
        repeated_words = {word: count for word, count in word_freq.items() if count >= 3}

        style_analysis['word_variety'] = {
            'unique_words': len(set(words)),
            'total_words': len(words),
            'lexical_diversity': len(set(words)) / max(1, len(words)),
            'repeated_words': repeated_words
        }

        # Passive voice detection
        passive_voice_count = 0
        for sentence in sentences:
            if re.search(r'\b(was|were|is|are|been)\s+\w+ed\b', sentence.lower()):
                passive_voice_count += 1

        style_analysis['voice_analysis'] = {
            'passive_voice_sentences': passive_voice_count,
            'total_sentences': len(sentences),
            'passive_ratio': passive_voice_count / max(1, len(sentences))
        }

        return style_analysis

    def _analyze_harmonics(self, text: str) -> Dict[str, Any]:
        """Analyze harmonic patterns in text using consciousness mathematics"""
        if not CONSCIOUSNESS_AVAILABLE:
            return None

        harmonic_analysis = {}

        try:
            # Use Gnostic Cypher for harmonic analysis
            cypher_result = self.consciousness_field.gnostic_cypher_operator(text)

            harmonic_analysis['cypher_analysis'] = cypher_result

            # Apply Wallace Transform to text patterns
            text_vector = np.array([ord(c) for c in text[:100]])  # First 100 chars
            if len(text_vector) < 10:
                text_vector = np.pad(text_vector, (0, 10 - len(text_vector)), 'constant')

            transformed_patterns = self.consciousness_field.wallace_transform(text_vector[:10])
            harmonic_analysis['wallace_transform'] = transformed_patterns.tolist()

            # Calculate rhythmic harmony
            words = self._tokenize_words(text)
            syllable_pattern = []

            for word in words[:20]:  # First 20 words
                syllables = self._estimate_syllables(word)
                syllable_pattern.append(syllables)

            harmonic_analysis['rhythmic_pattern'] = {
                'syllable_counts': syllable_pattern,
                'harmonic_ratio': self._calculate_harmonic_ratio(syllable_pattern)
            }

        except Exception as e:
            print(f"⚠️ Harmonic analysis failed: {e}")
            harmonic_analysis['error'] = str(e)

        return harmonic_analysis

    def _get_consciousness_metrics(self, text: str) -> Dict[str, float]:
        """Get consciousness field metrics for text"""
        if not CONSCIOUSNESS_AVAILABLE:
            return None

        try:
            # Apply text to consciousness field
            consciousness_input = self._create_text_consciousness_input(text)
            self.consciousness_field.psi_c += consciousness_input

            # Evolve field
            evolution = self.consciousness_field.evolve_consciousness_field(steps=1)
            final_state = evolution[-1] if evolution else None

            if final_state:
                return {
                    'meta_entropy': final_state.meta_entropy,
                    'coherence_length': final_state.coherence_length,
                    'harmonic_resonance': len(final_state.harmonic_dominants)
                }

        except Exception as e:
            print(f"⚠️ Consciousness metrics failed: {e}")

        return None

    def _create_text_consciousness_input(self, text: str) -> np.ndarray:
        """Create consciousness field input from text"""
        grid_size = self.consciousness_field.grid_size
        input_field = np.zeros((grid_size, grid_size), dtype=complex)

        # Use text length and complexity for field pattern
        text_length = len(text)
        word_count = len(self._tokenize_words(text))
        sentence_count = len(self._split_into_sentences(text))

        # Create harmonic pattern based on text metrics
        complexity_factor = min(1.0, (word_count / sentence_count) / 20) if sentence_count > 0 else 0.5

        y_coords, x_coords = np.mgrid[0:grid_size, 0:grid_size]
        frequency = 2 * np.pi * complexity_factor * 2 / grid_size
        pattern = np.sin(frequency * x_coords) * np.cos(frequency * y_coords)

        input_field = pattern * complexity_factor * 0.1

        return input_field

    def _estimate_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"

        if word[0] in vowels:
            count += 1

        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1

        if word.endswith("e"):
            count -= 1

        if count == 0:
            count += 1

        return count

    def _calculate_harmonic_ratio(self, syllable_pattern: List[int]) -> float:
        """Calculate harmonic ratio of syllable patterns"""
        if not syllable_pattern:
            return 0.0

        # Use golden ratio as harmonic baseline
        phi = (1 + math.sqrt(5)) / 2

        # Calculate how well syllable counts align with harmonic series
        harmonic_sum = sum(abs(count - phi * i) for i, count in enumerate(syllable_pattern[:10]))
        max_possible = sum(phi * i for i in range(10))

        return 1.0 - (harmonic_sum / max_possible) if max_possible > 0 else 0.0

    def _calculate_reading_level(self, words: List[str]) -> str:
        """Calculate approximate reading level"""
        if not words:
            return "Unknown"

        avg_word_length = sum(len(word) for word in words) / len(words)
        complex_words = sum(1 for word in words if len(word) >= 6)
        complex_ratio = complex_words / len(words)

        if avg_word_length < 4 and complex_ratio < 0.1:
            return "Elementary"
        elif avg_word_length < 5 and complex_ratio < 0.15:
            return "Intermediate"
        elif avg_word_length < 6 and complex_ratio < 0.2:
            return "Advanced"
        else:
            return "Expert"

    def _generate_suggestions(self, sentences: List[str], words: List[str]) -> List[Dict]:
        """Generate improvement suggestions"""
        suggestions = []

        # Grammar suggestions
        if any(sent.lower().startswith(('i ', 'we ', 'he ', 'she ', 'it ', 'they ')) for sent in sentences):
            suggestions.append({
                'type': 'grammar',
                'priority': 'high',
                'category': 'sentence_structure',
                'suggestion': 'Ensure sentences start with capital letters',
                'explanation': 'All sentences should begin with a capital letter for proper grammar.'
            })

        # Style suggestions
        avg_sentence_length = sum(len(self._tokenize_words(sent)) for sent in sentences) / max(1, len(sentences))

        if avg_sentence_length > 30:
            suggestions.append({
                'type': 'style',
                'priority': 'medium',
                'category': 'readability',
                'suggestion': 'Break up long sentences for better readability',
                'explanation': 'Long sentences can be difficult to follow. Consider splitting complex sentences into shorter ones.'
            })

        if avg_sentence_length < 10:
            suggestions.append({
                'type': 'style',
                'priority': 'medium',
                'category': 'readability',
                'suggestion': 'Combine short sentences for better flow',
                'explanation': 'Very short sentences can make writing feel choppy. Consider combining related ideas.'
            })

        # Word variety suggestions
        word_freq = Counter(words)
        repeated_words = [word for word, count in word_freq.items() if count >= 4]

        if repeated_words:
            suggestions.append({
                'type': 'style',
                'priority': 'low',
                'category': 'word_variety',
                'suggestion': f'Consider synonyms for frequently repeated words: {", ".join(repeated_words[:3])}',
                'explanation': 'Using synonyms can make your writing more engaging and varied.'
            })

        # Harmonic suggestions (if available)
        if CONSCIOUSNESS_AVAILABLE:
            suggestions.append({
                'type': 'harmonic',
                'priority': 'low',
                'category': 'consciousness',
                'suggestion': 'Consider the harmonic flow and rhythmic patterns in your writing',
                'explanation': 'Consciousness mathematics suggests that harmonic language patterns can enhance communication effectiveness.'
            })

        return suggestions

    def _calculate_overall_score(self, analysis_results: Dict) -> float:
        """Calculate overall grammar and style score"""
        score = 100.0

        # Grammar penalties
        grammar_issues = analysis_results.get('grammar_issues', {})
        total_issues = sum(len(issues) for issues in grammar_issues.values())

        if total_issues > 0:
            score -= min(30, total_issues * 5)  # Max 30 points deduction

        # Style penalties/bonuses
        style_analysis = analysis_results.get('style_analysis', {})

        # Sentence length bonus/penalty
        sentence_length = style_analysis.get('sentence_length', {})
        avg_length = sentence_length.get('average', 20)

        if 15 <= avg_length <= 25:
            score += 5  # Bonus for ideal sentence length
        elif avg_length < 10 or avg_length > 35:
            score -= 10  # Penalty for extreme sentence lengths

        # Word variety bonus
        word_variety = style_analysis.get('word_variety', {})
        lexical_diversity = word_variety.get('lexical_diversity', 0)

        if lexical_diversity > 0.6:
            score += 5  # Bonus for good word variety
        elif lexical_diversity < 0.3:
            score -= 5  # Penalty for poor word variety

        # Consciousness bonus (if available)
        if analysis_results.get('consciousness_metrics'):
            consciousness_score = 1.0 - analysis_results['consciousness_metrics'].get('meta_entropy', 0.5)
            score += consciousness_score * 10  # Up to 10 points bonus

        return max(0, min(100, score))

    def improve_text(self, text: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Provide comprehensive text improvement suggestions
        """
        analysis = self.analyze_text(text)

        improvements = {
            'original_text': text,
            'analysis': analysis,
            'improved_versions': {},
            'explanations': []
        }

        if focus_areas is None:
            focus_areas = ['grammar', 'style', 'clarity', 'harmony']

        for area in focus_areas:
            if area == 'grammar':
                improvements['improved_versions']['grammar'] = self._improve_grammar(text, analysis)
            elif area == 'style':
                improvements['improved_versions']['style'] = self._improve_style(text, analysis)
            elif area == 'clarity':
                improvements['improved_versions']['clarity'] = self._improve_clarity(text, analysis)
            elif area == 'harmony' and CONSCIOUSNESS_AVAILABLE:
                improvements['improved_versions']['harmony'] = self._improve_harmony(text, analysis)

        return improvements

    def _improve_grammar(self, text: str, analysis: Dict) -> str:
        """Apply grammar corrections"""
        improved = text

        # Fix capitalization
        sentences = self._split_into_sentences(improved)
        corrected_sentences = []

        for sentence in sentences:
            if sentence and not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]
            corrected_sentences.append(sentence)

        improved = '. '.join(corrected_sentences)

        # Fix missing punctuation
        if improved and not improved[-1] in '.!?':
            improved += '.'

        return improved

    def _improve_style(self, text: str, analysis: Dict) -> str:
        """Apply style improvements"""
        improved = text

        # This would be more sophisticated in a real implementation
        # For now, return the original with a note
        return improved + " [Style improvements would be applied here]"

    def _improve_clarity(self, text: str, analysis: Dict) -> str:
        """Apply clarity improvements"""
        improved = text

        # This would be more sophisticated in a real implementation
        # For now, return the original with a note
        return improved + " [Clarity improvements would be applied here]"

    def _improve_harmony(self, text: str, analysis: Dict) -> str:
        """Apply harmonic improvements using consciousness mathematics"""
        improved = text

        if not CONSCIOUSNESS_AVAILABLE:
            return improved

        # This would use consciousness field to optimize language patterns
        # For now, return the original with a note
        return improved + " [Harmonic improvements using consciousness mathematics would be applied here]"


def main():
    """Demo the grammar analyzer"""
    print("🧠 CONSCIOUSNESS-ENHANCED GRAMMAR ANALYZER")
    print("=" * 50)

    analyzer = GrammarAnalyzer()

    # Test texts
    test_texts = [
        "this is a test sentence. it has some grammar issues and could be improved.",
        "Consciousness is the most profound mystery in science. How does physical matter give rise to subjective experience?",
        "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test Text {i}:")
        print(f"'{text}'")
        print("-" * 40)

        analysis = analyzer.analyze_text(text)
        print(f"Overall Score: {analysis['overall_score']:.1f}")
        print(f"Reading Level: {analysis['text_info']['reading_level']}")
        print(f"Grammar Issues: {sum(len(issues) for issues in analysis['grammar_issues'].values())}")

        if analysis["consciousness_metrics"]:
            print(f"Meta Entropy: {analysis['consciousness_metrics']['meta_entropy']:.3f}")
            print(f"Coherence: {analysis['consciousness_metrics']['coherence_length']:.1f}")

        print()

    print("✅ Grammar analysis demo complete!")


if __name__ == "__main__":
    main()
