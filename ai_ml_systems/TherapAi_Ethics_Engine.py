#!/usr/bin/env python3
"""
TherapAi Ethics Engine: Core Modules & Logic
Author: Brad Wallace (ArtWithHeart) – Koba42
Description: Ethical reasoning and guardrails engine for decentralized therapeutic AI

This module provides a comprehensive ethical framework for AI systems, including:
- Real-time ethical decision making
- Emotional state assessment
- Harm detection and prevention
- Transparent logging and audit trails
- RESTful API interface
- Docker containerization support
"""

import json
import logging
import requests
import numpy as np
from enum import Enum
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import uuid
import hashlib
import sqlite3
from pathlib import Path
import docker
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('therapai_ethics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Ethical Directives Enum ---
class EthicalDirective(Enum):
    """Core ethical principles for AI therapeutic systems"""
    NON_HARM = "Do no harm"
    AUTONOMY = "Respect user agency"
    TRANSPARENCY = "Maintain explainability"
    RESTORATION = "Enable healing if harm occurs"
    INCLUSIVITY = "Respect neurodivergent, marginalized identities"
    PRIVACY = "Protect user confidentiality"
    BENEFICENCE = "Promote user well-being"
    JUSTICE = "Ensure fair treatment"

# --- Emotional States Enum ---
class EmotionalState(Enum):
    """Categorized emotional states for assessment"""
    CRISIS = "crisis"
    DISTRESS = "distress"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    ELEVATED = "elevated"

# --- Risk Levels ---
class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DecisionContext:
    """Context for ethical decision making"""
    user_id: str
    input_text: str
    emotional_score: float
    flagged_keywords: List[str]
    timestamp: datetime
    session_id: str
    user_demographics: Optional[Dict] = None
    conversation_history: Optional[List[str]] = None
    risk_factors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())

@dataclass
class EthicalAssessment:
    """Result of ethical evaluation"""
    violations: List[EthicalDirective]
    risk_level: RiskLevel
    recommended_actions: List[str]
    confidence_score: float
    reasoning: str
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class SentimentAnalyzer:
    """Real-time sentiment analysis using external APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache = {}
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple approaches"""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Basic sentiment analysis (can be replaced with API calls)
        sentiment_score = self._basic_sentiment_analysis(text)
        emotional_state = self._categorize_emotion(sentiment_score)
        
        result = {
            'sentiment_score': sentiment_score,
            'emotional_state': emotional_state.value,
            'confidence': 0.85,
            'keywords': self._extract_keywords(text)
        }
        
        # Cache result
        self.cache[text_hash] = result
        return result
    
    def _basic_sentiment_analysis(self, text: str) -> float:
        """Basic sentiment analysis implementation"""
        positive_words = {'good', 'great', 'happy', 'joy', 'love', 'wonderful', 'amazing', 'excellent'}
        negative_words = {'bad', 'terrible', 'sad', 'angry', 'hate', 'awful', 'horrible', 'depressed'}
        crisis_words = {'suicide', 'kill', 'die', 'end', 'hopeless', 'worthless', 'alone'}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        crisis_count = sum(1 for word in words if word in crisis_words)
        
        if crisis_count > 0:
            return -0.9
        elif len(words) == 0:
            return 0.0
        else:
            return (positive_count - negative_count) / len(words)
    
    def _categorize_emotion(self, score: float) -> EmotionalState:
        """Categorize emotional state based on sentiment score"""
        if score <= -0.7:
            return EmotionalState.CRISIS
        elif score <= -0.3:
            return EmotionalState.DISTRESS
        elif score <= 0.3:
            return EmotionalState.NEUTRAL
        elif score <= 0.7:
            return EmotionalState.POSITIVE
        else:
            return EmotionalState.ELEVATED
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        keywords = []
        text_lower = text.lower()
        
        # Crisis indicators
        crisis_indicators = ['suicide', 'kill', 'die', 'end', 'hopeless', 'worthless', 'alone', 'pain']
        for indicator in crisis_indicators:
            if indicator in text_lower:
                keywords.append(indicator)
        
        return keywords

class EthicsEngine:
    """Main ethics engine for therapeutic AI systems"""
    
    def __init__(self, db_path: str = "ethics_engine.db"):
        self.violation_log = []
        self.directives = [directive for directive in EthicalDirective]
        self.sentiment_analyzer = SentimentAnalyzer()
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ethical_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                timestamp TEXT,
                input_text TEXT,
                emotional_score REAL,
                violations TEXT,
                risk_level TEXT,
                recommended_actions TEXT,
                confidence_score REAL,
                reasoning TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def evaluate_input(self, context: DecisionContext) -> EthicalAssessment:
        """Evaluate input for ethical violations and risk assessment"""
        violations = []
        risk_factors = []
        recommended_actions = []
        
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(context.input_text)
        emotional_score = sentiment_result['sentiment_score']
        
        # Update context with analyzed data
        context.emotional_score = emotional_score
        context.flagged_keywords = sentiment_result['keywords']
        
        # Harm Detection
        if emotional_score < -0.75 or any(word in context.input_text.lower() for word in context.flagged_keywords):
            violations.append(EthicalDirective.NON_HARM)
            risk_factors.append("High emotional distress detected")
            recommended_actions.append("Immediate crisis intervention recommended")
        
        # Privacy and Confidentiality
        if self._contains_pii(context.input_text):
            violations.append(EthicalDirective.PRIVACY)
            risk_factors.append("Personal identifiable information detected")
            recommended_actions.append("Sanitize input and ensure data protection")
        
        # Autonomy Assessment
        if self._indicates_coercion(context.input_text):
            violations.append(EthicalDirective.AUTONOMY)
            risk_factors.append("Potential coercion or manipulation detected")
            recommended_actions.append("Ensure user consent and agency")
        
        # Determine risk level
        risk_level = self._assess_risk_level(violations, emotional_score, len(context.flagged_keywords))
        
        # Generate reasoning
        reasoning = self._generate_reasoning(violations, risk_level, context)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(context, violations)
        
        # Create assessment
        assessment = EthicalAssessment(
            violations=violations,
            risk_level=risk_level,
            recommended_actions=recommended_actions,
            confidence_score=confidence_score,
            reasoning=reasoning
        )
        
        # Log decision
        self.log_decision(context, assessment)
        
        return assessment
    
    def _contains_pii(self, text: str) -> bool:
        """Check for personally identifiable information"""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        import re
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _indicates_coercion(self, text: str) -> bool:
        """Check for signs of coercion or manipulation"""
        coercion_indicators = [
            'must', 'have to', 'forced', 'coerced', 'manipulated',
            'no choice', 'no option', 'trapped'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in coercion_indicators)
    
    def _assess_risk_level(self, violations: List[EthicalDirective], emotional_score: float, keyword_count: int) -> RiskLevel:
        """Assess overall risk level"""
        risk_score = 0
        
        # Violation-based risk
        risk_score += len(violations) * 0.3
        
        # Emotional distress risk
        if emotional_score < -0.8:
            risk_score += 0.4
        elif emotional_score < -0.5:
            risk_score += 0.2
        
        # Keyword-based risk
        risk_score += keyword_count * 0.1
        
        # Determine risk level
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_reasoning(self, violations: List[EthicalDirective], risk_level: RiskLevel, context: DecisionContext) -> str:
        """Generate human-readable reasoning for the assessment"""
        reasoning_parts = []
        
        if violations:
            reasoning_parts.append(f"Detected {len(violations)} ethical violation(s): {', '.join([v.name for v in violations])}")
        
        if context.emotional_score < -0.5:
            reasoning_parts.append(f"High emotional distress detected (score: {context.emotional_score:.2f})")
        
        if context.flagged_keywords:
            reasoning_parts.append(f"Concerning keywords identified: {', '.join(context.flagged_keywords)}")
        
        reasoning_parts.append(f"Overall risk level: {risk_level.value}")
        
        return ". ".join(reasoning_parts)
    
    def _calculate_confidence(self, context: DecisionContext, violations: List[EthicalDirective]) -> float:
        """Calculate confidence score for the assessment"""
        base_confidence = 0.7
        
        # Increase confidence with more context
        if context.conversation_history:
            base_confidence += 0.1
        
        if context.user_demographics:
            base_confidence += 0.1
        
        # Increase confidence with clear violations
        if violations:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def log_decision(self, context: DecisionContext, assessment: EthicalAssessment):
        """Log ethical decision to database and memory"""
        # Memory log
        log_entry = {
            'session_id': context.session_id,
            'timestamp': context.timestamp.isoformat(),
            'user_id': context.user_id,
            'input_text': context.input_text,
            'emotional_score': context.emotional_score,
            'violations': [v.name for v in assessment.violations],
            'risk_level': assessment.risk_level.value,
            'recommended_actions': assessment.recommended_actions,
            'confidence_score': assessment.confidence_score,
            'reasoning': assessment.reasoning
        }
        self.violation_log.append(log_entry)
        
        # Database log
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ethical_decisions 
            (session_id, user_id, timestamp, input_text, emotional_score, violations, 
             risk_level, recommended_actions, confidence_score, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            context.session_id,
            context.user_id,
            context.timestamp.isoformat(),
            context.input_text,
            context.emotional_score,
            json.dumps([v.name for v in assessment.violations]),
            assessment.risk_level.value,
            json.dumps(assessment.recommended_actions),
            assessment.confidence_score,
            assessment.reasoning
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Ethical decision logged for session {context.session_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about ethical decisions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total decisions
        cursor.execute("SELECT COUNT(*) FROM ethical_decisions")
        total_decisions = cursor.fetchone()[0]
        
        # Risk level distribution
        cursor.execute("SELECT risk_level, COUNT(*) FROM ethical_decisions GROUP BY risk_level")
        risk_distribution = dict(cursor.fetchall())
        
        # Recent violations
        cursor.execute("""
            SELECT violations FROM ethical_decisions 
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        recent_violations = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_decisions': total_decisions,
            'risk_distribution': risk_distribution,
            'recent_violations_24h': len(recent_violations)
        }

# --- Math Module: Reflective Index Score (RIS) ---
def reflective_index_score(emotional_vector: List[float], keyword_hits: int, 
                          violation_count: int, risk_factors: List[str]) -> float:
    """
    Calculates a composite ethical score.
    
    Args:
        emotional_vector: List of emotional tone values (range -1 to 1)
        keyword_hits: Number of flagged terms present
        violation_count: Number of ethical violations
        risk_factors: List of identified risk factors
    
    Returns:
        float: Normalized RIS score between -1 and 1
    """
    if not emotional_vector:
        emotional_vector = [0.0]
    
    # Base emotional score
    base = sum(emotional_vector) / len(emotional_vector)
    
    # Penalties
    keyword_penalty = keyword_hits * 0.05
    violation_penalty = violation_count * 0.1
    risk_penalty = len(risk_factors) * 0.02
    
    # Calculate RIS
    ris_score = base - keyword_penalty - violation_penalty - risk_penalty
    
    # Normalize to [-1, 1]
    return max(-1.0, min(ris_score, 1.0))

# --- RESTful API Interface ---
app = Flask(__name__)
CORS(app)

# Global ethics engine instance
ethics_engine = EthicsEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/evaluate', methods=['POST'])
def evaluate_input():
    """Evaluate input for ethical concerns"""
    try:
        data = request.get_json()
        
        # Create decision context
        context = DecisionContext(
            user_id=data.get('user_id', 'anonymous'),
            input_text=data.get('input_text', ''),
            emotional_score=data.get('emotional_score', 0.0),
            flagged_keywords=data.get('flagged_keywords', []),
            user_demographics=data.get('user_demographics'),
            conversation_history=data.get('conversation_history'),
            risk_factors=data.get('risk_factors')
        )
        
        # Evaluate
        assessment = ethics_engine.evaluate_input(context)
        
        # Return result
        return jsonify({
            'success': True,
            'assessment': {
                'violations': [v.name for v in assessment.violations],
                'risk_level': assessment.risk_level.value,
                'recommended_actions': assessment.recommended_actions,
                'confidence_score': assessment.confidence_score,
                'reasoning': assessment.reasoning,
                'timestamp': assessment.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get ethics engine statistics"""
    try:
        stats = ethics_engine.get_statistics()
        return jsonify({'success': True, 'statistics': stats})
    except Exception as e:
        logger.error(f"Error in statistics endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --- Docker Support ---
def create_dockerfile():
    """Create Dockerfile for containerization"""
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY TherapAi_Ethics_Engine.py .

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    # Create requirements.txt
    requirements = [
        'flask',
        'flask-cors',
        'numpy',
        'requests',
        'docker'
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    logger.info("Dockerfile and requirements.txt created")

# --- Usage Example ---
if __name__ == '__main__':
    # Create Docker files
    create_dockerfile()
    
    # Example usage
    engine = EthicsEngine()
    
    # Test case 1: Crisis situation
    context1 = DecisionContext(
        user_id="user123",
        input_text="I feel like everything is falling apart and I can't go on anymore.",
        emotional_score=-0.9,
        flagged_keywords=["suicide", "hopeless", "end"]
    )
    
    assessment1 = engine.evaluate_input(context1)
    print("=== Crisis Assessment ===")
    print(f"Violations: {[v.name for v in assessment1.violations]}")
    print(f"Risk Level: {assessment1.risk_level.value}")
    print(f"Recommended Actions: {assessment1.recommended_actions}")
    print(f"Reasoning: {assessment1.reasoning}")
    
    # Test case 2: Normal conversation
    context2 = DecisionContext(
        user_id="user456",
        input_text="I had a good day today and feel optimistic about tomorrow.",
        emotional_score=0.7,
        flagged_keywords=[]
    )
    
    assessment2 = engine.evaluate_input(context2)
    print("\n=== Normal Assessment ===")
    print(f"Violations: {[v.name for v in assessment2.violations]}")
    print(f"Risk Level: {assessment2.risk_level.value}")
    print(f"Confidence: {assessment2.confidence_score:.2f}")
    
    # Calculate RIS scores
    ris1 = reflective_index_score([-0.9], 3, len(assessment1.violations), assessment1.recommended_actions)
    ris2 = reflective_index_score([0.7], 0, len(assessment2.violations), assessment2.recommended_actions)
    
    print(f"\n=== RIS Scores ===")
    print(f"Crisis RIS: {ris1:.3f}")
    print(f"Normal RIS: {ris2:.3f}")
    
    # Get statistics
    stats = engine.get_statistics()
    print(f"\n=== Statistics ===")
    print(f"Total Decisions: {stats['total_decisions']}")
    print(f"Risk Distribution: {stats['risk_distribution']}")
    
    # Start API server
    print("\n=== Starting API Server ===")
    print("API available at: http://localhost:5000")
    print("Endpoints:")
    print("  POST /evaluate - Evaluate input for ethical concerns")
    print("  GET /statistics - Get engine statistics")
    print("  GET /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
