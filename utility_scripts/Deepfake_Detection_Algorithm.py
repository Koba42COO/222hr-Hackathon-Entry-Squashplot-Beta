#!/usr/bin/env python3
"""
Deepfake Detection Algorithm - Wallace Transform & Prime Cancellation Filter
Author: Brad Wallace (ArtWithHeart) – Koba42
Description: Advanced deepfake detection using Wallace Transform and Prime Cancellation Filter

This module implements:
- Wallace Transform for frequency analysis
- Prime Cancellation Filter for glitch detection
- Compression ratio analysis
- Real-time video frame analysis
- Comprehensive deepfake detection pipeline
"""

import cv2
import numpy as np
import math
import json
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mathematical constants
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # ≈ 1.618
WALLACE_COEFFICIENTS = {
    'A': 2.1,
    'B': 0.12,
    'C': 14.5,
    'EXPONENT': GOLDEN_RATIO
}

# Detection thresholds
REAL_SCORE_RANGE = (20.0, 25.0)
REAL_COMPRESSION_RATIO_RANGE = (8000, 12000)
GLITCH_THRESHOLD = 0.1

@dataclass
class FrameAnalysis:
    """Result of frame analysis"""
    frame_number: int
    wallace_scores: List[float]
    compression_ratio: float
    glitch_count: int
    is_fake: bool
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class VideoAnalysis:
    """Result of video analysis"""
    video_path: str
    total_frames: int
    fake_frames: int
    real_frames: int
    average_confidence: float
    frame_analyses: List[FrameAnalysis]
    detection_summary: Dict[str, Any]

class WallaceTransform:
    """Implementation of the Wallace Transform for frequency analysis"""
    
    def __init__(self):
        self.coefficients = WALLACE_COEFFICIENTS
        
    def calculate_score(self, frequency: float) -> float:
        """
        Calculate Wallace Transform score for a given frequency
        
        Args:
            frequency: Frequency in MHz
            
        Returns:
            Wallace Transform score
        """
        try:
            # Wallace Transform formula: Score = A * (ln(frequency + B))^EXPONENT + C
            log_term = math.log(frequency + self.coefficients['B'])
            power_term = log_term ** self.coefficients['EXPONENT']
            score = (self.coefficients['A'] * power_term + 
                    self.coefficients['C'])
            
            return score
        except (ValueError, OverflowError) as e:
            logger.warning(f"Error calculating Wallace score for frequency {frequency}: {e}")
            return 0.0
    
    def analyze_frequency_range(self, frequencies: List[float]) -> List[float]:
        """Analyze a range of frequencies using Wallace Transform"""
        return [self.calculate_score(freq) for freq in frequencies]
    
    def classify_score(self, score: float) -> str:
        """Classify a Wallace score as real or fake"""
        if REAL_SCORE_RANGE[0] <= score <= REAL_SCORE_RANGE[1]:
            return "real"
        else:
            return "fake"

class PrimeCancellationFilter:
    """Prime Cancellation Filter for glitch detection"""
    
    def __init__(self, max_prime: int = 200):
        self.primes = self._generate_primes(max_prime)
        self.wallace_transform = WallaceTransform()
        self.prime_scores = self._calculate_prime_scores()
        
    def _generate_primes(self, max_prime: int) -> List[int]:
        """Generate list of primes up to max_prime"""
        primes = []
        for num in range(2, max_prime + 1):
            if self._is_prime(num):
                primes.append(num)
        return primes
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def _calculate_prime_scores(self) -> Dict[int, float]:
        """Calculate Wallace scores for all primes"""
        prime_scores = {}
        for prime in self.primes:
            prime_scores[prime] = self.wallace_transform.calculate_score(prime)
        return prime_scores
    
    def detect_glitches(self, pixel_values: List[int]) -> List[Tuple[int, int, float]]:
        """
        Detect glitches using prime cancellation
        
        Args:
            pixel_values: List of pixel values (0-255)
            
        Returns:
            List of glitch tuples (prime1, prime2, sum_score)
        """
        glitches = []
        
        # Find pixel values that are close to primes
        prime_pixels = []
        for pixel in pixel_values:
            closest_prime = min(self.primes, key=lambda p: abs(p - pixel))
            if abs(closest_prime - pixel) <= 2:  # Within 2 units of a prime
                prime_pixels.append(closest_prime)
        
        # Check all pairs for glitches
        for i, prime1 in enumerate(prime_pixels):
            for prime2 in prime_pixels[i+1:]:
                score1 = self.prime_scores[prime1]
                score2 = self.prime_scores[prime2]
                sum_score = score1 + score2
                
                # Check if sum is near zero (glitch)
                if abs(sum_score) < GLITCH_THRESHOLD:
                    glitches.append((prime1, prime2, sum_score))
        
        return glitches
    
    def create_prime_grid(self, scores: List[float]) -> np.ndarray:
        """Create a grid of prime-based Wallace scores"""
        grid_size = len(self.primes)
        grid = np.zeros((grid_size, grid_size))
        
        for i, prime1 in enumerate(self.primes):
            for j, prime2 in enumerate(self.primes):
                score1 = self.prime_scores[prime1]
                score2 = self.prime_scores[prime2]
                grid[i, j] = score1 + score2
        
        return grid

class CompressionAnalyzer:
    """Compression ratio analysis for deepfake detection"""
    
    def __init__(self, compression_size: int = 21):
        self.compression_size = compression_size
        
    def calculate_compression_ratio(self, original_size: int, compressed_scores: List[float]) -> float:
        """
        Calculate compression ratio
        
        Args:
            original_size: Original data size in bytes
            compressed_scores: List of Wallace scores (compressed representation)
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        compressed_size = len(compressed_scores)
        if compressed_size == 0:
            return 0.0
        
        ratio = original_size / compressed_size
        return ratio
    
    def classify_compression_ratio(self, ratio: float) -> str:
        """Classify compression ratio as real or fake"""
        if REAL_COMPRESSION_RATIO_RANGE[0] <= ratio <= REAL_COMPRESSION_RATIO_RANGE[1]:
            return "real"
        else:
            return "fake"
    
    def compress_scores(self, scores: List[float]) -> List[float]:
        """Compress Wallace scores to fixed size"""
        if len(scores) <= self.compression_size:
            # Pad with zeros if too small
            return scores + [0.0] * (self.compression_size - len(scores))
        else:
            # Sample evenly if too large
            indices = np.linspace(0, len(scores) - 1, self.compression_size, dtype=int)
            return [scores[i] for i in indices]

class DeepfakeDetector:
    """Main deepfake detection system"""
    
    def __init__(self):
        self.wallace_transform = WallaceTransform()
        self.prime_filter = PrimeCancellationFilter()
        self.compression_analyzer = CompressionAnalyzer()
        
    def analyze_frame(self, frame: np.ndarray, frame_number: int = 0) -> FrameAnalysis:
        """
        Analyze a single video frame for deepfake detection
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number for tracking
            
        Returns:
            FrameAnalysis result
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            
            # Divide frame into 10x10 blocks
            block_size = 10
            height, width = gray_frame.shape
            blocks = []
            
            for i in range(0, height - block_size + 1, block_size):
                for j in range(0, width - block_size + 1, block_size):
                    block = gray_frame[i:i+block_size, j:j+block_size]
                    blocks.append(block)
            
            # Calculate frequencies for each block
            frequencies = []
            for block in blocks:
                # Average pixel value as frequency (MHz)
                avg_value = np.mean(block)
                frequency = avg_value / 100.0  # Convert to MHz
                frequencies.append(frequency)
            
            # Apply Wallace Transform
            wallace_scores = self.wallace_transform.analyze_frequency_range(frequencies)
            
            # Detect glitches using Prime Cancellation Filter
            pixel_values = gray_frame.flatten().tolist()
            glitches = self.prime_filter.detect_glitches(pixel_values)
            
            # Calculate compression ratio
            compressed_scores = self.compression_analyzer.compress_scores(wallace_scores)
            original_size = len(gray_frame.flatten())
            compression_ratio = self.compression_analyzer.calculate_compression_ratio(
                original_size, compressed_scores
            )
            
            # Determine if frame is fake
            is_fake = self._classify_frame(wallace_scores, glitches, compression_ratio)
            
            # Calculate confidence
            confidence = self._calculate_confidence(wallace_scores, glitches, compression_ratio)
            
            # Create metadata
            metadata = {
                'block_count': len(blocks),
                'glitch_details': [(p1, p2, score) for p1, p2, score in glitches],
                'compression_classification': self.compression_analyzer.classify_compression_ratio(compression_ratio),
                'average_wallace_score': np.mean(wallace_scores),
                'wallace_score_std': np.std(wallace_scores)
            }
            
            return FrameAnalysis(
                frame_number=frame_number,
                wallace_scores=wallace_scores,
                compression_ratio=compression_ratio,
                glitch_count=len(glitches),
                is_fake=is_fake,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame {frame_number}: {e}")
            return FrameAnalysis(
                frame_number=frame_number,
                wallace_scores=[],
                compression_ratio=0.0,
                glitch_count=0,
                is_fake=False,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _classify_frame(self, wallace_scores: List[float], glitches: List, compression_ratio: float) -> bool:
        """Classify frame as fake or real"""
        if not wallace_scores:
            return False
        
        # Check Wallace scores
        avg_score = np.mean(wallace_scores)
        score_classification = self.wallace_transform.classify_score(avg_score)
        
        # Check compression ratio
        compression_classification = self.compression_analyzer.classify_compression_ratio(compression_ratio)
        
        # Check for glitches
        has_glitches = len(glitches) > 0
        
        # Frame is fake if any indicator suggests so
        is_fake = (score_classification == "fake" or 
                  compression_classification == "fake" or 
                  has_glitches)
        
        return is_fake
    
    def _calculate_confidence(self, wallace_scores: List[float], glitches: List, compression_ratio: float) -> float:
        """Calculate detection confidence"""
        if not wallace_scores:
            return 0.0
        
        # Base confidence from Wallace scores
        avg_score = np.mean(wallace_scores)
        score_confidence = 1.0 - abs(avg_score - 22.5) / 25.0  # 22.5 is middle of real range
        
        # Glitch penalty
        glitch_penalty = min(len(glitches) * 0.1, 0.5)
        
        # Compression confidence
        if REAL_COMPRESSION_RATIO_RANGE[0] <= compression_ratio <= REAL_COMPRESSION_RATIO_RANGE[1]:
            compression_confidence = 1.0
        else:
            compression_confidence = 0.3
        
        # Combine confidences
        confidence = (score_confidence + compression_confidence) / 2 - glitch_penalty
        return max(0.0, min(1.0, confidence))
    
    def analyze_video(self, video_path: str, sample_rate: int = 1) -> VideoAnalysis:
        """
        Analyze entire video for deepfake detection
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame (1 = all frames)
            
        Returns:
            VideoAnalysis result
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_analyses = []
        frame_count = 0
        fake_count = 0
        real_count = 0
        total_confidence = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on sample_rate
                if frame_count % sample_rate == 0:
                    analysis = self.analyze_frame(frame, frame_count)
                    frame_analyses.append(analysis)
                    
                    if analysis.is_fake:
                        fake_count += 1
                    else:
                        real_count += 1
                    
                    total_confidence += analysis.confidence
                    
                    # Log progress
                    if frame_count % 100 == 0:
                        logger.info(f"Processed frame {frame_count}, fake: {analysis.is_fake}")
                
                frame_count += 1
                
        finally:
            cap.release()
        
        # Calculate summary statistics
        total_analyzed = len(frame_analyses)
        average_confidence = total_confidence / total_analyzed if total_analyzed > 0 else 0.0
        
        # Create detection summary
        detection_summary = {
            'total_frames': frame_count,
            'analyzed_frames': total_analyzed,
            'fake_percentage': (fake_count / total_analyzed * 100) if total_analyzed > 0 else 0.0,
            'real_percentage': (real_count / total_analyzed * 100) if total_analyzed > 0 else 0.0,
            'overall_classification': 'fake' if fake_count > real_count else 'real',
            'average_confidence': average_confidence
        }
        
        return VideoAnalysis(
            video_path=video_path,
            total_frames=frame_count,
            fake_frames=fake_count,
            real_frames=real_count,
            average_confidence=average_confidence,
            frame_analyses=frame_analyses,
            detection_summary=detection_summary
        )
    
    def visualize_analysis(self, video_analysis: VideoAnalysis, save_path: Optional[str] = None):
        """Visualize analysis results"""
        if not video_analysis.frame_analyses:
            logger.warning("No frame analyses to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        frame_numbers = [fa.frame_number for fa in video_analysis.frame_analyses]
        wallace_scores = [np.mean(fa.wallace_scores) for fa in video_analysis.frame_analyses]
        compression_ratios = [fa.compression_ratio for fa in video_analysis.frame_analyses]
        glitch_counts = [fa.glitch_count for fa in video_analysis.frame_analyses]
        is_fake = [fa.is_fake for fa in video_analysis.frame_analyses]
        
        # Plot 1: Wallace Scores over time
        axes[0, 0].plot(frame_numbers, wallace_scores, 'b-', alpha=0.7)
        axes[0, 0].axhline(y=REAL_SCORE_RANGE[0], color='g', linestyle='--', label='Real Range')
        axes[0, 0].axhline(y=REAL_SCORE_RANGE[1], color='g', linestyle='--')
        axes[0, 0].set_title('Wallace Scores Over Time')
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('Wallace Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Compression Ratios
        axes[0, 1].plot(frame_numbers, compression_ratios, 'r-', alpha=0.7)
        axes[0, 1].axhline(y=REAL_COMPRESSION_RATIO_RANGE[0], color='g', linestyle='--', label='Real Range')
        axes[0, 1].axhline(y=REAL_COMPRESSION_RATIO_RANGE[1], color='g', linestyle='--')
        axes[0, 1].set_title('Compression Ratios Over Time')
        axes[0, 1].set_xlabel('Frame Number')
        axes[0, 1].set_ylabel('Compression Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Glitch Counts
        axes[1, 0].plot(frame_numbers, glitch_counts, 'orange', alpha=0.7)
        axes[1, 0].set_title('Glitch Counts Over Time')
        axes[1, 0].set_xlabel('Frame Number')
        axes[1, 0].set_ylabel('Glitch Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Fake/Real Classification
        fake_frames = [fn for fn, fake in zip(frame_numbers, is_fake) if fake]
        real_frames = [fn for fn, fake in zip(frame_numbers, is_fake) if not fake]
        
        axes[1, 1].scatter(fake_frames, [1] * len(fake_frames), c='red', label='Fake', alpha=0.7)
        axes[1, 1].scatter(real_frames, [0] * len(real_frames), c='green', label='Real', alpha=0.7)
        axes[1, 1].set_title('Frame Classification')
        axes[1, 1].set_xlabel('Frame Number')
        axes[1, 1].set_ylabel('Classification (0=Real, 1=Fake)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def export_analysis(self, video_analysis: VideoAnalysis, filepath: str):
        """Export analysis results to JSON"""
        export_data = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'video_path': video_analysis.video_path,
                'detection_summary': video_analysis.detection_summary
            },
            'frame_analyses': []
        }
        
        for fa in video_analysis.frame_analyses:
            frame_data = {
                'frame_number': fa.frame_number,
                'wallace_scores': fa.wallace_scores,
                'compression_ratio': fa.compression_ratio,
                'glitch_count': fa.glitch_count,
                'is_fake': fa.is_fake,
                'confidence': fa.confidence,
                'metadata': fa.metadata
            }
            export_data['frame_analyses'].append(frame_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Analysis exported to {filepath}")

def main():
    """Main function for testing and demonstration"""
    print("=== Deepfake Detection Algorithm - Wallace Transform & Prime Cancellation Filter ===")
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Test Wallace Transform
    print("\n=== Testing Wallace Transform ===")
    test_frequencies = [7.0, 7.2, 1.3, 13.0, 17.0]
    for freq in test_frequencies:
        score = detector.wallace_transform.calculate_score(freq)
        classification = detector.wallace_transform.classify_score(score)
        print(f"Frequency: {freq} MHz -> Score: {score:.2f} -> {classification}")
    
    # Test Prime Cancellation Filter
    print("\n=== Testing Prime Cancellation Filter ===")
    test_pixels = [13, 17, 25, 30, 35, 40, 45, 50]
    glitches = detector.prime_filter.detect_glitches(test_pixels)
    print(f"Test pixels: {test_pixels}")
    print(f"Detected glitches: {glitches}")
    
    # Test Compression Analyzer
    print("\n=== Testing Compression Analyzer ===")
    test_scores = [20.98, 21.13, 18.45, 22.1, 19.8] * 20  # 100 scores
    compressed = detector.compression_analyzer.compress_scores(test_scores)
    ratio = detector.compression_analyzer.calculate_compression_ratio(100, compressed)
    classification = detector.compression_analyzer.classify_compression_ratio(ratio)
    print(f"Original size: 100, Compressed size: {len(compressed)}")
    print(f"Compression ratio: {ratio:.1f}:1 -> {classification}")
    
    # Example video analysis (if video file provided)
    video_path = input("\nEnter video file path (or press Enter to skip): ").strip()
    if video_path and Path(video_path).exists():
        print(f"\n=== Analyzing Video: {video_path} ===")
        
        try:
            # Analyze video
            analysis = detector.analyze_video(video_path, sample_rate=5)  # Every 5th frame
            
            # Print results
            print(f"Total frames: {analysis.total_frames}")
            print(f"Analyzed frames: {len(analysis.frame_analyses)}")
            print(f"Fake frames: {analysis.fake_frames}")
            print(f"Real frames: {analysis.real_frames}")
            print(f"Fake percentage: {analysis.detection_summary['fake_percentage']:.1f}%")
            print(f"Overall classification: {analysis.detection_summary['overall_classification']}")
            print(f"Average confidence: {analysis.average_confidence:.3f}")
            
            # Visualize results
            detector.visualize_analysis(analysis, save_path="deepfake_analysis.png")
            
            # Export results
            detector.export_analysis(analysis, "deepfake_analysis_results.json")
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
    else:
        print("No video file provided, skipping video analysis")
    
    print("\n=== Deepfake Detection Complete ===")

if __name__ == "__main__":
    main()
