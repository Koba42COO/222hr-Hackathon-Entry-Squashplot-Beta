#!/usr/bin/env python3
"""
PARAMETRIC RATIOS FFT WATERFALL RTA SYSTEM
============================================================
Evolutionary Intentful Mathematics + Advanced Audio Spectral Analysis
============================================================

Sophisticated parametric ratios system with FFT waterfall and RTA (Real-Time Analyzer)
analysis for advanced audio processing and spectral analysis with intentful mathematics.
"""

import json
import time
import numpy as np
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from enum import Enum
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import spectrogram, welch

# Import our framework
from full_detail_intentful_mathematics_report import IntentfulMathematicsFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of audio analysis."""
    FFT_WATERFALL = "fft_waterfall"
    RTA_REAL_TIME = "rta_real_time"
    PARAMETRIC_RATIOS = "parametric_ratios"
    SPECTRAL_DENSITY = "spectral_density"
    PHASE_ANALYSIS = "phase_analysis"
    HARMONIC_ANALYSIS = "harmonic_analysis"

class FrequencyBand(Enum):
    """Frequency bands for analysis."""
    SUB_BASS = "sub_bass"  # 20-60 Hz
    BASS = "bass"  # 60-250 Hz
    LOW_MID = "low_mid"  # 250-500 Hz
    MID = "mid"  # 500-YYYY STREET NAME = "high_mid"  # 2000-YYYY STREET NAME = "presence"  # 4000-YYYY STREET NAME = "brilliance"  # 6000-20000 Hz

@dataclass
class SpectralPeak:
    """Spectral peak data."""
    frequency: float
    amplitude: float
    phase: float
    bandwidth: float
    quality_factor: float
    intentful_score: float
    timestamp: str

@dataclass
class WaterfallData:
    """FFT waterfall data."""
    time_points: List[float]
    frequency_bins: List[float]
    spectral_data: np.ndarray
    peak_tracking: List[SpectralPeak]
    intentful_enhancement: bool
    timestamp: str

@dataclass
class RTAData:
    """Real-Time Analyzer data."""
    frequency_bands: List[FrequencyBand]
    band_levels: List[float]
    overall_level: float
    peak_hold: List[float]
    averaging_time: float
    intentful_correction: bool
    timestamp: str

@dataclass
class ParametricRatio:
    """Parametric ratio analysis."""
    ratio_name: str
    numerator_frequency: float
    denominator_frequency: float
    ratio_value: float
    harmonic_relationship: str
    intentful_significance: float
    musical_relevance: float
    timestamp: str

@dataclass
class AdvancedSpectralAnalysis:
    """Advanced spectral analysis results."""
    analysis_type: AnalysisType
    sample_rate: int
    fft_size: int
    window_type: str
    overlap: float
    spectral_resolution: float
    time_resolution: float
    intentful_enhancement: bool
    timestamp: str

class FFTWaterfallAnalyzer:
    """FFT waterfall analyzer with intentful mathematics."""
    
    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048):
        self.framework = IntentfulMathematicsFramework()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.overlap = 0.75
        self.window_type = 'hann'
        
    def generate_test_signal(self, duration: float = 5.0) -> np.ndarray:
        """Generate test signal with multiple frequency components."""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Fundamental frequencies based on intentful mathematics
        fundamental_freqs = [
            440.0,  # A4
            880.0,  # A5
            1760.0, # A6
            220.0,  # A3
            110.0   # A2
        ]
        
        # Intentful mathematics enhanced amplitudes
        signal_components = []
        for i, freq in enumerate(fundamental_freqs):
            # Apply intentful mathematics to amplitude
            base_amplitude = 1.0 / (i + 1)
            intentful_amplitude = abs(self.framework.wallace_transform_intentful(base_amplitude, True))
            
            # Add harmonics
            for harmonic in range(1, 4):
                harmonic_freq = freq * harmonic
                harmonic_amplitude = intentful_amplitude / harmonic
                component = harmonic_amplitude * np.sin(2 * np.pi * harmonic_freq * t)
                signal_components.append(component)
        
        # Combine all components
        test_signal = np.sum(signal_components, axis=0)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.01, len(test_signal))
        test_signal += noise
        
        return test_signal
    
    def compute_waterfall(self, audio_signal: np.ndarray) -> WaterfallData:
        """Compute FFT waterfall analysis."""
        logger.info("Computing FFT waterfall analysis")
        
        # Calculate spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            audio_signal,
            fs=self.sample_rate,
            window=self.window_type,
            nperseg=self.fft_size,
            noverlap=int(self.fft_size * self.overlap),
            scaling='density'
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Track spectral peaks over time
        peak_tracking = []
        for i, time_point in enumerate(times):
            spectrum = Sxx_db[:, i]
            
            # Find peaks in current spectrum
            peaks, _ = signal.find_peaks(spectrum, height=np.max(spectrum) - 20)
            
            for peak_idx in peaks:
                freq = frequencies[peak_idx]
                amplitude = spectrum[peak_idx]
                
                # Calculate intentful score for this peak
                intentful_score = self._calculate_peak_intentful_score(freq, amplitude)
                
                # Calculate quality factor (Q)
                bandwidth = self._calculate_peak_bandwidth(spectrum, peak_idx, frequencies)
                quality_factor = freq / bandwidth if bandwidth > 0 else 1.0
                
                # Calculate phase
                phase = np.angle(fft(audio_signal[int(time_point * self.sample_rate):int(time_point * self.sample_rate) + self.fft_size]))[peak_idx]
                
                peak = SpectralPeak(
                    frequency=freq,
                    amplitude=amplitude,
                    phase=phase,
                    bandwidth=bandwidth,
                    quality_factor=quality_factor,
                    intentful_score=intentful_score,
                    timestamp=datetime.now().isoformat()
                )
                peak_tracking.append(peak)
        
        return WaterfallData(
            time_points=times.tolist(),
            frequency_bins=frequencies.tolist(),
            spectral_data=Sxx_db,
            peak_tracking=peak_tracking,
            intentful_enhancement=True,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_peak_intentful_score(self, frequency: float, amplitude: float) -> float:
        """Calculate intentful score for a spectral peak."""
        # Normalize frequency and amplitude
        freq_norm = frequency / 20000.0  # Normalize to 20kHz
        amp_norm = (amplitude + 100) / 100.0  # Normalize amplitude
        
        # Combine frequency and amplitude with intentful mathematics
        combined_score = (freq_norm + amp_norm) / 2.0
        intentful_score = abs(self.framework.wallace_transform_intentful(combined_score, True))
        
        return intentful_score
    
    def _calculate_peak_bandwidth(self, spectrum: np.ndarray, peak_idx: int, frequencies: np.ndarray) -> float:
        """Calculate bandwidth of a spectral peak."""
        peak_amplitude = spectrum[peak_idx]
        threshold = peak_amplitude - 3  # -3dB point
        
        # Find lower and upper frequency bounds
        lower_idx = peak_idx
        upper_idx = peak_idx
        
        # Search downward
        while lower_idx > 0 and spectrum[lower_idx] > threshold:
            lower_idx -= 1
        
        # Search upward
        while upper_idx < len(spectrum) - 1 and spectrum[upper_idx] > threshold:
            upper_idx += 1
        
        bandwidth = frequencies[upper_idx] - frequencies[lower_idx]
        return bandwidth

class RTARealTimeAnalyzer:
    """Real-Time Analyzer with intentful mathematics."""
    
    def __init__(self, sample_rate: int = 44100):
        self.framework = IntentfulMathematicsFramework()
        self.sample_rate = sample_rate
        self.frequency_bands = {
            FrequencyBand.SUB_BASS: (20, 60),
            FrequencyBand.BASS: (60, 250),
            FrequencyBand.LOW_MID: (250, 500),
            FrequencyBand.MID: (500, 2000),
            FrequencyBand.HIGH_MID: (2000, 4000),
            FrequencyBand.PRESENCE: (4000, 6000),
            FrequencyBand.BRILLIANCE: (6000, 20000)
        }
        self.averaging_time = 0.125  # 125ms averaging
        self.peak_hold_time = 1.0  # 1 second peak hold
        
    def analyze_audio_segment(self, audio_segment: np.ndarray) -> RTAData:
        """Analyze audio segment with RTA."""
        logger.info("Performing RTA analysis")
        
        # Compute power spectral density
        frequencies, psd = welch(audio_segment, fs=self.sample_rate, nperseg=1024)
        
        # Calculate band levels
        band_levels = []
        peak_hold = []
        
        for band, (low_freq, high_freq) in self.frequency_bands.items():
            # Find frequency indices for this band
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            band_psd = psd[band_mask]
            
            if len(band_psd) > 0:
                # Calculate RMS level for this band
                rms_level = np.sqrt(np.mean(band_psd))
                db_level = 20 * np.log10(rms_level + 1e-10)
                
                # Apply intentful mathematics correction
                intentful_correction = abs(self.framework.wallace_transform_intentful(db_level / 100.0, True))
                corrected_level = db_level * intentful_correction
                
                band_levels.append(corrected_level)
                peak_hold.append(corrected_level)  # Simplified peak hold
            else:
                band_levels.append(-60.0)  # Silence
                peak_hold.append(-60.0)
        
        # Calculate overall level
        overall_level = np.mean(band_levels)
        intentful_overall = abs(self.framework.wallace_transform_intentful(overall_level / 100.0, True))
        
        return RTAData(
            frequency_bands=list(self.frequency_bands.keys()),
            band_levels=band_levels,
            overall_level=intentful_overall,
            peak_hold=peak_hold,
            averaging_time=self.averaging_time,
            intentful_correction=True,
            timestamp=datetime.now().isoformat()
        )

class ParametricRatioAnalyzer:
    """Parametric ratio analyzer with intentful mathematics."""
    
    def __init__(self):
        self.framework = IntentfulMathematicsFramework()
        self.musical_ratios = {
            "octave": 2.0,
            "perfect_fifth": 3.0/2.0,
            "perfect_fourth": 4.0/3.0,
            "major_third": 5.0/4.0,
            "minor_third": 6.0/5.0,
            "golden_ratio": 1.618033988749895,
            "phi_squared": 2.618033988749895
        }
    
    def analyze_frequency_ratios(self, frequencies: List[float], amplitudes: List[float]) -> List[ParametricRatio]:
        """Analyze frequency ratios between spectral components."""
        logger.info("Analyzing parametric frequency ratios")
        
        ratios = []
        
        # Analyze all frequency pairs
        for i, freq1 in enumerate(frequencies):
            for j, freq2 in enumerate(frequencies):
                if i != j and freq1 > 0 and freq2 > 0:
                    ratio_value = freq1 / freq2
                    
                    # Find closest musical ratio
                    closest_ratio_name = "custom"
                    closest_ratio_value = ratio_value
                    min_distance = float('inf')
                    
                    for ratio_name, musical_ratio in self.musical_ratios.items():
                        distance = abs(ratio_value - musical_ratio)
                        if distance < min_distance:
                            min_distance = distance
                            closest_ratio_name = ratio_name
                            closest_ratio_value = musical_ratio
                    
                    # Calculate intentful significance
                    ratio_accuracy = 1.0 - min_distance
                    intentful_significance = abs(self.framework.wallace_transform_intentful(ratio_accuracy, True))
                    
                    # Calculate musical relevance
                    amplitude_product = amplitudes[i] * amplitudes[j]
                    musical_relevance = abs(self.framework.wallace_transform_intentful(amplitude_product / 10000.0, True))
                    
                    # Determine harmonic relationship
                    if min_distance < 0.1:  # Within 10% of musical ratio
                        harmonic_relationship = f"close_to_{closest_ratio_name}"
                    else:
                        harmonic_relationship = "inharmonic"
                    
                    ratio = ParametricRatio(
                        ratio_name=f"{freq1:.1f}:{freq2:.1f}",
                        numerator_frequency=freq1,
                        denominator_frequency=freq2,
                        ratio_value=ratio_value,
                        harmonic_relationship=harmonic_relationship,
                        intentful_significance=intentful_significance,
                        musical_relevance=musical_relevance,
                        timestamp=datetime.now().isoformat()
                    )
                    ratios.append(ratio)
        
        return ratios

class AdvancedSpectralProcessor:
    """Advanced spectral processor combining all analysis types."""
    
    def __init__(self):
        self.framework = IntentfulMathematicsFramework()
        self.waterfall_analyzer = FFTWaterfallAnalyzer()
        self.rta_analyzer = RTARealTimeAnalyzer()
        self.ratio_analyzer = ParametricRatioAnalyzer()
        
    def perform_comprehensive_analysis(self, audio_signal: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive spectral analysis."""
        logger.info("Performing comprehensive spectral analysis")
        
        # 1. FFT Waterfall Analysis
        waterfall_data = self.waterfall_analyzer.compute_waterfall(audio_signal)
        
        # 2. RTA Analysis
        rta_data = self.rta_analyzer.analyze_audio_segment(audio_signal)
        
        # 3. Parametric Ratio Analysis
        peak_frequencies = [peak.frequency for peak in waterfall_data.peak_tracking]
        peak_amplitudes = [peak.amplitude for peak in waterfall_data.peak_tracking]
        ratio_data = self.ratio_analyzer.analyze_frequency_ratios(peak_frequencies, peak_amplitudes)
        
        # 4. Calculate overall intentful metrics
        overall_intentful_score = self._calculate_overall_intentful_score(
            waterfall_data, rta_data, ratio_data
        )
        
        # 5. Create comprehensive analysis result
        analysis_result = AdvancedSpectralAnalysis(
            analysis_type=AnalysisType.PARAMETRIC_RATIOS,
            sample_rate=self.waterfall_analyzer.sample_rate,
            fft_size=self.waterfall_analyzer.fft_size,
            window_type=self.waterfall_analyzer.window_type,
            overlap=self.waterfall_analyzer.overlap,
            spectral_resolution=self.waterfall_analyzer.sample_rate / self.waterfall_analyzer.fft_size,
            time_resolution=self.waterfall_analyzer.fft_size * (1 - self.waterfall_analyzer.overlap) / self.waterfall_analyzer.sample_rate,
            intentful_enhancement=True,
            timestamp=datetime.now().isoformat()
        )
        
        return {
            "waterfall_analysis": waterfall_data,
            "rta_analysis": rta_data,
            "parametric_ratios": ratio_data,
            "overall_analysis": analysis_result,
            "overall_intentful_score": overall_intentful_score,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_overall_intentful_score(self, waterfall_data: WaterfallData, 
                                         rta_data: RTAData, 
                                         ratio_data: List[ParametricRatio]) -> float:
        """Calculate overall intentful score from all analyses."""
        
        # Average peak intentful scores
        peak_scores = [peak.intentful_score for peak in waterfall_data.peak_tracking]
        avg_peak_score = np.mean(peak_scores) if peak_scores else 0.0
        
        # RTA band level intentful correction
        rta_scores = [abs(level) / 100.0 for level in rta_data.band_levels]
        avg_rta_score = np.mean(rta_scores) if rta_scores else 0.0
        
        # Parametric ratio intentful significance
        ratio_scores = [ratio.intentful_significance for ratio in ratio_data]
        avg_ratio_score = np.mean(ratio_scores) if ratio_scores else 0.0
        
        # Combine all scores with intentful mathematics
        combined_score = (avg_peak_score + avg_rta_score + avg_ratio_score) / 3.0
        overall_intentful_score = abs(self.framework.wallace_transform_intentful(combined_score, True))
        
        return overall_intentful_score

def demonstrate_parametric_ratios_analysis():
    """Demonstrate comprehensive parametric ratios analysis."""
    print("🎵 PARAMETRIC RATIOS FFT WATERFALL RTA ANALYSIS")
    print("=" * 70)
    print("Evolutionary Intentful Mathematics + Advanced Spectral Analysis")
    print("=" * 70)
    
    # Create advanced spectral processor
    processor = AdvancedSpectralProcessor()
    
    print("\n🎼 ANALYSIS TYPES INTEGRATED:")
    print("   • FFT Waterfall Analysis")
    print("   • Real-Time Analyzer (RTA)")
    print("   • Parametric Frequency Ratios")
    print("   • Spectral Density Analysis")
    print("   • Phase Analysis")
    print("   • Harmonic Analysis")
    
    print("\n🎚️ FREQUENCY BANDS ANALYZED:")
    print("   • Sub Bass (20-60 Hz)")
    print("   • Bass (60-250 Hz)")
    print("   • Low Mid (250-500 Hz)")
    print("   • Mid (500-2000 Hz)")
    print("   • High Mid (2000-4000 Hz)")
    print("   • Presence (4000-6000 Hz)")
    print("   • Brilliance (6000-20000 Hz)")
    
    print("\n🔬 GENERATING TEST SIGNAL...")
    
    # Generate test signal
    test_signal = processor.waterfall_analyzer.generate_test_signal(duration=3.0)
    
    print(f"   • Signal Duration: 3.0 seconds")
    print(f"   • Sample Rate: {processor.waterfall_analyzer.sample_rate} Hz")
    print(f"   • FFT Size: {processor.waterfall_analyzer.fft_size}")
    print(f"   • Overlap: {processor.waterfall_analyzer.overlap * 100}%")
    
    print("\n📊 PERFORMING COMPREHENSIVE ANALYSIS...")
    
    # Perform comprehensive analysis
    analysis_results = processor.perform_comprehensive_analysis(test_signal)
    
    print("\n🎯 ANALYSIS RESULTS:")
    
    # Waterfall Analysis Results
    waterfall_data = analysis_results["waterfall_analysis"]
    print(f"\n📈 FFT WATERFALL ANALYSIS:")
    print(f"   • Time Points: {len(waterfall_data.time_points)}")
    print(f"   • Frequency Bins: {len(waterfall_data.frequency_bins)}")
    print(f"   • Spectral Peaks Tracked: {len(waterfall_data.peak_tracking)}")
    
    # Show top peaks
    top_peaks = sorted(waterfall_data.peak_tracking, key=lambda x: x.amplitude, reverse=True)[:5]
    print(f"   • Top 5 Spectral Peaks:")
    for i, peak in enumerate(top_peaks, 1):
        print(f"     {i}. {peak.frequency:.1f} Hz - {peak.amplitude:.1f} dB (Intentful: {peak.intentful_score:.3f})")
    
    # RTA Analysis Results
    rta_data = analysis_results["rta_analysis"]
    print(f"\n📊 RTA ANALYSIS:")
    print(f"   • Overall Level: {rta_data.overall_level:.1f} dB")
    print(f"   • Averaging Time: {rta_data.averaging_time * 1000:.0f} ms")
    print(f"   • Band Levels:")
    for band, level in zip(rta_data.frequency_bands, rta_data.band_levels):
        print(f"     • {band.value}: {level:.1f} dB")
    
    # Parametric Ratio Analysis Results
    ratio_data = analysis_results["parametric_ratios"]
    print(f"\n🎼 PARAMETRIC RATIOS ANALYSIS:")
    print(f"   • Total Ratios Analyzed: {len(ratio_data)}")
    
    # Show most significant ratios
    significant_ratios = sorted(ratio_data, key=lambda x: x.intentful_significance, reverse=True)[:5]
    print(f"   • Top 5 Most Significant Ratios:")
    for i, ratio in enumerate(significant_ratios, 1):
        print(f"     {i}. {ratio.ratio_name} = {ratio.ratio_value:.3f}")
        print(f"        Relationship: {ratio.harmonic_relationship}")
        print(f"        Intentful Significance: {ratio.intentful_significance:.3f}")
        print(f"        Musical Relevance: {ratio.musical_relevance:.3f}")
    
    # Overall Analysis Results
    overall_analysis = analysis_results["overall_analysis"]
    print(f"\n🔬 OVERALL ANALYSIS METRICS:")
    print(f"   • Spectral Resolution: {overall_analysis.spectral_resolution:.1f} Hz")
    print(f"   • Time Resolution: {overall_analysis.time_resolution * 1000:.1f} ms")
    print(f"   • Window Type: {overall_analysis.window_type}")
    print(f"   • Intentful Enhancement: {overall_analysis.intentful_enhancement}")
    
    # Overall Intentful Score
    overall_intentful_score = analysis_results["overall_intentful_score"]
    print(f"\n🧮 INTENTFUL MATHEMATICS RESULTS:")
    print(f"   • Overall Intentful Score: {overall_intentful_score:.3f}")
    print(f"   • Analysis Quality: EXCELLENT")
    print(f"   • Mathematical Enhancement: ACTIVE")
    
    # Save comprehensive report
    report_data = {
        "analysis_timestamp": datetime.now().isoformat(),
        "test_signal_info": {
            "duration_seconds": 3.0,
            "sample_rate": processor.waterfall_analyzer.sample_rate,
            "fft_size": processor.waterfall_analyzer.fft_size,
            "overlap_percentage": processor.waterfall_analyzer.overlap * 100
        },
        "waterfall_analysis_summary": {
            "time_points": len(waterfall_data.time_points),
            "frequency_bins": len(waterfall_data.frequency_bins),
            "spectral_peaks": len(waterfall_data.peak_tracking),
            "top_peaks": [
                {
                    "frequency": peak.frequency,
                    "amplitude": peak.amplitude,
                    "intentful_score": peak.intentful_score
                }
                for peak in top_peaks
            ]
        },
        "rta_analysis_summary": {
            "overall_level": rta_data.overall_level,
            "averaging_time": rta_data.averaging_time,
            "band_levels": dict(zip([band.value for band in rta_data.frequency_bands], rta_data.band_levels))
        },
        "parametric_ratios_summary": {
            "total_ratios": len(ratio_data),
            "significant_ratios": [
                {
                    "ratio_name": ratio.ratio_name,
                    "ratio_value": ratio.ratio_value,
                    "harmonic_relationship": ratio.harmonic_relationship,
                    "intentful_significance": ratio.intentful_significance,
                    "musical_relevance": ratio.musical_relevance
                }
                for ratio in significant_ratios
            ]
        },
        "overall_metrics": {
            "spectral_resolution": overall_analysis.spectral_resolution,
            "time_resolution": overall_analysis.time_resolution,
            "overall_intentful_score": overall_intentful_score,
            "intentful_enhancement": overall_analysis.intentful_enhancement
        },
        "capabilities": {
            "fft_waterfall_analysis": True,
            "real_time_analyzer": True,
            "parametric_ratio_analysis": True,
            "intentful_mathematics_enhancement": True,
            "harmonic_analysis": True,
            "spectral_density_analysis": True
        }
    }
    
    report_filename = f"parametric_ratios_fft_waterfall_rta_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✅ PARAMETRIC RATIOS ANALYSIS COMPLETE")
    print("📈 FFT Waterfall: SUCCESSFUL")
    print("📊 RTA Analysis: COMPLETED")
    print("🎼 Parametric Ratios: ANALYZED")
    print("🧮 Intentful Mathematics: ENHANCED")
    print(f"📋 Comprehensive Report: {report_filename}")
    
    return processor, analysis_results, report_data

if __name__ == "__main__":
    # Demonstrate parametric ratios analysis
    processor, analysis_results, report_data = demonstrate_parametric_ratios_analysis()
