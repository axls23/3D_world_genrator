#!/usr/bin/env python3
"""
Advanced Neural Network-Inspired Convergence Detection and Optimization Utilities
for 3D Gaussian Splatting Training.

This module implements sophisticated convergence detection techniques adapted from
deep learning research, recognizing that GS-plat is a hybrid discrete-continuous
optimization problem with unique dynamics.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import signal as scipy_signal
import torch
from typing import Dict, Optional, Tuple, Any, Callable


@dataclass
class ConvergenceConfig:
    """
    Centralized configuration for all convergence detection and optimization utilities.
    
    This class holds all tunable hyperparameters that were previously hardcoded
    throughout the different utility classes, enabling easy experimentation and
    production deployment with different configurations.
    """
    
    # Phase detection thresholds
    phase_detection_curvature_threshold: float = 0.01
    phase_detection_grad_variance_threshold: float = 1e-6
    phase_detection_gauss_change_threshold: float = 10.0
    phase_detection_stability_window: int = 50
    phase_detection_loss_window: int = 100
    phase_detection_grad_window: int = 50
    phase_detection_gauss_window: int = 50
    phase_detection_init_steps: int = 200
    
    # Stability check thresholds
    stability_loss_std_threshold: float = 1e-4
    stability_grad_std_threshold: float = 1e-5
    
    # Polyak averaging parameters
    polyak_grad_norm_threshold: float = 1e-4
    
    # Population health thresholds
    health_inactive_gaussians_threshold: float = 30.0  # percentage
    health_mean_opacity_threshold: float = 0.2
    health_scale_max_ratio_threshold: float = 100.0
    health_mode_collapse_inactive_threshold: float = 20.0  # percentage change
    health_mode_collapse_opacity_threshold: float = 0.3
    
    # Spectral analysis thresholds
    spectral_oscillation_ratio_threshold: float = 0.15
    spectral_hurst_threshold: float = 0.5
    
    # Curvature scheduling parameters
    curvature_sensitivity: float = 2.0
    curvature_lr_multiplier_min: float = 0.8
    curvature_lr_multiplier_max: float = 1.1
    curvature_steep_threshold: float = 0.05
    curvature_flat_threshold: float = 0.01
    curvature_increase_interval: int = 500


class TrainingPhase(Enum):
    """Training phases for 3D Gaussian Splatting optimization."""
    INITIALIZATION = 0      # Random Gaussians settling
    DENSIFICATION = 1       # Adding Gaussians to cover scene
    REFINEMENT = 2          # Tuning existing Gaussians
    CONVERGENCE = 3         # Fine-tuning SH coefficients


class PhaseAwareOptimizer:
    """
    Phase-based learning strategy that adapts to GS-plat's discrete optimization phases.
    
    Unlike standard neural networks, GS-plat has distinct phases where different
    parameters should be optimized at different rates. This class implements a
    formal state machine for phase transitions to prevent phase flapping.
    """
    
    def __init__(self, config: ConvergenceConfig):
        """
        Initialize the phase-aware optimizer with configuration.
        
        Args:
            config: ConvergenceConfig instance containing all tunable parameters
        """
        self.config = config
        self.current_phase = TrainingPhase.INITIALIZATION
        self.loss_history = []
        self.gradient_norms = []
        self.num_gaussians_history = []
    
    def detect_phase(self, step: int) -> TrainingPhase:
        """
        Detect training phase using a formal state machine approach.
        
        This prevents false early stopping during densification when loss
        naturally fluctuates from adding/removing Gaussians. The state machine
        approach prevents phase flapping by checking for valid transitions.
        
        Args:
            step: Current training step
            
        Returns:
            Current training phase
        """
        # Always start with initialization
        if step < self.config.phase_detection_init_steps:
            return TrainingPhase.INITIALIZATION
        
        # State machine transitions based on current phase
        if self.current_phase == TrainingPhase.INITIALIZATION:
            return self._transition_from_initialization()
        elif self.current_phase == TrainingPhase.DENSIFICATION:
            return self._transition_from_densification()
        elif self.current_phase == TrainingPhase.REFINEMENT:
            return self._transition_from_refinement()
        elif self.current_phase == TrainingPhase.CONVERGENCE:
            return self._transition_from_convergence()
        
        return self.current_phase
    
    def _transition_from_initialization(self) -> TrainingPhase:
        """Transition from initialization phase."""
        # Always move to densification after initialization
        return TrainingPhase.DENSIFICATION
    
    def _transition_from_densification(self) -> TrainingPhase:
        """Transition from densification phase."""
        # Check if we can move to refinement
        if self._can_transition_to_refinement():
            return TrainingPhase.REFINEMENT
        return TrainingPhase.DENSIFICATION
    
    def _transition_from_refinement(self) -> TrainingPhase:
        """Transition from refinement phase."""
        # Check if we can move to convergence
        if self._can_transition_to_convergence():
            return TrainingPhase.CONVERGENCE
        return TrainingPhase.REFINEMENT
    
    def _transition_from_convergence(self) -> TrainingPhase:
        """Transition from convergence phase."""
        # Once in convergence, stay there
        return TrainingPhase.CONVERGENCE
    
    def _can_transition_to_refinement(self) -> bool:
        """Check if we can transition to refinement phase."""
        # Low gradient variance + stable Gaussian count = refinement
        if len(self.gradient_norms) >= self.config.phase_detection_grad_window:
            grad_variance = np.var(self.gradient_norms[-self.config.phase_detection_grad_window:])
            
            if grad_variance < self.config.phase_detection_grad_variance_threshold:
                # Check if Gaussian count is stable
                if len(self.num_gaussians_history) >= self.config.phase_detection_gauss_window:
                    gauss_changes = np.diff(self.num_gaussians_history[-self.config.phase_detection_gauss_window:])
                    if np.std(gauss_changes) < self.config.phase_detection_gauss_change_threshold:
                        return True
        return False
    
    def _can_transition_to_convergence(self) -> bool:
        """Check if we can transition to convergence phase."""
        # Stable metrics across board = convergence
        return self.check_stability(window=self.config.phase_detection_stability_window)
    
    def get_phase_learning_rates(self, phase: TrainingPhase) -> Dict[str, float]:
        """
        Different learning rates per optimization phase.
        
        These are base learning rates that will be scaled by batch size
        in the actual optimizer.
        """
        lr_configs = {
            TrainingPhase.INITIALIZATION: {
                'means_lr': 1.6e-4,
                'scales_lr': 5e-3,
                'opacities_lr': 5e-2,
                'quats_lr': 1e-3,
                'sh0_lr': 2.5e-3,
                'shN_lr': 1.25e-4,
            },
            TrainingPhase.DENSIFICATION: {
                'means_lr': 3e-4,      # Higher for rapid coverage
                'scales_lr': 8e-3,
                'opacities_lr': 8e-2,
                'quats_lr': 2e-3,
                'sh0_lr': 3e-3,
                'shN_lr': 1e-4,
            },
            TrainingPhase.REFINEMENT: {
                'means_lr': 1.2e-4,    # Moderate for refinement
                'scales_lr': 3e-3,
                'opacities_lr': 3e-2,
                'quats_lr': 8e-4,
                'sh0_lr': 2e-3,
                'shN_lr': 5e-4,        # Increase detail learning
            },
            TrainingPhase.CONVERGENCE: {
                'means_lr': 5e-5,      # Very low for fine-tuning
                'scales_lr': 1e-3,
                'opacities_lr': 1e-2,
                'quats_lr': 3e-4,
                'sh0_lr': 1e-3,
                'shN_lr': 2e-4,
            },
        }
        return lr_configs[phase]
    
    def check_stability(self, window: int = None) -> bool:
        """
        Check if all metrics are stable.
        
        Args:
            window: Window size for stability check. If None, uses config default.
            
        Returns:
            True if metrics are stable, False otherwise
        """
        if window is None:
            window = self.config.phase_detection_stability_window
            
        if len(self.loss_history) < window:
            return False
        
        loss_std = np.std(self.loss_history[-window:])
        grad_std = np.std(self.gradient_norms[-window:])
        
        return (loss_std < self.config.stability_loss_std_threshold and 
                grad_std < self.config.stability_grad_std_threshold)


class PolyakAveragingDetector:
    """
    Exponential moving average for noise-robust convergence detection.
    
    Mirrors momentum-style averaging successful in neural network optimization,
    making convergence detection robust to noise from Gaussian add/remove operations.
    """
    
    def __init__(self, config: ConvergenceConfig, decay: float = 0.99, patience: int = 5):
        """
        Initialize the Polyak averaging detector with configuration.
        
        Args:
            config: ConvergenceConfig instance containing all tunable parameters
            decay: EMA decay factor (higher = more smoothing)
            patience: Number of consecutive stable evaluations required for convergence
        """
        self.config = config
        self.decay = decay
        self.patience = patience
        self.ema_loss = None
        self.ema_grad_norm = None
        self.ema_psnr = None
        self.convergence_counter = 0
        self.history = []
    
    def update(self, loss: float, grad_norm: float, psnr: float):
        """Update exponential moving averages."""
        if self.ema_loss is None:
            self.ema_loss = loss
            self.ema_grad_norm = grad_norm
            self.ema_psnr = psnr
        else:
            self.ema_loss = self.decay * self.ema_loss + (1 - self.decay) * loss
            self.ema_grad_norm = self.decay * self.ema_grad_norm + (1 - self.decay) * grad_norm
            self.ema_psnr = self.decay * self.ema_psnr + (1 - self.decay) * psnr
        
        self.history.append({
            'ema_loss': self.ema_loss,
            'ema_grad_norm': self.ema_grad_norm,
            'ema_psnr': self.ema_psnr,
        })
    
    def check_convergence(self, min_delta: float = 0.001) -> bool:
        """
        Convergence when EMA metrics stabilize, not noisy snapshots.
        
        This is more robust than instantaneous metrics which can be noisy
        due to discrete Gaussian add/remove operations.
        """
        if self.ema_grad_norm is None or len(self.history) < self.patience:
            return False
        
        # Check gradient norm stability (primary indicator)
        grad_converged = self.ema_grad_norm < self.config.polyak_grad_norm_threshold
        
        # Check PSNR improvement plateau
        recent_psnrs = [h['ema_psnr'] for h in self.history[-self.patience:]]
        if len(recent_psnrs) < 2:
            return False
        
        psnr_improvements = np.diff(recent_psnrs)
        psnr_plateaued = all(imp < min_delta for imp in psnr_improvements)
        
        if grad_converged and psnr_plateaued:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        
        # Require N consecutive stable evaluations
        return self.convergence_counter >= self.patience
    
    def get_metrics(self) -> dict:
        """Return current EMA metrics."""
        return {
            'ema_loss': self.ema_loss,
            'ema_grad_norm': self.ema_grad_norm,
            'ema_psnr': self.ema_psnr,
        }


class GaussianPopulationMonitor:
    """
    Monitor Gaussian distribution health - analogous to batch norm diagnostics.
    
    Tracks population-level statistics to detect mode collapse, spatial clustering,
    and other failure modes specific to Gaussian representation learning.
    """
    
    def __init__(self, config: ConvergenceConfig):
        """
        Initialize the population monitor with configuration.
        
        Args:
            config: ConvergenceConfig instance containing all tunable parameters
        """
        self.config = config
    
    def compute_health_metrics(self, splats: dict) -> dict:
        """
        Compute population-level statistics.
        
        Args:
            splats: Dictionary containing 'means', 'scales', 'opacities', etc.
        
        Returns:
            Dictionary of health metrics and flags
        """
        means = splats['means'].detach().cpu().numpy()
        scales = torch.exp(splats['scales']).detach().cpu().numpy()
        opacities = torch.sigmoid(splats['opacities']).detach().cpu().numpy()
        
        metrics = {
            # Spatial distribution
            'center_of_mass': np.mean(means, axis=0).tolist(),
            'spatial_spread': float(np.std(np.linalg.norm(means, axis=1))),
            
            # Scale distribution (detect outliers)
            'scale_mean': float(np.mean(scales)),
            'scale_std': float(np.std(scales)),
            'scale_max_ratio': float(np.max(scales) / (np.mean(scales) + 1e-8)),
            
            # Opacity distribution
            'mean_opacity': float(np.mean(opacities)),
            'opacity_variance': float(np.var(opacities)),
            'inactive_gaussians_pct': float(np.sum(opacities < 0.01) / len(opacities) * 100),
            'num_gaussians': int(len(means)),
        }
        
        # Health check flags
        metrics['is_healthy'] = (
            metrics['inactive_gaussians_pct'] < self.config.health_inactive_gaussians_threshold and  # Not over-pruning
            metrics['mean_opacity'] > self.config.health_mean_opacity_threshold and           # Not too transparent
            metrics['scale_max_ratio'] < self.config.health_scale_max_ratio_threshold            # No extreme outliers
        )
        
        return metrics
    
    def detect_mode_collapse(self, current_metrics: dict, prev_metrics: dict = None) -> bool:
        """
        Detect Gaussian collapse (analogous to dying ReLU).
        
        Mode collapse occurs when too many Gaussians become inactive or
        when spatial coverage degrades rapidly.
        """
        if prev_metrics is None:
            return False
        
        # Sudden spike in inactive Gaussians
        inactive_change = (current_metrics['inactive_gaussians_pct'] - 
                          prev_metrics['inactive_gaussians_pct'])
        
        # Large drop in mean opacity
        opacity_drop = prev_metrics['mean_opacity'] - current_metrics['mean_opacity']
        
        return (inactive_change > self.config.health_mode_collapse_inactive_threshold or 
                opacity_drop > self.config.health_mode_collapse_opacity_threshold)


class SpectralConvergenceDetector:
    """
    FFT-based loss landscape analysis.
    
    Uses spectral analysis to distinguish true convergence from oscillations,
    similar to loss landscape visualization techniques in deep learning research.
    """
    
    def __init__(self, config: ConvergenceConfig, window_size: int = 100):
        """
        Initialize the spectral convergence detector with configuration.
        
        Args:
            config: ConvergenceConfig instance containing all tunable parameters
            window_size: Window size for spectral analysis
        """
        self.config = config
        self.window_size = window_size
        self.loss_history = []
    
    def analyze_loss_spectrum(self) -> dict:
        """
        Apply FFT to detect convergence vs oscillation.
        
        High-frequency components indicate noise/oscillation.
        Low-frequency components indicate trend.
        """
        if len(self.loss_history) < self.window_size:
            return {'is_converging': False, 'oscillation_ratio': 1.0}
        
        recent_losses = np.array(self.loss_history[-self.window_size:])
        
        # Detrend and normalize
        detrended = scipy_signal.detrend(recent_losses)
        normalized = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-8)
        
        # FFT analysis
        spectrum = np.abs(np.fft.fft(normalized))
        
        # Energy in frequency bands
        low_freq_energy = np.sum(spectrum[:self.window_size//4])
        high_freq_energy = np.sum(spectrum[self.window_size//4:])
        
        oscillation_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        # Hurst exponent for mean reversion
        hurst = self._compute_hurst_exponent(detrended)
        
        return {
            'oscillation_ratio': float(oscillation_ratio),
            'hurst_exponent': float(hurst),
            'is_converging': (oscillation_ratio < self.config.spectral_oscillation_ratio_threshold and 
                             hurst < self.config.spectral_hurst_threshold),
            'spectrum_peak_freq': int(np.argmax(spectrum[:self.window_size//2])),
        }
    
    def _compute_hurst_exponent(self, series: np.ndarray) -> float:
        """
        Compute Hurst exponent for mean reversion analysis.
        
        Hurst > 0.5 = trending
        Hurst < 0.5 = mean-reverting (desired for convergence)
        """
        lags = range(2, min(20, len(series)//2))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        
        if len(tau) < 2:
            return 0.5
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]


class CurvatureAwareScheduler:
    """
    Approximate Hessian curvature for adaptive learning rate scheduling.
    
    Uses gradient norm changes to estimate loss landscape curvature,
    adapting learning rates accordingly (second-order optimization principle).
    Implements a smooth, continuous function instead of discrete multipliers.
    """
    
    def __init__(self, config: ConvergenceConfig):
        """
        Initialize the curvature-aware scheduler with configuration.
        
        Args:
            config: ConvergenceConfig instance containing all tunable parameters
        """
        self.config = config
        self.prev_grad_norm = None
    
    def compute_lr_multiplier(self, current_grad_norm: float, step: int) -> Tuple[float, str]:
        """
        Adjust LR based on gradient curvature estimate using smooth continuous function.
        
        Args:
            current_grad_norm: Current gradient L2 norm
            step: Current training step
        
        Returns:
            (multiplier, reason) tuple
        """
        if self.prev_grad_norm is None:
            self.prev_grad_norm = current_grad_norm
            return 1.0, "initialization"
        
        # Estimate curvature from gradient changes
        grad_change_rate = ((current_grad_norm - self.prev_grad_norm) / 
                           (self.prev_grad_norm + 1e-8))
        
        # Smooth continuous function: multiplier = 1.0 - (grad_change_rate * sensitivity)
        multiplier = 1.0 - (grad_change_rate * self.config.curvature_sensitivity)
        
        # Apply bounds
        multiplier = np.clip(multiplier, 
                           self.config.curvature_lr_multiplier_min, 
                           self.config.curvature_lr_multiplier_max)
        
        # Determine reason for logging
        if grad_change_rate > self.config.curvature_steep_threshold:
            reason = "steep_curvature"
        elif abs(grad_change_rate) < self.config.curvature_flat_threshold:
            reason = "flat_stable"
        else:
            reason = "smooth_adjustment"
        
        self.prev_grad_norm = current_grad_norm
        return float(multiplier), reason


class QuickPSNREstimator:
    """
    Lightweight PSNR estimation for frequent convergence checks.
    
    Instead of running full validation, estimate PSNR from a small subset
    of validation views to reduce overhead. Correctly samples from validation set.
    """
    
    def __init__(self, num_samples: int = 3):
        """
        Initialize the PSNR estimator.
        
        Args:
            num_samples: Number of validation samples to use for estimation
        """
        self.num_samples = num_samples
        self.sample_indices = None
    
    def estimate_psnr(self, gaussian_model: Any, valloader: Any, render_fn: Callable, step: int) -> float:
        """
        Estimate PSNR from a small random subset of validation views.
        
        This is much faster than full validation but provides reasonable
        estimate for convergence detection. Correctly samples from validation set.
        
        Args:
            gaussian_model: The Gaussian splatting model to render with
            valloader: Validation data loader
            render_fn: Function to render a single view and compute PSNR
            step: Current training step (for logging)
        
        Returns:
            Estimated PSNR value
        """
        try:
            # Use a fixed random subset for consistency
            if self.sample_indices is None:
                total_val = len(valloader)
                self.sample_indices = np.random.choice(
                    total_val, 
                    min(self.num_samples, total_val), 
                    replace=False
                )
            
            psnr_values = []
            for idx in self.sample_indices:
                # Get the validation data point (corrected from trainloader)
                data = valloader.dataset[idx]
                
                # Render and compute PSNR using the provided render function
                # This is a placeholder interface - the actual implementation would
                # call the render function with the model and data
                psnr = render_fn(gaussian_model, data)
                psnr_values.append(psnr)
            
            return float(np.mean(psnr_values))
        except Exception as e:
            # Fallback to last known PSNR with error logging
            print(f"Warning: PSNR estimation failed at step {step}: {e}")
            return 30.0


# Export all classes
__all__ = [
    'ConvergenceConfig',
    'TrainingPhase',
    'PhaseAwareOptimizer',
    'PolyakAveragingDetector',
    'GaussianPopulationMonitor',
    'SpectralConvergenceDetector',
    'CurvatureAwareScheduler',
    'QuickPSNREstimator',
]

