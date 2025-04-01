import copy
import datetime
import uuid
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.functional as F
import contextlib

# Configure logging
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def tempquant(model, target_module, bits, quant_method='symmetric', per_channel=False):
    """
    Enhanced context manager for temporary quantization during sensitivity analysis.
    
    Args:
        model: Model containing the module to quantize
        target_module: Name of module to quantize
        bits: Bit precision for quantization (2, 3, 4, 8, etc.)
        quant_method: Quantization method ('symmetric', 'asymmetric', or 'power_of_2')
        per_channel: Whether to use per-channel quantization (True) or per-tensor (False)
    
    Yields:
        None: Yields control back to the context while module is quantized
    
    Raises:
        ValueError: If target module not found or quantization parameters invalid
    """
    # Store original parameters
    orig_state = {}
    target_mod = None
    
    # Find target module
    for name, module in model.named_modules():
        if name == target_module:
            target_mod = module
            # Store original parameters
            for param_name, param in module.named_parameters():
                orig_state[param_name] = param.data.clone()
            break
    
    if target_mod is None:
        raise ValueError(f"Module {target_module} not found in model")
    
    try:
        # Validate quantization parameters
        if bits not in [2, 3, 4, 8]:
            raise ValueError(f"Bit precision {bits} not supported. Use 2, 3, 4, or 8 bits.")
        
        if quant_method not in ['symmetric', 'asymmetric', 'power_of_2']:
            raise ValueError(f"Quantization method {quant_method} not supported.")
        
        # Apply appropriate quantization based on bit depth
        if bits == 8:
            # 8-bit quantization using PyTorch's quantization
            with torch.no_grad():
                # Create dynamic or static quantization config based on module type
                if isinstance(target_mod, (nn.RNN, nn.LSTM, nn.GRU)):
                    # For recurrent layers, use dynamic quantization
                    torch.quantization.quantize_dynamic(
                        target_mod,  # Module to quantize
                        {type(target_mod)},  # Type of layer to quantize
                        dtype=torch.qint8  # Quantization type
                    )
                else:
                    # For other layers, use static or dynamic based on module type
                    module_types = {nn.Linear, nn.Conv2d}
                    
                    # For linear and conv modules, try to apply static quantization if possible
                    if isinstance(target_mod, tuple(module_types)):
                        # Use per-channel quantization if requested and supported
                        qconfig = torch.quantization.get_default_qconfig('fbgemm' if per_channel else 'qnnpack')
                        
                        # Create a temporary submodule for quantization
                        temp_module = nn.Sequential(target_mod)
                        temp_module.qconfig = qconfig
                        
                        # Prepare and convert
                        torch.quantization.prepare(temp_module, inplace=True)
                        # Note: In a real scenario, calibration data would be used here
                        torch.quantization.convert(temp_module, inplace=True)
                        
                        # Update the target module from the quantized version
                        target_mod = temp_module[0]
                    else:
                        # Fall back to dynamic quantization
                        torch.quantization.quantize_dynamic(
                            target_mod,
                            {nn.Linear, nn.Conv2d, nn.RNN, nn.LSTM, nn.GRU},
                            dtype=torch.qint8
                        )
        else:
            # Manual implementation for lower bit precision
            for param_name, param in target_mod.named_parameters():
                if param.dim() > 0:  # Skip scalar parameters
                    with torch.no_grad():
                        # Apply different quantization methods
                        if quant_method == 'symmetric':
                            # Symmetric quantization
                            self._apply_symmetric_quantization(param, bits, per_channel)
                        elif quant_method == 'asymmetric':
                            # Asymmetric quantization
                            self._apply_asymmetric_quantization(param, bits, per_channel)
                        elif quant_method == 'power_of_2':
                            # Power of 2 scale quantization (for efficient implementation)
                            self._apply_power_of_2_quantization(param, bits, per_channel)
        
        # Yield to allow evaluation of the temporarily quantized model
        yield
    finally:
        # Restore original parameters
        for name, module in model.named_modules():
            if name == target_module:
                for param_name, param in module.named_parameters():
                    if param_name in orig_state:
                        param.data.copy_(orig_state[param_name])

def _apply_symmetric_quantization(param, bits, per_channel=False):
    """Apply symmetric quantization to parameter tensor"""
    # Calculate quantization range
    n_levels = 2**bits
    min_level = -(n_levels // 2)
    max_level = (n_levels // 2) - 1
    
    if per_channel and param.dim() > 1:
        # Per-channel quantization (along first dimension)
        dim = 0  # Typically weights are [out_channels, in_channels, ...]
        scales = param.abs().amax(dim=tuple(range(1, param.dim())), keepdim=True) / max_level
        scales = scales.clamp(min=1e-8)  # Prevent division by zero
        
        # Quantize and dequantize
        param_q = torch.round(param / scales).clamp_(min_level, max_level)
        param.copy_(param_q * scales)
    else:
        # Per-tensor quantization
        scale = param.abs().max() / max_level
        if scale == 0:
            scale = 1e-8  # Prevent division by zero
            
        # Quantize and dequantize
        param_q = torch.round(param / scale).clamp_(min_level, max_level)
        param.copy_(param_q * scale)

def _apply_asymmetric_quantization(param, bits, per_channel=False):
    """Apply asymmetric quantization to parameter tensor"""
    # Calculate quantization range
    n_levels = 2**bits
    
    if per_channel and param.dim() > 1:
        # Per-channel quantization
        dim = 0  # Typically weights are [out_channels, in_channels, ...]
        
        # Calculate min and max per channel
        max_val = param.amax(dim=tuple(range(1, param.dim())), keepdim=True)
        min_val = param.amin(dim=tuple(range(1, param.dim())), keepdim=True)
        
        # Handle case where min == max
        is_same = (max_val == min_val)
        min_val = torch.where(is_same, max_val - 1.0, min_val)
        
        # Calculate scale and zero point
        scales = (max_val - min_val) / (n_levels - 1)
        scales = scales.clamp(min=1e-8)  # Prevent division by zero
        zero_points = torch.round(-min_val / scales)
        
        # Quantize and dequantize
        param_q = torch.round(param / scales + zero_points).clamp_(0, n_levels - 1)
        param.copy_((param_q - zero_points) * scales)
    else:
        # Per-tensor quantization
        max_val = param.max()
        min_val = param.min()
        
        # Handle case where min == max
        if max_val == min_val:
            min_val = max_val - 1.0
            
        # Calculate scale and zero point
        scale = (max_val - min_val) / (n_levels - 1)
        if scale == 0:
            scale = 1e-8  # Prevent division by zero
        zero_point = torch.round(-min_val / scale)
        
        # Quantize and dequantize
        param_q = torch.round(param / scale + zero_point).clamp_(0, n_levels - 1)
        param.copy_((param_q - zero_point) * scale)

def _apply_power_of_2_quantization(param, bits, per_channel=False):
    """Apply power-of-2 scale quantization for efficient implementation"""
    # Calculate quantization range
    n_levels = 2**bits
    min_level = -(n_levels // 2)
    max_level = (n_levels // 2) - 1
    
    if per_channel and param.dim() > 1:
        # Per-channel quantization
        dim = 0  # Typically weights are [out_channels, in_channels, ...]
        
        # Calculate max per channel
        max_val = param.abs().amax(dim=tuple(range(1, param.dim())), keepdim=True)
        
        # Find nearest power of 2 for scale
        log2 = torch.log2(max_val)
        log2_ceil = torch.ceil(log2)
        scales = 2 ** log2_ceil / max_level
        scales = scales.clamp(min=1e-8)  # Prevent division by zero
        
        # Quantize and dequantize
        param_q = torch.round(param / scales).clamp_(min_level, max_level)
        param.copy_(param_q * scales)
    else:
        # Per-tensor quantization
        max_val = param.abs().max()
        
        # Find nearest power of 2 for scale
        if max_val > 0:
            log2 = torch.log2(max_val)
            log2_ceil = torch.ceil(log2)
            scale = 2 ** log2_ceil / max_level
        else:
            scale = 1e-8  # Prevent division by zero
            
        # Quantize and dequantize
        param_q = torch.round(param / scale).clamp_(min_level, max_level)
        param.copy_(param_q * scale)


class ComplianceAwareMixedPrecisionQuantizer:
    """
    Mixed-precision quantizer that preserves compliance-critical components
    while aggressively quantizing non-critical components.
    
    Features:
    1. Automatically identifies compliance-critical modules
    2. Performs sensitivity analysis to determine optimal bit precision
    3. Applies mixed precision quantization across the model
    4. Supports PyTorch's quantization framework
    5. Provides detailed reporting on quantization decisions
    """
    
    def __init__(self, model, compliance_evaluator, config=None):
        """
        Initialize the mixed-precision quantizer.
        
        Args:
            model: The model to quantize
            compliance_evaluator: Evaluator for compliance metrics
            config: Configuration dictionary
        """
        self.model = model
        self.compliance_evaluator = compliance_evaluator
        self.config = config or {}
        self.precision_mapping = {}
        self.sensitivity_results = {}
        
        # Set up configurable parameters with defaults
        self._setup_parameters()
        
        # Identify compliance-critical modules
        self.compliance_modules = self._identify_compliance_modules()
        logger.info(f"Identified {len(self.compliance_modules)} compliance-critical modules")
        
        # Record baseline metrics
        self.baseline_metrics = {
            "params": self._count_parameters(model),
            "memory": self._estimate_memory_footprint(model)
        }
        
    def _setup_parameters(self):
        """Set up configurable parameters with defaults"""
        self.params = {
            # Available precision options
            "precisions": self.config.get("precisions", [8, 4, 2]),
            
            # Quantization backend
            "quant_backend": self.config.get("quant_backend", 'fbgemm'),
            
            # Supported operations for built-in quantization
            "supported_ops": self.config.get("supported_ops", {nn.Linear, nn.Conv2d, nn.ReLU}),
            
            # Quantization method ('symmetric', 'asymmetric', 'power_of_2')
            "quant_method": self.config.get("quant_method", 'symmetric'),
            
            # Whether to use per-channel quantization when possible
            "per_channel": self.config.get("per_channel", True),
            
            # Minimum compliance threshold as percentage of baseline
            "compliance_threshold": self.config.get("compliance_threshold", 0.99),
            
            # Keywords for identifying compliance modules by name
            "compliance_keywords": self.config.get("compliance_keywords", [
                'compliance', 'regulatory', 'rule', 'constraint', 
                'filter', 'verify', 'check', 'gate'
            ]),
            
            # Whether to preserve input/output modules at higher precision
            "preserve_io_precision": self.config.get("preserve_io_precision", True),
            
            # Number of calibration samples to use for quantization
            "calibration_samples": self.config.get("calibration_samples", 100),
            
            # Whether to enable quantization-aware fine-tuning
            "enable_qat": self.config.get("enable_qat", False),
            
            # QAT training parameters
            "qat_epochs": self.config.get("qat_epochs", 5),
            "qat_learning_rate": self.config.get("qat_learning_rate", 1e-5),
        }

    def _identify_compliance_modules(self):
        """
        Identify compliance-critical modules in the model using multiple methods:
        1. Naming patterns 
        2. Structural analysis
        3. IO modules preservation (optional)
        
        Returns:
            List of module names deemed compliance-critical
        """
        compliance_modules = set()
        
        # 1. Identify modules by name pattern
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in self.params["compliance_keywords"]):
                compliance_modules.add(name)
                
                # Also add parent module for context preservation
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    compliance_modules.add(parent_name)
        
        # 2. Identify IO modules if configured to preserve them
        if self.params["preserve_io_precision"]:
            # Find input modules (typically first few layers)
            for name, module in self.model.named_modules():
                # Use heuristics to identify input modules
                if isinstance(module, (nn.Embedding, nn.Linear, nn.Conv2d)):
                    if not any(name.startswith(parent) for parent in compliance_modules 
                             if parent != name):
                        # This is a top-level module not under a compliance module
                        compliance_modules.add(name)
                        break  # Only add the first one
            
            # Find output modules (typically final layers)
            output_candidates = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    output_candidates.append(name)
            
            # Take the last linear layer as output
            if output_candidates:
                compliance_modules.add(output_candidates[-1])
        
        # 3. Advanced: Use model structure analysis
        # This would require tracing the model to find bottlenecks
        # but for simplicity, we'll skip this in this implementation
        
        return list(compliance_modules)

    def analyze_sensitivity(self, calibration_data, bits_options=None, num_batches=10):
        """
        Analyze quantization sensitivity of different model components.
        
        Args:
            calibration_data: Dataset for calibration
            bits_options: Different bit-width options to test
            num_batches: Number of batches to evaluate for faster analysis
            
        Returns:
            Sensitivity analysis for each module
        """
        if bits_options is None:
            bits_options = self.params["precisions"]
            
        sensitivity_results = {}
        
        # Get baseline compliance score with full precision
        self.model.eval()  # Set model to evaluation mode
        try:
            baseline_score = self.compliance_evaluator.evaluate(
                self.model, calibration_data, num_batches=num_batches
            )
            # Ensure we store the baseline score for later use
            self.baseline_score = baseline_score
            logger.info(f"Baseline compliance score: {baseline_score:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating baseline compliance: {e}")
            raise RuntimeError(f"Failed to establish baseline compliance: {e}")
        
        # Create a histogram of module types to report distribution
        module_types = {}
        for name, module in self.model.named_modules():
            if not list(module.parameters()):  # Skip modules without parameters
                continue
                
            module_type = module.__class__.__name__
            module_types[module_type] = module_types.get(module_type, 0) + 1
            
        logger.info(f"Module type distribution: {module_types}")
        
        # Process modules in batches to avoid excessive memory usage
        # Group modules by type for more efficient analysis
        module_groups = self._group_modules_by_type()
        
        # Analyze each module group
        for group_name, module_names in module_groups.items():
            logger.info(f"Analyzing sensitivity for module group: {group_name} ({len(module_names)} modules)")
            
            # Select a representative sample of modules from large groups
            max_sample_size = 10  # To avoid excessive testing time
            if len(module_names) > max_sample_size and group_name != "compliance_critical":
                import random
                sampled_names = random.sample(module_names, max_sample_size)
                logger.info(f"Testing {max_sample_size} sampled modules out of {len(module_names)}")
            else:
                sampled_names = module_names
            
            # Test each module with different precision levels
            for module_name in sampled_names:
                if not self._has_parameters(module_name):
                    continue
                    
                logger.info(f"Testing module: {module_name}")
                module_sensitivity = {}
                
                for bits in bits_options:
                    try:
                        # Temporarily quantize this module and evaluate compliance
                        with tempquant(
                            self.model, 
                            target_module=module_name, 
                            bits=bits,
                            quant_method=self.params["quant_method"],
                            per_channel=self.params["per_channel"]
                        ):
                            # Evaluate compliance with temporary quantization
                            quant_score = self.compliance_evaluator.evaluate(
                                self.model, calibration_data, num_batches=num_batches
                            )
                            
                        # Calculate relative compliance drop
                        compliance_drop = (baseline_score - quant_score) / baseline_score
                        
                        # Store results
                        module_sensitivity[bits] = {
                            'compliance_score': quant_score,
                            'compliance_drop': compliance_drop,
                            'sensitivity': self._calculate_sensitivity_score(compliance_drop, bits)
                        }
                        
                        logger.info(f"Module {module_name} with {bits}-bit precision: score={quant_score:.4f}, drop={compliance_drop:.4%}")
                        
                    except Exception as e:
                        logger.warning(f"Error testing {module_name} with {bits}-bit quantization: {str(e)}")
                        # Use a placeholder for modules that fail quantization
                        module_sensitivity[bits] = {
                            'compliance_score': 0.0,
                            'compliance_drop': 1.0,
                            'sensitivity': 1.0,
                            'error': str(e)
                        }
                        
                sensitivity_results[module_name] = module_sensitivity
            
            # If we sampled modules, apply the average sensitivity to other modules in the group
            if len(sampled_names) < len(module_names) and sampled_names:
                # Calculate average sensitivity for each bit precision
                avg_sensitivities = {}
                for bits in bits_options:
                    sensitivities = [
                        sensitivity_results[name][bits]['sensitivity'] 
                        for name in sampled_names 
                        if name in sensitivity_results and bits in sensitivity_results[name]
                    ]
                    if sensitivities:
                        avg_sensitivities[bits] = sum(sensitivities) / len(sensitivities)
                    else:
                        avg_sensitivities[bits] = 1.0  # Conservative default
                
                # Apply the average sensitivity to non-tested modules
                for name in module_names:
                    if name not in sampled_names:
                        module_sensitivity = {}
                        for bits in bits_options:
                            module_sensitivity[bits] = {
                                'compliance_score': baseline_score * (1 - avg_sensitivities[bits] / 10),  # Estimate
                                'compliance_drop': avg_sensitivities[bits] / 10,  # Conservative estimate
                                'sensitivity': avg_sensitivities[bits],
                                'estimated': True  # Mark as estimated
                            }
                        sensitivity_results[name] = module_sensitivity
        
        # Store results for later use
        self.sensitivity_results = sensitivity_results
        
        return sensitivity_results
    
    def _group_modules_by_type(self):
        """Group modules by type to optimize sensitivity analysis"""
        groups = {
            "compliance_critical": [],
            "embedding": [],
            "linear": [],
            "conv": [],
            "norm": [],
            "other": []
        }
        
        for name, module in self.model.named_modules():
            if not self._has_parameters(name):
                continue
                
            if name in self.compliance_modules:
                groups["compliance_critical"].append(name)
            elif isinstance(module, nn.Embedding):
                groups["embedding"].append(name)
            elif isinstance(module, nn.Linear):
                groups["linear"].append(name)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                groups["conv"].append(name)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                groups["norm"].append(name)
            else:
                groups["other"].append(name)
                
        return groups
    
    def _has_parameters(self, module_name):
        """Check if a module has parameters to quantize"""
        for name, module in self.model.named_modules():
            if name == module_name:
                return any(p.numel() > 0 for p in module.parameters())
        return False
    
    def _calculate_sensitivity_score(self, compliance_drop, bits):
        """
        Calculate sensitivity score based on compliance drop and bit precision.
        
        A higher sensitivity score means the module is more sensitive to quantization.
        """
        # Weight by bit depth - lower precision should lead to higher sensitivity
        bit_factor = (8 / bits) - 0.5  # 8-bit: 0.5, 4-bit: 1.5, 2-bit: 3.5
        
        # Calculate basic sensitivity - exponential to emphasize large drops
        basic_sensitivity = compliance_drop * bit_factor
        
        # Apply exponential scaling to emphasize critical drops
        if compliance_drop > 0.1:  # More than 10% drop is considered critical
            enhanced_sensitivity = np.exp(compliance_drop * 2) * bit_factor
            return max(basic_sensitivity, enhanced_sensitivity)
        
        return basic_sensitivity
        
    def _determine_optimal_precision(self, sensitivity_results=None, compliance_threshold=None):
        """
        Determine optimal precision for each module based on sensitivity.
        
        Args:
            sensitivity_results: Results from sensitivity analysis
            compliance_threshold: Minimum acceptable compliance after quantization
            
        Returns:
            Mapping of module names to optimal bit precision
        """
        if sensitivity_results is None:
            sensitivity_results = self.sensitivity_results
            
        if compliance_threshold is None:
            compliance_threshold = self.params["compliance_threshold"]
            
        precision_mapping = {}
        
        # Always prioritize compliance modules - use highest precision
        highest_precision = max(self.params["precisions"])
        for name in self.compliance_modules:
            precision_mapping[name] = highest_precision
            logger.info(f"Setting compliance-critical module {name} to {highest_precision}-bit precision")
        
        # For all other modules, find optimal precision based on sensitivity
        baseline_score = getattr(self, 'baseline_score', 1.0)
        
        for name, sensitivities in sensitivity_results.items():
            # Skip if already set (e.g., compliance module)
            if name in precision_mapping:
                continue
                
            # For non-critical modules, find the lowest precision that maintains compliance
            found_precision = False
            
            # Sort bit options from lowest to highest
            for bits in sorted(self.params["precisions"]):
                if bits not in sensitivities:
                    continue
                    
                sensitivity_info = sensitivities[bits]
                
                # Skip if there was an error with this precision level
                if 'error' in sensitivity_info:
                    continue
                    
                score = sensitivity_info.get('compliance_score', 0)
                
                # Check if this precision maintains acceptable compliance
                if score >= compliance_threshold * baseline_score:
                    precision_mapping[name] = bits
                    found_precision = True
                    break
                    
            # If no precision level is acceptable, use highest precision
            if not found_precision:
                precision_mapping[name] = highest_precision
                logger.warning(f"No suitable precision found for {name}, using {highest_precision}-bit")
                
        return precision_mapping
    
    def quantize(self, calibration_data=None, compliance_threshold=None):
        """
        Quantize model with mixed precision based on compliance sensitivity.
        
        Args:
            calibration_data: Dataset for calibration (if not already analyzed)
            compliance_threshold: Minimum acceptable compliance after quantization
            
        Returns:
            Quantized model with mixed precision
        """
        # Set compliance threshold if provided
        if compliance_threshold is not None:
            self.params["compliance_threshold"] = compliance_threshold
        
        # Run sensitivity analysis if not already done
        if not self.sensitivity_results and calibration_data is not None:
            logger.info("Running sensitivity analysis...")
            self.sensitivity_results = self.analyze_sensitivity(calibration_data)
            
        if not self.sensitivity_results:
            raise ValueError("Sensitivity analysis must be run before quantization")
            
        # Determine optimal precision for each module
        self.precision_mapping = self._determine_optimal_precision(
            self.sensitivity_results, self.params["compliance_threshold"]
        )
        
        # Create a copy of the model for quantization
        logger.info(f"Applying mixed precision quantization with {len(self.precision_mapping)} modules")
        quantized_model = copy.deepcopy(self.model)
        
        # Prepare model for quantization
        quantized_model.eval()
        
        # Apply calibration if needed
        if calibration_data is not None and 8 in self.params["precisions"]:
            logger.info("Calibrating model for quantization")
            self._calibrate_model(quantized_model, calibration_data)
        
        # Apply module-specific quantization
        for name, module in quantized_model.named_modules():
            if name in self.precision_mapping:
                bits = self.precision_mapping[name]
                
                try:
                    # Apply quantization for the module
                    self._quantize_module(
                        module, 
                        bits, 
                        name,
                        method=self.params["quant_method"],
                        per_channel=self.params["per_channel"]
                    )
                    logger.info(f"Quantized module {name} to {bits}-bit precision")
                except Exception as e:
                    logger.error(f"Error quantizing module {name}: {e}")
                
        # Trace which modules were successfully quantized
        actual_precision = {
            name: self._get_actual_precision(quantized_model, name) 
            for name in self.precision_mapping.keys()
        }
        
        # Record which modules were successfully quantized
        self.quantized_modules = {
            name: bits for name, bits in actual_precision.items()
            if bits != 32  # 32-bit indicates not quantized
        }
        
        logger.info(f"Successfully quantized {len(self.quantized_modules)} modules")
        
        # Perform quantization-aware training if enabled
        if self.params["enable_qat"] and calibration_data is not None:
            logger.info("Performing quantization-aware training...")
            quantized_model = self._perform_quantization_aware_training(
                quantized_model, 
                calibration_data
            )
        
        return quantized_model
    
    def _calibrate_model(self, model, calibration_data):
        """
        Calibrate model with data for static quantization.
        
        This runs a forward pass with calibration data to collect
        activation statistics for quantization.
        
        Args:
            model: Model to calibrate
            calibration_data: Dataset for calibration
        """
        # Create a calibration function that will run the model with calibration data
        def calibrate(model):
            model.eval()
            with torch.no_grad():
                # Process a limited number of batches
                num_batches = min(len(calibration_data), self.params["calibration_samples"])
                
                for i, batch in enumerate(calibration_data):
                    if i >= num_batches:
                        break
                        
                    # Forward pass to collect statistics
                    if isinstance(batch, dict):
                        # Handle dictionary input
                        inputs = {k: v for k, v in batch.items() 
                                 if k in ['input_ids', 'attention_mask', 'inputs']}
                        _ = model(**inputs)
                    else:
                        # Handle tensor input
                        _ = model(batch)
        
        # Record modules that need calibration (8-bit quantized modules)
        modules_to_calibrate = [
            name for name, bits in self.precision_mapping.items()
            if bits == 8
        ]
        
        # Only execute if we have modules to calibrate
        if not modules_to_calibrate:
            return
            
        # Prepare observers for modules that will use 8-bit quantization
        for name, module in model.named_modules():
            if name in modules_to_calibrate:
                # Check if the module is of a type that supports PyTorch quantization
                if type(module) in self.params["supported_ops"]:
                    # Configure the module for quantization
                    module.qconfig = torch.quantization.get_default_qconfig(self.params["quant_backend"])
        
        # Prepare model for calibration
        prepared_model = torch.quantization.prepare(model, inplace=False)
        
        # Run calibration
        try:
            calibrate(prepared_model)
            logger.info("Calibration completed successfully")
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            # Continue with uncalibrated model
    
    def _quantize_module(self, module, bits, module_name, method='symmetric', per_channel=True):
        """
        Apply quantization to a single module with specified bit-width.
        
        Args:
            module: The PyTorch module to quantize
            bits: Target bit precision
            module_name: Name of the module for logging
            method: Quantization method ('symmetric', 'asymmetric', 'power_of_2')
            per_channel: Whether to use per-channel quantization when possible
        """
        # Skip quantization for modules with no parameters
        if not any(p.numel() > 0 for p in module.parameters()):
            logger.debug(f"Skipping quantization for {module_name} - no parameters")
            return

        # For 8-bit quantization, use PyTorch's built-in quantization
        if bits == 8:
            # Check if module type is supported for quantization
            module_type = module.__class__
            
            if module_type in self.params["supported_ops"]:
                try:
                    # Configure the module for quantization
                    module.qconfig = torch.quantization.get_default_qconfig(self.params["quant_backend"])
                    
                    # Prepare the module for quantization
                    torch.quantization.prepare(module, inplace=True)
                    
                    # Convert to quantized module
                    torch.quantization.convert(module, inplace=True)
                    
                    logger.info(f"Successfully applied 8-bit quantization to {module_name}")
                except Exception as e:
                    logger.warning(f"Error applying PyTorch quantization to {module_name}: {str(e)}")
                    # Fall back to manual implementation
                    self._apply_manual_quantization(module, bits, method, per_channel)
            else:
                # Fall back to manual quantization for unsupported ops
                logger.debug(f"Using manual quantization for unsupported module type: {module_type}")
                self._apply_manual_quantization(module, bits, method, per_channel)
        else:
            # Use manual quantization for lower bit-widths
            self._apply_manual_quantization(module, bits, method, per_channel)
    
    def _apply_manual_quantization(self, module, bits, method='symmetric', per_channel=True):
        """
        Apply manual quantization to module parameters.
        
        Args:
            module: Module to quantize
            bits: Number of bits (2, 3, 4)
            method: Quantization method
            per_channel: Whether to use per-channel quantization
        """
        for param_name, param in module.named_parameters():
            if param.dim() > 0:  # Skip scalar parameters
                with torch.no_grad():
                    if method == 'symmetric':
                        _apply_symmetric_quantization(param, bits, per_channel)
                    elif method == 'asymmetric':
                        _apply_asymmetric_quantization(param, bits, per_channel)
                    elif method == 'power_of_2':
                        _apply_power_of_2_quantization(param, bits, per_channel)
                    else:
                        raise ValueError(f"Unsupported quantization method: {method}")
                    
                    # Store quantization parameters for potential dequantization
                    if not hasattr(module, 'quantization_params'):
                        module.quantization_params = {}
                    
                    module.quantization_params[param_name] = {
                        'bits': bits,
                        'method': method,
                        'per_channel': per_channel
                    }
    
    def _perform_quantization_aware_training(self, model, dataset, num_epochs=None, learning_rate=None):
        """
        Perform quantization-aware training to fine-tune the quantized model.
        
        Args:
            model: Quantized model to fine-tune
            dataset: Training dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate for QAT
            
        Returns:
            Fine-tuned quantized model
        """
        # Use configured values if not provided
        num_epochs = num_epochs or self.params["qat_epochs"]
        learning_rate = learning_rate or self.params["qat_learning_rate"]
        
        try:
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Simple training loop
            model.train()
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in dataset:
                    # Forward pass
                    if isinstance(batch, dict):
                        # Assume model accepts inputs like a HuggingFace model
                        outputs = model(
                            input_ids=batch.get('input_ids'),
                            attention_mask=batch.get('attention_mask'),
                            labels=batch.get('labels')
                        )
                        loss = outputs.loss
                    else:
                        # Simple case - assume batch has inputs and targets
                        inputs, targets = batch
                        outputs = model(inputs)
                        loss = F.cross_entropy(outputs, targets)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                # Log progress
                avg_loss = epoch_loss / max(1, num_batches)
                logger.info(f"QAT Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                
                # Check compliance on a subset
                try:
                    compliance_score = self.compliance_evaluator.evaluate(
                        model, dataset, num_batches=min(10, len(dataset))
                    )
                    logger.info(f"QAT Epoch {epoch+1}, Compliance: {compliance_score:.4f}")
                except Exception as e:
                    logger.warning(f"Error evaluating compliance during QAT: {e}")
            
            logger.info("Quantization-aware training completed")
            return model
        
        except Exception as e:
            logger.error(f"Error during quantization-aware training: {e}")
            # Return the original model if QAT fails
            return model
    
    def _get_actual_precision(self, model, module_name):
        """
        Determine the actual precision of a module after quantization.
        
        Args:
            model: Quantized model
            module_name: Name of the module to check
            
        Returns:
            Bit precision (or 32 if not quantized)
        """
        # Find the module
        target_module = None
        for name, module in model.named_modules():
            if name == module_name:
                target_module = module
                break
                
        if target_module is None:
            return 32  # Not found, assume full precision
            
        # Check if the module has quantization parameters
        if hasattr(target_module, 'quantization_params'):
            # Return the minimum bit width across all parameters
            return min(params['bits'] for params in target_module.quantization_params.values())
            
        # Check if this is a PyTorch quantized module
        if hasattr(target_module, '_modules'):
            for submodule in target_module._modules.values():
                if 'quantized' in submodule.__class__.__name__.lower():
                    return 8  # PyTorch quantization is 8-bit
                    
        return 32  # No quantization detected
    
    def generate_report(self):
        """
        Generate detailed report on quantization decisions.
        
        Returns:
            Dictionary with detailed report information
        """
        if not hasattr(self, 'sensitivity_results'):
            return {"error": "No sensitivity analysis results available"}
            
        report = {
            "model_info": {
                "original_parameters": self.baseline_metrics["params"],
                "original_size_mb": self.baseline_metrics["memory"],
                "module_count": sum(1 for _ in self.model.named_modules() if self._has_parameters(_)),
                "quantized_module_count": len(self.quantized_modules) if hasattr(self, 'quantized_modules') else 0
            },
            "compliance_modules": {
                "count": len(self.compliance_modules),
                "modules": self.compliance_modules
            },
            "precision_distribution": self._get_precision_distribution(),
            "memory_savings": self._calculate_memory_savings(),
            "module_details": self._get_module_details(),
            "compliance_critical_modules": self._get_compliance_critical_details(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _get_precision_distribution(self):
        """Get distribution of precision levels across the model"""
        if not hasattr(self, 'precision_mapping'):
            return {}
            
        distribution = {}
        for bits in set(self.precision_mapping.values()):
            count = sum(1 for b in self.precision_mapping.values() if b == bits)
            distribution[int(bits)] = count
            
        return distribution
    
    def _calculate_memory_savings(self):
        """Calculate memory savings from quantization"""
        if not hasattr(self, 'precision_mapping'):
            return {}
            
        original_bits = 32  # Assuming FP32 original model
        
        # Count parameters by precision
        param_counts = {}
        for name, module in self.model.named_modules():
            if name in self.precision_mapping:
                bits = self.precision_mapping[name]
                params = sum(p.numel() for p in module.parameters())
                
                if bits not in param_counts:
                    param_counts[bits] = 0
                param_counts[bits] += params
        
        # Calculate original and new storage requirements
        original_size = sum(param_counts.values()) * (original_bits / 8)  # in bytes
        new_size = sum(count * (bits / 8) for bits, count in param_counts.items())
        
        savings = 1.0 - (new_size / original_size)
        
        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": new_size / (1024 * 1024),
            "savings_percentage": savings * 100,
            "parameter_distribution": {int(k): v for k, v in param_counts.items()}
        }
    
    def _get_module_details(self):
        """Get detailed information about each module's quantization"""
        if not hasattr(self, 'precision_mapping') or not hasattr(self, 'sensitivity_results'):
            return {}
            
        details = {}
        for name in self.precision_mapping:
            bits = self.precision_mapping[name]
            is_compliance = name in self.compliance_modules
            
            sensitivity = "Unknown"
            if name in self.sensitivity_results and bits in self.sensitivity_results[name]:
                sensitivity_data = self.sensitivity_results[name][bits]
                if isinstance(sensitivity_data, dict) and 'sensitivity' in sensitivity_data:
                    sensitivity = sensitivity_data['sensitivity']
                
            details[name] = {
                "precision_bits": int(bits),
                "is_compliance_critical": is_compliance,
                "sensitivity": sensitivity
            }
            
        return details
        
    def _get_compliance_critical_details(self):
        """Get detailed report on compliance-critical modules"""
        if not self.compliance_modules:
            return {}
            
        critical_details = {}
        
        # Calculate what percentage of the model is considered compliance-critical
        total_params = sum(p.numel() for p in self.model.parameters())
        critical_params = 0
        
        for name in self.compliance_modules:
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module_params = sum(p.numel() for p in module.parameters())
                    critical_params += module_params
                    
                    bits = self.precision_mapping.get(name, 32) if hasattr(self, 'precision_mapping') else 32
                    
                    critical_details[name] = {
                        "parameters": module_params,
                        "percentage_of_model": (module_params / total_params) * 100,
                        "bits": int(bits),
                        "module_type": type(module).__name__
                    }
                    
        critical_details["summary"] = {
            "critical_parameters": critical_params,
            "total_parameters": total_params,
            "percentage_critical": (critical_params / total_params) * 100,
            "average_bits": (
                sum(details["bits"] for name, details in critical_details.items() 
                    if name != "summary") / 
                max(1, len(critical_details) - 1)
            ) if critical_details else 32
        }
        
        return critical_details
    
    def _generate_recommendations(self):
        """Generate recommendations based on quantization results"""
        if not hasattr(self, 'precision_mapping') or not hasattr(self, 'sensitivity_results'):
            return ["Complete sensitivity analysis first to get recommendations"]
            
        recommendations = []
        
        # Check memory savings
        memory_savings = self._calculate_memory_savings()
        savings_percentage = memory_savings.get("savings_percentage", 0)
        
        if savings_percentage < 20:
            recommendations.append(
                "Consider using more aggressive quantization - current memory savings are less than 20%"
            )
            
        # Check if too many modules are at full precision
        precision_dist = self._get_precision_distribution()
        high_precision_count = precision_dist.get(8, 0)
        total_modules = sum(precision_dist.values())
        
        if high_precision_count / max(1, total_modules) > 0.7:
            recommendations.append(
                "Too many modules (>70%) at 8-bit precision. Consider relaxing compliance threshold to reduce model size further."
            )
            
        # Check compliance modules
        critical_details = self._get_compliance_critical_details()
        critical_percentage = critical_details.get("summary", {}).get("percentage_critical", 0)
        
        if critical_percentage < 10:
            recommendations.append(
                "Consider identifying more compliance-critical modules - current selection is less than 10% of the model."
            )
            
        # Check quantization method
        if self.params["quant_method"] == "symmetric" and savings_percentage < 50:
            recommendations.append(
                "Try asymmetric quantization method which may offer better accuracy-compression trade-off."
            )
            
        # Check for QAT
        if not self.params["enable_qat"] and any(b < 8 for b in precision_dist.keys()):
            recommendations.append(
                "Enable quantization-aware training (QAT) for better accuracy when using 4-bit or 2-bit quantization."
            )
            
        # If no special recommendations, give general advice
        if not recommendations:
            recommendations.append(
                "Current quantization configuration appears optimal. Consider experimenting with different bit precision combinations for specific modules if needed."
            )
            
        return recommendations
    
    def _count_parameters(self, model):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in model.parameters())
    
    def _estimate_memory_footprint(self, model):
        """Estimate memory footprint of model in MB"""
        params = sum(p.numel() for p in model.parameters())
        # Assume 4 bytes per parameter for FP32
        return (params * 4) / (1024 * 1024)
