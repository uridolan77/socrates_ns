import copy
import datetime
import uuid
import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.functional as F
import contextlib

# Enhanced context manager for temporary quantization during sensitivity analysis
@contextlib.contextmanager
def tempquant(model, target_module, bits):
    """
    Temporarily quantize a specific module for evaluation with enhanced support for
    different bit precision levels and quantization schemes.
    
    Args:
        model: Model containing the module to quantize
        target_module: Name of module to quantize
        bits: Bit precision for quantization (2, 4, 8, etc.)
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
    
    if target_mod is None:
        raise ValueError(f"Module {target_module} not found in model")
    
    try:
        # Apply appropriate quantization based on bit depth
        if bits == 8:
            # 8-bit quantization using PyTorch's quantization
            with torch.no_grad():
                # Convert to dynamic quantization mode
                torch.quantization.quantize_dynamic(
                    target_mod,  # Module to quantize
                    {nn.Linear, nn.Conv2d},  # Types of layers to quantize
                    dtype=torch.qint8  # Quantization type
                )
        elif bits == 4:
            # 4-bit quantization (manual implementation)
            for param_name, param in target_mod.named_parameters():
                if param.dim() > 0:  # Skip scalar parameters
                    with torch.no_grad():
                        # Calculate quantization scale
                        max_val = float(param.abs().max())
                        scale = (2**(bits-1) - 1) / max_val if max_val > 0 else 1.0
                        
                        # Quantize weights
                        quantized = torch.round(param.data * scale)
                        quantized = torch.clamp(quantized, -2**(bits-1), 2**(bits-1)-1)
                        param.data = quantized / scale
        elif bits == 2:
            # 2-bit quantization (manual implementation for extreme compression)
            for param_name, param in target_mod.named_parameters():
                if param.dim() > 0:  # Skip scalar parameters
                    with torch.no_grad():
                        # Calculate quantization scale
                        max_val = float(param.abs().max())
                        scale = (2**(bits-1) - 1) / max_val if max_val > 0 else 1.0
                        
                        # Quantize weights to just -1, 0, 1
                        quantized = torch.round(param.data * scale)
                        quantized = torch.clamp(quantized, -2**(bits-1), 2**(bits-1)-1)
                        param.data = quantized / scale
        else:
            # Default fallback for other bit values
            for param_name, param in target_mod.named_parameters():
                if param.dim() > 0:  # Skip scalar parameters
                    with torch.no_grad():
                        # Calculate quantization scale
                        max_val = float(param.abs().max())
                        scale = (2**(bits-1) - 1) / max_val if max_val > 0 else 1.0
                        
                        # Quantize weights
                        quantized = torch.round(param.data * scale)
                        quantized = torch.clamp(quantized, -2**(bits-1), 2**(bits-1)-1)
                        param.data = quantized / scale
        
        # Yield to allow evaluation of the temporarily quantized model
        yield
    finally:
        # Restore original parameters
        for name, module in model.named_modules():
            if name == target_module:
                for param_name, param in module.named_parameters():
                    if param_name in orig_state:
                        param.data = orig_state[param_name]


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
    
    def __init__(self, model, compliance_evaluator):
        self.model = model
        self.compliance_evaluator = compliance_evaluator
        self.precision_mapping = {}
        self.compliance_modules = self._identify_compliance_modules()
        
        # Configure quantization options
        self.available_precisions = [8, 4, 2]  # Available bit precision options
        
        # Initialize quantization APIs
        self.quant_backend = 'fbgemm'  # Use fbgemm for server quantization (alternative: qnnpack for mobile)
        self.supported_ops = {nn.Linear, nn.Conv2d, nn.ReLU}  # Ops that support quantization
        
    def _identify_compliance_modules(self):
        """
        Identify compliance-critical modules in the model using naming patterns
        and structural analysis.
        """
        compliance_modules = []
        
        # Identify modules by name pattern
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in [
                'compliance', 'regulatory', 'rule', 'constraint', 
                'filter', 'verify', 'check', 'gate'
            ]):
                compliance_modules.append(name)
                
        # Analyze model structure to find potential compliance bottlenecks
        # This would identify modules that act as information bottlenecks in compliance workflows
        
        return compliance_modules

    def analyze_sensitivity(self, calibration_data, bits_options=None):
        """
        Analyze quantization sensitivity of different model components.
        
        Args:
            calibration_data: Dataset for calibration
            bits_options: Different bit-width options to test (default: [8, 4, 2])
            
        Returns:
            Sensitivity analysis for each module
        """
        if bits_options is None:
            bits_options = self.available_precisions
            
        sensitivity_results = {}
        
        # Get baseline compliance score with full precision
        self.model.eval()  # Set model to evaluation mode
        baseline_score = self.compliance_evaluator.evaluate(self.model, calibration_data)
        print(f"Baseline compliance score: {baseline_score:.4f}")
        
        # Create a histogram of module types to report distribution
        module_types = {}
        for name, module in self.model.named_modules():
            if not list(module.parameters()):  # Skip modules without parameters
                continue
                
            module_type = module.__class__.__name__
            module_types[module_type] = module_types.get(module_type, 0) + 1
            
        print(f"Module type distribution: {module_types}")
        
        # Test each module with different precision levels
        for name, module in self.model.named_modules():
            if not list(module.parameters()):  # Skip modules without parameters
                continue
                
            module_sensitivity = {}
            
            for bits in bits_options:
                try:
                    # Temporarily quantize this module and evaluate compliance
                    with tempquant(self.model, target_module=name, bits=bits):
                        # Evaluate compliance with temporary quantization
                        quant_score = self.compliance_evaluator.evaluate(
                            self.model, calibration_data
                        )
                        
                    # Calculate relative compliance drop
                    compliance_drop = (baseline_score - quant_score) / baseline_score
                    
                    # Store results
                    module_sensitivity[bits] = {
                        'compliance_score': quant_score,
                        'compliance_drop': compliance_drop,
                        'sensitivity': self._calculate_sensitivity_score(compliance_drop, bits)
                    }
                    
                    print(f"Module {name} with {bits}-bit precision: score={quant_score:.4f}, drop={compliance_drop:.4%}")
                    
                except Exception as e:
                    print(f"Error testing {name} with {bits}-bit quantization: {str(e)}")
                    # Use a placeholder for modules that fail quantization
                    module_sensitivity[bits] = {
                        'compliance_score': 0.0,
                        'compliance_drop': 1.0,
                        'sensitivity': 1.0,
                        'error': str(e)
                    }
                    
            sensitivity_results[name] = module_sensitivity
            
        return sensitivity_results
    
    def _calculate_sensitivity_score(self, compliance_drop, bits):
        """Calculate sensitivity score based on compliance drop and bit precision"""
        # Higher sensitivity score means more sensitive to quantization
        # Weight by bit depth - lower precision should lead to higher sensitivity
        bit_factor = (8 / bits) - 0.5  # 8-bit: 0.5, 4-bit: 1.5, 2-bit: 3.5
        return compliance_drop * bit_factor
        
    def _determine_optimal_precision(self, sensitivity_results, compliance_threshold=0.99):
        """
        Determine optimal precision for each module based on sensitivity
        
        Args:
            sensitivity_results: Results from sensitivity analysis
            compliance_threshold: Minimum acceptable compliance after quantization
            
        Returns:
            Mapping of module names to optimal bit precision
        """
        precision_mapping = {}
        baseline_score = self.compliance_evaluator.baseline_score
        
        for name, sensitivities in sensitivity_results.items():
            # Default to high precision for compliance modules
            if name in self.compliance_modules:
                precision_mapping[name] = max(self.available_precisions)
                continue
                
            # For non-critical modules, find the lowest precision that maintains compliance
            found_precision = False
            
            # Sort bit options from lowest to highest
            for bits in sorted(sensitivities.keys()):
                sensitivity_info = sensitivities[bits]
                
                # Skip if there was an error with this precision level
                if 'error' in sensitivity_info:
                    continue
                    
                score = sensitivity_info['compliance_score']
                
                # Check if this precision maintains acceptable compliance
                if score >= compliance_threshold * baseline_score:
                    precision_mapping[name] = bits
                    found_precision = True
                    break
                    
            # If no precision level is acceptable, use highest precision
            if not found_precision and self.available_precisions:
                precision_mapping[name] = max(self.available_precisions)
                
        return precision_mapping
    
    def quantize(self, calibration_data=None, compliance_threshold=0.99):
        """
        Quantize model with mixed precision based on compliance sensitivity
        
        Args:
            calibration_data: Dataset for calibration (if not already analyzed)
            compliance_threshold: Minimum acceptable compliance after quantization
            
        Returns:
            Quantized model with mixed precision
        """
        # Run sensitivity analysis if not already done
        if not hasattr(self, 'sensitivity_results') and calibration_data is not None:
            self.sensitivity_results = self.analyze_sensitivity(calibration_data)
            
        # Determine optimal precision for each module
        self.precision_mapping = self._determine_optimal_precision(
            self.sensitivity_results, compliance_threshold
        )
        
        # Apply mixed-precision quantization
        print(f"Applying mixed precision quantization with {len(self.precision_mapping)} modules")
        quantized_model = copy.deepcopy(self.model)
        
        # Prepare model for quantization
        quantized_model.eval()
        
        # Create quantization configuration
        quantization_config = torch.quantization.get_default_qconfig(self.quant_backend)
        
        # Apply module-specific quantization
        for name, module in quantized_model.named_modules():
            if name in self.precision_mapping:
                bits = self.precision_mapping[name]
                
                # Apply custom quantization for the module
                self._quantize_module(module, bits, name)
                
        # Trace any failed or skipped modules
        actual_precision = {name: self._get_actual_precision(quantized_model, name) 
                          for name in self.precision_mapping.keys()}
        
        # Record which modules were successfully quantized
        self.quantized_modules = {name: bits for name, bits in actual_precision.items()
                               if bits != 32}  # 32-bit indicates not quantized
                               
        print(f"Successfully quantized {len(self.quantized_modules)} modules")
        
        return quantized_model
    
    def _quantize_module(self, module, bits, module_name):
        """
        Apply quantization to a single module with specified bit-width
        
        Args:
            module: The PyTorch module to quantize
            bits: Target bit precision
            module_name: Name of the module for logging
        """
        # Skip quantization for modules with no parameters
        if not list(module.parameters()):
            return

        # For 8-bit quantization, use PyTorch's built-in quantization
        if bits == 8:
            # Check if module type is supported for quantization
            module_type = module.__class__
            
            if module_type in self.supported_ops:
                try:
                    # Configure the module for quantization
                    module.qconfig = torch.quantization.get_default_qconfig(self.quant_backend)
                    
                    # Prepare the module for quantization
                    torch.quantization.prepare(module, inplace=True)
                    
                    # Convert to quantized module
                    torch.quantization.convert(module, inplace=True)
                    
                    print(f"Successfully applied 8-bit quantization to {module_name}")
                except Exception as e:
                    print(f"Error applying PyTorch quantization to {module_name}: {str(e)}")
                    # Fall back to manual implementation
                    self._apply_manual_quantization(module, bits)
            else:
                # Fall back to manual quantization for unsupported ops
                self._apply_manual_quantization(module, bits)
        else:
            # Use manual quantization for 4-bit and 2-bit
            self._apply_manual_quantization(module, bits)
            
    def _apply_manual_quantization(self, module, bits):
        """Apply manual quantization to module parameters"""
        for param_name, param in module.named_parameters():
            if param.dim() > 0:  # Skip scalar parameters
                with torch.no_grad():
                    # Calculate quantization scale based on parameter range
                    max_val = float(param.abs().max())
                    scale = (2**(bits-1) - 1) / max_val if max_val > 0 else 1.0
                    
                    # Quantize the weights
                    quantized = torch.round(param.data * scale)
                    quantized = torch.clamp(quantized, -2**(bits-1), 2**(bits-1)-1)
                    param.data = quantized / scale
                    
                    # Store quantization parameters for potential dequantization
                    if not hasattr(module, 'quantization_params'):
                        module.quantization_params = {}
                    
                    module.quantization_params[param_name] = {
                        'bits': bits,
                        'scale': scale,
                        'max_val': max_val
                    }
    
    def _get_actual_precision(self, model, module_name):
        """Determine the actual precision of a module after quantization"""
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
        """Generate detailed report on quantization decisions"""
        if not hasattr(self, 'sensitivity_results'):
            return "No sensitivity analysis results available"
            
        report = {
            "compliance_modules": self.compliance_modules,
            "precision_distribution": self._get_precision_distribution(),
            "memory_savings": self._calculate_memory_savings(),
            "module_details": self._get_module_details(),
            "compliance_critical_modules": self._get_compliance_critical_details()
        }
        
        return report
    
    def _get_precision_distribution(self):
        """Get distribution of precision levels across the model"""
        distribution = {}
        for bits in set(self.precision_mapping.values()):
            count = sum(1 for b in self.precision_mapping.values() if b == bits)
            distribution[bits] = count
            
        return distribution
    
    def _calculate_memory_savings(self):
        """Calculate memory savings from quantization"""
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
            "parameter_distribution": param_counts
        }
    
    def _get_module_details(self):
        """Get detailed information about each module's quantization"""
        details = {}
        for name in self.precision_mapping:
            bits = self.precision_mapping[name]
            is_compliance = name in self.compliance_modules
            
            if hasattr(self, 'sensitivity_results') and name in self.sensitivity_results:
                sensitivity = self.sensitivity_results[name][bits]['sensitivity']
            else:
                sensitivity = "Unknown"
                
            details[name] = {
                "precision_bits": bits,
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
                    
                    bits = self.precision_mapping.get(name, 32)
                    
                    critical_details[name] = {
                        "parameters": module_params,
                        "percentage_of_model": (module_params / total_params) * 100,
                        "bits": bits
                    }
                    
        critical_details["summary"] = {
            "critical_parameters": critical_params,
            "total_parameters": total_params,
            "percentage_critical": (critical_params / total_params) * 100,
            "average_bits": sum(details["bits"] for details in critical_details.values()) / len(critical_details) 
                            if critical_details else 32
        }
        
        return critical_details