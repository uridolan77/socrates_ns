import sys
import os
import optimizer
import copy
import datetime
from dataclasses import dataclass, field, batch
from typing import List, Dict, Any, Optional, Tuple 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from src.compliance.utils.balanced_sampler import ComplianceBalancedSampler
from src.neuro.dataset_mock import (
    load_general_compliance_dataset,
    load_framework_dataset,
    load_compliance_edge_cases,
    load_compliance_adversarial
)

# Ensure the correct Python environment is being used
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Add the current directory to the Python path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class ComplianceAwareDistillation:
    """
    Knowledge distillation framework that preserves compliance capabilities
    while creating smaller, efficient models for specific regulatory domains
    """
    
    def __init__(self, teacher_model, student_architecture, regulatory_framework):
        """
        Initialize the compliance-aware distillation framework
        
        Args:
            teacher_model: Large, fully-capable compliance model
            student_architecture: Architecture specification for student model
            regulatory_framework: Target regulatory framework (e.g., 'GDPR', 'HIPAA')
        """
        self.teacher = teacher_model
        self.student = self._initialize_student(student_architecture)
        self.regulatory_framework = regulatory_framework
        self.framework_datasets = self._load_framework_datasets()
        
        # Track compliance metrics during distillation
        self.compliance_metrics = {
            "baseline": self._evaluate_teacher_compliance(),
            "history": []
        }
        
    def _initialize_student(self, architecture):
        """Initialize student model based on architecture spec"""
        # Implementation would create appropriate student model
        # Could be from scratch or a pretrained smaller model
        # Define or import SmallerComplianceModel
        # Placeholder implementation for SmallerComplianceModel
        class SmallerComplianceModel(nn.Module):
            def __init__(self, architecture):
                super(SmallerComplianceModel, self).__init__()
                # Define the architecture of the smaller model here
                self.architecture = architecture
                self.dummy_layer = nn.Linear(10, 10)  # Example layer

            def forward(self, x):
                return self.dummy_layer(x)

        return SmallerComplianceModel(architecture)

    def _load_framework_datasets(self):
        """Load datasets specific to the target regulatory framework"""
        datasets = {
            "general": load_general_compliance_dataset(),
            "framework_specific": load_framework_dataset(self.regulatory_framework),
            "edge_cases": load_compliance_edge_cases(self.regulatory_framework),
            "adversarial": load_compliance_adversarial(self.regulatory_framework)
        }
        
        return datasets
    
    def _evaluate_teacher_compliance(self):
        """Evaluate teacher model compliance as baseline"""
        metrics = {}
        
        for dataset_name, dataset in self.framework_datasets.items():
            metrics[dataset_name] = self.evaluate_compliance(
                self.teacher, dataset, self.regulatory_framework
            )
            
        metrics["overall"] = sum(metrics.values()) / len(metrics)
        return metrics
        
    def distill(self, epochs=50, batch_size=32, temperature=2.0, alpha=0.5):
        """
        Perform compliance-aware knowledge distillation
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            temperature: Softmax temperature for distillation
            alpha: Weight balancing distillation and compliance losses
            
        Returns:
            Trained student model with preserved compliance capabilities
        """
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-4)
        
        # Setup learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=epochs
        )
        
        # Calculate effective training steps
        total_steps = sum(len(d) // batch_size for d in self.framework_datasets.values()) * epochs
        print(f"Starting distillation for {self.regulatory_framework} compliance")
        print(f"Training for {epochs} epochs, {total_steps} total steps")
        
        # Track best model
        best_model = None
        best_compliance = 0.0
        
        for epoch in range(epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(
                epoch, batch_size, temperature, alpha
            )
            
            # Evaluate compliance on validation sets
            compliance_metrics = self._evaluate_student_compliance()
            
            # Track history
            self.compliance_metrics["history"].append({
                "epoch": epoch,
                "train": train_metrics,
                "compliance": compliance_metrics
            })
            
            # Update best model if improved
            if compliance_metrics["overall"] > best_compliance:
                best_compliance = compliance_metrics["overall"]
                best_model = copy.deepcopy(self.student)
                print(f"Epoch {epoch}: New best model with compliance score {best_compliance:.4f}")
            
            # Log progress
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train loss: {train_metrics['total_loss']:.4f}")
            print(f"  Distillation loss: {train_metrics['distillation_loss']:.4f}")
            print(f"  Compliance loss: {train_metrics['compliance_loss']:.4f}")
            print(f"  Overall compliance: {compliance_metrics['overall']:.4f}")
            print(f"  Teacher compliance: {self.compliance_metrics['baseline']['overall']:.4f}")
            print(f"  Compliance ratio: {compliance_metrics['overall'] / self.compliance_metrics['baseline']['overall']:.4f}")
            
            # Update learning rate
            scheduler.step()
            
        # Restore best model
        if best_model is not None:
            self.student = best_model
            
        # Final compliance evaluation
        final_compliance = self._evaluate_student_compliance()
        self.compliance_metrics["final"] = final_compliance
        
        return self.student
    
    def _train_epoch(self, epoch, batch_size, temperature, alpha):
        """Train student model for one epoch"""
        self.student.train()
        self.teacher.eval()
        
        # Initialize metrics
        metrics = {
            "total_loss": 0.0,
            "distillation_loss": 0.0,
            "compliance_loss": 0.0,
            "steps": 0
        }
        
        # Create regulatory dataset sampler that balances different compliance aspects
        sampler = ComplianceBalancedSampler(self.framework_datasets)
        
        for batch in sampler.get_batches(batch_size):
            # Forward pass through teacher model
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    batch["input_ids"], 
                    batch["attention_mask"]
                )
                
            # Forward pass through student model
            student_outputs = self.student(
                batch["input_ids"],
                batch["attention_mask"]
            )
            
            # Compute distillation loss
            distillation_loss = self._compute_distillation_loss(
                student_outputs, teacher_outputs, temperature
            )
            
            # Compute compliance loss
            compliance_loss = self._compute_compliance_loss(
                student_outputs, batch
            )
            
            # Combined loss
            loss = alpha * distillation_loss + (1 - alpha) * compliance_loss
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update metrics
            metrics["total_loss"] += loss.item()
            metrics["distillation_loss"] += distillation_loss.item()
            metrics["compliance_loss"] += compliance_loss.item()
            metrics["steps"] += 1
            
            # Dynamic alpha adjustment based on compliance gap
            if metrics["steps"] % 100 == 0:
                # Periodically evaluate compliance
                compliance = self._evaluate_student_compliance(sample=True)
                teacher_compliance = self.compliance_metrics["baseline"]["overall"]
                
                # Adjust alpha based on compliance gap
                compliance_ratio = compliance["overall"] / teacher_compliance
                if compliance_ratio < 0.9:
                    # Increase focus on compliance if falling behind
                    alpha = max(0.1, alpha - 0.05)
                elif compliance_ratio > 0.99:
                    # Increase focus on distillation if compliance is good
                    alpha = min(0.9, alpha + 0.05)
        
        # Average metrics
        for key in ["total_loss", "distillation_loss", "compliance_loss"]:
            metrics[key] /= metrics["steps"]
            
        return metrics
    
    def _compute_distillation_loss(self, student_outputs, teacher_outputs, temperature):
        """
        Compute knowledge distillation loss
        
        Uses temperature-scaled softmax to create soft targets from teacher
        """
        # Get logits from outputs
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Apply temperature scaling
        scaled_student = student_logits / temperature
        scaled_teacher = teacher_logits / temperature
        
        # Compute KL divergence loss
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        loss = loss_fn(
            F.log_softmax(scaled_student, dim=-1),
            F.softmax(scaled_teacher, dim=-1)
        )
        
        # Scale by temperature²
        return loss * (temperature ** 2)
    
    def _compute_compliance_loss(self, student_outputs, batch):
        """
        Compute specialized compliance loss
        
        Enforces regulatory constraints directly during training
        """
        # Standard task loss (e.g., token prediction)
        task_loss = F.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            batch["labels"].view(-1)
        )
        
        # Regulatory constraint loss
        regulatory_loss = self._regulatory_constraint_loss(
            student_outputs, batch
        )
        
        # Combine losses with higher weight on regulatory constraints
        combined_loss = 0.3 * task_loss + 0.7 * regulatory_loss
        
        return combined_loss
    
    def _regulatory_constraint_loss(self, outputs, batch):
        """
        Compute loss based on regulatory constraints
        
        Implements specific loss terms for the target regulatory framework
        """
        # Extract relevant constraint information from batch
        constraints = batch.get("regulatory_constraints", {})
        
        # Initialize constraint losses
        constraint_losses = []
        
        # Example: GDPR-specific constraints
        if self.regulatory_framework == "GDPR":
            # Data minimization constraint
            if "data_minimization" in constraints:
                min_loss = self._compute_data_minimization_loss(
                    outputs, constraints["data_minimization"]
                )
                constraint_losses.append(min_loss)
                
            # Purpose limitation constraint
            if "purpose_limitation" in constraints:
                purpose_loss = self._compute_purpose_limitation_loss(
                    outputs, constraints["purpose_limitation"]
                )
                constraint_losses.append(purpose_loss)
            
            # Additional GDPR-specific constraints...
            
        # Example: HIPAA-specific constraints
        elif self.regulatory_framework == "HIPAA":
            # PHI protection constraint
            if "phi_protection" in constraints:
                phi_loss = self._compute_phi_protection_loss(
                    outputs, constraints["phi_protection"]
                )
                constraint_losses.append(phi_loss)
                
            # Additional HIPAA-specific constraints...
            
        # Default to generic regulatory constraints if none specific
        if not constraint_losses:
            constraint_losses.append(self._compute_generic_regulatory_loss(outputs, batch))
            
        # Combine all constraint losses
        if constraint_losses:
            return sum(constraint_losses) / len(constraint_losses)
        else:
            # Return zero tensor if no constraints
            return torch.tensor(0.0, device=outputs.logits.device)
    
    def _compute_data_minimization_loss(self, outputs, constraint):
        """Compute loss enforcing GDPR data minimization principle"""
        # Extract predicted tokens with high probability
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(5, dim=-1)
        
        # Check if predicted tokens contain personal data markers
        personal_data_tokens = constraint.get("personal_data_tokens", [])
        personal_data_ids = [self.tokenizer.convert_tokens_to_ids(token) 
                            for token in personal_data_tokens]
        
        # Calculate penalty for generating personal data tokens not in input
        penalty = 0.0
        for token_id in personal_data_ids:
            # Check if token_id appears in top_indices and isn't in input tokens
            contains_token = (top_indices == token_id).any(dim=-1)
            in_input = (constraint["input_ids"] == token_id).any(dim=-1)
            unnecessary_disclosure = contains_token & ~in_input
            if unnecessary_disclosure.any():
                penalty += unnecessary_disclosure.float().mean()
        
        return penalty

        
    def _compute_purpose_limitation_loss(self, outputs, constraint):
        """Compute loss enforcing GDPR purpose limitation principle"""
        # Implementation would encourage outputs that adhere to stated purpose
        return torch.tensor(0.1, device=outputs.logits.device)  # Placeholder
    
    def _compute_phi_protection_loss(self, outputs, constraint):
        """Compute loss enforcing HIPAA PHI protection requirements"""
        # Implementation would penalize disclosure of protected health information
        return torch.tensor(0.1, device=outputs.logits.device)  # Placeholder
    
    def _compute_generic_regulatory_loss(self, outputs, batch):
        """Compute generic regulatory compliance loss"""
        # Implementation would apply general compliance principles
        return torch.tensor(0.1, device=outputs.logits.device)  # Placeholder

  
    @staticmethod
    def evaluate_compliance(model, dataset, framework):
        """Evaluate model compliance on dataset for specific framework"""
        # Placeholder implementation for compliance evaluation
        # Replace with actual logic to evaluate compliance
        return 0.95  # Example compliance score

    def count_parameters(model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def sample_dataset(self, dataset, max_samples=100):
        """Create smaller sample of dataset for quick evaluation"""
        # Implementation would sample subset of dataset
        return dataset  # Placeholder

    # Helper estimation functions
    def count_parameters(model):
        """Count trainable parameters in model"""
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 0  # Default for models without parameter attribute

    def estimate_memory_footprint(model):
        """Estimate memory footprint of model in MB"""
        if hasattr(model, 'parameters'):
            params = sum(p.numel() for p in model.parameters())
            # Assume 4 bytes per parameter for FP32
            return (params * 4) / (1024 * 1024)
        return 0  # Default for models without parameter attribute

    def _evaluate_student_compliance(self, sample=False):
        """
        Evaluate student model compliance with regulatory framework
        
        Args:
            sample: If True, use smaller sample for efficiency during training
            
        Returns:
            Compliance metrics dictionary
        """
        self.student.eval()
        metrics = {}
        
        # Use either full datasets or sampled subsets
        datasets = {}
        if sample:
            # Use small samples during training
            for name, dataset in self.framework_datasets.items():
                datasets[name] = self.sample_dataset(dataset, max_samples=100)
        else:
            datasets = self.framework_datasets
        
        # Evaluate on each dataset
        for dataset_name, dataset in datasets.items():
            metrics[dataset_name] = ComplianceAwareDistillation.evaluate_compliance(
                self.student, dataset, self.regulatory_framework
            )
            
        # Calculate overall compliance
        metrics["overall"] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def generate_compliance_report(self):
        """
        Generate detailed compliance comparison report between teacher and student
        
        Returns:
            Dictionary with detailed compliance metrics and visualizations
        """
        # Verify we have completed distillation
        if "final" not in self.compliance_metrics:
            return {"error": "Distillation not completed yet"}
            
        # Compile comprehensive report
        report = {
            "model_comparison": {
                "teacher_parameters": ComplianceAwareDistillation.count_parameters(self.teacher),
                "student_parameters": ComplianceAwareDistillation.count_parameters(self.student),
                "compression_ratio": ComplianceAwareDistillation.count_parameters(self.teacher) / ComplianceAwareDistillation.count_parameters(self.student),
            },
            "compliance_comparison": {
                "teacher": self.compliance_metrics["baseline"],
                "student": self.compliance_metrics["final"],
                "ratio": {
                    k: self.compliance_metrics["final"][k] / self.compliance_metrics["baseline"][k]
                    for k in self.compliance_metrics["baseline"]
                }
            },
            "training_history": self.compliance_metrics["history"],
            "framework_specific": self._generate_framework_specific_report(),
            "conclusion": self._generate_conclusion()
        }
        
        return report
    
    def _generate_framework_specific_report(self):
        """Generate regulatory framework-specific detailed analysis"""
        # Implementation would provide detailed framework-specific metrics
        return {"framework": self.regulatory_framework}  # Placeholder
    
    def _generate_conclusion(self):
        """Generate conclusion and recommendations"""
        # Implementation would analyze results and provide recommendations
        student_overall = self.compliance_metrics["final"]["overall"]
        teacher_overall = self.compliance_metrics["baseline"]["overall"]
        ratio = student_overall / teacher_overall
        
        if ratio > 0.95:
            status = "Excellent"
            message = "Student model successfully preserves compliance capabilities"
        elif ratio > 0.9:
            status = "Good"
            message = "Student model retains most compliance capabilities"
        elif ratio > 0.8:
            status = "Acceptable"
            message = "Student model meets minimum compliance requirements but could be improved"
        else:
            status = "Insufficient"
            message = "Student model fails to adequately preserve compliance capabilities"
            
        return {
            "status": status,
            "message": message,
            "compliance_preservation": f"{ratio:.2%}",
            "recommendations": self._generate_recommendations(ratio)
        }
    
    def _generate_recommendations(self, compliance_ratio):
        """Generate specific recommendations based on results"""
        # Implementation would provide targeted recommendations
        recommendations = []
        
        if compliance_ratio < 0.9:
            recommendations.append(
                "Increase model capacity in compliance-critical components"
            )
            
        if compliance_ratio < 0.95:
            recommendations.append(
                "Fine-tune with specialized compliance datasets"
            )
            
        return recommendations or ["No specific recommendations required"]
    

    @staticmethod
    def calculate_similarity_score(predictions, expected):
        """
        Calculate semantic similarity between predictions and expected outputs
        
        Args:
            predictions: List of model text predictions
            expected: List of expected outputs
            
        Returns:
            Similarity score between 0-1
        """
        if not predictions or not expected:
            return 0.7  # Default moderate score if inputs are missing
        
        # In a real implementation, we would:
        # 1. Use sentence embeddings to compute vector representations
        # 2. Calculate cosine similarity between prediction and expected vectors
        
        # Simplified implementation using word overlap
        scores = []
        
        for i, pred in enumerate(predictions):
            if i < len(expected):
                exp = expected[i]
                
                # Tokenize to words
                pred_words = set(pred.lower().split())
                exp_words = set(exp.lower().split())
                
                # Calculate Jaccard similarity
                if pred_words or exp_words:
                    intersection = pred_words.intersection(exp_words)
                    union = pred_words.union(exp_words)
                    similarity = len(intersection) / len(union)
                else:
                    similarity = 1.0  # Both empty means perfect match
                    
                scores.append(similarity)
            else:
                # No expected output for this prediction
                scores.append(0.7)
        
        # Return average similarity across all pairs
        return sum(scores) / len(scores) if scores else 0.7

    def evaluate_phi_protection(self, predictions, phi_tokens, expected_redactions):
        """
        Evaluate compliance with HIPAA PHI protection requirements
        
        Higher score if PHI is properly redacted/protected
        
        Args:
            predictions: List of model text predictions
            phi_tokens: List of PHI tokens to check for (should be redacted)
            expected_redactions: List of expected redaction tokens/formats
            
        Returns:
            Compliance score between 0-1
        """
        if not predictions:
            return 0.7  # Default moderate score if predictions are missing
        
        scores = []
        
        for i, pred in enumerate(predictions):
            pred_score = 1.0
            phi_disclosures = 0
            redaction_misses = 0
            
            # Check for presence of PHI tokens (negative)
            for token in phi_tokens:
                if token in pred:
                    phi_disclosures += 1
            
            # Check for proper use of redactions (positive)
            if i < len(expected_redactions):
                expected = expected_redactions[i]
                for redaction in expected:
                    if redaction not in pred:
                        redaction_misses += 1
            
            # Calculate penalty for PHI disclosures (severe)
            if phi_disclosures > 0:
                disclosure_penalty = 0.15 * phi_disclosures + 0.1 * (phi_disclosures ** 2)
                pred_score -= disclosure_penalty
            
            # Calculate penalty for missing redactions (moderate)
            if redaction_misses > 0:
                redaction_penalty = 0.1 * redaction_misses
                pred_score -= redaction_penalty
            
            # Bonus for proper redaction
            if phi_disclosures == 0 and redaction_misses == 0:
                pred_score = min(1.0, pred_score + 0.05)
            
            scores.append(max(0.0, min(1.0, pred_score)))
        
        # Average score across all predictions
        return sum(scores) / len(scores) if scores else 0.7



    # Completed implementations for the critical methods in ComplianceAwareDistillation

    def _compute_data_minimization_loss(self, outputs, constraint):
        """
        Compute loss enforcing GDPR data minimization principle.
        
        This loss penalizes the model for generating unnecessary personal data 
        tokens that weren't present in the input, encouraging minimal data usage.
        
        Args:
            outputs: Model outputs containing logits
            constraint: Dictionary containing data minimization constraints
            
        Returns:
            Tensor containing the data minimization loss
        """
        # Extract predicted tokens with high probability
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(5, dim=-1)
        
        # Get personal data token IDs
        personal_data_tokens = constraint.get("personal_data_tokens", [])
        personal_data_ids = [self.tokenizer.convert_tokens_to_ids(token) 
                            for token in personal_data_tokens]
        
        # Calculate penalty for generating personal data tokens not in input
        penalty = torch.tensor(0.0, device=logits.device)
        for token_id in personal_data_ids:
            # Check if token_id appears in top_indices and isn't in input tokens
            contains_token = (top_indices == token_id).any(dim=-1)
            in_input = (constraint["input_ids"] == token_id).any(dim=-1)
            unnecessary_disclosure = contains_token & ~in_input
            
            if unnecessary_disclosure.any():
                # Apply stronger penalty for tokens with higher probability
                token_positions = torch.where(unnecessary_disclosure)[0]
                for pos in token_positions:
                    # Find the position of the token in the top_k predictions
                    token_pos_in_topk = torch.where(top_indices[pos] == token_id)[0]
                    if len(token_pos_in_topk) > 0:
                        # Higher penalty for higher-ranked tokens
                        token_prob = top_probs[pos, token_pos_in_topk[0]]
                        penalty += token_prob * 2.0  # Scale by probability
        
        # Add entropy regularization to encourage more certain predictions
        # for non-personal data (more focused distribution)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        
        # Combined penalty with entropy regularization
        combined_penalty = penalty + 0.01 * entropy
        
        return combined_penalty


    def _compute_purpose_limitation_loss(self, outputs, constraint):
        """
        Compute loss enforcing GDPR purpose limitation principle.
        
        This loss encourages outputs to adhere to the stated purpose by penalizing
        responses that drift from the authorized purpose scope.
        
        Args:
            outputs: Model outputs containing logits
            constraint: Dictionary containing purpose limitation constraints
            
        Returns:
            Tensor containing the purpose limitation loss
        """
        logits = outputs.logits
        device = logits.device
        
        # Extract purpose-related information from constraints
        stated_purpose = constraint.get("stated_purpose", "")
        purpose_embedding = constraint.get("purpose_embedding", None)
        
        # If we don't have purpose embedding, calculate it
        if purpose_embedding is None and stated_purpose:
            # This would be replaced with actual embedding generation
            # For simplicity, we assume purpose_embedder is available
            purpose_embedding = self.purpose_embedder(stated_purpose)
        
        # If we still don't have purpose embedding, return minimal loss
        if purpose_embedding is None:
            return torch.tensor(0.1, device=device)
        
        # Get output sequence embeddings
        # Assuming we can extract embeddings from the student model
        output_embeddings = self.student.get_sequence_embedding(outputs)
        
        # Calculate cosine similarity between output and purpose embeddings
        similarity = F.cosine_similarity(output_embeddings, purpose_embedding, dim=-1)
        
        # Loss is higher when similarity is lower (penalize drift from purpose)
        purpose_loss = 1.0 - similarity.mean()
        
        # Additional checks for specific unauthorized purposes
        unauthorized_purposes = constraint.get("unauthorized_purposes", [])
        unauthorized_embeddings = [self.purpose_embedder(p) for p in unauthorized_purposes]
        
        # Penalize similarity with unauthorized purposes
        unauthorized_penalty = 0.0
        for unauth_emb in unauthorized_embeddings:
            unauth_similarity = F.cosine_similarity(output_embeddings, unauth_emb, dim=-1)
            unauthorized_penalty += torch.clamp(unauth_similarity.mean(), min=0.0)
        
        # Final combined loss
        combined_loss = purpose_loss + 0.5 * unauthorized_penalty
        
        return combined_loss


    def _compute_phi_protection_loss(self, outputs, constraint):
        """
        Compute loss enforcing HIPAA PHI protection requirements.
        
        This loss penalizes the disclosure of protected health information,
        encouraging the model to maintain patient privacy.
        
        Args:
            outputs: Model outputs containing logits
            constraint: Dictionary containing PHI protection constraints
            
        Returns:
            Tensor containing the PHI protection loss
        """
        logits = outputs.logits
        device = logits.device
        
        # Get probability distribution over vocabulary
        probs = F.softmax(logits, dim=-1)
        
        # Extract PHI-related tokens from constraint
        phi_tokens = constraint.get("phi_tokens", [])
        phi_token_ids = [self.tokenizer.convert_tokens_to_ids(token) 
                        for token in phi_tokens]
        
        # Get PHI redaction mapping (e.g., "John" -> "[PATIENT_NAME]")
        redaction_map = constraint.get("phi_redaction_map", {})
        redaction_ids = {}
        for phi, redaction in redaction_map.items():
            phi_id = self.tokenizer.convert_tokens_to_ids(phi)
            redaction_id = self.tokenizer.convert_tokens_to_ids(redaction)
            redaction_ids[phi_id] = redaction_id
        
        # Calculate PHI disclosure penalty
        phi_penalty = torch.tensor(0.0, device=device)
        
        # Penalize probabilities assigned to PHI tokens
        for phi_id in phi_token_ids:
            # Get the probability of generating this PHI token
            phi_prob = probs[:, :, phi_id].mean()
            
            # If this token has a redaction alternative, use that in the penalty calculation
            if phi_id in redaction_ids:
                redaction_id = redaction_ids[phi_id]
                redaction_prob = probs[:, :, redaction_id].mean()
                
                # Penalize if PHI probability is higher than redaction probability
                penalty = torch.max(phi_prob - redaction_prob, torch.tensor(0.0, device=device))
                phi_penalty += penalty * 2.0  # Stronger penalty for PHI with available redaction
            else:
                # Standard penalty for PHI tokens
                phi_penalty += phi_prob
        
        # Check for context-specific PHI disclosures
        context_allowed_phi = constraint.get("context_allowed_phi", False)
        if not context_allowed_phi:
            # Apply stronger penalty for any PHI in the output
            phi_penalty *= 2.0
        
        # Additional penalty for sensitive combinations (e.g., name+diagnosis)
        sensitive_combinations = constraint.get("sensitive_combinations", [])
        combination_penalty = torch.tensor(0.0, device=device)
        
        for combo in sensitive_combinations:
            combo_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in combo]
            
            # Check for presence of whole combination in top predictions
            presence = [(probs[:, :, token_id] > 0.05).any() for token_id in combo_ids]
            if all(presence):
                # All tokens in sensitive combination are present with significant probability
                combination_penalty += torch.tensor(1.0, device=device)
        
        # Final combined PHI protection loss
        phi_loss = phi_penalty + combination_penalty
        
        return phi_loss


    @staticmethod
    def evaluate_compliance(self, model, dataset, framework):
        """
        Evaluate model compliance on dataset for specific framework
        
        Args:
            model: The model to evaluate
            dataset: Compliance evaluation dataset
            framework: Regulatory framework (e.g., 'GDPR', 'HIPAA')
            
        Returns:
            Float between 0-1 representing compliance score
        """
        model.eval()
        total_score = 0.0
        framework_metrics = {}
        
        # Define framework-specific evaluation criteria
        if framework == "GDPR":
            criteria = {
                "data_minimization": 0.25,
                "purpose_limitation": 0.25,
                "accuracy": 0.15,
                "storage_limitation": 0.15,
                "confidentiality": 0.20
            }
        elif framework == "HIPAA":
            criteria = {
                "phi_protection": 0.35,
                "authorization": 0.25,
                "use_disclosure": 0.20,
                "safeguards": 0.20
            }
        else:
            # Generic compliance criteria
            criteria = {
                "data_protection": 0.4,
                "consent_handling": 0.3,
                "regulatory_alignment": 0.3
            }
        
        # Evaluate each criterion
        with torch.no_grad():
            for batch in dataset:
                # Evaluate individual criteria
                criterion_scores = {}
                
                for criterion, weight in criteria.items():
                    # Evaluate this specific criterion
                    criterion_score = self.evaluate_criterion(model, batch, criterion, framework)
                    criterion_scores[criterion] = criterion_score
                    
                    # Add to weighted total
                    total_score += criterion_score * weight
                
                # Track metrics per batch
                for criterion, score in criterion_scores.items():
                    if criterion not in framework_metrics:
                        framework_metrics[criterion] = []
                    framework_metrics[criterion].append(score)
        
        # Average metrics across all batches
        for criterion in framework_metrics:
            framework_metrics[criterion] = sum(framework_metrics[criterion]) / len(framework_metrics[criterion])
        
        # Calculate final weighted compliance score
        final_score = sum(framework_metrics[criterion] * weight for criterion, weight in criteria.items())
        
        return final_score


    def _evaluate_data_minimization(self, predictions, personal_data_tokens, input_texts):
        """
        Evaluate compliance with GDPR data minimization.

        Args:
            predictions: List of model text predictions.
            personal_data_tokens: List of personal data tokens to check.
            input_texts: List of input text strings.

        Returns:
            Compliance score between 0-1.
        """
        if not predictions or not personal_data_tokens or not input_texts:
            return 0.7  # Default moderate score if inputs are missing

        scores = []

        for pred, input_text in zip(predictions, input_texts):
            pred_score = 1.0
            unnecessary_disclosures = 0

            # Check each personal data token
            for token in personal_data_tokens:
                # If token in prediction but not in input, reduce score
                if token in pred and token not in input_text:
                    unnecessary_disclosures += 1
            score = self._evaluate_data_minimization(predictions, personal_data_tokens, batch["input_texts"])
            # Calculate score based on number of unnecessary disclosures
            # More severe penalty for multiple disclosures
            if unnecessary_disclosures > 0:
                penalty = 0.1 * unnecessary_disclosures + 0.05 * (unnecessary_disclosures ** 2)
                pred_score -= penalty

            scores.append(max(0.0, min(1.0, pred_score)))

        # Average score across all predictions
        return sum(scores) / len(scores) if scores else 0.7

    def _evaluate_criterion(self, model, batch, criterion, framework):
        """
        Evaluate a specific compliance criterion on a batch
        
        Args:
            model: The model to evaluate
            batch: Data batch
            criterion: The specific criterion to evaluate
            framework: Regulatory framework
            
        Returns:
            Score for this criterion (0-1)
        """
        # Process batch with model
        outputs = model(batch["input_ids"], batch["attention_mask"])
        
        # Decode outputs to text
        predictions = self.decode_outputs(outputs, model.tokenizer)
        
        # Get ground truth expected behaviors
        expected = batch.get(f"{criterion}_expected", [])
        
        # GDPR criteria evaluation
        if criterion == "data_minimization":
            # Check if model avoids generating personal data not in input
            personal_data_tokens = batch.get("personal_data_tokens", [])
            score = self._evaluate_data_minimization(predictions, personal_data_tokens, batch["input_texts"])
            
        elif criterion == "purpose_limitation":
            # Check if model respects stated purpose limitations
            purposes = batch.get("stated_purposes", [])
            score = self.evaluate_purpose_limitation(predictions, purposes)
        
        # HIPAA criteria evaluation
        elif criterion == "phi_protection":
            # Check if model properly protects PHI
            phi_tokens = batch.get("phi_tokens", [])
            redactions = batch.get("phi_redactions", [])
            score = self.evaluate_phi_protection(predictions, phi_tokens, redactions)
        
        # Generic criteria
        else:
            # Compare predictions against expected outputs for this criterion
            score = self.calculate_similarity_score(predictions, expected)
        
        return score


    @staticmethod
    def decode_outputs(outputs, tokenizer):
        """
        Convert model output logits to text predictions
        
        Args:
            outputs: Model outputs containing logits
            tokenizer: Tokenizer for decoding token IDs
            
        Returns:
            List of decoded text predictions
        """
        # Get most likely tokens
        token_ids = outputs.logits.argmax(dim=-1)
        
        # Decode to text
        predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
        
        return predictions


    def evaluate_purpose_limitation(self, predictions, stated_purposes):
        """
        Evaluate compliance with GDPR purpose limitation
        
        Higher score if predictions align with stated purposes
        
        Args:
            predictions: List of model text predictions
            stated_purposes: List of authorized purpose statements
            
        Returns:
            Compliance score between 0-1
        """
        if not predictions or not stated_purposes:
            return 0.7  # Default moderate score if inputs are missing
        
        scores = []
        purpose_keywords = {}
        
        # Extract keywords from purposes for simple keyword-based evaluation
        for i, purpose in enumerate(stated_purposes):
            # Simple keyword extraction (in real implementation, use proper NLP)
            keywords = purpose.lower().split()
            # Remove common words
            stopwords = ["the", "a", "an", "in", "to", "for", "of", "and", "or", "with"]
            keywords = [word for word in keywords if word not in stopwords and len(word) > 3]
            purpose_keywords[i] = keywords
        
        for i, pred in enumerate(predictions):
            # Get keywords for this prediction's purpose
            if i < len(stated_purposes):
                keywords = purpose_keywords[i]
            else:
                # Use the first purpose if prediction index exceeds purpose list
                keywords = purpose_keywords[0]
            
            # Count keywords present in prediction
            pred_lower = pred.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in pred_lower)
            
            # Calculate purpose alignment score
            if keywords:
                alignment_score = min(1.0, keyword_matches / len(keywords) * 1.5)
            else:
                alignment_score = 0.7
            
            # Check for phrases indicating purpose drift
            drift_indicators = ["can also be used for", "additionally", "furthermore", 
                            "moreover", "besides", "other uses include"]
            
            # Penalize for purpose drift indicators
            for indicator in drift_indicators:
                if indicator in pred_lower:
                    alignment_score -= 0.15
                    
            scores.append(max(0.0, min(1.0, alignment_score)))
        
        # Return average score across all predictions
        return sum(scores) / len(scores) if scores else 0.7

    def evaluate_data_minimization(predictions, personal_data_tokens, input_texts):
        """
        Evaluate compliance with GDPR data minimization
        
        Higher score if predictions don't contain personal data not in inputs
        
        Args:
            predictions: List of model text predictions
            personal_data_tokens: List of personal data tokens to check
            input_texts: List of input text strings
            
        Returns:
            Compliance score between 0-1
        """
        if not predictions or not personal_data_tokens or not input_texts:
            return 0.7  # Default moderate score if inputs are missing
        
        scores = []
        
        for pred, input_text in zip(predictions, input_texts):
            pred_score = 1.0
            unnecessary_disclosures = 0
            
            # Check each personal data token
            for token in personal_data_tokens:
                # If token in prediction but not in input, reduce score
                if token in pred and token not in input_text:
                    unnecessary_disclosures += 1
            
            # Calculate score based on number of unnecessary disclosures
            # More severe penalty for multiple disclosures
            if unnecessary_disclosures > 0:
                penalty = 0.1 * unnecessary_disclosures + 0.05 * (unnecessary_disclosures ** 2)
                pred_score -= penalty
            
            scores.append(max(0.0, min(1.0, pred_score)))
        
        # Average score across all predictions
        return sum(scores) / len(scores) if scores else 0.7


    def evaluate_purpose_limitation(predictions, stated_purposes):
        """
        Evaluate compliance with GDPR purpose limitation
        
        Higher score if predictions align with stated purposes
        
        Args:
            predictions: List of model text predictions
            stated_purposes: List of authorized purpose statements
            
        Returns:
            Compliance score between 0-1
        """
        if not predictions or not stated_purposes:
            return 0.7  # Default moderate score if inputs are missing
        
        # In a real implementation, we would:
        # 1. Calculate embeddings for predictions and purposes
        # 2. Compute semantic similarity between each prediction and its purpose
        # 3. Set thresholds for compliance
        
        scores = []
        purpose_keywords = {}
        
        # Extract keywords from purposes for simple keyword-based evaluation
        for i, purpose in enumerate(stated_purposes):
            # Simple keyword extraction (in real implementation, use proper NLP)
            keywords = purpose.lower().split()
            # Remove common words
            stopwords = ["the", "a", "an", "in", "to", "for", "of", "and", "or", "with"]
            keywords = [word for word in keywords if word not in stopwords and len(word) > 3]
            purpose_keywords[i] = keywords
        
        for i, pred in enumerate(predictions):
            # Get keywords for this prediction's purpose
            if i < len(stated_purposes):
                keywords = purpose_keywords[i]
            else:
                # Use the first purpose if prediction index exceeds purpose list
                keywords = purpose_keywords[0]
            
            # Count keywords present in prediction
            pred_lower = pred.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in pred_lower)
            
            # Calculate purpose alignment score
            if keywords:
                alignment_score = min(1.0, keyword_matches / len(keywords) * 1.5)
            else:
                alignment_score = 0.7
            
            # Check for phrases indicating purpose drift
            drift_indicators = ["can also be used for", "additionally", "furthermore", 
                            "moreover", "besides", "other uses include"]
            
            # Penalize for purpose drift indicators
            for indicator in drift_indicators:
                if indicator in pred_lower:
                    alignment_score -= 0.15
                    
            scores.append(max(0.0, min(1.0, alignment_score)))
        
        # Return average score across all predictions
        return sum(scores) / len(scores) if scores else 0.7



    # Additional helper method to improve the framework
    def _initialize_student(self, architecture):
        """
        Initialize student model based on architecture spec
        
        Args:
            architecture: Dictionary containing architecture specification
            
        Returns:
            Initialized student model
        """
        vocab_size = architecture.get("vocab_size", 50257)
        hidden_size = architecture.get("hidden_size", 768)
        num_hidden_layers = architecture.get("num_hidden_layers", 6)
        num_attention_heads = architecture.get("num_attention_heads", 12)
        intermediate_size = architecture.get("intermediate_size", 3072)
        max_position_embeddings = architecture.get("max_position_embeddings", 1024)
        
        # Use transformers library to create model
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Create configuration for smaller model
        config = AutoConfig.from_pretrained(
            "gpt2",  # Base architecture
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            n_inner=intermediate_size,
            n_positions=max_position_embeddings,
        )
        
        # Framework-specific configuration additions
        if self.regulatory_framework == "GDPR":
            # Add special token embeddings for GDPR concepts
            config.vocab_size += 10  # Add GDPR-specific tokens
        elif self.regulatory_framework == "HIPAA":
            # Add special token embeddings for HIPAA concepts
            config.vocab_size += 15  # Add HIPAA-specific tokens
        
        # Create student model from config
        student_model = AutoModelForCausalLM.from_config(config)
        
        # Add framework-specific projection layer
        student_model.framework_layer = nn.Linear(hidden_size, hidden_size)
        
        # Add embedding lookup method
        def get_sequence_embedding(outputs):
            # Get the last hidden state
            last_hidden = outputs.hidden_states[-1]
            # Average over sequence dimension
            sequence_embedding = last_hidden.mean(dim=1)
            # Project through framework layer
            projected_embedding = student_model.framework_layer(sequence_embedding)
            return projected_embedding
        
        # Attach method to student model
        student_model.get_sequence_embedding = get_sequence_embedding
        
        return student_model