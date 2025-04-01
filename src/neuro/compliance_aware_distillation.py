import sys
import os
import copy
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    get_linear_schedule_with_warmup,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)

# Setup logging
logger = logging.getLogger(__name__)

class ComplianceEvaluator:
    """Evaluates model compliance with regulatory frameworks"""
    
    def __init__(self, tokenizer, regulatory_framework):
        self.tokenizer = tokenizer
        self.regulatory_framework = regulatory_framework
        
    def evaluate_compliance(self, model, dataset, detailed=False):
        """Evaluate model compliance on dataset for regulatory framework"""
        model.eval()
        framework = self.regulatory_framework
        compliance_score = 0.0
        criterion_scores = {}
        
        # Define framework-specific evaluation criteria and weights
        criteria, weights = self._get_framework_criteria()
        
        # Evaluate each criterion
        with torch.no_grad():
            for batch in dataset:
                batch_scores = {}
                
                # Evaluate each criterion
                for criterion, weight in weights.items():
                    criterion_score = self._evaluate_criterion(model, batch, criterion)
                    
                    if criterion not in criterion_scores:
                        criterion_scores[criterion] = []
                    criterion_scores[criterion].append(criterion_score)
        
        # Average metrics across all batches
        for criterion in criterion_scores:
            criterion_scores[criterion] = sum(criterion_scores[criterion]) / len(criterion_scores[criterion])
        
        # Calculate final weighted compliance score
        compliance_score = sum(criterion_scores[criterion] * weight for criterion, weight in weights.items())
        
        if detailed:
            return {
                "overall_score": compliance_score,
                "criterion_scores": criterion_scores
            }
        return compliance_score
    
    def _get_framework_criteria(self):
        """Get evaluation criteria and weights for the framework"""
        framework = self.regulatory_framework
        
        if framework == "GDPR":
            criteria = ["data_minimization", "purpose_limitation", "accuracy", "storage_limitation", "confidentiality"]
            weights = {
                "data_minimization": 0.25,
                "purpose_limitation": 0.25,
                "accuracy": 0.15,
                "storage_limitation": 0.15,
                "confidentiality": 0.20
            }
        elif framework == "HIPAA":
            criteria = ["phi_protection", "authorization", "use_disclosure", "safeguards"]
            weights = {
                "phi_protection": 0.35,
                "authorization": 0.25,
                "use_disclosure": 0.20,
                "safeguards": 0.20
            }
        else:
            # Generic compliance criteria
            criteria = ["data_protection", "consent_handling", "regulatory_alignment"]
            weights = {
                "data_protection": 0.4,
                "consent_handling": 0.3,
                "regulatory_alignment": 0.3
            }
        
        return criteria, weights
    
    def _evaluate_criterion(self, model, batch, criterion):
        """Evaluate a specific compliance criterion on a batch"""
        # Process batch with model
        outputs = model(batch["input_ids"], batch["attention_mask"])
        
        # Decode outputs to text
        predictions = self._decode_outputs(outputs)
        
        # Get expected behaviors for this criterion
        expected = batch.get(f"{criterion}_expected", [])
        
        # Route to appropriate evaluation function
        if criterion == "data_minimization":
            return self._evaluate_data_minimization(
                predictions, 
                batch.get("personal_data_tokens", []), 
                batch.get("input_texts", [])
            )
        elif criterion == "purpose_limitation":
            return self._evaluate_purpose_limitation(
                predictions, 
                batch.get("stated_purposes", [])
            )
        elif criterion == "phi_protection":
            return self._evaluate_phi_protection(
                predictions, 
                batch.get("phi_tokens", []), 
                batch.get("phi_redactions", [])
            )
        else:
            # Generic criterion - compare against expected outputs
            return self._calculate_similarity_score(predictions, expected)
    
    def _decode_outputs(self, outputs):
        """Convert model output logits to text predictions"""
        # Get most likely tokens
        token_ids = outputs.logits.argmax(dim=-1)
        
        # Decode to text
        predictions = [
            self.tokenizer.decode(ids, skip_special_tokens=True) 
            for ids in token_ids
        ]
        
        return predictions
    
    def _evaluate_data_minimization(self, predictions, personal_data_tokens, input_texts):
        """Evaluate compliance with GDPR data minimization"""
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
            
            # Calculate score based on unnecessary disclosures
            if unnecessary_disclosures > 0:
                penalty = 0.1 * unnecessary_disclosures + 0.05 * (unnecessary_disclosures ** 2)
                pred_score -= penalty
            
            scores.append(max(0.0, min(1.0, pred_score)))
        
        # Average score across all predictions
        return sum(scores) / len(scores) if scores else 0.7
    
    def _evaluate_purpose_limitation(self, predictions, stated_purposes):
        """Evaluate compliance with GDPR purpose limitation"""
        if not predictions or not stated_purposes:
            return 0.7  # Default moderate score if inputs are missing
        
        scores = []
        purpose_keywords = {}
        
        # Extract keywords from purposes
        for i, purpose in enumerate(stated_purposes):
            keywords = purpose.lower().split()
            stopwords = ["the", "a", "an", "in", "to", "for", "of", "and", "or", "with"]
            keywords = [w for w in keywords if w not in stopwords and len(w) > 3]
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
            drift_indicators = [
                "can also be used for", "additionally", "furthermore", 
                "moreover", "besides", "other uses include"
            ]
            
            # Penalize for purpose drift indicators
            for indicator in drift_indicators:
                if indicator in pred_lower:
                    alignment_score -= 0.15
                    
            scores.append(max(0.0, min(1.0, alignment_score)))
        
        # Return average score across all predictions
        return sum(scores) / len(scores) if scores else 0.7
    
    def _evaluate_phi_protection(self, predictions, phi_tokens, expected_redactions):
        """Evaluate compliance with HIPAA PHI protection requirements"""
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
    
    def _calculate_similarity_score(self, predictions, expected):
        """Calculate semantic similarity between predictions and expected outputs"""
        if not predictions or not expected:
            return 0.7  # Default moderate score if inputs are missing
        
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


class ComplianceAwareDistillation:
    """
    Knowledge distillation framework that preserves compliance capabilities
    while creating smaller, efficient models for specific regulatory domains
    """
    
    def __init__(
        self, 
        teacher_model, 
        student_architecture, 
        regulatory_framework,
        tokenizer=None,
        config=None
    ):
        """
        Initialize the compliance-aware distillation framework
        
        Args:
            teacher_model: Large, fully-capable compliance model
            student_architecture: Architecture specification for student model
            regulatory_framework: Target regulatory framework (e.g., 'GDPR', 'HIPAA')
            tokenizer: Tokenizer to use (will extract from teacher if None)
            config: Optional configuration dictionary with parameters
        """
        self.teacher = teacher_model
        self.config = config or {}
        
        # Extract or initialize tokenizer
        self.tokenizer = tokenizer or self._extract_tokenizer_from_teacher()
        
        # Initialize student model from architecture spec
        self.student = self._initialize_student(student_architecture)
        
        self.regulatory_framework = regulatory_framework
        
        # Load the various datasets needed for distillation
        self.framework_datasets = self._load_framework_datasets()
        
        # Initialize the compliance evaluator
        self.evaluator = ComplianceEvaluator(self.tokenizer, self.regulatory_framework)
        
        # Track compliance metrics during distillation
        self.compliance_metrics = {
            "baseline": self._evaluate_teacher_compliance(),
            "history": []
        }
        
        # Set up configurable parameters with defaults
        self._setup_parameters()
    
    def _setup_parameters(self):
        """Set up configurable parameters with defaults"""
        self.params = {
            # Distillation parameters
            "temperature": self.config.get("temperature", 2.0),
            "alpha": self.config.get("alpha", 0.5),  # Balance between distillation and compliance losses
            
            # Compliance loss weights
            "task_loss_weight": self.config.get("task_loss_weight", 0.3),
            "regulatory_loss_weight": self.config.get("regulatory_loss_weight", 0.7),
            
            # Learning parameters
            "learning_rate": self.config.get("learning_rate", 1e-4),
            "warmup_steps": self.config.get("warmup_steps", 100),
            
            # Dynamic alpha adjustment parameters
            "alpha_adjustment_frequency": self.config.get("alpha_adjustment_frequency", 100),
            "alpha_adjustment_step": self.config.get("alpha_adjustment_step", 0.05),
            "min_alpha": self.config.get("min_alpha", 0.1),
            "max_alpha": self.config.get("max_alpha", 0.9),
            "compliance_target_ratio": self.config.get("compliance_target_ratio", 0.95),
        }
    
    def _extract_tokenizer_from_teacher(self):
        """Extract tokenizer from teacher model if possible"""
        if hasattr(self.teacher, "tokenizer"):
            return self.teacher.tokenizer
        
        # Attempt to load tokenizer based on model type
        try:
            # For HuggingFace models
            model_name = self.teacher.config._name_or_path
            return AutoTokenizer.from_pretrained(model_name)
        except (AttributeError, ImportError, ValueError) as e:
            logger.warning(f"Could not extract tokenizer from teacher model: {e}")
            logger.warning("Please provide a tokenizer explicitly")
            return None
    
    def _initialize_student(self, architecture):
        """
        Initialize student model based on architecture spec
        
        Args:
            architecture: Dictionary containing architecture specification
            
        Returns:
            Initialized student model
        """
        logger.info(f"Initializing student model with architecture: {architecture}")
        
        # Extract architecture parameters with defaults
        vocab_size = architecture.get("vocab_size", 50257)  # Default GPT-2 vocab size
        hidden_size = architecture.get("hidden_size", 768)
        num_hidden_layers = architecture.get("num_hidden_layers", 6)
        num_attention_heads = architecture.get("num_attention_heads", 12)
        intermediate_size = architecture.get("intermediate_size", 3072)
        max_position_embeddings = architecture.get("max_position_embeddings", 1024)
        base_model = architecture.get("base_model", "gpt2")
        
        try:
            # Create configuration for student model
            config = AutoConfig.from_pretrained(
                base_model,
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
            
            logger.info(f"Successfully initialized student model with {self.count_parameters(student_model)} parameters")
            return student_model
            
        except Exception as e:
            logger.error(f"Error initializing student model: {e}")
            raise ValueError(f"Failed to initialize student model: {e}")
    
    def _load_framework_datasets(self):
        """Load datasets specific to the target regulatory framework"""
        try:
            from src.neuro.dataset_mock import (
                load_general_compliance_dataset,
                load_framework_dataset,
                load_compliance_edge_cases,
                load_compliance_adversarial
            )
            
            datasets = {
                "general": load_general_compliance_dataset(),
                "framework_specific": load_framework_dataset(self.regulatory_framework),
                "edge_cases": load_compliance_edge_cases(self.regulatory_framework),
                "adversarial": load_compliance_adversarial(self.regulatory_framework)
            }
            
            logger.info(f"Loaded framework datasets for {self.regulatory_framework}")
            for name, dataset in datasets.items():
                logger.info(f"  - {name}: {len(dataset)} samples")
                
            return datasets
        except ImportError as e:
            logger.error(f"Error loading framework datasets: {e}")
            raise ImportError(f"Failed to load framework datasets: {e}")
    
    def _evaluate_teacher_compliance(self):
        """Evaluate teacher model compliance as baseline"""
        logger.info("Evaluating teacher model compliance as baseline")
        metrics = {}
        
        for dataset_name, dataset in self.framework_datasets.items():
            metrics[dataset_name] = self.evaluator.evaluate_compliance(
                self.teacher, dataset, detailed=True
            )
            
        # Overall score is average of dataset scores
        metrics["overall"] = sum(
            m["overall_score"] if isinstance(m, dict) else m 
            for m in metrics.values()
        ) / len(metrics)
        
        logger.info(f"Teacher baseline compliance score: {metrics['overall']:.4f}")
        return metrics
    
    def distill(self, epochs=50, batch_size=32, learning_rate=None, save_path=None):
        """
        Perform compliance-aware knowledge distillation
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate (overrides config)
            save_path: Path to save the best model
            
        Returns:
            Trained student model with preserved compliance capabilities
        """
        # Update learning rate if provided
        if learning_rate is not None:
            self.params["learning_rate"] = learning_rate
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.student.parameters(), 
            lr=self.params["learning_rate"]
        )
        
        # Setup learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.params["warmup_steps"], 
            num_training_steps=epochs
        )
        
        # Create balanced sampler for compliance data
        try:
            from src.compliance.utils.balanced_sampler import ComplianceBalancedSampler
            sampler = ComplianceBalancedSampler(self.framework_datasets)
        except ImportError:
            # Fall back to simple concatenated dataset
            logger.warning("ComplianceBalancedSampler not available, using simple sampling")
            sampler = self._create_simple_sampler(self.framework_datasets, batch_size)
        
        # Calculate effective training steps
        total_steps = sum(len(d) // batch_size for d in self.framework_datasets.values()) * epochs
        logger.info(f"Starting distillation for {self.regulatory_framework} compliance")
        logger.info(f"Training for {epochs} epochs, {total_steps} total steps")
        
        # Track best model
        best_model = None
        best_compliance = 0.0
        
        for epoch in range(epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(
                epoch, batch_size, optimizer, scheduler, sampler
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
            overall_compliance = compliance_metrics["overall"]
            if overall_compliance > best_compliance:
                best_compliance = overall_compliance
                best_model = copy.deepcopy(self.student)
                logger.info(f"Epoch {epoch}: New best model with compliance score {best_compliance:.4f}")
                
                # Save best model if path provided
                if save_path:
                    self._save_model(best_model, save_path, epoch, best_compliance)
            
            # Log progress
            self._log_training_progress(epoch, epochs, train_metrics, compliance_metrics)
            
        # Restore best model
        if best_model is not None:
            self.student = best_model
            
        # Final compliance evaluation
        final_compliance = self._evaluate_student_compliance(detailed=True)
        self.compliance_metrics["final"] = final_compliance
        
        return self.student
    
    def _create_simple_sampler(self, datasets, batch_size):
        """Create a simple dataset sampler as fallback"""
        class SimpleSampler:
            def __init__(self, datasets, batch_size):
                self.datasets = datasets
                self.batch_size = batch_size
                
            def get_batches(self, batch_size=None):
                batch_size = batch_size or self.batch_size
                all_batches = []
                
                for dataset in self.datasets.values():
                    samples = list(dataset)
                    for i in range(0, len(samples), batch_size):
                        batch = samples[i:i+batch_size]
                        if len(batch) == batch_size:  # Only use full batches
                            all_batches.append(batch)
                
                # Shuffle batches
                import random
                random.shuffle(all_batches)
                return all_batches
                
        return SimpleSampler(datasets, batch_size)
    
    def _train_epoch(self, epoch, batch_size, optimizer, scheduler, sampler):
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
        
        # Get current alpha value
        alpha = self.params["alpha"]
        
        # Train on batches
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
                student_outputs, 
                teacher_outputs, 
                self.params["temperature"]
            )
            
            # Compute compliance loss
            compliance_loss = self._compute_compliance_loss(
                student_outputs, 
                batch
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
            if metrics["steps"] % self.params["alpha_adjustment_frequency"] == 0:
                alpha = self._adjust_alpha_dynamically(metrics["steps"])
        
        # Average metrics
        for key in ["total_loss", "distillation_loss", "compliance_loss"]:
            metrics[key] /= metrics["steps"]
            
        # Store final alpha value
        metrics["alpha"] = alpha
        self.params["alpha"] = alpha
            
        return metrics
    
    def _adjust_alpha_dynamically(self, step):
        """
        Dynamically adjust alpha based on compliance gap
        
        This balances distillation and compliance objectives during training
        """
        # Get current alpha
        alpha = self.params["alpha"]
        
        # Periodically evaluate compliance
        compliance = self._evaluate_student_compliance(sample=True)
        teacher_compliance = self.compliance_metrics["baseline"]["overall"]
        
        # Adjust alpha based on compliance gap
        compliance_ratio = compliance["overall"] / teacher_compliance
        target_ratio = self.params["compliance_target_ratio"]
        
        if compliance_ratio < target_ratio:
            # Increase focus on compliance if falling behind
            alpha = max(
                self.params["min_alpha"], 
                alpha - self.params["alpha_adjustment_step"]
            )
            logger.info(f"Step {step}: Decreasing alpha to {alpha:.2f} (compliance ratio: {compliance_ratio:.2f})")
        elif compliance_ratio > 0.99:
            # Increase focus on distillation if compliance is good
            alpha = min(
                self.params["max_alpha"], 
                alpha + self.params["alpha_adjustment_step"]
            )
            logger.info(f"Step {step}: Increasing alpha to {alpha:.2f} (compliance ratio: {compliance_ratio:.2f})")
        
        return alpha
    
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
            batch["labels"].view(-1),
            ignore_index=-100  # Ignore padding
        )
        
        # Regulatory constraint loss
        regulatory_loss = self._compute_regulatory_constraint_loss(
            student_outputs, 
            batch
        )
        
        # Combine losses with configurable weights
        task_weight = self.params["task_loss_weight"]
        regulatory_weight = self.params["regulatory_loss_weight"]
        
        combined_loss = task_weight * task_loss + regulatory_weight * regulatory_loss
        
        return combined_loss
    
    def _compute_regulatory_constraint_loss(self, outputs, batch):
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
        
        # Example: HIPAA-specific constraints
        elif self.regulatory_framework == "HIPAA":
            # PHI protection constraint
            if "phi_protection" in constraints:
                phi_loss = self._compute_phi_protection_loss(
                    outputs, constraints["phi_protection"]
                )
                constraint_losses.append(phi_loss)
        
        # Default to generic regulatory constraints if none specific
        if not constraint_losses:
            constraint_losses.append(
                self._compute_generic_regulatory_loss(outputs, batch)
            )
            
        # Combine all constraint losses
        if constraint_losses:
            return sum(constraint_losses) / len(constraint_losses)
        else:
            # Return zero tensor if no constraints
            return torch.tensor(0.0, device=outputs.logits.device)
    
    def _compute_data_minimization_loss(self, outputs, constraint):
        """
        Compute loss enforcing GDPR data minimization principle.
        
        This loss penalizes the model for generating unnecessary personal data 
        tokens that weren't present in the input, encouraging minimal data usage.
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
            if hasattr(self, 'purpose_embedder'):
                purpose_embedding = self.purpose_embedder(stated_purpose)
            else:
                # Fall back if no embedder available
                return torch.tensor(0.1, device=device)
        
        # If we still don't have purpose embedding, return minimal loss
        if purpose_embedding is None:
            return torch.tensor(0.1, device=device)
        
        # Get output sequence embeddings
        # Assuming we can extract embeddings from the student model
        if hasattr(self.student, 'get_sequence_embedding'):
            output_embeddings = self.student.get_sequence_embedding(outputs)
        else:
            # Fall back if no embedding method available
            return torch.tensor(0.1, device=device)
        
        # Calculate cosine similarity between output and purpose embeddings
        similarity = F.cosine_similarity(output_embeddings, purpose_embedding, dim=-1)
        
        # Loss is higher when similarity is lower (penalize drift from purpose)
        purpose_loss = 1.0 - similarity.mean()
        
        # Additional checks for specific unauthorized purposes
        unauthorized_purposes = constraint.get("unauthorized_purposes", [])
        unauthorized_penalty = 0.0
        
        if hasattr(self, 'purpose_embedder') and unauthorized_purposes:
            unauthorized_embeddings = [
                self.purpose_embedder(p) for p in unauthorized_purposes
            ]
            
            # Penalize similarity with unauthorized purposes
            for unauth_emb in unauthorized_embeddings:
                unauth_similarity = F.cosine_similarity(
                    output_embeddings, unauth_emb, dim=-1
                )
                unauthorized_penalty += torch.clamp(unauth_similarity.mean(), min=0.0)
        
        # Final combined loss
        combined_loss = purpose_loss + 0.5 * unauthorized_penalty
        
        return combined_loss
    
    def _compute_phi_protection_loss(self, outputs, constraint):
        """
        Compute loss enforcing HIPAA PHI protection requirements.
        
        This loss penalizes the disclosure of protected health information,
        encouraging the model to maintain patient privacy.
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
                penalty = torch.max(
                    phi_prob - redaction_prob, 
                    torch.tensor(0.0, device=device)
                )
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
    
    def _compute_generic_regulatory_loss(self, outputs, batch):
        """
        Compute generic regulatory compliance loss when specific constraints are not provided.
        """
        device = outputs.logits.device
        
        # Default loss weight
        base_loss = torch.tensor(0.1, device=device)
        
        # Check if batch contains any regulatory context
        regulatory_context = batch.get("regulatory_context", {})
        
        if not regulatory_context:
            return base_loss
            
        # Extract regulatory information
        domain = regulatory_context.get("domain", "")
        sensitivity = regulatory_context.get("sensitivity", "low")
        
        # Apply sensitivity multiplier
        sensitivity_multiplier = {
            "low": 1.0,
            "medium": 2.0,
            "high": 3.0,
            "critical": 5.0
        }.get(sensitivity, 1.0)
        
        # Get probability distribution
        probs = F.softmax(outputs.logits, dim=-1)
        
        # Compute entropy (lower entropy = more focused predictions)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        
        # Encourage lower entropy (more confidence) for regulatory content
        entropy_loss = 0.05 * entropy * sensitivity_multiplier
        
        # Compute final generic regulatory loss
        generic_loss = base_loss + entropy_loss
        
        return generic_loss
    
    def _evaluate_student_compliance(self, sample=False, detailed=False):
        """
        Evaluate student model compliance with regulatory framework
        
        Args:
            sample: If True, use smaller sample for efficiency during training
            detailed: If True, return detailed metrics
            
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
                datasets[name] = self._sample_dataset(dataset, max_samples=100)
        else:
            datasets = self.framework_datasets
        
        # Evaluate on each dataset
        for dataset_name, dataset in datasets.items():
            metrics[dataset_name] = self.evaluator.evaluate_compliance(
                self.student, dataset, detailed=detailed
            )
            
        # Calculate overall compliance
        metrics["overall"] = sum(
            m["overall_score"] if isinstance(m, dict) else m 
            for m in metrics.values()
        ) / len(metrics)
        
        return metrics
    
    def _sample_dataset(self, dataset, max_samples=100):
        """Create smaller sample of dataset for quick evaluation"""
        if len(dataset) <= max_samples:
            return dataset
            
        # Simple sampling - in a real implementation, this would be stratified
        import random
        samples = list(dataset)
        return random.sample(samples, max_samples)
    
    def _log_training_progress(self, epoch, epochs, train_metrics, compliance_metrics):
        """Log training progress"""
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"  Train loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  Distillation loss: {train_metrics['distillation_loss']:.4f}")
        logger.info(f"  Compliance loss: {train_metrics['compliance_loss']:.4f}")
        logger.info(f"  Alpha: {self.params['alpha']:.4f}")
        logger.info(f"  Overall compliance: {compliance_metrics['overall']:.4f}")
        
        teacher_compliance = self.compliance_metrics["baseline"]["overall"]
        logger.info(f"  Teacher compliance: {teacher_compliance:.4f}")
        
        compliance_ratio = compliance_metrics["overall"] / teacher_compliance
        logger.info(f"  Compliance ratio: {compliance_ratio:.4f}")
    
    def _save_model(self, model, path, epoch, compliance_score):
        """Save model checkpoint"""
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(path, exist_ok=True)
            
            # Save with epoch and score in filename
            filename = f"student_{self.regulatory_framework}_e{epoch}_c{compliance_score:.4f}.pt"
            full_path = os.path.join(path, filename)
            
            # Save model
            torch.save(model.state_dict(), full_path)
            logger.info(f"Saved model checkpoint to {full_path}")
            
            # Also save as latest
            latest_path = os.path.join(path, f"student_{self.regulatory_framework}_latest.pt")
            torch.save(model.state_dict(), latest_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
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
                "teacher_parameters": self.count_parameters(self.teacher),
                "student_parameters": self.count_parameters(self.student),
                "compression_ratio": self.count_parameters(self.teacher) / max(1, self.count_parameters(self.student)),
                "memory_footprint": {
                    "teacher": self.estimate_memory_footprint(self.teacher),
                    "student": self.estimate_memory_footprint(self.student),
                    "reduction_percent": (1 - self.estimate_memory_footprint(self.student) / 
                                      max(1, self.estimate_memory_footprint(self.teacher))) * 100
                }
            },
            "compliance_comparison": {
                "teacher": self.compliance_metrics["baseline"],
                "student": self.compliance_metrics["final"],
                "ratio": {
                    k: (self.compliance_metrics["final"][k] / self.compliance_metrics["baseline"][k]
                        if isinstance(self.compliance_metrics["final"][k], (int, float)) else
                        self.compliance_metrics["final"][k]["overall_score"] / 
                        self.compliance_metrics["baseline"][k]["overall_score"])
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
        framework = self.regulatory_framework
        
        # Basic framework info
        report = {
            "framework": framework,
            "framework_description": self._get_framework_description(framework)
        }
        
        # Add framework-specific metrics
        if framework == "GDPR":
            report.update({
                "data_minimization_score": self._calculate_framework_specific_metric("data_minimization"),
                "purpose_limitation_score": self._calculate_framework_specific_metric("purpose_limitation"),
                "key_principles": [
                    "Lawfulness, fairness and transparency",
                    "Purpose limitation",
                    "Data minimization",
                    "Accuracy",
                    "Storage limitation",
                    "Integrity and confidentiality"
                ]
            })
        elif framework == "HIPAA":
            report.update({
                "phi_protection_score": self._calculate_framework_specific_metric("phi_protection"),
                "authorization_score": self._calculate_framework_specific_metric("authorization"),
                "key_principles": [
                    "Privacy Rule",
                    "Security Rule",
                    "Breach Notification Rule",
                    "Patient Rights"
                ]
            })
        
        return report
    
    def _get_framework_description(self, framework):
        """Get description for a regulatory framework"""
        descriptions = {
            "GDPR": "General Data Protection Regulation: A regulation in EU law on data protection and privacy in the European Union and the European Economic Area.",
            "HIPAA": "Health Insurance Portability and Accountability Act: US legislation that provides data privacy and security provisions for safeguarding medical information.",
            "EU_AI_Act": "European Union Artificial Intelligence Act: A proposed regulation establishing harmonized rules on artificial intelligence in the EU.",
            "FDA": "Food and Drug Administration: US agency responsible for protecting public health including regulations for AI/ML in medical devices.",
            "ISO_21434": "International standard for cybersecurity engineering for road vehicles and their components."
        }
        
        return descriptions.get(framework, f"Regulatory framework: {framework}")
    
    def _calculate_framework_specific_metric(self, metric_name):
        """Calculate framework-specific compliance metric from evaluation data"""
        # Extract metric scores from final evaluation if available
        final = self.compliance_metrics.get("final", {})
        
        if isinstance(final, dict) and "criterion_scores" in final:
            return final["criterion_scores"].get(metric_name, 0.0)
        
        # Try to extract from dataset evaluations
        for dataset_name, metrics in final.items():
            if isinstance(metrics, dict) and "criterion_scores" in metrics:
                if metric_name in metrics["criterion_scores"]:
                    return metrics["criterion_scores"][metric_name]
        
        # Default if not found
        return 0.0
    
    def _generate_conclusion(self):
        """Generate conclusion and recommendations"""
        student_overall = self.compliance_metrics["final"]["overall"] if isinstance(self.compliance_metrics["final"]["overall"], (int, float)) else self.compliance_metrics["final"]["overall"]["overall_score"]
        teacher_overall = self.compliance_metrics["baseline"]["overall"] if isinstance(self.compliance_metrics["baseline"]["overall"], (int, float)) else self.compliance_metrics["baseline"]["overall"]["overall_score"]
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
            "recommendations": self._generate_recommendations(ratio),
            "model_size_reduction": f"{(1 - self.count_parameters(self.student) / max(1, self.count_parameters(self.teacher))):.2%}"
        }
    
    def _generate_recommendations(self, compliance_ratio):
        """Generate specific recommendations based on results"""
        recommendations = []
        
        if compliance_ratio < 0.9:
            recommendations.append(
                "Increase model capacity in compliance-critical components"
            )
            
        if compliance_ratio < 0.95:
            recommendations.append(
                "Fine-tune with specialized compliance datasets"
            )
        
        if compliance_ratio < 0.9:
            recommendations.append(
                "Consider using a larger student model or adding dedicated compliance layers"
            )
            
        if compliance_ratio < 0.8:
            recommendations.append(
                "Retrain with lower alpha value to prioritize compliance over knowledge distillation"
            )
        
        if compliance_ratio > 0.95:
            recommendations.append(
                "Consider more aggressive compression as compliance is well-preserved"
            )
            
        return recommendations or ["No specific recommendations required"]
    
    def count_parameters(self, model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def estimate_memory_footprint(self, model):
        """Estimate memory footprint of model in MB"""
        params = sum(p.numel() for p in model.parameters())
        # Assume 4 bytes per parameter for FP32
        return (params * 4) / (1024 * 1024)
