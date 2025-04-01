import datetime
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import z3
import spacy
from lark import Lark, Transformer, v_args

logger = logging.getLogger(__name__)

# --- Z3 Helper Types ---
Z3SymbolTable = Dict[str, Union[z3.SortRef, z3.FuncDeclRef, z3.ExprRef]]
Z3FactList = List[z3.BoolRef]
# --- End Z3 Helper Types ---

class SymbolicParser:
    """
    Parser for symbolic logic formulas using lark-parser.
    Converts string representations to Z3 expressions.
    """
    
    def __init__(self, reasoner):
        """Initialize the parser with a reference to the SymbolicReasoner."""
        self.reasoner = reasoner
        self.logger = logging.getLogger(__name__)
        
        # Define the grammar for the symbolic logic language
        self.grammar = r"""
            ?formula: quantifier | connective | predicate | var | bool_const
            
            quantifier: forall | exists
            forall: "ForAll" "(" var_decls "," formula ")"
            exists: "Exists" "(" var_decls "," formula ")"
            
            var_decls: var_decl (";" var_decl)*
            var_decl: IDENTIFIER ":" IDENTIFIER
            
            connective: implies | and_expr | or_expr | not_expr
            implies: "Implies" "(" formula "," formula ")"
            and_expr: "And" "(" [formula ("," formula)*] ")"
            or_expr: "Or" "(" [formula ("," formula)*] ")"
            not_expr: "Not" "(" formula ")"
            
            predicate: IDENTIFIER "(" [arg ("," arg)*] ")"
            
            arg: formula | IDENTIFIER
            
            var: IDENTIFIER
            bool_const: "true" | "false"
            
            IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
            
            %import common.WS
            %ignore WS
        """
        
        # Initialize the Lark parser
        self.parser = Lark(self.grammar, start="formula", parser="earley")
        
        # Initialize the transformer for converting the parse tree to Z3 expressions
        self.transformer = LogicTransformer(self.reasoner)
        
    def parse_logic_formula(self, formula_str: str) -> z3.ExprRef:
        """
        Parse a logic formula string into a Z3 expression.
        
        Args:
            formula_str: String representation of the formula
            
        Returns:
            Z3 expression representing the parsed formula
            
        Raises:
            ValueError: If the formula cannot be parsed
        """
        try:
            # Parse the formula string using Lark
            parse_tree = self.parser.parse(formula_str.strip())
            self.logger.debug(f"Successfully parsed formula: {formula_str}")
            
            # Transform the parse tree into a Z3 expression
            z3_expr = self.transformer.transform(parse_tree)
            return z3_expr
            
        except Exception as e:
            self.logger.error(f"Failed to parse logic formula '{formula_str}': {e}", exc_info=True)
            raise ValueError(f"Formula parsing error: {e}") from e

@v_args(inline=True)
class LogicTransformer(Transformer):
    """Transforms the parse tree produced by Lark into Z3 expressions."""
    
    def __init__(self, reasoner):
        super().__init__()
        self.reasoner = reasoner
        self.bound_vars = {}
        
    def formula(self, expr):
        """Process a formula node (this might not be needed depending on grammar)"""
        return expr
        
    def forall(self, var_decls, formula):
        """Process a universal quantifier"""
        # var_decls is a list of (name, sort) tuples
        z3_vars = []
        
        # Create new scope for variables
        old_bound_vars = self.bound_vars.copy()
        
        # Create bound variables
        for var_name, sort_name in var_decls:
            z3_sort = self.reasoner._get_z3_sort(sort_name)
            z3_var = z3.Const(var_name, z3_sort)
            self.bound_vars[var_name] = z3_var
            z3_vars.append(z3_var)
            
        # Process the body with the bound variables
        body_expr = formula
        
        # Restore previous variable scope
        self.bound_vars = old_bound_vars
        
        # Create the quantified expression
        return z3.ForAll(z3_vars, body_expr)
        
    def exists(self, var_decls, formula):
        """Process an existential quantifier"""
        # var_decls is a list of (name, sort) tuples
        z3_vars = []
        
        # Create new scope for variables
        old_bound_vars = self.bound_vars.copy()
        
        # Create bound variables
        for var_name, sort_name in var_decls:
            z3_sort = self.reasoner._get_z3_sort(sort_name)
            z3_var = z3.Const(var_name, z3_sort)
            self.bound_vars[var_name] = z3_var
            z3_vars.append(z3_var)
            
        # Process the body with the bound variables
        body_expr = formula
        
        # Restore previous variable scope
        self.bound_vars = old_bound_vars
        
        # Create the quantified expression
        return z3.Exists(z3_vars, body_expr)
        
    def var_decls(self, *var_decl_items):
        """Process variable declarations"""
        return list(var_decl_items)
        
    def var_decl(self, var_name, sort_name):
        """Process a single variable declaration"""
        return (var_name.value, sort_name.value)
        
    def implies(self, left, right):
        """Process an implication"""
        return z3.Implies(left, right)
        
    def and_expr(self, *args):
        """Process a conjunction"""
        return z3.And(args) if args else z3.BoolVal(True)
        
    def or_expr(self, *args):
        """Process a disjunction"""
        return z3.Or(args) if args else z3.BoolVal(False)
        
    def not_expr(self, arg):
        """Process a negation"""
        return z3.Not(arg)
        
    def predicate(self, name, *args):
        """Process a predicate application"""
        name = name.value
        
        # Process arguments
        z3_args = []
        for arg in args:
            # If arg is already a Z3 expression (from a nested formula)
            if hasattr(arg, 'sort'):
                z3_args.append(arg)
            # If arg is a variable name (string)
            elif isinstance(arg, str) or hasattr(arg, 'value'):
                arg_name = arg if isinstance(arg, str) else arg.value
                # Check if it's a bound variable
                if arg_name in self.bound_vars:
                    z3_args.append(self.bound_vars[arg_name])
                else:
                    # Create a new constant with Entity sort
                    entity_sort = self.reasoner._get_z3_sort('Entity')
                    z3_args.append(z3.Const(arg_name, entity_sort))
        
        # Get argument sorts
        arg_sorts = [arg.sort() for arg in z3_args]
        
        # Get function declaration
        func = self.reasoner._get_z3_func(name, *arg_sorts)
        
        # Apply function to arguments
        return func(*z3_args)
        
    def var(self, name):
        """Process a variable reference"""
        name = name.value
        if name in self.bound_vars:
            return self.bound_vars[name]
        
        # If variable is not bound, assume it's a boolean variable
        return z3.Bool(name)
        
    def bool_const(self, value):
        """Process a boolean constant"""
        return z3.BoolVal(value.value == "true")
        
    def arg(self, value):
        """Process an argument"""
        return value

class SymbolicReasoner:
    """
    Symbolic reasoning engine for compliance verification using formal logic (Z3).
    """
    DEFAULT_Z3_TIMEOUT_MS = 5000 # 5 seconds

    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge_base: Optional[Any] = None):
        self.config = config or {}
        self.knowledge_base = knowledge_base # Placeholder for KB interaction
        self.z3_timeout = self.config.get('z3_timeout_ms', self.DEFAULT_Z3_TIMEOUT_MS)
        
        # Load NLP model
        self.nlp = self._load_nlp_model()

        self.rules = self._load_rules()
        self.rule_cache = {}  # Cache for rule applicability

        # Z3 context management
        self.z3_sorts: Dict[str, z3.SortRef] = {}
        self.z3_funcs: Dict[str, z3.FuncDeclRef] = {}
        self._setup_z3_base_context()

        # Initialize the symbolic parser
        self.symbolic_parser = SymbolicParser(self)

        logger.info(f"SymbolicReasoner initialized.")

    def _load_nlp_model(self) -> Any:
        """Load the spaCy NLP model."""
        model_name = self.config.get("spacy_model", "en_core_web_sm") # Small model default
        try:
            return spacy.load(model_name)
        except OSError:
            logger.error(f"Could not load spaCy model '{model_name}'. Ensure it's downloaded.")
            raise

    def _setup_z3_base_context(self):
        """Initialize common Z3 sorts."""
        # Define base sorts (types) for entities, concepts, activities etc.
        self.z3_sorts['Entity'] = z3.DeclareSort('Entity')
        self.z3_sorts['Concept'] = z3.DeclareSort('Concept')
        self.z3_sorts['Activity'] = z3.DeclareSort('Activity')
        self.z3_sorts['DataSource'] = z3.DeclareSort('DataSource') # Example
        # Base BoolSort and StringSort are built-in

    def _get_z3_sort(self, sort_name: str) -> z3.SortRef:
        """Get or declare a Z3.sort"""
        if sort_name not in self.z3_sorts:
             logger.debug(f"Dynamically declaring Z3 sort: {sort_name}")
             self.z3_sorts[sort_name] = z3.DeclareSort(sort_name)
        return self.z3_sorts[sort_name]

    def _get_z3_func(self, func_name: str, *sig: z3.SortRef) -> z3.FuncDeclRef:
        """Get or declare a Z3 function (predicate)."""
        sig_key = tuple(s.sexpr() for s in sig) # Create hashable key from signature
        key = (func_name, sig_key)
        if key not in self.z3_funcs:
            logger.debug(f"Dynamically declaring Z3 function: {func_name}{sig}")
            self.z3_funcs[key] = z3.Function(func_name, *sig, z3.BoolSort()) # Assume boolean predicates
        return self.z3_funcs[key]

    def _load_rules(self):
        """Load symbolic reasoning rules."""
        # (Keep existing rule loading logic)
        # TODO: Consider loading rules from external source (JSON, YAML, DB)
        rules = {
            "gdpr_data_minimization": {
                "id": "gdpr_data_minimization", "framework": "GDPR", "severity": "high",
                "description": "Personal data must be adequate, relevant and limited to what is necessary",
                # NOTE: Using 'Entity' sort for 'x' now
                "symbolic_representation": "ForAll(x:Entity, Implies(PersonalData(x), And(Adequate(x), Relevant(x), Necessary(x))))",
                "pattern": r"\b(?:collect|store|process|use)\b.{0,30}\b(?:all|every|any|extensive)\b.{0,30}\b(?:data|information)\b",
                "keywords": ["collect", "store", "process", "extensive", "all data", "minimization"]
            },
            "gdpr_consent": {
                "id": "gdpr_consent", "framework": "GDPR", "severity": "critical",
                "description": "Processing based on consent must be demonstrably given",
                # NOTE: Using 'Activity' and 'ConsentDoc' (example) sorts
                "symbolic_representation": "ForAll(x:Activity, Implies(And(ProcessingActivity(x), BasedOnConsent(x)), Exists(y:ConsentDoc, And(Consent(y), Demonstrable(y), GivenFor(y, x)))))",
                "pattern": r"\b(?:without|no|implicit|lack of)\b.{0,30}\b(?:consent|permission|authorization|agreement)\b",
                "keywords": ["consent", "permission", "authorization", "explicit", "opt-in"]
            },
            "hipaa_phi_disclosure": {
                "id": "hipaa_phi_disclosure", "framework": "HIPAA", "severity": "critical",
                "description": "Protected health information should only be disclosed with authorization",
                # NOTE: Using 'PHI_Data' and 'AuthDoc' (example) sorts
                "symbolic_representation": "ForAll(x:PHI_Data, Implies(And(PHI(x), Disclosed(x)), Exists(y:AuthDoc, And(Authorization(y), CoversDisclosure(y, x)))))",
                "pattern": r"\b(?:health|medical|patient).{0,50}\b(?:disclose|share|reveal|transmit|send)\b",
                "keywords": ["PHI", "health information", "medical data", "disclose", "share", "authorization"]
            }
            # TODO: Add more rules, potentially load from external source
        }
        logger.info(f"Loaded {len(rules)} symbolic reasoning rules.")
        return rules

    def evaluate_compliance(self, text: str, applicable_rules: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate text compliance using Z3 formal logic first, falling back to patterns/keywords.
        """
        if not text or not applicable_rules:
            return self._create_result(is_compliant=True)

        overall_compliant = True
        all_violations = []
        all_reasoning_steps = []
        rule_results = {}

        for rule_id in applicable_rules:
            rule = self.rules.get(rule_id)
            if not rule:
                logger.warning(f"Rule ID '{rule_id}' provided but not found in loaded rules.")
                continue

            logger.debug(f"Evaluating rule '{rule_id}' for text segment.")
            rule_compliant = True
            rule_reasoning = []
            rule_violations = []
            method_used = "heuristic" # Default to heuristic
            confidence = 0.7 # Default confidence for fallback

            # --- Attempt Z3 Evaluation ---
            if "symbolic_representation" in rule:
                try:
                    logger.debug(f"Attempting Z3 evaluation for rule '{rule_id}'.")
                    # 1. Generate Z3 facts from text and context
                    facts = self._generate_z3_facts(text, context)
                    if not facts:
                         logger.debug(f"No relevant Z3 facts generated for rule '{rule_id}'. Skipping Z3 check.")
                         raise ValueError("No Z3 facts generated") # Treat as failure for fallback

                    # 2. Parse rule formula
                    rule_formula = self._parse_logic_formula(rule["symbolic_representation"])

                    # 3. Evaluate using Z3 solver
                    z3_result = self._evaluate_logical_rule_with_z3(rule_formula, facts, rule)
                    rule_compliant = z3_result["is_compliant"]
                    rule_reasoning.append(z3_result["reasoning"])
                    if not rule_compliant:
                        rule_violations.append(z3_result["violation_details"])
                    method_used = "formal_logic"
                    confidence = z3_result.get("confidence", 1.0) # Z3 gives high confidence
                    logger.debug(f"Z3 evaluation for rule '{rule_id}' completed. Compliant: {rule_compliant}")

                except Exception as e:
                    logger.error(f"Unexpected error during Z3 evaluation of rule '{rule_id}': {e}", exc_info=True)
                    rule_reasoning.append({"step": "Z3 check failed", "detail": f"Error: {e}"})
                    
                    # Fall back to heuristic evaluation
                    logger.debug(f"Using heuristic evaluation (pattern/keyword) for rule '{rule_id}'.")
                    heuristic_result = self._evaluate_heuristic_rule(text, rule)
                    if not heuristic_result["is_compliant"]:
                         rule_compliant = False
                         rule_violations.extend(heuristic_result["violations"])
                    rule_reasoning.extend(heuristic_result["reasoning"])
                    method_used = "heuristic"
                    confidence = 0.7 # Lower confidence for heuristics
            else:
                # No symbolic representation, use heuristic only
                logger.debug(f"No symbolic representation for rule '{rule_id}'. Using heuristic evaluation only.")
                heuristic_result = self._evaluate_heuristic_rule(text, rule)
                if not heuristic_result["is_compliant"]:
                     rule_compliant = False
                     rule_violations.extend(heuristic_result["violations"])
                rule_reasoning.extend(heuristic_result["reasoning"])
                method_used = "heuristic"
                confidence = 0.7 # Lower confidence for heuristics

            # --- Aggregate Rule Results ---
            rule_results[rule_id] = {
                "is_compliant": rule_compliant,
                "method": method_used,
                "confidence": confidence,
                "reasoning": rule_reasoning,
                "violations": rule_violations
            }
            all_reasoning_steps.extend([{"rule_id": rule_id, **step} for step in rule_reasoning])
            if not rule_compliant:
                overall_compliant = False
                all_violations.extend([{ "rule_id": rule_id, "severity": rule.get("severity", "medium"), **v} for v in rule_violations])

        # --- Calculate Overall Score ---
        compliance_score = self._calculate_compliance_score(rule_results, applicable_rules)

        return self._create_result(
            is_compliant=overall_compliant,
            violations=all_violations,
            reasoning_steps=all_reasoning_steps,
            compliance_score=compliance_score,
            rule_results=rule_results # Include detailed results per rule
        )

    def _evaluate_heuristic_rule(self, text: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate rule using pattern and keyword matching."""
        violations = []
        reasoning = []
        is_compliant = True

        # 1. Check Pattern
        pattern = rule.get("pattern")
        matched_by_pattern = False
        if pattern:
            try:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
                if matches:
                    matched_by_pattern = True
                    is_compliant = False
                    for match in matches:
                         violation_detail = {
                             "description": rule.get("description", ""),
                             "matched_text": match.group(0),
                             "position": (match.start(), match.end()),
                             "method": "pattern"
                         }
                         violations.append(violation_detail)
                         reasoning.append({
                             "step": "Pattern Match",
                             "detail": f"Pattern '{pattern}' matched at {match.span()}: '{match.group(0)}'",
                             "conclusion": "Potential Violation"
                         })
            except re.error as e:
                logger.warning(f"Invalid regex pattern in rule {rule.get('id', 'N/A')}: {pattern} - {e}")
                reasoning.append({"step": "Pattern Check Failed", "detail": f"Invalid regex: {e}"})

        # 2. Check Keywords (if no pattern match or as additional evidence)
        # Decide if keywords act as fallback or independent check
        check_keywords = rule.get("keywords") and (not matched_by_pattern or self.config.get("keywords_always_check", False))
        if check_keywords:
            keywords = rule.get("keywords", [])
            text_lower = text.lower()
            matched_keywords = [kw for kw in keywords if kw.lower() in text_lower] # Simple substring match

            if matched_keywords:
                # Decide if keywords alone trigger violation or just add reasoning
                if self.config.get("keywords_trigger_violation", True) and not matched_by_pattern:
                     is_compliant = False
                     violation_detail = {
                         "description": rule.get("description", ""),
                         "matched_keywords": matched_keywords,
                         "method": "keyword"
                     }
                     violations.append(violation_detail)

                reasoning.append({
                    "step": "Keyword Match",
                    "detail": f"Keywords detected: {', '.join(matched_keywords)}",
                    "conclusion": "Potential Violation" if is_compliant is False else "Supporting Evidence (Keywords)"
                })

        # 3. No violation detected by heuristics
        if is_compliant:
            reasoning.append({
                "step": "Heuristic Check",
                "detail": "No violation patterns or triggering keywords detected.",
                "conclusion": "Compliant (Heuristic)"
            })

        return {
            "is_compliant": is_compliant,
            "violations": violations,
            "reasoning": reasoning
        }


    def _evaluate_logical_rule_with_z3(self, rule_formula: z3.ExprRef, facts: Z3FactList, rule_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a pre-parsed Z3 rule formula against Z3 facts.

        Args:
            rule_formula: The Z3 expression representing the rule.
            facts: A list of Z3 expressions representing the facts from the text.
            rule_meta: The original rule dictionary for context.

        Returns:
            Dictionary with evaluation results.
        """
        solver = z3.Solver()
        solver.set(timeout=self.z3_timeout) # Set timeout

        # Add facts derived from the text
        if facts:
            solver.add(facts)
        else:
            # If no facts are relevant, the rule might be vacuously true or irrelevant.
            # Assume compliant if no facts contradict it, but log this.
            logger.debug(f"Rule {rule_meta.get('id', 'N/A')}: No relevant Z3 facts generated. Assuming compliant by default for Z3 check.")
            return {
                "is_compliant": True,
                "method": "formal_logic",
                "confidence": 0.9, # Slightly lower confidence as no facts supported it
                "reasoning": {"step": "Z3 Check", "detail": "No relevant facts generated.", "conclusion": "Compliant (Vacuously/No Evidence)"}
            }

        # Check unsatisfiability of Facts AND Not(Rule)
        negated_rule = z3.Not(rule_formula)
        solver.add(negated_rule)

        check_result = solver.check()

        if check_result == z3.unsat:
            # Facts AND Not(Rule) is unsatisfiable => Facts imply Rule
            return {
                "is_compliant": True,
                "method": "formal_logic",
                "confidence": 1.0,
                "reasoning": {"step": "Z3 Check", "detail": f"Facts logically imply the rule '{rule_meta.get('id', 'N/A')}'.", "conclusion": "Compliant (Formal Logic)"}
            }
        elif check_result == z3.sat:
            # Facts AND Not(Rule) is satisfiable => Facts *do not* imply Rule (Violation Found)
            model = solver.model()
            witness = self._extract_violation_witness(model, facts, negated_rule)
            violation_detail = {
                "description": rule_meta.get("description", "Rule violated"),
                "witness": witness,
                "formula": rule_meta.get("symbolic_representation", "N/A"),
                "method": "formal_logic"
            }
            return {
                "is_compliant": False,
                "method": "formal_logic",
                "confidence": 1.0,
                "reasoning": {"step": "Z3 Check", "detail": f"Violation found for rule '{rule_meta.get('id', 'N/A')}'. Counterexample exists.", "conclusion": "Violation (Formal Logic)"},
                "violation_details": violation_detail
            }
        elif check_result == z3.unknown:
            # Solver timed out or was inconclusive
            logger.warning(f"Z3 solver returned 'unknown' for rule {rule_meta.get('id', 'N/A')}. Timeout: {self.z3_timeout}ms.")
            raise z3.Z3Exception(f"Solver timed out or was inconclusive after {self.z3_timeout}ms")
        else:
             # Should not happen
             raise z3.Z3Exception(f"Unexpected Z3 solver result: {check_result}")


    def _generate_z3_facts(self, text: str, context: Optional[Dict[str, Any]] = None) -> Z3FactList:
        """
        Generate a list of Z3 assertions (facts) from the input text using NLP.
        """
        logger.debug("Generating Z3 facts using spaCy NLP pipeline.")
        facts: Z3FactList = []
        doc = self.nlp(text)
        entity_map: Dict[str, z3.ExprRef] = {} # Map entity text/span to Z3 constant

        # --- Extract Entities and Assert Types ---
        EntitySort = self._get_z3_sort('Entity')
        for ent in doc.ents:
            # Create a unique Z3 constant for each entity mention
            ent_id = f"ent_{ent.start_char}" # Unique ID based on position
            z3_ent = z3.Const(ent_id, EntitySort)
            entity_map[ent_id] = z3_ent

            # Assert entity type (map spaCy labels to our Z3 predicates)
            predicate_name = self._map_spacy_label_to_predicate(ent.label_)
            if predicate_name:
                 EntityTypeFunc = self._get_z3_func(predicate_name, EntitySort)
                 facts.append(EntityTypeFunc(z3_ent))
                 logger.debug(f"Fact: {predicate_name}({ent_id} '{ent.text}')")

                 # Add specific predicates like PersonalData, PHI based on type
                 if predicate_name in ["Person", "Email", "PhoneNumber", "Location"]: # Example mapping
                      PdFunc = self._get_z3_func('PersonalData', EntitySort)
                      facts.append(PdFunc(z3_ent))
                      logger.debug(f"Fact: PersonalData({ent_id})")
                 if predicate_name in ["MedicalCondition", "Medication", "PatientInfo"]: # Example mapping
                      PhiFunc = self._get_z3_func('PHI', EntitySort) # Assuming PHI applies to 'Entity' sort
                      facts.append(PhiFunc(z3_ent))
                      logger.debug(f"Fact: PHI({ent_id})")


        # --- Extract Relations using Dependency Parsing ---
        # Simple example: Find Subject-Verb-Object triples
        ActivitySort = self._get_z3_sort('Activity') # For actions/verbs
        for token in doc:
             if "subj" in token.dep_: # Find subject
                  subject = token
                  verb = token.head
                  objects = [child for child in verb.children if "obj" in child.dep_]

                  if objects:
                       obj = objects[0] # Take first object for simplicity

                       # Find corresponding Z3 constants for subject/object if they are entities
                       subj_ent_id = f"ent_{subject.idx}"
                       obj_ent_id = f"ent_{obj.idx}"

                       z3_subj = entity_map.get(subj_ent_id)
                       z3_obj = entity_map.get(obj_ent_id)

                       # Create a predicate for the verb/action
                       # Normalize verb lemma
                       action_predicate_name = verb.lemma_.capitalize()
                       # Need to decide the signature based on subj/obj types
                       # Example: Relation(Entity, Entity) or Action(Entity)?
                       if z3_subj and z3_obj:
                            # Relation between two entities
                            RelationFunc = self._get_z3_func(action_predicate_name, EntitySort, EntitySort)
                            facts.append(RelationFunc(z3_subj, z3_obj))
                            logger.debug(f"Fact: {action_predicate_name}({subj_ent_id}, {obj_ent_id})")
                       elif z3_subj:
                            # Action performed by subject (object might not be entity)
                            ActionFunc = self._get_z3_func(action_predicate_name, EntitySort)
                            facts.append(ActionFunc(z3_subj))
                            logger.debug(f"Fact: {action_predicate_name}({subj_ent_id})")


        # --- Extract Concepts (using keywords or more advanced methods) ---
        concepts = self._extract_concepts(text) # Use existing basic method or enhance
        ConceptSort = self._get_z3_sort('Concept')
        HasConceptFunc = self._get_z3_func('HasConcept', ConceptSort) # Global concept presence
        for concept_name in concepts:
             z3_concept = z3.Const(f"concept_{concept_name}", ConceptSort)
             # Assert concept exists globally for now
             # Could refine to link concepts to specific text spans/entities if NLP supports it
             facts.append(HasConceptFunc(z3_concept))
             logger.debug(f"Fact: HasConcept(concept_{concept_name})")

             # Map generic concepts to specific Z3 predicates used in rules
             if concept_name == "PersonalData":
                  # Find entities marked as PD and assert this concept? Redundant?
                  pass
             elif concept_name == "ProcessingActivity":
                  # Find verbs like process, collect, store and assert ProcessingActivity(verb_activity)?
                  pass
             # ... map other concepts to relevant predicates ...


        # --- Add Contextual Facts ---
        if context:
            domain = context.get("domain")
            # Example: Assert the domain using a specific predicate
            if domain:
                 DomainFunc = self._get_z3_func('IsInDomain', z3.StringSort())
                 facts.append(DomainFunc(z3.StringVal(domain)))
                 logger.debug(f"Fact: IsInDomain('{domain}')")
            # Add other relevant context as facts

        logger.info(f"Generated {len(facts)} Z3 facts from text.")
        return facts


    def _map_spacy_label_to_predicate(self, label: str) -> Optional[str]:
        """Maps spaCy NER labels to Z3 predicate names."""
        # Customize this mapping based on your Z3 ontology and spaCy model
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location", # Geo-political entity
            "LOC": "Location",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Money",
            "PERCENT": "Percent",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "EMAIL": "Email", # Custom or from extended model
            "PHONE": "PhoneNumber", # Custom or from extended model
            # Domain specific (examples)
            "DISEASE": "MedicalCondition",
            "MEDICATION": "Medication",
            "PATIENT": "PatientInfo",
        }
        return mapping.get(label)


    def _parse_logic_formula(self, formula_str: str) -> z3.ExprRef:
        """
        Parse a logic formula string into a Z3 expression using the SymbolicParser.
        """
        return self.symbolic_parser.parse_logic_formula(formula_str)


    def _extract_violation_witness(self, model: z3.ModelRef, facts: Z3FactList, negated_rule: z3.BoolRef) -> str:
        """Extracts a human-readable counterexample from the Z3 model."""
        if not model: return "No model available for witness."

        witness_parts = ["Violation Witness:"]
        try:
             # Evaluate the facts in the model (should all be true)
             # Evaluate the negated rule (should be true in this model)
             witness_parts.append(f"  Negated Rule holds: {model.eval(negated_rule, model_completion=True)}")

             # Show values of variables involved in the violation
             # Need to know which variables are relevant (e.g., universally quantified ones in original rule)
             # This requires inspecting the structure of `negated_rule` which is complex.
             # Simplification: Show interpretation of constants and functions present in the facts.
             witness_parts.append("  Relevant Facts leading to violation:")
             relevant_consts = set()
             relevant_funcs = set()

             # Inspect facts to find relevant items
             def collect_symbols(expr):
                 if z3.is_const(expr) and not z3.is_true(expr) and not z3.is_false(expr):
                      relevant_consts.add(str(expr))
                 elif z3.is_app(expr): # Function application
                      relevant_funcs.add(expr.decl().name())
                      for child in expr.children():
                           collect_symbols(child)

             for fact in facts:
                  if model.eval(fact, model_completion=True): # Only show facts true in the model
                       collect_symbols(fact)
                       witness_parts.append(f"    - {fact}") # Show the fact itself

             # Display model interpretation for relevant symbols
             if relevant_consts: witness_parts.append("  Model Interpretation (Constants):")
             for const_name in sorted(list(relevant_consts)):
                  # Need to actually get the constant object back to evaluate... difficult from string name alone
                  # witness_parts.append(f"    {const_name} = {model[const_name]}") # This won't work directly
                  pass # Skipping detailed constant interpretation for now

             # This provides a starting point but needs refinement based on rule structure.
             return "\n".join(witness_parts)

        except Exception as e:
            logger.error(f"Error extracting witness from Z3 model: {e}")
            return f"Could not extract detailed witness. Model exists indicating violation. Error: {e}"


    def _create_result(self, is_compliant: bool, violations: List = [], reasoning_steps: List = [], compliance_score: float = 1.0, rule_results: Optional[Dict] = None) -> Dict[str, Any]:
         """Helper to format the evaluation result."""
         return {
            "is_compliant": is_compliant,
            "violations": violations,
            "reasoning_steps": reasoning_steps,
            "compliance_score": compliance_score,
            "rule_results": rule_results or {}
         }

    # --- Heuristic Methods (Keep existing implementations, potentially refine) ---
    def get_applicable_rules(self, text, frameworks=None, context=None):
        # (Keep existing implementation - caching, keyword/pattern check)
        # Enhancement: Could add semantic similarity check here if embeddings are available
        cache_key = self._generate_cache_key(text, frameworks, context)
        if cache_key in self.rule_cache: return self.rule_cache[cache_key]

        applicable_rules = []
        framework_filter = set(frameworks) if frameworks else None

        for rule_id, rule in self.rules.items():
            if framework_filter and rule.get("framework") not in framework_filter: continue
            # Basic domain check
            if context and "domain" in context:
                 rule_domains = rule.get("applicable_domains", ["general"]) # Assume domains list in rule
                 if "all" not in rule_domains and context["domain"] not in rule_domains: continue

            is_applicable = False
            # 1. Keyword Check
            keywords = rule.get("keywords", [])
            if keywords and any(keyword.lower() in text.lower() for keyword in keywords):
                is_applicable = True

            # 2. Pattern Check (if not already applicable)
            if not is_applicable and rule.get("pattern"):
                try:
                    if re.search(rule["pattern"], text, re.IGNORECASE | re.DOTALL):
                        is_applicable = True
                except re.error: pass # Ignore bad patterns here

            # 3. TODO: Add Semantic Check (if embeddings available)
            # if not is_applicable and self.embeddings_available:
            #    if self._check_semantic_relevance(text, rule): is_applicable = True

            if is_applicable:
                 applicable_rules.append(rule_id)

        # Sort rules by severity
        applicable_rules.sort(
            key=lambda rid: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(self.rules.get(rid, {}).get("severity", "medium"), 2)
        )
        self.rule_cache[cache_key] = applicable_rules
        return applicable_rules

    def _generate_cache_key(self, text, frameworks, context):
        # (Keep existing implementation)
        text_sample = text[:100] if text else ""
        frameworks_str = ",".join(sorted(frameworks)) if frameworks else "all"
        context_str = ""
        if context:
            domain = context.get("domain", "general")
            content_type = context.get("content_type", "text")
            context_str = f"{domain}:{content_type}"
        combined = f"{text_sample}|{frameworks_str}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _split_into_sentences(self, text):
        # Use NLP for sentence splitting
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _extract_triples(self, sentence):
        # (Keep existing basic implementation - ideally replace/enhance with NLP dependency parsing in _generate_z3_facts)
        triples = []
        svo_pattern = r'([\w\s]+?)\s+((?:is|are|was|were|has|have|had|do|does|did|will|would|shall|should|may|might|can|could)\s+[\w\s]+?|[\w]+?(?:s|es|ed|ing)?)\s+([\w\s]+)'
        matches = re.finditer(svo_pattern, sentence, re.IGNORECASE)
        for match in matches:
            subject, predicate, obj = match.groups()
            if subject and predicate and obj: triples.append({"subject": subject.strip(), "predicate": predicate.strip(), "object": obj.strip()})
        return triples

    def _extract_entities(self, sentence):
        # Use spaCy for entity extraction
        doc = self.nlp(sentence)
        entities = []
        for ent in doc.ents:
            entities.append({
                "type": ent.label_,
                "text": ent.text,
                "position": (ent.start_char, ent.end_char)
            })
        return entities

    def _extract_concepts(self, sentence):
        # (Keep existing implementation)
        concepts = []
        concept_keywords = {
            "Consent": ["consent", "permission", "authorize", "agree"],
            "PersonalData": ["personal data", "personal information", "pii"],
            "ProcessingActivity": ["process", "collect", "store", "use", "share", "disclose"],
            "PHI": ["health information", "medical data", "patient data", "phi"],
            "Authorization": ["authorization", "authorized", "approved"]
        }
        sentence_lower = sentence.lower()
        for concept, keywords in concept_keywords.items():
            if any(kw in sentence_lower for kw in keywords):
                concepts.append(concept)
        return list(set(concepts)) # Return unique concepts

    def _calculate_compliance_score(self, rule_results, applicable_rules):
        # (Keep existing implementation)
        if not rule_results or not applicable_rules: return 1.0
        severity_weights = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
        total_weight = 0
        weighted_score_sum = 0
        applicable_rule_count = len(applicable_rules) # Use count of rules *intended* to run

        for rule_id in applicable_rules: # Iterate through rules that *should* have run
            result = rule_results.get(rule_id)
            rule = self.rules.get(rule_id, {})
            severity = rule.get("severity", "medium")
            weight = severity_weights.get(severity, 0.5)
            total_weight += weight

            if result:
                score = 1.0 if result.get("is_compliant", False) else 0.0
                confidence = result.get("confidence", 0.7) # Default confidence if method failed?
                weighted_score_sum += weight * score * confidence
            else:
                # Rule was applicable but didn't produce a result (e.g., error?)
                # Penalize? Or treat as compliant with low confidence? Let's penalize slightly.
                weighted_score_sum += weight * 0.5 * 0.5 # Assume 50% compliant, 50% confidence

        # Normalize score based on the weight of *applicable* rules
        if total_weight > 0:
            return weighted_score_sum / total_weight
        else:
            return 1.0 # No applicable rules with weight? Compliant.
