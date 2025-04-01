from lark import Lark, Transformer, v_args, Token, Tree, UnexpectedToken, UnexpectedCharacters
import z3
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set

class SymbolicParser:
    """
    Advanced parser for symbolic logic formulas using lark-parser.
    Converts string representations to Z3 expressions with robust error handling,
    type checking, and support for a wide range of logical, arithmetic, and relational operators.
    """
    
    def __init__(self, reasoner):
        """Initialize the parser with a reference to the SymbolicReasoner."""
        self.reasoner = reasoner
        self.logger = logging.getLogger(__name__)
        
        # Define the grammar for a comprehensive symbolic logic language
        self.grammar = r"""
            ?start: formula
            
            // Formulas
            ?formula: quantifier_expr
                    | connective_expr
                    | relation_expr
                    | predicate_expr
                    | var
                    | bool_const
                    | "(" formula ")"                       -> paren_formula
            
            // Quantifiers
            ?quantifier_expr: forall_expr
                           | exists_expr
            
            forall_expr: "ForAll" "(" var_decls "," formula ")"
            exists_expr: "Exists" "(" var_decls "," formula ")"
            
            var_decls: var_decl (";" var_decl)*
            var_decl: IDENTIFIER ":" sort_expr
            
            ?sort_expr: IDENTIFIER                          -> simple_sort
                      | IDENTIFIER "(" sort_expr ("," sort_expr)* ")"  -> parameterized_sort
            
            // Logical connectives (in order of precedence)
            ?connective_expr: implies_expr
                           | iff_expr
                           | xor_expr
                           | or_expr
                           | and_expr
                           | not_expr
            
            implies_expr: "Implies" "(" formula "," formula ")"
            iff_expr: "Iff" "(" formula "," formula ")"
            xor_expr: "Xor" "(" formula "," formula ")"
            or_expr: "Or" "(" [formula ("," formula)*] ")"
            and_expr: "And" "(" [formula ("," formula)*] ")"
            not_expr: "Not" "(" formula ")"
            
            // Relational expressions (comparisons)
            ?relation_expr: eq_expr
                         | neq_expr
                         | lt_expr
                         | le_expr
                         | gt_expr
                         | ge_expr
            
            eq_expr: "=" "(" term "," term ")"
            neq_expr: "!=" "(" term "," term ")"
            lt_expr: "<" "(" term "," term ")"
            le_expr: "<=" "(" term "," term ")"
            gt_expr: ">" "(" term "," term ")"
            ge_expr: ">=" "(" term "," term ")"
            
            // Predicate application
            predicate_expr: IDENTIFIER "(" [term ("," term)*] ")"
            
            // Terms
            ?term: formula                                  -> bool_term
                | arith_expr
                | array_expr
                | func_app
                | constant
                | var
                | "(" term ")"                              -> paren_term
            
            // Arithmetic expressions
            ?arith_expr: add_expr
                       | sub_expr
                       | mul_expr
                       | div_expr
                       | mod_expr
                       | neg_expr
                       | int_const
                       | real_const
            
            add_expr: "+" "(" term "," term ")"
            sub_expr: "-" "(" term "," term ")"
            mul_expr: "*" "(" term "," term ")"
            div_expr: "/" "(" term "," term ")"
            mod_expr: "%" "(" term "," term ")"
            neg_expr: "-" term
            
            // Array expressions
            ?array_expr: array_select
                       | array_store
            
            array_select: "Select" "(" term "," term ")"
            array_store: "Store" "(" term "," term "," term ")"
            
            // Function application
            func_app: IDENTIFIER "(" [term ("," term)*] ")"
            
            // Constants
            ?constant: bool_const
                     | int_const
                     | real_const
                     | string_const
            
            bool_const: "true" | "false"
            int_const: INT
            real_const: FLOAT
            string_const: ESCAPED_STRING
            
            // Variables
            var: IDENTIFIER
            
            // Comments
            COMMENT: "//" /[^\n]/*
                   | "/*" /(.|\n)+?/ "*/"
            
            // Tokens
            IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
            
            %import common.INT
            %import common.FLOAT
            %import common.ESCAPED_STRING
            %import common.WS
            
            %ignore WS
            %ignore COMMENT
        """
        
        # Initialize the Lark parser with better error handling
        self.parser = Lark(self.grammar, parser="earley", debug=True, propagate_positions=True)
        
        # Initialize the transformer
        self.transformer = AdvancedLogicTransformer(self.reasoner)
    
    def parse_logic_formula(self, formula_str: str) -> z3.ExprRef:
        """
        Parse a logic formula string into a Z3 expression.
        
        Args:
            formula_str: String representation of the formula
            
        Returns:
            Z3 expression representing the parsed formula
            
        Raises:
            ValueError: If the formula cannot be parsed or has semantic errors
        """
        try:
            # Validate the formula string
            self._validate_input(formula_str)
            
            # Parse the formula string using Lark
            parse_tree = self.parser.parse(formula_str.strip())
            self.logger.debug(f"Successfully parsed formula: {formula_str}")
            
            # Transform the parse tree into a Z3 expression
            z3_expr = self.transformer.transform(parse_tree)
            
            # Perform type validation on the result
            self._validate_expression(z3_expr)
            
            return z3_expr
            
        except UnexpectedToken as e:
            position_info = f" at line {e.line}, column {e.column}"
            expected = f", expected: {', '.join(e.expected)}" if e.expected else ""
            error_msg = f"Syntax error{position_info}: Unexpected token '{e.token}'{expected}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
            
        except UnexpectedCharacters as e:
            position_info = f" at line {e.line}, column {e.column}"
            error_msg = f"Syntax error{position_info}: Unexpected character '{e.char}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
            
        except Exception as e:
            self.logger.error(f"Failed to parse logic formula '{formula_str}': {e}", exc_info=True)
            raise ValueError(f"Formula parsing error: {e}") from e
    
    def _validate_input(self, formula_str: str) -> None:
        """Perform basic validation on the input formula string."""
        if not formula_str:
            raise ValueError("Empty formula string")
        
        # Check for balanced parentheses
        paren_count = 0
        for char in formula_str:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    raise ValueError("Unbalanced parentheses: too many closing parentheses")
        
        if paren_count > 0:
            raise ValueError("Unbalanced parentheses: missing closing parentheses")
    
    def _validate_expression(self, expr: z3.ExprRef) -> None:
        """Validate the generated Z3 expression for logical soundness."""
        # This could be extended with more sophisticated semantic validation
        if not expr.sort() == z3.BoolSort() and not isinstance(expr, z3.BoolRef):
            self.logger.warning(f"Top-level expression does not have boolean sort: {expr}")


@v_args(inline=True)
class AdvancedLogicTransformer(Transformer):
    """
    Transforms the parse tree produced by Lark into Z3 expressions
    with robust type checking and error handling.
    """
    
    def __init__(self, reasoner):
        super().__init__()
        self.reasoner = reasoner
        self.bound_vars = {}  # Maps variable names to Z3 constants
        self.logger = logging.getLogger(__name__ + ".transformer")
    
    # --- Formula handlers ---
    
    def paren_formula(self, formula):
        """Handle parenthesized formulas."""
        return formula
    
    # --- Quantifier handlers ---
    
    def forall_expr(self, var_decls, formula):
        """Process a universal quantifier."""
        z3_vars, old_vars = self._process_var_decls(var_decls)
        
        # Create the quantified expression
        result = z3.ForAll(z3_vars, formula)
        
        # Restore previous variable scope
        self.bound_vars = old_vars
        
        return result
    
    def exists_expr(self, var_decls, formula):
        """Process an existential quantifier."""
        z3_vars, old_vars = self._process_var_decls(var_decls)
        
        # Create the quantified expression
        result = z3.Exists(z3_vars, formula)
        
        # Restore previous variable scope
        self.bound_vars = old_vars
        
        return result
    
    def _process_var_decls(self, var_decls):
        """Process variable declarations for quantifiers."""
        # Create new scope for variables
        old_bound_vars = self.bound_vars.copy()
        z3_vars = []
        
        # Create bound variables
        for var_name, sort_expr in var_decls:
            z3_sort = self._resolve_sort(sort_expr)
            z3_var = z3.Const(var_name, z3_sort)
            self.bound_vars[var_name] = z3_var
            z3_vars.append(z3_var)
        
        return z3_vars, old_bound_vars
    
    def var_decls(self, *var_decl_items):
        """Process multiple variable declarations."""
        return list(var_decl_items)
    
    def var_decl(self, var_name, sort_expr):
        """Process a single variable declaration."""
        return (var_name.value, sort_expr)
    
    def simple_sort(self, sort_name):
        """Process a simple sort."""
        return sort_name.value
    
    def parameterized_sort(self, sort_name, *params):
        """Process a parameterized sort. This is for future extension."""
        # Currently Z3 Python API doesn't easily support parameterized sorts
        # This would need custom handling based on the sort name
        self.logger.warning(f"Parameterized sorts are not fully supported: {sort_name.value}({', '.join(map(str, params))})")
        return sort_name.value
    
    # --- Logical connective handlers ---
    
    def implies_expr(self, left, right):
        """Process an implication."""
        self._assert_bool_sort(left, "Implies left operand")
        self._assert_bool_sort(right, "Implies right operand")
        return z3.Implies(left, right)
    
    def iff_expr(self, left, right):
        """Process a bidirectional implication (if and only if)."""
        self._assert_bool_sort(left, "Iff left operand")
        self._assert_bool_sort(right, "Iff right operand")
        return left == right
    
    def xor_expr(self, left, right):
        """Process an exclusive or."""
        self._assert_bool_sort(left, "Xor left operand")
        self._assert_bool_sort(right, "Xor right operand")
        return z3.Xor(left, right)
    
    def and_expr(self, *args):
        """Process a conjunction."""
        for i, arg in enumerate(args):
            self._assert_bool_sort(arg, f"And argument {i+1}")
        return z3.And(args) if args else z3.BoolVal(True)
    
    def or_expr(self, *args):
        """Process a disjunction."""
        for i, arg in enumerate(args):
            self._assert_bool_sort(arg, f"Or argument {i+1}")
        return z3.Or(args) if args else z3.BoolVal(False)
    
    def not_expr(self, arg):
        """Process a negation."""
        self._assert_bool_sort(arg, "Not argument")
        return z3.Not(arg)
    
    # --- Relational expression handlers ---
    
    def eq_expr(self, left, right):
        """Process an equality comparison."""
        self._assert_compatible_sorts(left, right, "Equality")
        return left == right
    
    def neq_expr(self, left, right):
        """Process an inequality comparison."""
        self._assert_compatible_sorts(left, right, "Inequality")
        return left != right
    
    def lt_expr(self, left, right):
        """Process a less-than comparison."""
        self._assert_arithmetic_sort(left, "Less-than left operand")
        self._assert_arithmetic_sort(right, "Less-than right operand")
        self._assert_compatible_sorts(left, right, "Less-than")
        return left < right
    
    def le_expr(self, left, right):
        """Process a less-than-or-equal comparison."""
        self._assert_arithmetic_sort(left, "Less-than-or-equal left operand")
        self._assert_arithmetic_sort(right, "Less-than-or-equal right operand")
        self._assert_compatible_sorts(left, right, "Less-than-or-equal")
        return left <= right
    
    def gt_expr(self, left, right):
        """Process a greater-than comparison."""
        self._assert_arithmetic_sort(left, "Greater-than left operand")
        self._assert_arithmetic_sort(right, "Greater-than right operand")
        self._assert_compatible_sorts(left, right, "Greater-than")
        return left > right
    
    def ge_expr(self, left, right):
        """Process a greater-than-or-equal comparison."""
        self._assert_arithmetic_sort(left, "Greater-than-or-equal left operand")
        self._assert_arithmetic_sort(right, "Greater-than-or-equal right operand")
        self._assert_compatible_sorts(left, right, "Greater-than-or-equal")
        return left >= right
    
    # --- Predicate application handler ---
    
    def predicate_expr(self, name, *args):
        """Process a predicate application (returns boolean)."""
        name = name.value
        z3_args = [self._convert_to_term(arg) for arg in args]
        
        # Get argument sorts
        arg_sorts = [arg.sort() for arg in z3_args]
        
        # Get function declaration
        func = self.reasoner._get_z3_func(name, *arg_sorts)
        
        # Apply function to arguments
        return func(*z3_args)
    
    def bool_term(self, formula):
        """Convert a boolean formula to a term."""
        self._assert_bool_sort(formula, "Boolean term")
        return formula
    
    # --- Term handlers ---
    
    def paren_term(self, term):
        """Handle parenthesized terms."""
        return term
    
    # --- Arithmetic expression handlers ---
    
    def add_expr(self, left, right):
        """Process an addition."""
        self._assert_arithmetic_sort(left, "Addition left operand")
        self._assert_arithmetic_sort(right, "Addition right operand")
        return left + right
    
    def sub_expr(self, left, right):
        """Process a subtraction."""
        self._assert_arithmetic_sort(left, "Subtraction left operand")
        self._assert_arithmetic_sort(right, "Subtraction right operand")
        return left - right
    
    def mul_expr(self, left, right):
        """Process a multiplication."""
        self._assert_arithmetic_sort(left, "Multiplication left operand")
        self._assert_arithmetic_sort(right, "Multiplication right operand")
        return left * right
    
    def div_expr(self, left, right):
        """Process a division."""
        self._assert_arithmetic_sort(left, "Division left operand")
        self._assert_arithmetic_sort(right, "Division right operand")
        return left / right
    
    def mod_expr(self, left, right):
        """Process a modulo operation."""
        self._assert_int_sort(left, "Modulo left operand")
        self._assert_int_sort(right, "Modulo right operand")
        return left % right
    
    def neg_expr(self, term):
        """Process a negation."""
        self._assert_arithmetic_sort(term, "Arithmetic negation operand")
        return -term
    
    # --- Array expression handlers ---
    
    def array_select(self, array, index):
        """Process an array selection."""
        # Check if array has array sort
        if not z3.is_array(array):
            raise ValueError(f"Select operation requires an array, got {array.sort()}")
        return z3.Select(array, index)
    
    def array_store(self, array, index, value):
        """Process an array store operation."""
        # Check if array has array sort
        if not z3.is_array(array):
            raise ValueError(f"Store operation requires an array, got {array.sort()}")
        return z3.Store(array, index, value)
    
    # --- Function application handler ---
    
    def func_app(self, name, *args):
        """Process a function application (non-predicate)."""
        name = name.value
        z3_args = [self._convert_to_term(arg) for arg in args]
        
        # Get argument sorts
        arg_sorts = [arg.sort() for arg in z3_args]
        
        # TODO: Handle non-boolean function returns
        # Currently, we're treating all functions as returning boolean
        # This requires extending the reasoner's function registry
        
        # Get function declaration
        func = self.reasoner._get_z3_func(name, *arg_sorts)
        
        # Apply function to arguments
        return func(*z3_args)
    
    # --- Constant handlers ---
    
    def bool_const(self, value):
        """Process a boolean constant."""
        return z3.BoolVal(value.value == "true")
    
    def int_const(self, value):
        """Process an integer constant."""
        return z3.IntVal(int(value.value))
    
    def real_const(self, value):
        """Process a real number constant."""
        return z3.RealVal(float(value.value))
    
    def string_const(self, value):
        """Process a string constant."""
        # Remove quotes from the string value
        str_value = value.value[1:-1]
        return z3.StringVal(str_value)
    
    # --- Variable handler ---
    
    def var(self, name):
        """Process a variable reference."""
        name = name.value
        if name in self.bound_vars:
            return self.bound_vars[name]
        
        # If variable is not bound, assume it's a constant of Entity sort
        # This could be customized based on context
        entity_sort = self.reasoner._get_z3_sort('Entity')
        return z3.Const(name, entity_sort)
    
    # --- Helper methods ---
    
    def _resolve_sort(self, sort_name):
        """Resolve a sort name to a Z3 sort."""
        if isinstance(sort_name, tuple):
            # Handle parameterized sorts (not fully implemented)
            base_sort, *params = sort_name
            # For now, just use the base sort
            return self.reasoner._get_z3_sort(base_sort)
        else:
            return self.reasoner._get_z3_sort(sort_name)
    
    def _convert_to_term(self, value):
        """Ensure a value is a valid Z3 term."""
        if hasattr(value, 'sort'):
            return value
        
        # If it's a string or token, try to resolve as variable
        if isinstance(value, str) or hasattr(value, 'value'):
            var_name = value if isinstance(value, str) else value.value
            if var_name in self.bound_vars:
                return self.bound_vars[var_name]
            
            # Default to Entity sort for unbound variables
            entity_sort = self.reasoner._get_z3_sort('Entity')
            return z3.Const(var_name, entity_sort)
        
        raise ValueError(f"Cannot convert to Z3 term: {value}")
    
    def _assert_bool_sort(self, expr, context="Expression"):
        """Assert that an expression has boolean sort."""
        if not z3.is_bool(expr):
            raise ValueError(f"{context} must have boolean sort, got {expr.sort()}")
    
    def _assert_arithmetic_sort(self, expr, context="Expression"):
        """Assert that an expression has arithmetic (int or real) sort."""
        if not (z3.is_int(expr) or z3.is_real(expr)):
            raise ValueError(f"{context} must have numeric sort (Int or Real), got {expr.sort()}")
    
    def _assert_int_sort(self, expr, context="Expression"):
        """Assert that an expression has integer sort."""
        if not z3.is_int(expr):
            raise ValueError(f"{context} must have integer sort, got {expr.sort()}")
    
    def _assert_compatible_sorts(self, left, right, op_name="Operation"):
        """Assert that two expressions have compatible sorts for an operation."""
        if left.sort() != right.sort():
            raise ValueError(f"{op_name} requires operands of the same sort, got {left.sort()} and {right.sort()}")
