import inspect
import os
import re
import json
import markdown
from typing import Dict, List, Any, Optional, Callable, Type, Union
import importlib

class ComplianceDocumentationGenerator:
    """
    Advanced documentation generator for compliance frameworks.
    Creates comprehensive documentation including API reference,
    usage examples, compliance strategies, and regulatory guides.
    """
    
    def __init__(self, 
                package_name: str, 
                output_dir: str = "docs",
                include_private: bool = False,
                template_dir: str = "doc_templates"):
        """
        Initialize the documentation generator
        
        Args:
            package_name: Name of the package to document
            output_dir: Directory for output documentation
            include_private: Whether to include private members (starting with _)
            template_dir: Directory containing documentation templates
        """
        self.package_name = package_name
        self.output_dir = output_dir
        self.include_private = include_private
        self.template_dir = template_dir
        
        # Import the target package
        self.package = importlib.import_module(package_name)
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track processed modules
        self.processed_modules = set()
        
        # Setup template engine
        self._load_templates()
    
    def _load_templates(self):
        """Load documentation templates"""
        self.templates = {}
        template_files = [
            "module_template.md",
            "class_template.md",
            "function_template.md",
            "index_template.md",
            "regulatory_template.md"
        ]
        
        for template_file in template_files:
            template_path = os.path.join(self.template_dir, template_file)
            if os.path.exists(template_path):
                with open(template_path, "r") as f:
                    self.templates[template_file] = f.read()
            else:
                # Create default template
                self.templates[template_file] = self._create_default_template(template_file)
    
    def _create_default_template(self, template_name: str) -> str:
        """Create a default template if not found"""
        if template_name == "module_template.md":
            return """# {module_name}

{module_doc}

## Classes

{class_list}

## Functions

{function_list}
"""
        elif template_name == "class_template.md":
            return """## {class_name}

{class_doc}

### Methods

{method_list}

### Properties

{property_list}
"""
        elif template_name == "function_template.md":
            return """### {function_name}

{function_doc}

**Parameters:**
{param_list}

**Returns:**
{returns}

**Example:**
```python
{example}
```
"""
        elif template_name == "index_template.md":
            return """# {package_name} Documentation

{package_doc}

## Modules

{module_list}

## Regulatory Frameworks

{framework_list}
"""
        elif template_name == "regulatory_template.md":
            return """# {framework_name}

{framework_doc}

## Key Requirements

{requirement_list}

## Implementation Strategy

{implementation_strategy}
"""
        return ""
    
    def generate_documentation(self, 
                             regulatory_info: Dict[str, Any] = None,
                             examples: Dict[str, str] = None):
        """
        Generate documentation for the package
        
        Args:
            regulatory_info: Information about regulatory frameworks
            examples: Example code snippets for functions/classes
        """
        print(f"Generating documentation for {self.package_name}...")
        
        # Create module documentation
        self._document_package(self.package, examples)
        
        # Create regulatory documentation
        if regulatory_info:
            self._create_regulatory_guides(regulatory_info)
            
        # Create index file
        self._create_index(regulatory_info)
        
        print(f"Documentation generated in {self.output_dir}")
    
    def _document_package(self, package, examples: Dict[str, str] = None):
        """Document a package and its modules recursively"""
        package_path = package.__path__[0] if hasattr(package, "__path__") else None
        if not package_path:
            return
            
        # Get all Python modules in the package
        for root, dirs, files in os.walk(package_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    module_path = os.path.join(root, file)
                    relative_path = os.path.relpath(module_path, package_path)
                    module_name = os.path.splitext(relative_path)[0].replace(os.path.sep, ".")
                    full_module_name = f"{package.__name__}.{module_name}"
                    
                    # Skip if already processed
                    if full_module_name in self.processed_modules:
                        continue
                        
                    # Import module
                    try:
                        module = importlib.import_module(full_module_name)
                        self._document_module(module, examples)
                        self.processed_modules.add(full_module_name)
                    except ImportError as e:
                        print(f"Error importing {full_module_name}: {e}")
    
    def _document_module(self, module, examples: Dict[str, str] = None):
        """Generate documentation for a single module"""
        module_name = module.__name__
        module_doc = inspect.getdoc(module) or "No documentation available."
        
        # Get all classes and functions in the module
        classes = []
        functions = []
        
        for name, obj in inspect.getmembers(module):
            # Skip private members unless included
            if name.startswith("_") and not self.include_private:
                continue
                
            # Document classes
            if inspect.isclass(obj) and obj.__module__ == module_name:
                classes.append(self._document_class(obj, examples))
                
            # Document functions
            elif inspect.isfunction(obj) and obj.__module__ == module_name:
                functions.append(self._document_function(obj, examples))
        
        # Create module documentation
        module_doc_path = os.path.join(self.output_dir, f"{module_name.replace('.', '_')}.md")
        
        with open(module_doc_path, "w") as f:
            f.write(self.templates["module_template.md"].format(
                module_name=module_name,
                module_doc=module_doc,
                class_list="\n".join(classes),
                function_list="\n".join(functions)
            ))
    
    def _document_class(self, cls, examples: Dict[str, str] = None) -> str:
        """Document a class and its methods"""
        class_name = cls.__name__
        class_doc = inspect.getdoc(cls) or "No documentation available."
        
        # Get methods
        methods = []
        properties = []
        
        for name, obj in inspect.getmembers(cls):
            # Skip private members unless included
            if name.startswith("_") and not self.include_private:
                continue
                
            # Document methods
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                methods.append(self._document_method(obj, class_name, examples))
                
            # Document properties
            elif isinstance(obj, property):
                properties.append(f"* **{name}**: {inspect.getdoc(obj) or 'No documentation available.'}")
        
        # Format class documentation
        class_doc_md = self.templates["class_template.md"].format(
            class_name=class_name,
            class_doc=class_doc,
            method_list="\n".join(methods),
            property_list="\n".join(properties) if properties else "No properties documented."
        )
        
        return class_doc_md
    
    def _document_method(self, method, class_name: str, examples: Dict[str, str] = None) -> str:
        """Document a method"""
        method_name = method.__name__
        
        # Create a function-like document for the method
        function_doc = self._document_function(
            method, 
            examples, 
            prefix=f"{class_name}."
        )
        
        return function_doc
    
    def _document_function(self, 
                         func, 
                         examples: Dict[str, str] = None, 
                         prefix: str = "") -> str:
        """Document a function"""
        func_name = func.__name__
        full_name = f"{prefix}{func_name}"
        func_doc = inspect.getdoc(func) or "No documentation available."
        
        # Get parameters
        try:
            signature = inspect.signature(func)
            params = []
            
            for name, param in signature.parameters.items():
                if name == "self":
                    continue
                    
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
                param_type = str(param_type).replace("<class '", "").replace("'>", "")
                
                default = ""
                if param.default != inspect.Parameter.empty:
                    default = f" (default: {param.default})"
                    
                params.append(f"* **{name}** ({param_type}){default}")
                
            # Get return type
            returns = "None"
            if signature.return_annotation != inspect.Signature.empty:
                returns = str(signature.return_annotation).replace("<class '", "").replace("'>", "")
                
        except (ValueError, TypeError):
            # Fallback for functions that can't be inspected
            params = ["Unable to determine parameters"]
            returns = "Unable to determine return type"
            
        # Get example if available
        example = "# No example available"
        if examples and full_name in examples:
            example = examples[full_name]
            
        # Format function documentation
        func_doc_md = self.templates["function_template.md"].format(
            function_name=full_name,
            function_doc=func_doc,
            param_list="\n".join(params) if params else "No parameters.",
            returns=returns,
            example=example
        )
        
        return func_doc_md
    
    def _create_regulatory_guides(self, regulatory_info: Dict[str, Any]):
        """Create regulatory framework guides"""
        for framework_id, framework_data in regulatory_info.items():
            # Format requirements
            requirements = []
            for req in framework_data.get("requirements", []):
                requirements.append(f"* **{req['id']}**: {req['description']}")
                
            # Format implementation strategy
            implementation = framework_data.get("implementation_strategy", "No implementation strategy available.")
            
            # Create regulatory guide
            guide_path = os.path.join(self.output_dir, f"regulatory_{framework_id.lower()}.md")
            
            with open(guide_path, "w") as f:
                f.write(self.templates["regulatory_template.md"].format(
                    framework_name=framework_data.get("name", framework_id),
                    framework_doc=framework_data.get("description", "No description available."),
                    requirement_list="\n".join(requirements) if requirements else "No specific requirements documented.",
                    implementation_strategy=implementation
                ))
    
    def _create_index(self, regulatory_info: Dict[str, Any] = None):
        """Create index documentation file"""
        package_doc = inspect.getdoc(self.package) or "No documentation available."
        
        # List all modules
        modules = []
        for module_name in sorted(self.processed_modules):
            short_name = module_name.replace(f"{self.package_name}.", "")
            modules.append(f"* [{short_name}]({module_name.replace('.', '_')}.md)")
            
        # List all regulatory frameworks
        frameworks = []
        if regulatory_info:
            for framework_id, framework_data in regulatory_info.items():
                frameworks.append(f"* [{framework_data.get('name', framework_id)}](regulatory_{framework_id.lower()}.md)")
        
        # Create index file
        index_path = os.path.join(self.output_dir, "index.md")
        
        with open(index_path, "w") as f:
            f.write(self.templates["index_template.md"].format(
                package_name=self.package_name,
                package_doc=package_doc,
                module_list="\n".join(modules) if modules else "No modules documented.",
                framework_list="\n".join(frameworks) if frameworks else "No regulatory frameworks documented."
            ))
            
        # Create HTML version of index
        self._convert_to_html("index.md")
    
    def _convert_to_html(self, markdown_file: str):
        """Convert a markdown file to HTML"""
        md_path = os.path.join(self.output_dir, markdown_file)
        html_path = os.path.join(self.output_dir, markdown_file.replace(".md", ".html"))
        
        with open(md_path, "r") as f:
            md_content = f.read()
            
        html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add simple styling
        styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{markdown_file.replace(".md", "")}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 1em; max-width: 1200px; margin: 0 auto; color: #333; }}
        pre {{ background-color: #f6f8fa; padding: 1em; overflow-x: auto; border-radius: 3px; }}
        code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; background-color: rgba(27,31,35,0.05); padding: 0.2em 0.4em; border-radius: 3px; }}
        pre code {{ background-color: transparent; padding: 0; }}
        h1, h2, h3, h4, h5, h6 {{ margin-top: 1.5em; margin-bottom: 0.5em; }}
        a {{ color: #0366d6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
        th {{ background-color: #f6f8fa; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
        
        with open(html_path, "w") as f:
            f.write(styled_html)

# Example usage:
# doc_gen = ComplianceDocumentationGenerator("src.compliance")
# examples = {
#     "ComplianceVerifier.verify_content": "verifier = ComplianceVerifier(config)\nresult = verifier.verify_content('some text')",
#     "ComplianceAwareDistillation.distill": "distiller = ComplianceAwareDistillation(teacher_model, student_model, 'GDPR')\nstudent = distiller.distill(epochs=10)"
# }
# regulatory_info = {
#     "GDPR": {
#         "name": "General Data Protection Regulation",
#         "description": "EU regulation on data protection and privacy",
#         "requirements": [
#             {"id": "Art. 5", "description": "Principles relating to processing of personal data"},
#             {"id": "Art. 6", "description": "Lawfulness of processing"}
#         ],
#         "implementation_strategy": "The framework implements GDPR compliance through..."
#     }
# }
# doc_gen.generate_documentation(regulatory_info, examples)