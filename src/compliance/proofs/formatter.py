import json
from html import escape as html_escape

class ProofFormatter:
    """
    Formats compliance verification proofs into different representations
    for auditability, explanation generation, and regulatory documentation.
    """
    def __init__(self):
        self.format_handlers = {
            "json": self.format_as_json,
            "text": self.format_as_text,
            "html": self.format_as_html,
            "markdown": self.format_as_markdown
        }
        
    def format(self, proof_trace, format_type="text"):
        """
        Format a proof trace in the specified format.
        
        Args:
            proof_trace: Compliance verification proof trace
            format_type: Format type ("json", "text", "html", "markdown")
            
        Returns:
            Formatted proof trace
        """
        handler = self.format_handlers.get(format_type.lower(), self.format_as_text)
        return handler(proof_trace)
        
    def format_as_json(self, proof_trace):
        """Format proof trace as JSON"""
        # Simple JSON formatting with indentation
        return json.dumps(proof_trace, indent=2)
        
    def format_as_text(self, proof_trace):
        """Format proof trace as human-readable text"""
        text = "Compliance Verification Proof\n"
        text += "===========================\n\n"
        
        # Add input summary
        text += f"Input: {proof_trace.get('input', '')[:100]}\n"
        if len(proof_trace.get('input', '')) > 100:
            text += "...(truncated)\n"
        text += "\n"
        
        # Add frameworks
        frameworks = proof_trace.get('frameworks', [])
        if frameworks:
            text += f"Frameworks: {', '.join(frameworks)}\n"
            
        # Add compliance mode
        text += f"Compliance Mode: {proof_trace.get('mode', 'standard')}\n\n"
        
        # Add verification steps
        steps = proof_trace.get('steps', [])
        if steps:
            text += "Verification Steps:\n"
            text += "-------------------\n"
            
            for i, step in enumerate(steps, 1):
                step_type = step.get('step_type', '')
                
                if step_type == 'framework_start':
                    text += f"\n[{i}] Framework: {step.get('framework_id', 'unknown')}\n"
                    text += f"    {step.get('description', '')}\n"
                    
                elif step_type == 'rule_verification':
                    text += f"\n  [{i}] Rule: {step.get('rule_id', 'unknown')}\n"
                    text += f"      Description: {step.get('rule_text', '')}\n"
                    text += f"      Result: {step.get('intermediate_conclusion', '')}\n"
                    if 'justification' in step:
                        text += f"      Justification: {step.get('justification', '')}\n"
                        
                elif step_type == 'framework_conclusion':
                    text += f"\n  [{i}] Framework Conclusion: {step.get('intermediate_conclusion', '')}\n"
                    text += f"      Compliance Score: {step.get('compliance_score', 0.0):.2f}\n"
                    text += f"      Violations: {step.get('violation_count', 0)}\n"
                    
                elif step_type == 'aggregation':
                    text += f"\n[{i}] Overall Result: {step.get('intermediate_conclusion', '')}\n"
                    text += f"    Compliance Score: {step.get('compliance_score', 0.0):.2f}\n"
                else:
                    text += f"\n[{i}] {step.get('description', 'Step')}\n"
                    if 'intermediate_conclusion' in step:
                        text += f"    Result: {step.get('intermediate_conclusion', '')}\n"
        
        # Add final conclusion
        conclusion = proof_trace.get('conclusion', {})
        text += "\nFinal Conclusion:\n"
        text += "-----------------\n"
        text += f"Compliance Status: {'Compliant' if conclusion.get('is_compliant', False) else 'Non-compliant'}\n"
        text += f"Overall Score: {conclusion.get('compliance_score', 0.0):.2f}\n"
        text += f"Violations: {conclusion.get('violation_count', 0)}\n"
        text += f"Justification: {conclusion.get('justification', 'No justification provided')}\n"
        
        # Add timestamp
        if 'timestamp' in conclusion:
            text += f"\nVerification Completed: {conclusion.get('timestamp', '')}\n"
            
        return text
        
    def format_as_html(self, proof_trace):
        """Format proof trace as HTML"""
        html = "<div class='compliance-proof'>\n"
        html += "<h2>Compliance Verification Proof</h2>\n"
        
        # Add input summary
        html += "<div class='input-summary'>\n"
        html += "<h3>Input</h3>\n"
        html += f"<p>{html_escape(proof_trace.get('input', '')[:100])}"
        if len(proof_trace.get('input', '')) > 100:
            html += "...(truncated)"
        html += "</p>\n"
        html += "</div>\n"
        
        # Add frameworks and mode
        html += "<div class='verification-context'>\n"
        frameworks = proof_trace.get('frameworks', [])
        if frameworks:
            html += f"<p><strong>Frameworks:</strong> {', '.join(html_escape(fw) for fw in frameworks)}</p>\n"
        html += f"<p><strong>Compliance Mode:</strong> {html_escape(proof_trace.get('mode', 'standard'))}</p>\n"
        html += "</div>\n"
        
        # Add verification steps
        steps = proof_trace.get('steps', [])
        if steps:
            html += "<h3>Verification Steps</h3>\n"
            html += "<div class='verification-steps'>\n"
            
            current_framework = None
            for step in steps:
                step_type = step.get('step_type', '')
                
                if step_type == 'framework_start':
                    # Close previous framework div if there was one
                    if current_framework is not None:
                        html += "</div>\n"
                        
                    framework_id = step.get('framework_id', 'unknown')
                    current_framework = framework_id
                    html += f"<div class='framework' id='framework-{html_escape(framework_id)}'>\n"
                    html += f"<h4>Framework: {html_escape(framework_id)}</h4>\n"
                    html += f"<p>{html_escape(step.get('description', ''))}</p>\n"
                    
                elif step_type == 'rule_verification':
                    rule_id = step.get('rule_id', 'unknown')
                    conclusion = step.get('intermediate_conclusion', '')
                    is_compliant = 'violates' not in conclusion.lower()
                    
                    html += f"<div class='rule {is_compliant and 'compliant' or 'non-compliant'}'>\n"
                    html += f"<p><strong>Rule:</strong> {html_escape(rule_id)}</p>\n"
                    html += f"<p><strong>Description:</strong> {html_escape(step.get('rule_text', ''))}</p>\n"
                    html += f"<p><strong>Result:</strong> {html_escape(conclusion)}</p>\n"
                    if 'justification' in step:
                        html += f"<p><strong>Justification:</strong> {html_escape(step.get('justification', ''))}</p>\n"
                    html += "</div>\n"
                    
                elif step_type == 'framework_conclusion':
                    html += "<div class='framework-conclusion'>\n"
                    html += f"<p><strong>Framework Conclusion:</strong> {html_escape(step.get('intermediate_conclusion', ''))}</p>\n"
                    html += f"<p><strong>Compliance Score:</strong> {step.get('compliance_score', 0.0):.2f}</p>\n"
                    html += f"<p><strong>Violations:</strong> {step.get('violation_count', 0)}</p>\n"
                    html += "</div>\n"
                    
                elif step_type == 'aggregation':
                    # Close framework div if there was one
                    if current_framework is not None:
                        html += "</div>\n"
                        current_framework = None
                        
                    html += "<div class='aggregation'>\n"
                    html += f"<h4>Overall Result</h4>\n"
                    html += f"<p>{html_escape(step.get('intermediate_conclusion', ''))}</p>\n"
                    html += f"<p><strong>Compliance Score:</strong> {step.get('compliance_score', 0.0):.2f}</p>\n"
                    html += "</div>\n"
            
            # Close framework div if still open
            if current_framework is not None:
                html += "</div>\n"
                
            html += "</div>\n"  # Close verification-steps
        
        # Add final conclusion
        conclusion = proof_trace.get('conclusion', {})
        is_compliant = conclusion.get('is_compliant', False)
        
        html += f"<div class='conclusion {is_compliant and 'compliant' or 'non-compliant'}'>\n"
        html += "<h3>Final Conclusion</h3>\n"
        html += f"<p><strong>Compliance Status:</strong> {'Compliant' if is_compliant else 'Non-compliant'}</p>\n"
        html += f"<p><strong>Overall Score:</strong> {conclusion.get('compliance_score', 0.0):.2f}</p>\n"
        html += f"<p><strong>Violations:</strong> {conclusion.get('violation_count', 0)}</p>\n"
        html += f"<p><strong>Justification:</strong> {html_escape(conclusion.get('justification', 'No justification provided'))}</p>\n"
        
        # Add timestamp
        if 'timestamp' in conclusion:
            html += f"<p><small>Verification Completed: {html_escape(conclusion.get('timestamp', ''))}</small></p>\n"
            
        html += "</div>\n"  # Close conclusion
        
        html += "</div>\n"  # Close compliance-proof
        return html
        
    def format_as_markdown(self, proof_trace):
        """Format proof trace as Markdown"""
        md = "# Compliance Verification Proof\n\n"
        
        # Add input summary
        md += "## Input\n\n"
        md += f"`{proof_trace.get('input', '')[:100]}`"
        if len(proof_trace.get('input', '')) > 100:
            md += "...(truncated)"
        md += "\n\n"
        
        # Add frameworks and mode
        frameworks = proof_trace.get('frameworks', [])
        if frameworks:
            md += f"**Frameworks:** {', '.join(frameworks)}\n\n"
        md += f"**Compliance Mode:** {proof_trace.get('mode', 'standard')}\n\n"
        
        # Add verification steps
        steps = proof_trace.get('steps', [])
        if steps:
            md += "## Verification Steps\n\n"
            
            current_framework = None
            for step in steps:
                step_type = step.get('step_type', '')
                
                if step_type == 'framework_start':
                    framework_id = step.get('framework_id', 'unknown')
                    current_framework = framework_id
                    md += f"### Framework: {framework_id}\n\n"
                    md += f"{step.get('description', '')}\n\n"
                    
                elif step_type == 'rule_verification':
                    rule_id = step.get('rule_id', 'unknown')
                    conclusion = step.get('intermediate_conclusion', '')
                    is_compliant = 'violates' not in conclusion.lower()
                    
                    md += f"#### Rule: {rule_id}\n\n"
                    md += f"* **Description:** {step.get('rule_text', '')}\n"
                    md += f"* **Result:** {conclusion}\n"
                    if 'justification' in step:
                        md += f"* **Justification:** {step.get('justification', '')}\n"
                    md += "\n"
                    
                elif step_type == 'framework_conclusion':
                    md += "#### Framework Conclusion\n\n"
                    md += f"* **Result:** {step.get('intermediate_conclusion', '')}\n"
                    md += f"* **Compliance Score:** {step.get('compliance_score', 0.0):.2f}\n"
                    md += f"* **Violations:** {step.get('violation_count', 0)}\n\n"
                    
                elif step_type == 'aggregation':
                    md += "### Overall Result\n\n"
                    md += f"{step.get('intermediate_conclusion', '')}\n\n"
                    md += f"**Compliance Score:** {step.get('compliance_score', 0.0):.2f}\n\n"
        
        # Add final conclusion
        conclusion = proof_trace.get('conclusion', {})
        md += "## Final Conclusion\n\n"
        md += f"**Compliance Status:** {'Compliant' if conclusion.get('is_compliant', False) else 'Non-compliant'}\n\n"
        md += f"**Overall Score:** {conclusion.get('compliance_score', 0.0):.2f}\n\n"
        md += f"**Violations:** {conclusion.get('violation_count', 0)}\n\n"
        md += f"**Justification:** {conclusion.get('justification', 'No justification provided')}\n\n"
        
        # Add timestamp
        if 'timestamp' in conclusion:
            md += f"*Verification Completed: {conclusion.get('timestamp', '')}*\n"
            
        return md

# Helper function for HTML escaping
def html_escape(text):
    """Escape HTML special characters"""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")