import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any

@dataclass
class BenchmarkConfig:
    """Configuration for compliance benchmarking"""
    batch_sizes: List[int] = None
    regulatory_frameworks: List[str] = None
    text_lengths: List[int] = None
    concurrency_levels: List[int] = None
    repetitions: int = 5
    warmup_iterations: int = 2
    
    def __post_init__(self):
        """Set default values if not provided"""
        self.batch_sizes = self.batch_sizes or [1, 4, 16, 64]
        self.regulatory_frameworks = self.regulatory_frameworks or ["GDPR", "HIPAA", "CCPA"]
        self.text_lengths = self.text_lengths or [100, 500, 2000, 5000]
        self.concurrency_levels = self.concurrency_levels or [1, 2, 4, 8, 16]


class ComplianceBenchmarker:
    """
    Framework for benchmarking compliance system performance across
    various dimensions and load conditions.
    """
    
    def __init__(self, compliance_system, config: BenchmarkConfig = None):
        """
        Initialize the benchmarker with a compliance system and configuration.
        
        Args:
            compliance_system: The compliance system to benchmark (e.g., CompliantLanguageModelProcessor)
            config: Configuration for benchmarking parameters
        """
        self.system = compliance_system
        self.config = config or BenchmarkConfig()
        self.results = []
        
    def run_latency_benchmark(self) -> pd.DataFrame:
        """
        Benchmark system latency across different text lengths and regulatory frameworks.
        
        Returns:
            DataFrame with benchmark results
        """
        print("Running latency benchmark...")
        results = []
        
        # Warm up to avoid cold-start penalties
        for _ in range(self.config.warmup_iterations):
            self._run_single_latency_test("GDPR", 100)
            
        # Run actual benchmarks
        for framework in self.config.regulatory_frameworks:
            for length in self.config.text_lengths:
                framework_results = []
                for _ in range(self.config.repetitions):
                    latency = self._run_single_latency_test(framework, length)
                    framework_results.append(latency)
                
                # Calculate statistics
                mean_latency = np.mean(framework_results)
                p50 = np.percentile(framework_results, 50)
                p95 = np.percentile(framework_results, 95)
                p99 = np.percentile(framework_results, 99)
                
                results.append({
                    "framework": framework,
                    "text_length": length,
                    "mean_latency_ms": mean_latency,
                    "p50_latency_ms": p50,
                    "p95_latency_ms": p95,
                    "p99_latency_ms": p99
                })
                
        df = pd.DataFrame(results)
        self.results.append(("latency", df))
        return df
    
    def run_throughput_benchmark(self) -> pd.DataFrame:
        """
        Benchmark system throughput with different batch sizes and concurrency levels.
        
        Returns:
            DataFrame with benchmark results
        """
        print("Running throughput benchmark...")
        results = []
        
        # Run throughput tests
        for batch_size in self.config.batch_sizes:
            for concurrency in self.config.concurrency_levels:
                # Skip unrealistic combinations (very large batch with high concurrency)
                if batch_size * concurrency > 512:
                    continue
                    
                throughput, avg_latency = self._run_throughput_test(batch_size, concurrency)
                results.append({
                    "batch_size": batch_size,
                    "concurrency": concurrency,
                    "throughput_requests_per_sec": throughput,
                    "avg_latency_ms": avg_latency
                })
                
        df = pd.DataFrame(results)
        self.results.append(("throughput", df))
        return df
    
    def run_memory_benchmark(self) -> pd.DataFrame:
        """
        Benchmark memory usage with different configurations.
        
        Returns:
            DataFrame with benchmark results
        """
        print("Running memory benchmark...")
        results = []
        
        # Import here to avoid unnecessary dependency if not using this method
        import psutil
        process = psutil.Process()
        
        # Run memory tests
        for framework in self.config.regulatory_frameworks:
            for length in self.config.text_lengths:
                # Clear any cached data before test
                if hasattr(self.system, 'clear_caches'):
                    self.system.clear_caches()
                
                # Baseline memory
                baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Generate text and measure memory
                _ = self._run_single_latency_test(framework, length)
                
                # Measure memory usage after processing
                memory_usage = process.memory_info().rss / (1024 * 1024) - baseline_memory
                
                results.append({
                    "framework": framework,
                    "text_length": length,
                    "memory_usage_mb": memory_usage
                })
                
        df = pd.DataFrame(results)
        self.results.append(("memory", df))
        return df
                
    def generate_report(self, output_file: str = "compliance_benchmark_report.html") -> str:
        """
        Generate a comprehensive HTML report of all benchmark results.
        
        Args:
            output_file: Path to output HTML file
            
        Returns:
            Path to the generated report
        """
        matplotlib.use('Agg')
        
        # Create report
        html = ["<html><head><title>Compliance System Benchmark Report</title>"]
        html.append("<style>body{font-family:Arial;margin:20px;line-height:1.6}")
        html.append("table{border-collapse:collapse;width:100%;margin-bottom:20px}")
        html.append("th,td{border:1px solid #ddd;padding:8px;text-align:left}")
        html.append("th{background-color:#f2f2f2}")
        html.append("tr:nth-child(even){background-color:#f9f9f9}")
        html.append("h1,h2,h3{color:#333}</style></head><body>")
        html.append("<h1>Compliance System Benchmark Report</h1>")
        
        # System information
        import platform
        import datetime
        html.append("<h2>System Information</h2>")
        html.append("<table>")
        html.append(f"<tr><th>Date</th><td>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>")
        html.append(f"<tr><th>Platform</th><td>{platform.platform()}</td></tr>")
        html.append(f"<tr><th>Python</th><td>{platform.python_version()}</td></tr>")
        html.append(f"<tr><th>Processor</th><td>{platform.processor()}</td></tr>")
        html.append("</table>")
        
        # Configuration information
        html.append("<h2>Benchmark Configuration</h2>")
        html.append("<table>")
        for key, value in self.config.__dict__.items():
            html.append(f"<tr><th>{key}</th><td>{value}</td></tr>")
        html.append("</table>")
        
        # Results
        for result_type, df in self.results:
            html.append(f"<h2>{result_type.capitalize()} Benchmark Results</h2>")
            
            # Add table view of data
            html.append(df.to_html())
            
            # Add visualizations
            if result_type == "latency":
                fig, ax = plt.subplots(figsize=(10, 6))
                for framework in df['framework'].unique():
                    subset = df[df['framework'] == framework]
                    ax.plot(subset['text_length'], subset['mean_latency_ms'], 
                            marker='o', label=f"{framework}")
                ax.set_xlabel('Text Length (tokens)')
                ax.set_ylabel('Latency (ms)')
                ax.set_title('Latency by Text Length and Framework')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                img_path = f"latency_by_framework.png"
                plt.savefig(img_path)
                html.append(f"<img src='{img_path}' style='max-width:100%'>")
                plt.close()
                
            elif result_type == "throughput":
                # Create heatmap of throughput by batch size and concurrency
                pivot_df = df.pivot(index="batch_size", columns="concurrency", 
                                   values="throughput_requests_per_sec")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(pivot_df.values, cmap='viridis')
                
                # Set labels
                ax.set_xticks(np.arange(len(pivot_df.columns)))
                ax.set_yticks(np.arange(len(pivot_df.index)))
                ax.set_xticklabels(pivot_df.columns)
                ax.set_yticklabels(pivot_df.index)
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Throughput (req/s)", rotation=-90, va="bottom")
                
                # Add values to cells
                for i in range(len(pivot_df.index)):
                    for j in range(len(pivot_df.columns)):
                        try:
                            text = ax.text(j, i, round(pivot_df.iloc[i, j], 1),
                                          ha="center", va="center", color="w")
                        except:
                            pass
                
                ax.set_title("Throughput (requests/sec) by Batch Size and Concurrency")
                ax.set_xlabel("Concurrency Level")
                ax.set_ylabel("Batch Size")
                
                img_path = f"throughput_heatmap.png"
                plt.savefig(img_path)
                html.append(f"<img src='{img_path}' style='max-width:100%'>")
                plt.close()
                
            elif result_type == "memory":
                fig, ax = plt.subplots(figsize=(10, 6))
                for framework in df['framework'].unique():
                    subset = df[df['framework'] == framework]
                    ax.plot(subset['text_length'], subset['memory_usage_mb'], 
                            marker='o', label=f"{framework}")
                ax.set_xlabel('Text Length (tokens)')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title('Memory Usage by Text Length and Framework')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                img_path = f"memory_by_framework.png"
                plt.savefig(img_path)
                html.append(f"<img src='{img_path}' style='max-width:100%'>")
                plt.close()
        
        # Close HTML
        html.append("</body></html>")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write("\n".join(html))
            
        print(f"Benchmark report saved to {output_file}")
        return output_file
    
    def _run_single_latency_test(self, framework: str, text_length: int) -> float:
        """Run a single latency test for a given framework and text length"""
        # Generate test text of specified length
        test_text = self._generate_test_text(text_length)
        
        # Create context with specified framework
        context = {"regulatory_frameworks": [framework]}
        
        # Measure processing time
        start_time = time.time()
        _ = self.system.generate_compliant_text(
            test_text, 
            context=context,
            compliance_mode="strict"
        )
        end_time = time.time()
        
        # Return latency in milliseconds
        return (end_time - start_time) * 1000
    
    def _run_throughput_test(self, batch_size: int, concurrency: int) -> Tuple[float, float]:
        """
        Run a throughput test with specified batch size and concurrency.
        
        Returns:
            Tuple of (throughput in req/sec, average latency in ms)
        """
        # Generate test texts
        test_texts = [self._generate_test_text(500) for _ in range(batch_size)]
        
        # Create contexts with randomly selected frameworks
        contexts = []
        for _ in range(batch_size):
            framework = np.random.choice(self.config.regulatory_frameworks)
            contexts.append({"regulatory_frameworks": [framework]})
        
        # Function to process one request
        def process_request(idx):
            start_time = time.time()
            _ = self.system.generate_compliant_text(
                test_texts[idx % batch_size],
                context=contexts[idx % batch_size],
                compliance_mode="strict"
            )
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to ms
        
        # Run with specified concurrency
        latencies = []
        total_requests = batch_size * 5  # Process 5 batches
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for latency in executor.map(process_request, range(total_requests)):
                latencies.append(latency)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = total_requests / total_time if total_time > 0 else 0
        avg_latency = np.mean(latencies)
        
        return throughput, avg_latency
        
    def _generate_test_text(self, length: int) -> str:
        """Generate a test text of approximately the specified length"""
        # This is a simplified text generator - in a real implementation,
        # you would want more realistic text generation
        import string
        import random
        
        # Common English words for more realistic text
        words = [
            "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "with", 
            "as", "on", "by", "this", "be", "at", "which", "have", "from", "an", 
            "they", "we", "their", "has", "would", "what", "will", "there", "if",
            "can", "all", "some", "when", "very", "just", "your", "any", "been"
        ]
        
        # Add some domain-specific words
        domain_words = [
            "data", "privacy", "regulation", "compliance", "policy", "user",
            "information", "personal", "processing", "consent", "rights",
            "protection", "security", "access", "authorization", "patient",
            "health", "medical", "treatment", "diagnosis", "financial"
        ]
        
        words.extend(domain_words)
        
        # Generate text
        text_words = []
        current_length = 0
        
        # Start with a domain-specific sentence
        starters = [
            "The following document outlines our compliance approach.",
            "Users must consent before data processing occurs.",
            "All personal information must be protected.",
            "Patient health records require special handling.",
            "Financial data must be securely stored."
        ]
        
        text_words.append(random.choice(starters))
        current_length += len(text_words[0])
        
        # Add words until we reach desired length
        while current_length < length:
            word = random.choice(words)
            text_words.append(word)
            current_length += len(word) + 1  # +1 for space
            
            # Occasionally add punctuation
            if random.random() < 0.1:
                text_words[-1] += random.choice([",", ".", ";", ":"])
                
            # Occasionally start a new sentence
            if random.random() < 0.05:
                text_words.append(".")
                
                # Capitalize next word
                if len(text_words) < length:
                    next_word = random.choice(words).capitalize()
                    text_words.append(next_word)
                    current_length += len(next_word) + 1
        
        return " ".join(text_words)


# Example usage:
# compliance_system = CompliantLanguageModelProcessor(...)
# config = BenchmarkConfig(
#     batch_sizes=[1, 4, 16],
#     regulatory_frameworks=["GDPR", "HIPAA"],
#     text_lengths=[100, 500, 1000],
#     concurrency_levels=[1, 2, 4]
# )
# benchmarker = ComplianceBenchmarker(compliance_system, config)
# latency_results = benchmarker.run_latency_benchmark()
# throughput_results = benchmarker.run_throughput_benchmark()
# memory_results = benchmarker.run_memory_benchmark()
# report_path = benchmarker.generate_report()
