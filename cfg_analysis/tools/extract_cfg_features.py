#!/usr/bin/env python3
"""
CFG Feature Extraction from C Source Code
Extracts 20+ handcrafted features for workload characterization
"""

import os
import json
import subprocess
import re
from pathlib import Path
from collections import defaultdict
import networkx as nx
from pycparser import c_parser, c_ast, parse_file

class CFGFeatureExtractor:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.parser = c_parser.CParser()
        
    def preprocess_c_file(self, filepath):
        """Preprocess C file using gcc to handle includes and macros"""
        try:
            result = subprocess.run(
                ['gcc', '-E', '-P', str(filepath)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
            else:
                # Fallback: try with pycparser's fake headers
                return parse_file(str(filepath), use_cpp=True)
        except Exception as e:
            print(f"Warning: Preprocessing failed for {filepath}: {e}")
            with open(filepath, 'r') as f:
                return f.read()
    
    def extract_cfg_from_ast(self, ast_node, func_name=None):
        """Extract CFG information from parsed AST"""
        cfg_info = {
            'num_loops': 0,
            'max_loop_depth': 0,
            'num_branches': 0,
            'num_function_calls': 0,
            'num_array_accesses': 0,
            'num_pointer_ops': 0,
            'num_arithmetic_ops': 0,
            'num_memory_ops': 0,
            'has_recursion': False,
            'has_parallel_pragma': False,
        }
        
        class CFGVisitor(c_ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
                self.max_loop_depth = 0
                self.num_loops = 0
                self.num_branches = 0
                self.num_function_calls = 0
                self.num_array_accesses = 0
                self.num_pointer_ops = 0
                self.num_arithmetic_ops = 0
                self.num_memory_ops = 0
                self.function_names = set()
                self.current_function = None
                self.has_parallel_pragma = False
                
            def visit_For(self, node):
                self.loop_depth += 1
                self.num_loops += 1
                self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_While(self, node):
                self.loop_depth += 1
                self.num_loops += 1
                self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_DoWhile(self, node):
                self.loop_depth += 1
                self.num_loops += 1
                self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_If(self, node):
                self.num_branches += 1
                self.generic_visit(node)
                
            def visit_Switch(self, node):
                self.num_branches += 1
                self.generic_visit(node)
                
            def visit_FuncCall(self, node):
                self.num_function_calls += 1
                if hasattr(node.name, 'name'):
                    if node.name.name == self.current_function:
                        cfg_info['has_recursion'] = True
                self.generic_visit(node)
                
            def visit_ArrayRef(self, node):
                self.num_array_accesses += 1
                self.num_memory_ops += 1
                self.generic_visit(node)
                
            def visit_UnaryOp(self, node):
                if node.op in ['*', '&', '++', '--']:
                    self.num_pointer_ops += 1
                self.generic_visit(node)
                
            def visit_BinaryOp(self, node):
                if node.op in ['+', '-', '*', '/', '%', '<<', '>>']:
                    self.num_arithmetic_ops += 1
                self.generic_visit(node)
                
            def visit_Assignment(self, node):
                self.num_memory_ops += 1
                self.generic_visit(node)
                
            def visit_FuncDef(self, node):
                self.current_function = node.decl.name
                self.function_names.add(node.decl.name)
                self.generic_visit(node)
                
            def visit_Pragma(self, node):
                if 'omp' in node.string.lower() or 'parallel' in node.string.lower():
                    self.has_parallel_pragma = True
                self.generic_visit(node)
        
        visitor = CFGVisitor()
        visitor.visit(ast_node)
        
        cfg_info['num_loops'] = visitor.num_loops
        cfg_info['max_loop_depth'] = visitor.max_loop_depth
        cfg_info['num_branches'] = visitor.num_branches
        cfg_info['num_function_calls'] = visitor.num_function_calls
        cfg_info['num_array_accesses'] = visitor.num_array_accesses
        cfg_info['num_pointer_ops'] = visitor.num_pointer_ops
        cfg_info['num_arithmetic_ops'] = visitor.num_arithmetic_ops
        cfg_info['num_memory_ops'] = visitor.num_memory_ops
        cfg_info['has_parallel_pragma'] = visitor.has_parallel_pragma
        
        return cfg_info
    
    def compute_derived_features(self, cfg_info):
        """Compute derived metrics from basic CFG info"""
        features = cfg_info.copy()
        
        # Cyclomatic complexity approximation: edges - nodes + 2*components
        # Simplified as: branches + loops + 1
        features['cyclomatic_complexity'] = (
            cfg_info['num_branches'] + 
            cfg_info['num_loops'] + 1
        )
        
        # Arithmetic intensity (ops per memory access)
        mem_ops = max(cfg_info['num_memory_ops'], 1)
        features['arithmetic_intensity'] = cfg_info['num_arithmetic_ops'] / mem_ops
        
        # Branch density
        total_ops = cfg_info['num_arithmetic_ops'] + cfg_info['num_memory_ops'] + 1
        features['branch_density'] = cfg_info['num_branches'] / total_ops
        
        # Loop intensity
        features['loop_intensity'] = cfg_info['num_loops'] / max(cfg_info['num_branches'] + 1, 1)
        
        # Memory intensity
        features['memory_intensity'] = cfg_info['num_memory_ops'] / total_ops
        
        # Array access ratio
        features['array_access_ratio'] = (
            cfg_info['num_array_accesses'] / max(cfg_info['num_memory_ops'], 1)
        )
        
        # Pointer complexity
        features['pointer_complexity'] = cfg_info['num_pointer_ops'] / total_ops
        
        # Function call overhead
        features['call_overhead'] = cfg_info['num_function_calls'] / total_ops
        
        # Convert booleans to int
        features['has_recursion'] = int(features['has_recursion'])
        features['has_parallel_pragma'] = int(features['has_parallel_pragma'])
        
        return features
    
    def analyze_source_file(self, filepath):
        """Main analysis function for a single C file"""
        print(f"Analyzing: {filepath.name}")
        
        try:
            # Parse the file
            ast = parse_file(str(filepath), use_cpp=True)
            
            # Extract CFG features
            cfg_info = self.extract_cfg_from_ast(ast)
            
            # Compute derived features
            features = self.compute_derived_features(cfg_info)
            
            # Add file metadata
            features['source_file'] = filepath.name
            features['source_lines'] = len(open(filepath).readlines())
            
            return features
            
        except Exception as e:
            print(f"Error analyzing {filepath.name}: {e}")
            return None
    
    def analyze_all_benchmarks(self):
        """Analyze all C files in source directory"""
        results = {}
        
        # Find all C files
        c_files = list(self.source_dir.glob("*.c"))
        
        if not c_files:
            print(f"No C files found in {self.source_dir}")
            return results
        
        print(f"Found {len(c_files)} C source files\n")
        
        for c_file in sorted(c_files):
            benchmark_name = c_file.stem  # filename without extension
            features = self.analyze_source_file(c_file)
            
            if features:
                results[benchmark_name] = features
                print(f"✓ Extracted {len(features)} features\n")
        
        return results
    
    def save_features(self, features_dict, output_file="cfg_features.json"):
        """Save extracted features to JSON"""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(features_dict, f, indent=2)
        
        print(f"\n✓ Features saved to: {output_path}")
        
        # Also save as CSV for easy inspection
        csv_path = output_path.with_suffix('.csv')
        self.save_features_csv(features_dict, csv_path)
    
    def save_features_csv(self, features_dict, output_file):
        """Save features as CSV for easy viewing"""
        import csv
        
        if not features_dict:
            return
        
        # Get all feature names from first benchmark
        feature_names = list(next(iter(features_dict.values())).keys())
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['benchmark'] + feature_names)
            
            # Data rows
            for benchmark, features in sorted(features_dict.items()):
                row = [benchmark] + [features.get(k, '') for k in feature_names]
                writer.writerow(row)
        
        print(f"✓ CSV saved to: {output_file}")
    
    def print_summary(self, features_dict):
        """Print summary statistics"""
        if not features_dict:
            return
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total benchmarks analyzed: {len(features_dict)}")
        
        # Get feature statistics
        import numpy as np
        
        numeric_features = [
            'num_loops', 'max_loop_depth', 'num_branches', 
            'cyclomatic_complexity', 'arithmetic_intensity'
        ]
        
        print("\nKey Statistics:")
        for feature in numeric_features:
            values = [f[feature] for f in features_dict.values() if feature in f]
            if values:
                print(f"  {feature:30s}: min={min(values):6.2f}, "
                      f"max={max(values):6.2f}, mean={np.mean(values):6.2f}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract CFG features from C code')
    parser.add_argument(
        '--source_dir', 
        type=str, 
        default='cfg_analysis/source_code',
        help='Directory containing C source files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='cfg_analysis/features',
        help='Directory to save extracted features'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run extraction
    extractor = CFGFeatureExtractor(args.source_dir, args.output_dir)
    features = extractor.analyze_all_benchmarks()
    
    if features:
        extractor.save_features(features)
        extractor.print_summary(features)
    else:
        print("No features extracted. Check your source files.")


if __name__ == "__main__":
    main()