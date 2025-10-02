#!/usr/bin/env python3
"""
CFG Feature Extraction from C Source Code (No Preprocessing Version)
Extracts features without needing header files
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict

class SimpleCFGExtractor:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
    def analyze_source_file(self, filepath):
        """Extract features using simple pattern matching"""
        print(f"Analyzing: {filepath.name}")
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            features = {
                'source_file': filepath.name,
                'source_lines': len(content.split('\n')),
            }
            
            # Count loops
            features['num_for_loops'] = len(re.findall(r'\bfor\s*\(', content))
            features['num_while_loops'] = len(re.findall(r'\bwhile\s*\(', content))
            features['num_do_while'] = len(re.findall(r'\bdo\s*\{', content))
            features['num_loops'] = features['num_for_loops'] + features['num_while_loops'] + features['num_do_while']
            
            # Estimate loop depth by counting nested braces in loop contexts
            features['max_loop_depth'] = self.estimate_loop_depth(content)
            
            # Count branches
            features['num_if'] = len(re.findall(r'\bif\s*\(', content))
            features['num_else'] = len(re.findall(r'\belse\b', content))
            features['num_switch'] = len(re.findall(r'\bswitch\s*\(', content))
            features['num_branches'] = features['num_if'] + features['num_switch']
            
            # Count function calls (approximation)
            features['num_function_calls'] = len(re.findall(r'\w+\s*\([^)]*\)\s*;', content))
            
            # Count array accesses
            features['num_array_accesses'] = len(re.findall(r'\w+\s*\[', content))
            
            # Count pointer operations
            features['num_pointer_deref'] = len(re.findall(r'\*\w+', content))
            features['num_address_of'] = len(re.findall(r'&\w+', content))
            features['num_pointer_ops'] = features['num_pointer_deref'] + features['num_address_of']
            
            # Count arithmetic operations
            arith_ops = [r'\+', r'-', r'\*', r'/', r'%', r'<<', r'>>']
            features['num_arithmetic_ops'] = sum(len(re.findall(op, content)) for op in arith_ops)
            
            # Count assignments (memory writes)
            features['num_assignments'] = len(re.findall(r'\w+\s*=', content))
            
            # Total memory operations
            features['num_memory_ops'] = features['num_assignments'] + features['num_array_accesses']
            
            # Check for recursion (function calling itself)
            functions = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', content)
            features['has_recursion'] = 0
            for func_name in functions:
                if func_name in content[content.find(func_name):]:
                    # Check if function name appears in its own body
                    func_start = content.find(f'{func_name}')
                    func_pattern = re.search(f'{func_name}\s*\([^)]*\)\s*\{{', content[func_start:])
                    if func_pattern:
                        func_body_start = func_start + func_pattern.end()
                        if func_name in content[func_body_start:func_body_start+10000]:
                            features['has_recursion'] = 1
                            break
            
            # Check for parallel pragmas
            features['has_parallel_pragma'] = 1 if re.search(r'#pragma\s+omp', content) else 0
            
            # Derived features
            features['cyclomatic_complexity'] = features['num_branches'] + features['num_loops'] + 1
            
            mem_ops = max(features['num_memory_ops'], 1)
            features['arithmetic_intensity'] = features['num_arithmetic_ops'] / mem_ops
            
            total_ops = features['num_arithmetic_ops'] + features['num_memory_ops'] + 1
            features['branch_density'] = features['num_branches'] / total_ops
            features['loop_intensity'] = features['num_loops'] / max(features['num_branches'] + 1, 1)
            features['memory_intensity'] = features['num_memory_ops'] / total_ops
            features['array_access_ratio'] = features['num_array_accesses'] / mem_ops
            features['pointer_complexity'] = features['num_pointer_ops'] / total_ops
            features['call_overhead'] = features['num_function_calls'] / total_ops
            
            print(f"  ✓ Extracted {len(features)} features\n")
            return features
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            return None
    
    def estimate_loop_depth(self, content):
        """Estimate maximum loop nesting depth"""
        max_depth = 0
        current_depth = 0
        
        # Simple state machine
        in_loop = False
        for line in content.split('\n'):
            # Check for loop keywords
            if re.search(r'\b(for|while)\s*\(', line):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            # Count closing braces to estimate when loops end
            if '}' in line and current_depth > 0:
                current_depth = max(0, current_depth - line.count('}'))
        
        return max(max_depth, 1)
    
    def analyze_all_benchmarks(self):
        """Analyze all C files in source directory"""
        results = {}
        
        c_files = list(self.source_dir.glob("*.c"))
        
        if not c_files:
            print(f"No C files found in {self.source_dir}")
            return results
        
        print(f"Found {len(c_files)} C source files\n")
        
        for c_file in sorted(c_files):
            benchmark_name = c_file.stem
            features = self.analyze_source_file(c_file)
            
            if features:
                results[benchmark_name] = features
        
        return results
    
    def save_features(self, features_dict, output_file="cfg_features.json"):
        """Save extracted features to JSON"""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(features_dict, f, indent=2)
        
        print(f"\n✓ Features saved to: {output_path}")
        
        # Also save as CSV
        csv_path = output_path.with_suffix('.csv')
        self.save_features_csv(features_dict, csv_path)
    
    def save_features_csv(self, features_dict, output_file):
        """Save features as CSV"""
        import csv
        
        if not features_dict:
            return
        
        feature_names = list(next(iter(features_dict.values())).keys())
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['benchmark'] + feature_names)
            
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
        
        import numpy as np
        
        key_features = ['num_loops', 'max_loop_depth', 'num_branches', 
                       'cyclomatic_complexity', 'arithmetic_intensity']
        
        print("\nKey Statistics:")
        for feature in key_features:
            values = [f[feature] for f in features_dict.values() if feature in f]
            if values:
                print(f"  {feature:30s}: min={min(values):6.2f}, "
                      f"max={max(values):6.2f}, mean={np.mean(values):6.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract CFG features from C code')
    parser.add_argument('--source_dir', type=str, default='../source_code')
    parser.add_argument('--output_dir', type=str, default='../features')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    extractor = SimpleCFGExtractor(args.source_dir, args.output_dir)
    features = extractor.analyze_all_benchmarks()
    
    if features:
        extractor.save_features(features)
        extractor.print_summary(features)
    else:
        print("No features extracted. Check your source files.")


if __name__ == "__main__":
    main()
