#!/usr/bin/env python3
"""
Fix linting issues in benchmark_datasets.py
"""
import re

def fix_benchmark_datasets():
    file_path = "lab_v10/src/common/benchmark_datasets.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix typing imports
    content = re.sub(r'Tuple\[', 'tuple[', content)
    content = re.sub(r'Dict\[', 'dict[', content)
    content = re.sub(r'List\[', 'list[', content)
    
    # Fix whitespace in docstrings
    content = re.sub(r'\n        \n', '\n\n', content)
    
    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix unused variable in loop
    content = re.sub(r'for t in range\(steps\):', 'for _t in range(steps):', content)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed linting issues in {file_path}")

if __name__ == "__main__":
    fix_benchmark_datasets()