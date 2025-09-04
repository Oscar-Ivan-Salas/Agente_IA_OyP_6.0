import os
import re
from pathlib import Path

def update_file_imports(file_path):
    """Update relative imports to absolute imports in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already using absolute imports
    if 'from gateway.' in content:
        return False
    
    # Update relative imports to absolute
    updated_content = re.sub(
        r'from \.(\w+) import',
        r'from gateway.\1 import',
        content
    )
    
    # Update relative imports with multiple dots
    updated_content = re.sub(
        r'from \.\.(\w+) import',
        r'from gateway.\1 import',
        updated_content
    )
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True
    return False

def update_imports_in_directory(directory):
    """Update imports in all Python files in a directory."""
    updated_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_file_imports(file_path):
                    updated_files.append(file_path)
    return updated_files

if __name__ == "__main__":
    gateway_dir = os.path.join(os.path.dirname(__file__), 'gateway')
    if os.path.exists(gateway_dir):
        updated = update_imports_in_directory(gateway_dir)
        if updated:
            print("Updated imports in the following files:")
            for file in updated:
                print(f"- {file}")
        else:
            print("No files needed import updates.")
    else:
        print(f"Gateway directory not found at: {gateway_dir}")
