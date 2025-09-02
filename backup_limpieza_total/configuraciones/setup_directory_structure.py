import os
from pathlib import Path

def create_directory_structure():
    # Base directories
    base_dirs = [
        'data/uploads',
        'data/processed',
        'data/models',
        'data/cache',
        'logs',
        'tests/unit',
        'tests/integration',
        'tests/e2e',
        'templates/emails',
        'templates/reports',
        'docs/api',
        'docs/deployment',
        'docs/guides',
        'scripts/backup',
        'scripts/deployment',
        'scripts/monitoring',
        'docker/compose',
        'docker/configs',
        'docker/images',
        'docker/volumes'
    ]
    
    # Create base directories
    for directory in base_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create empty __init__.py files to make them Python packages
    for dirpath, dirnames, _ in os.walk('.'):
        if 'venv' in dirpath or '.git' in dirpath:
            continue
        init_file = os.path.join(dirpath, '__init__.py')
        if not os.path.exists(init_file) and any(f.endswith('.py') for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))):
            with open(init_file, 'w') as f:
                pass
            print(f"Created: {init_file}")
    
    # Create .gitkeep in empty directories
    for dirpath, dirnames, filenames in os.walk('.'):
        if 'venv' in dirpath or '.git' in dirpath:
            continue
        if not dirnames and not filenames:
            gitkeep = os.path.join(dirpath, '.gitkeep')
            with open(gitkeep, 'w') as f:
                pass
            print(f"Created: {gitkeep}")
    
    print("\nDirectory structure created successfully!")

if __name__ == "__main__":
    create_directory_structure()
