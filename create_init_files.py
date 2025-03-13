import os

def create_init_files(root_dir):
    """Create __init__.py files in all directories and subdirectories."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip .git directory
        if '.git' in dirpath:
            continue
            
        # Create __init__.py if it doesn't exist
        init_file = os.path.join(dirpath, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                # Add a comment to identify the module
                module_path = os.path.relpath(dirpath, root_dir).replace(os.sep, '.')
                if module_path == '.':
                    f.write(f"# AiMee - Main Package\n")
                else:
                    f.write(f"# AiMee - {module_path} Package\n")
            print(f"Created: {init_file}")

if __name__ == "__main__":
    create_init_files(os.getcwd())
    print("Created __init__.py files in all directories.") 