# import_finder.py

import os
import ast

def find_imports(directory):
    imports = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    imports.add(n.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom):
                                imports.add(node.module.split('.')[0])
                    except:
                        print(f"Could not parse {file}")
    return imports

if __name__ == "__main__":
    project_dir = "."  # Current directory, change if needed
    all_imports = find_imports(project_dir)
    print("All imports found in the project:")
    for imp in sorted(all_imports):
        print(imp)