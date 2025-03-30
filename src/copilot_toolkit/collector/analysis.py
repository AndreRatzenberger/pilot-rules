# src/pilot_rules/collector/analysis.py
import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union

# Import utility function - use relative import within the package
from .utils import get_file_metadata

# --- Python Component Extraction ---
def extract_python_components(file_path: str) -> Dict[str, Any]:
    """Extract classes, functions, and imports from Python files."""
    components = {
        "classes": [],
        "functions": [],
        "imports": [],
        "docstring": None
    }

    # Ensure it's a python file before trying to parse
    if not file_path.lower().endswith('.py'):
        return components # Return empty structure for non-python files

    try:
        # Read with error handling for encoding issues
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        tree = ast.parse(content)

        # Extract module docstring
        components["docstring"] = ast.get_docstring(tree) # Returns None if no docstring

        # Extract top-level classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "methods": [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                }
                components["classes"].append(class_info)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                 # We consider all functions directly under the module body as "top-level" here
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    # Simplified arg extraction (just names)
                    "args": [arg.arg for arg in node.args.args]
                }
                components["functions"].append(func_info)

        # Extract all imports (simplified representation)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Store 'import x' or 'import x as y'
                    components["imports"].append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom):
                module_part = node.module or ""
                level_dots = "." * node.level
                # Store 'from .mod import x' or 'from mod import x as y'
                imported_names = []
                for alias in node.names:
                     name_part = alias.name
                     if alias.asname:
                         name_part += f" as {alias.asname}"
                     imported_names.append(name_part)

                components["imports"].append(f"from {level_dots}{module_part} import {', '.join(imported_names)}")

    except SyntaxError as e:
         print(f"Warning: Could not parse Python components in {file_path} due to SyntaxError: {e}")
    except Exception as e:
        print(f"Warning: Could not parse Python components in {file_path}: {e}")

    return components


# --- Dependency Analysis ---

def get_module_prefixes(module_name: str) -> List[str]:
    """
    Generate all possible module prefixes for a given module name.
    For example, 'a.b.c' would return ['a.b.c', 'a.b', 'a']
    """
    parts = module_name.split('.')
    return ['.'.join(parts[:i]) for i in range(len(parts), 0, -1)]

def analyze_code_dependencies(files: List[str]) -> Dict[str, Set[str]]:
    """Analyze dependencies between Python files based on imports."""
    # Filter to only analyze python files within the provided list
    python_files = {f for f in files if f.lower().endswith('.py')}
    if not python_files:
        return {} # No Python files to analyze

    dependencies: Dict[str, Set[str]] = {file: set() for file in python_files}
    module_map: Dict[str, str] = {} # Map potential module names to absolute file paths
    project_root = Path.cwd().resolve() # Assume CWD is project root for relative imports

    # --- Build Module Map (heuristic) ---
    # Map files within the project to their potential Python module paths
    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            # Attempt to create a module path relative to the project root
            relative_path = file_path.relative_to(project_root)
            parts = list(relative_path.parts)
            module_name = None
            if parts[-1] == '__init__.py':
                module_parts = parts[:-1]
                if module_parts: # Avoid mapping root __init__.py as empty string
                     module_name = '.'.join(module_parts)
            elif parts[-1].endswith('.py'):
                module_parts = parts[:-1] + [parts[-1][:-3]] # Remove .py
                module_name = '.'.join(module_parts)

            if module_name:
                 # print(f"Mapping module '{module_name}' to '{file_path_str}'") # Debug
                 module_map[module_name] = file_path_str

        except ValueError:
            # File is outside the assumed project root, less reliable mapping
            # Map only by filename stem if not already mapped? Risky.
            # print(f"Debug: File {file_path_str} is outside project root {project_root}")
            pass

    # --- Analyze Imports in Each File ---
    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            tree = ast.parse(code)

            for node in ast.walk(tree):
                imported_module_str = None
                target_file: Optional[str] = None

                # Handle 'import x' or 'import x.y'
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module_str = alias.name
                        # Check full name and prefixes against our map
                        for prefix in get_module_prefixes(imported_module_str):
                             if prefix in module_map:
                                 target_file = module_map[prefix]
                                 # Ensure the target is actually one of the collected python files
                                 if target_file in python_files and target_file != file_path_str:
                                      dependencies[file_path_str].add(target_file)
                                 break # Found the longest matching prefix

                # Handle 'from x import y' or 'from .x import y'
                elif isinstance(node, ast.ImportFrom):
                    level = node.level
                    module_base = node.module or ""

                    if level == 0: # Absolute import: 'from package import module'
                         imported_module_str = module_base
                         for prefix in get_module_prefixes(imported_module_str):
                             if prefix in module_map:
                                 target_file = module_map[prefix]
                                 if target_file in python_files and target_file != file_path_str:
                                     dependencies[file_path_str].add(target_file)
                                 break
                    else: # Relative import: 'from . import x', 'from ..y import z'
                        current_dir = file_path.parent
                        base_path = current_dir
                        # Navigate up for '..' (level 2 means one level up, etc.)
                        for _ in range(level - 1):
                            base_path = base_path.parent

                        # Try to resolve the relative path
                        relative_module_parts = module_base.split('.') if module_base else []
                        target_path_base = base_path
                        for part in relative_module_parts:
                            target_path_base = target_path_base / part

                        # Check if the resolved path corresponds to a known file/module
                        # Check 1: Is it a directory with __init__.py?
                        init_py_path = (target_path_base / '__init__.py').resolve()
                        init_py_str = str(init_py_path)
                        if init_py_str in python_files and init_py_str != file_path_str:
                            dependencies[file_path_str].add(init_py_str)
                            target_file = init_py_str # Mark as found

                        # Check 2: Is it a .py file directly?
                        module_py_path = target_path_base.with_suffix('.py').resolve()
                        module_py_str = str(module_py_path)
                        if not target_file and module_py_str in python_files and module_py_str != file_path_str:
                             dependencies[file_path_str].add(module_py_str)
                             target_file = module_py_str

                        # Note: This relative import resolution is basic and might miss complex cases.
                        # We are primarily checking if the base module path (e.g., `.`, `..utils`) exists.

        except SyntaxError as e:
            print(f"Warning: Skipping import analysis in {file_path_str} due to SyntaxError: {e}")
        except Exception as e:
            print(f"Warning: Could not analyze imports in {file_path_str}: {e}")

    # Final cleanup: remove self-references (though logic above tries to avoid them)
    for file in list(dependencies.keys()):
        dependencies[file].discard(file)

    return dependencies


# --- Pattern Detection ---
def get_common_patterns(files: List[str]) -> Dict[str, Any]:
    """Identify common design patterns in the codebase (Python focused heuristic)."""
    patterns: Dict[str, Union[List[str], Dict[str, List[str]]]] = {
        "singleton": [],
        "factory": [],
        "observer": [],
        "decorator": [],
        "mvc_components": {
            "models": [],
            "views": [],
            "controllers": []
        }
    }
    python_files = [f for f in files if f.lower().endswith('.py')]

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower() # Read content once
            file_basename_lower = os.path.basename(file_path).lower()
            file_path_parts_lower = {p.lower() for p in Path(file_path).parts}

            # Basic keyword/structure checks (can be improved significantly)
            # Check for singleton pattern (simple heuristic)
            if ("instance = none" in content or "_instance = none" in content) and ("__new__" in content or " getinstance " in content):
                 if "singleton" not in patterns: patterns["singleton"] = []
                 patterns["singleton"].append(file_path)

            # Check for factory pattern
            if "factory" in file_basename_lower or ("def create_" in content and " return " in content) or ("def make_" in content and " return " in content):
                if "factory" not in patterns: patterns["factory"] = []
                patterns["factory"].append(file_path)

            # Check for observer pattern
            if ("observer" in content or "listener" in content) and ("notify" in content or "update" in content or "addeventlistener" in content or "subscribe" in content):
                 if "observer" not in patterns: patterns["observer"] = []
                 patterns["observer"].append(file_path)

            # Check for decorator pattern definition (crude)
            if "def wrapper(" in content and "return wrapper" in content and "@" in content: # Check for @ usage too
                 if "decorator" not in patterns: patterns["decorator"] = []
                 patterns["decorator"].append(file_path) # Might be too broad

            # Check for MVC components based on naming conventions (filename or path)
            # Initialize mvc_components if needed
            if "mvc_components" not in patterns:
                 patterns["mvc_components"] = {"models": [], "views": [], "controllers": []}

            if "model" in file_basename_lower or "models" in file_path_parts_lower:
                patterns["mvc_components"]["models"].append(file_path)
            if "view" in file_basename_lower or "views" in file_path_parts_lower or "template" in file_basename_lower:
                patterns["mvc_components"]["views"].append(file_path)
            if "controller" in file_basename_lower or "controllers" in file_path_parts_lower or "handler" in file_basename_lower or "route" in file_basename_lower:
                patterns["mvc_components"]["controllers"].append(file_path)

        except Exception as e:
            # print(f"Warning: Could not analyze patterns in {file_path}: {e}") # Can be noisy
            continue # Ignore files that can't be read or processed

    # --- Clean up empty categories ---
    cleaned_patterns: Dict[str, Any] = {}
    for key, value in patterns.items():
        if isinstance(value, list):
            if value: # Keep if list is not empty
                 cleaned_patterns[key] = value
        elif isinstance(value, dict):
            # For nested dicts like MVC, keep the parent key if any child list is non-empty
            non_empty_sub_patterns = {
                subkey: sublist for subkey, sublist in value.items() if isinstance(sublist, list) and sublist
            }
            if non_empty_sub_patterns: # Keep if dict has non-empty lists
                 cleaned_patterns[key] = non_empty_sub_patterns

    return cleaned_patterns


# --- Key File Identification ---
def find_key_files(files: List[str], dependencies: Dict[str, Set[str]]) -> List[str]:
    """Identify key files based on dependencies and naming conventions."""
    print("Scoring files to identify key ones...")
    scores = {file: 0.0 for file in files} # Use float for potentially fractional scores
    report_base_path = Path.cwd() # Use CWD for relative path context in scoring

    # --- Score by incoming dependencies (Python only) ---
    python_files = {f for f in files if f.lower().endswith('.py')}
    dependent_count = {file: 0 for file in python_files}
    for file, deps in dependencies.items(): # dependencies should already be Python-only
        if file not in python_files: continue
        for dep in deps:
            if dep in dependent_count: # Target file must also be Python
                dependent_count[dep] += 1

    for file, count in dependent_count.items():
        # Higher score for files depended upon by many others
        scores[file] += count * 2.0
        # print(f"  Score bump (deps): {Path(file).name} +{count * 2.0} (depended by {count})")


    # --- Score by naming conventions and location ---
    for file in files:
        p = Path(file)
        try:
             rel_path = p.relative_to(report_base_path)
             rel_parts = {part.lower() for part in rel_path.parts}
        except ValueError:
             rel_parts = set() # File outside project root

        base_name = p.name.lower()
        parent_dir_name = p.parent.name.lower()

        # Boost core/entry point files
        if base_name in ["main.py", "app.py", "cli.py", "manage.py", "server.py", "__main__.py"]:
            scores[file] += 5.0
        elif base_name == "__init__.py":
             # Boost __init__ slightly, more if it's higher up
             depth = len(p.parent.parts) - len(report_base_path.parts)
             scores[file] += max(0.5, 3.0 - depth * 0.5) # Higher score for top-level __init__
        elif base_name.startswith("settings.") or base_name.startswith("config."):
             scores[file] += 4.0
        elif base_name.startswith("test_") or "test" in rel_parts:
             scores[file] -= 1.0 # Lower score for test files unless highly depended upon

        # Config/Settings patterns
        if any(config_name in base_name for config_name in ["config", "settings", "constant", "conf."]):
            scores[file] += 2.0

        # Base classes, core logic patterns
        if any(pattern in base_name for pattern in ["base.", "abstract", "interface", "factory", "core."]):
            scores[file] += 2.0
        if "core" in rel_parts or parent_dir_name == "core":
             scores[file] += 1.5

        # Utilities
        if any(util_name in base_name for util_name in ["util", "helper", "common", "tool", "shared"]):
            scores[file] += 1.0

        # Location boost (e.g., direct children of 'src')
        if "src" in rel_parts and len(rel_parts) <= 2: # e.g. src/module.py
            scores[file] += 0.5


    # --- Score by file size / complexity (crude) ---
    for file in files:
        try:
            metadata = get_file_metadata(file) # Use imported function
            line_count = metadata.get("line_count", 0)
            if line_count > 0:
                 # Add points for size, capping to avoid huge files dominating
                 scores[file] += min(line_count / 100.0, 4.0)

            # Bonus for very significant files (lines), penalize tiny files
            if line_count > 400:
                scores[file] += 1.5
            elif line_count < 10 and line_count > 0:
                 scores[file] -= 0.5
        except Exception:
            pass # Ignore if metadata fails


    # --- Score by file type ---
    for file in files:
         ext = Path(file).suffix.lower()
         if ext == ".py": scores[file] += 1.0 # Python is often central
         elif ext in [".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"]: scores[file] += 0.5 # Config files
         elif ext in [".md", ".rst", ".txt"]: scores[file] += 0.1 # Docs provide context
         elif ext in [".sh", ".bat"]: scores[file] += 0.3 # Scripts can be important


    # --- Adjustments ---
    for file in files:
         p = Path(file)
         # Example/Doc adjustment
         if "example" in p.name.lower() or "demo" in p.name.lower() or "doc" in p.parts:
             scores[file] *= 0.5 # Reduce score for examples/docs unless very central


    # --- Sort and Select Top N ---
    # Filter out files with zero or negative scores? Optional, but can clean up.
    # eligible_files = {f: s for f, s in scores.items() if s > 0}
    # sorted_files = sorted(eligible_files, key=eligible_files.get, reverse=True)
    # Using all files for now:
    sorted_files = sorted(files, key=lambda f: scores.get(f, 0.0), reverse=True)

    # Determine number of key files (e.g., top 15%, min 3, max 15)
    total_files = len(files)
    num_key_files = max(min(total_files // 7, 15), min(3, total_files))

    print(f"Selected top {num_key_files} files as key files.")
    # Debugging top scores:
    # for i in range(min(10, len(sorted_files))):
    #      f = sorted_files[i]
    #      print(f"  {i+1}. {Path(f).name}: {scores.get(f, 0.0):.2f}")

    return sorted_files[:num_key_files]