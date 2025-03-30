# Code Repository Analysis

Generated on 2025-03-30 20:46:48.564

## Repository Summary

- **Extensions analyzed**: `.py`
- **Number of files analyzed**: 9
- **Analysis Root (for display)**: `.`
- **Total lines of code (approx)**: 2445

## Project Structure (Relative View)

```
pilot-rules/
└── src/
    └── pilot_rules/
        ├── __init__.py
        ├── code_collector.py
        ├── main.py
        └── collector/
            ├── __init__.py
            ├── analysis.py
            ├── config.py
            ├── discovery.py
            ├── reporting.py
            └── utils.py
```

## Key Files

These files appear central based on dependencies, naming, and size:

### src/pilot_rules/collector/config.py

- **Lines**: 197
- **Size**: 8.58 KB
- **Last modified**: 2025-03-30 20:38:09
- **Used by**: 1 other analyzed Python file(s)

**Functions**:
- `parse_include_exclude_args(...)`
- `load_config_from_toml(...)`
- `process_config_and_args(...)`

**Content**:
```py
# src/pilot_rules/collector/config.py
import tomli
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_OUTPUT_FILENAME = "repository_analysis.md"
DEFAULT_INCLUDE_SPEC = "py:." # Default to python files in current dir

def parse_include_exclude_args(args: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Parses include/exclude arguments like 'py,js:src' or '*:temp'."""
    parsed = []
    if not args:
        return parsed

    for arg in args:
        if ':' not in arg:
            raise ValueError(f"Invalid include/exclude format: '{arg}'. Expected 'EXTS:PATH' or '*:PATTERN'.")

        exts_str, path_pattern = arg.split(':', 1)
        extensions = [ext.strip().lower().lstrip('.') for ext in exts_str.split(',') if ext.strip()]
        if not extensions:
            raise ValueError(f"No extensions specified in '{arg}'. Use '*' for all.")

        # Use '*' as a special marker for all extensions
        if '*' in extensions:
             extensions = ['*']

        # Normalize path pattern to use forward slashes for consistency
        # Keep it relative for now, resolve later if needed
        path_pattern = Path(path_pattern).as_posix()

        parsed.append({
            "extensions": extensions, # List of extensions (lowercase, no dot), or ['*']
            "pattern": path_pattern # Path or pattern string (relative or absolute)
        })
    return parsed


def load_config_from_toml(config_path: Path) -> Tuple[List[Dict], List[Dict], Optional[str]]:
    """Loads sources, excludes, and output path from a TOML file."""
    config_sources = []
    config_excludes = []
    config_output = None

    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)

        # --- Parse sources ---
        raw_sources = config_data.get('source', [])
        if not isinstance(raw_sources, list):
             raise ValueError("Invalid config: 'source' must be an array of tables.")

        for i, src_table in enumerate(raw_sources):
            if not isinstance(src_table, dict):
                 raise ValueError(f"Invalid config: Item {i} in 'source' array is not a table.")

            exts = src_table.get('exts', ['*']) # Default to all if not specified
            root = src_table.get('root', '.')
            exclude_patterns = src_table.get('exclude', []) # Excludes within a source block

            if not isinstance(exts, list) or not all(isinstance(e, str) for e in exts):
                 raise ValueError(f"Invalid config: 'exts' must be a list of strings in source #{i+1}")
            if not isinstance(root, str):
                 raise ValueError(f"Invalid config: 'root' must be a string in source #{i+1}.")
            if not isinstance(exclude_patterns, list) or not all(isinstance(p, str) for p in exclude_patterns):
                  raise ValueError(f"Invalid config: 'exclude' must be a list of strings in source #{i+1}")

            # Normalize extensions: lowercase, no leading dot
            normalized_exts = [e.lower().lstrip('.') for e in exts]
            if '*' in normalized_exts:
                 normalized_exts = ['*'] # Treat ['*'] as the 'all' marker

            # Store source config
            config_sources.append({
                'root': root, # Keep relative for now, resolve later
                'extensions': normalized_exts,
            })

            # Add source-specific excludes to the global excludes list
            # Assume format '*:<pattern>' for excludes defined within a source block
            for pattern in exclude_patterns:
                 config_excludes.append({'extensions': ['*'], 'pattern': Path(pattern).as_posix()})


        # --- Parse global output ---
        config_output = config_data.get('output')
        if config_output and not isinstance(config_output, str):
            raise ValueError("Invalid config: 'output' must be a string.")

        # --- Parse global excludes (optional top-level section) ---
        raw_global_excludes = config_data.get('exclude', [])
        if not isinstance(raw_global_excludes, list):
             raise ValueError("Invalid config: Top-level 'exclude' must be an array.")
        for i, ex_table in enumerate(raw_global_excludes):
             if not isinstance(ex_table, dict):
                  raise ValueError(f"Invalid config: Item {i} in top-level 'exclude' array is not a table.")
             exts = ex_table.get('exts', ['*'])
             pattern = ex_table.get('pattern')
             if pattern is None:
                  raise ValueError(f"Invalid config: 'pattern' missing in top-level exclude #{i+1}")
             if not isinstance(pattern, str):
                  raise ValueError(f"Invalid config: 'pattern' must be a string in top-level exclude #{i+1}")
             if not isinstance(exts, list) or not all(isinstance(e, str) for e in exts):
                 raise ValueError(f"Invalid config: 'exts' must be a list of strings in top-level exclude #{i+1}")

             normalized_exts = [e.lower().lstrip('.') for e in exts]
             if '*' in normalized_exts:
                  normalized_exts = ['*']

             config_excludes.append({'extensions': normalized_exts, 'pattern': Path(pattern).as_posix()})


    except tomli.TOMLDecodeError as e:
        raise ValueError(f"Error parsing TOML config file '{config_path}': {e}")
    except FileNotFoundError:
         raise ValueError(f"Config file not found: '{config_path}'")

    return config_sources, config_excludes, config_output


def process_config_and_args(
    include_args: Optional[List[str]],
    exclude_args: Optional[List[str]],
    output_arg: Optional[str], # Output from CLI args might be None if default used
    config_arg: Optional[str]
) -> Tuple[List[Dict], List[Dict], Path]:
    """
    Loads config, parses CLI args, merges them, and resolves paths.

    Returns:
        Tuple: (final_sources, final_excludes, final_output_path)
               Sources/Excludes contain resolved root paths and normalized patterns/extensions.
    """
    config_sources = []
    config_excludes = []
    config_output = None

    # 1. Load Config File (if provided)
    if config_arg:
        config_path = Path(config_arg)
        if config_path.is_file():
            config_sources, config_excludes, config_output = load_config_from_toml(config_path)
        else:
            # Argparse should handle file existence, but double-check
            raise ValueError(f"Config file path specified but not found or not a file: '{config_arg}'")

    # 2. Parse CLI arguments
    cli_includes = parse_include_exclude_args(include_args)
    cli_excludes = parse_include_exclude_args(exclude_args)
    # Use output_arg directly (it incorporates the argparse default if not provided)
    cli_output = output_arg if output_arg else DEFAULT_OUTPUT_FILENAME


    # 3. Combine sources: CLI overrides config sources entirely if provided.
    final_sources_specs = []
    if cli_includes:
        print("Using include sources from command line arguments.")
        final_sources_specs = cli_includes # Use CLI specs directly
    elif config_sources:
        print("Using include sources from configuration file.")
        final_sources_specs = config_sources # Use config specs
    else:
        print(f"No includes specified via CLI or config, defaulting to '{DEFAULT_INCLUDE_SPEC}'.")
        final_sources_specs = parse_include_exclude_args([DEFAULT_INCLUDE_SPEC])


    # 4. Combine excludes: CLI appends to config excludes
    # Exclude patterns remain relative path strings for fnmatch
    final_excludes = config_excludes + cli_excludes
    if final_excludes:
         print(f"Applying {len(final_excludes)} exclusion rule(s).")


    # 5. Determine final output path: CLI > Config > Default
    # cli_output already incorporates the default if needed
    final_output_str = cli_output if cli_output else config_output
    if not final_output_str: # Should not happen if argparse default is set
        final_output_str = DEFAULT_OUTPUT_FILENAME

    # Resolve output path relative to CWD
    final_output_path = Path(final_output_str).resolve()
    print(f"Final output path: {final_output_path}")

    # 6. Resolve source roots relative to CWD *after* deciding which specs to use
    resolved_final_sources = []
    for spec in final_sources_specs:
         # spec['pattern'] here is the root directory from include args or config
         resolved_root = Path(spec['pattern']).resolve()
         resolved_final_sources.append({
             'root': resolved_root,
             'extensions': spec['extensions'] # Keep normalized extensions
         })


    return resolved_final_sources, final_excludes, final_output_path
```

### src/pilot_rules/collector/analysis.py

- **Lines**: 403
- **Size**: 18.76 KB
- **Last modified**: 2025-03-30 20:38:55
- **Used by**: 2 other analyzed Python file(s)

**Functions**:
- `extract_python_components(...)`
- `get_module_prefixes(...)`
- `analyze_code_dependencies(...)`
- `get_common_patterns(...)`
- `find_key_files(...)`

**Content**:
```py
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
```

### src/pilot_rules/main.py

- **Lines**: 247
- **Size**: 9.42 KB
- **Last modified**: 2025-03-30 20:42:31
- **Used by**: 1 other analyzed Python file(s)

**Functions**:
- `get_version(...)`
- `display_guide(...)`
- `copy_template(...)`
- `main(...)`

**Content**:
```py
# src/pilot_rules/main.py
import os
import shutil
import argparse
import sys
from pathlib import Path
from typing import Optional

# Third-party imports
import tomli
from rich.console import Console
from rich.markdown import Markdown

from pilot_rules import collector

# --- Import the refactored collector entry point ---


# --- Helper Functions for Scaffolding ---

def get_version() -> str:
    """
    Get the current version from pyproject.toml.
    Searches upwards from the current file's location.

    Returns:
        str: The current version number or a fallback.
    """
    try:
        # Start searching from the package directory upwards
        current_dir = Path(__file__).parent
        while current_dir != current_dir.parent: # Stop at root directory '/'
            pyproject_path = current_dir / "pyproject.toml"
            if pyproject_path.exists():
                print(f"DEBUG: Found pyproject.toml at {pyproject_path}") # Debug print
                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomli.load(f)
                version = pyproject_data.get("project", {}).get("version", "0.0.0")
                if version != "0.0.0":
                    return version
                # If version is placeholder, keep searching upwards
            current_dir = current_dir.parent

        # If not found after searching upwards
        print("DEBUG: pyproject.toml with version not found.") # Debug print
        return "0.0.0" # Fallback if not found
    except Exception as e:
        print(f"DEBUG: Error getting version: {e}") # Debug print
        import traceback
        traceback.print_exc() # Print error during dev
        return "0.0.0" # Fallback on error


def display_guide(guide_path: Path) -> None:
    """
    Display the markdown guide using rich formatting.

    Args:
        guide_path: Path to the markdown guide file.
    """
    console = Console()
    if not guide_path.is_file():
        console.print(f"[red]Error: Guide file not found at '{guide_path}'[/red]")
        return

    try:
        with open(guide_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        markdown = Markdown(markdown_content)
        console.print("\n")
        console.rule("[bold blue]Getting Started Guide")
        console.print("\n")
        console.print(markdown)
        console.print("\n")
        console.rule("[bold blue]End of Guide")

    except Exception as e:
        console.print(f"[red]Error displaying guide '{guide_path}': {str(e)}[/red]")


def copy_template(template_type: str, root_dir: Path) -> Optional[Path]:
    """
    Copy template files based on the specified type ('cursor' or 'copilot').

    Args:
        template_type: Either 'cursor' or 'copilot'.
        root_dir: The root directory (usually CWD) where to copy the templates.

    Returns:
        Path to the relevant guide file if successful, None otherwise.
    """
    console = Console()
    package_dir = Path(__file__).parent # Directory where main.py is located
    templates_dir = package_dir / "templates"
    guides_dir = package_dir / "guides"

    source_dir: Optional[Path] = None
    target_dir: Optional[Path] = None
    guide_file: Optional[Path] = None

    if template_type == "cursor":
        source_dir = templates_dir / "cursor"
        target_dir = root_dir / ".cursor" # Target is relative to root_dir (CWD)
        guide_file = guides_dir / "cursor.md"
    elif template_type == "copilot":
        source_dir = templates_dir / "github"
        target_dir = root_dir / ".github" # Target is relative to root_dir (CWD)
        guide_file = guides_dir / "copilot.md"
    else:
        # This case should not be reached due to argparse mutual exclusion
        console.print(f"[red]Internal Error: Unknown template type '{template_type}'[/red]")
        return None

    if not source_dir or not source_dir.is_dir():
        console.print(f"[red]Error: Template source directory not found: '{source_dir}'[/red]")
        return None
    if not guide_file or not guide_file.is_file():
        console.print(f"[red]Error: Guide file not found: '{guide_file}'[/red]")
        # Decide whether to proceed without a guide or stop
        # return None # Stop if guide is essential

    # Create target directory if it doesn't exist
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(f"[red]Error: Could not create target directory '{target_dir}': {e}[/red]")
        return None

    # Copy the contents
    console.print(f"Copying {template_type} templates to '{target_dir}'...")
    try:
        for item in source_dir.iterdir():
            target_path = target_dir / item.name
            if item.is_file():
                shutil.copy2(item, target_path)
                # console.print(f"  Copied file: {item.name}") # Optional verbose logging
            elif item.is_dir():
                shutil.copytree(item, target_path, dirs_exist_ok=True)
                # console.print(f"  Copied directory: {item.name}") # Optional verbose logging
        console.print(f"[green]Successfully copied {template_type} templates.[/green]")
        return guide_file # Return path to guide file on success
    except Exception as e:
        console.print(f"[red]Error copying templates from '{source_dir}' to '{target_dir}': {e}[/red]")
        return None


# --- Main Application Logic ---

def main():
    """
    Entry point for the pilot-rules CLI application.
    Handles argument parsing and delegates tasks to scaffolding or collection modules.
    """
    console = Console()
    version = get_version()
    console.print(f"[bold blue]Pilot Rules v{version}[/bold blue]")
    console.print("[blue]www.whiteduck.de[/blue]") # Replace or remove if needed
    console.print()

    parser = argparse.ArgumentParser(
        description="Manage Pilot Rules templates or collect code for analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Mutually Exclusive Actions ---
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--cursor", action="store_true", help="Scaffold Cursor templates (.cursor)")
    action_group.add_argument("--copilot", action="store_true", help="Scaffold Copilot templates (.github)")
    action_group.add_argument("--collect", action="store_true", help="Collect code from the repository")

    # --- Options for Code Collection ---
    collect_group = parser.add_argument_group('Code Collection Options (used with --collect)')
    collect_group.add_argument(
        "--include",
        action="append",
        metavar="EXTS:FOLDER",
        help="Specify files to include. Format: 'ext1,ext2:./folder' or '*:.'."
             " Can be used multiple times. Default: 'py:.' if no includes provided."
    )
    collect_group.add_argument(
        "--exclude",
        action="append",
        metavar="EXTS_OR_*:PATTERN",
        help="Specify path patterns to exclude. Format: 'py:temp' or '*:node_modules'."
             " '*' matches any extension. Can be used multiple times."
    )
    collect_group.add_argument(
        "--output",
        default=None, # Default is handled inside the collector logic now
        metavar="FILEPATH",
        help=f"Path to the output Markdown file (default: '{collector.config.DEFAULT_OUTPUT_FILENAME}')"
    )
    collect_group.add_argument(
        "--config",
        metavar="TOML_FILE",
        help="Path to a .toml configuration file for collection settings."
    )

    args = parser.parse_args()

    # Root directory for scaffolding is the current working directory
    scaffold_root_dir = Path.cwd()
    guide_file_to_display: Optional[Path] = None

    try:
        if args.collect:
            console.print("[cyan]Starting code collection process...[/cyan]")
            # --- Call the refactored collector function ---
            # Errors within the collector (ValueError, etc.) will be caught below
            collector.run_collection(
                include_args=args.include,
                exclude_args=args.exclude,
                output_arg=args.output, # Pass CLI arg (can be None)
                config_arg=args.config
            )
            console.print("[green]Code collection process finished.[/green]")

        elif args.cursor:
            guide_file_to_display = copy_template("cursor", scaffold_root_dir)
            # Success/Error messages printed within copy_template

        elif args.copilot:
            guide_file_to_display = copy_template("copilot", scaffold_root_dir)
            # Success/Error messages printed within copy_template

        # Display guide only if scaffolding was successful and returned a guide path
        if guide_file_to_display:
            display_guide(guide_file_to_display)

    except FileNotFoundError as e:
         # Should primarily be caught within helpers now, but keep as fallback
         console.print(f"[red]Error: Required file or directory not found: {str(e)}[/red]", file=sys.stderr)
         exit(1)
    except ValueError as e: # Catch config errors propagated from collector
         console.print(f"[red]Configuration Error: {str(e)}[/red]", file=sys.stderr)
         exit(1)
    except Exception as e:
        # Catch-all for unexpected errors in main logic or propagated from helpers/collector
        console.print(f"[red]An unexpected error occurred: {str(e)}[/red]", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit(1)

# --- Standard Python entry point check ---
if __name__ == "__main__":
    main()
```

## Design Patterns (Python Heuristics)

Potential patterns identified based on naming and structure:

### Singleton Pattern

- `src/pilot_rules/code_collector.py`
- `src/pilot_rules/collector/analysis.py`

### Factory Pattern

- `src/pilot_rules/code_collector.py`
- `src/pilot_rules/collector/analysis.py`

### Observer Pattern

- `src/pilot_rules/code_collector.py`
- `src/pilot_rules/collector/analysis.py`

### Decorator Pattern

- `src/pilot_rules/code_collector.py`
- `src/pilot_rules/collector/analysis.py`


## All Analyzed Files (excluding key files)

### src/pilot_rules/__init__.py

- **Lines**: 4
- **Size**: 0.08 KB

**Content**:
```py
from .main import main as scaffolder_main

def main() -> None:
    scaffolder_main()
```

### src/pilot_rules/code_collector.py

- **Lines**: 1048
- **Size**: 47.82 KB

**Content**:
```py
#!/usr/bin/env python3
"""
Code Repository Analyzer

Generates a comprehensive Markdown document of a code repository,
optimized for LLM consumption and understanding. Handles multiple file
types, exclusions, and configuration files.
"""

import os
import sys
import glob
import datetime
import re
import ast
import fnmatch # For wildcard path matching
import tomli # For reading config file
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union

# --- Existing Helper Functions (get_file_metadata, extract_python_components, etc.) ---
# These functions generally remain the same, but we'll call them conditionally
# or update their usage slightly.

# Keep get_file_metadata as is
def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a file."""
    metadata = {
        "path": file_path,
        "size_bytes": 0,
        "line_count": 0,
        "last_modified": "Unknown",
        "created": "Unknown",
    }

    try:
        p = Path(file_path)
        stats = p.stat()
        metadata["size_bytes"] = stats.st_size
        metadata["last_modified"] = datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        # ctime is platform dependent (creation on Windows, metadata change on Unix)
        # Use mtime as a reliable fallback for "created" if ctime is older than mtime
        ctime = stats.st_ctime
        mtime = stats.st_mtime
        best_ctime = ctime if ctime <= mtime else mtime # Heuristic
        metadata["created"] = datetime.datetime.fromtimestamp(best_ctime).strftime("%Y-%m-%d %H:%M:%S")

        try:
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                metadata["line_count"] = len(content.splitlines())
        except Exception as read_err:
            print(f"Warning: Could not read content/count lines for {file_path}: {read_err}")
            metadata["line_count"] = 0 # Indicate unreadable/binary?

    except Exception as e:
        print(f"Warning: Could not get complete metadata for {file_path}: {e}")

    return metadata

# Keep extract_python_components as is, but call only for .py files
def extract_python_components(file_path: str) -> Dict[str, Any]:
    """Extract classes, functions, and imports from Python files."""
    # ... (existing implementation) ...
    components = {
        "classes": [],
        "functions": [],
        "imports": [],
        "docstring": None
    }

    # Ensure it's a python file before trying to parse
    if not file_path.lower().endswith('.py'):
        return components

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        # Extract module docstring
        if ast.get_docstring(tree):
            components["docstring"] = ast.get_docstring(tree)

        # Helper to determine if a function is top-level or a method
        def is_top_level_function(node, tree):
            for parent_node in ast.walk(tree):
                if isinstance(parent_node, ast.ClassDef):
                    for child in parent_node.body:
                         # Check identity using 'is' for direct reference comparison
                        if child is node:
                            return False
            return True

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
                 # Check if it's truly top-level (not a method)
                 # This check might be complex; let's list all for now and rely on context
                # if is_top_level_function(node, tree): # Simpler: List all functions found at top level of module body
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "args": [arg.arg for arg in node.args.args if hasattr(arg, 'arg')] # Simplified arg extraction
                }
                components["functions"].append(func_info)

        # Extract all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    components["imports"].append(alias.name) # Store the imported name/alias
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Handle relative imports representation
                relative_prefix = "." * node.level
                full_module_path = relative_prefix + module
                for alias in node.names:
                    # Store like 'from .module import name'
                    components["imports"].append(f"from {full_module_path} import {alias.name}")

    except SyntaxError as e:
         print(f"Warning: Could not parse Python components in {file_path} due to SyntaxError: {e}")
    except Exception as e:
        print(f"Warning: Could not parse Python components in {file_path}: {e}")

    return components


# Keep analyze_code_dependencies as is, but call only if .py files are included
def analyze_code_dependencies(files: List[str]) -> Dict[str, Set[str]]:
    """Analyze dependencies between Python files based on imports."""
    # ... (existing implementation) ...
    # Filter to only analyze python files within the provided list
    python_files = [f for f in files if f.lower().endswith('.py')]
    if not python_files:
        return {} # No Python files to analyze

    dependencies = {file: set() for file in python_files}
    module_map = {}
    # Simplified module mapping - relies on relative paths from CWD or structured project
    project_root = Path.cwd() # Assume CWD is project root for simplicity here

    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            # Attempt to create a module path relative to the project root
            relative_path = file_path.relative_to(project_root)
            parts = list(relative_path.parts)
            if parts[-1] == '__init__.py':
                parts.pop() # Module is the directory name
                if not parts: continue # Skip root __init__.py mapping?
                module_name = '.'.join(parts)
            elif parts[-1].endswith('.py'):
                parts[-1] = parts[-1][:-3] # Remove .py
                module_name = '.'.join(parts)
            else:
                continue # Not a standard python module file

            if module_name:
                 module_map[module_name] = str(file_path) # Map full module name to absolute path
                 # Add shorter name if not conflicting? Risky. Stick to full paths.

        except ValueError:
            # File is outside the assumed project root, handle simple name mapping
            base_name = file_path.stem
            if base_name != '__init__' and base_name not in module_map:
                 module_map[base_name] = str(file_path)

    # Now analyze imports in each Python file
    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)

            for node in ast.walk(tree):
                imported_module_str = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module_str = alias.name
                        # Check full name and prefixes
                        for prefix in get_module_prefixes(imported_module_str):
                             if prefix in module_map:
                                 # Check if the dependency is actually within our collected files
                                 dep_path = module_map[prefix]
                                 if dep_path in python_files:
                                     dependencies[file_path_str].add(dep_path)
                                 break # Found the longest matching prefix

                elif isinstance(node, ast.ImportFrom):
                    level = node.level
                    module_base = node.module or ""

                    if level == 0: # Absolute import
                         imported_module_str = module_base
                         for prefix in get_module_prefixes(imported_module_str):
                             if prefix in module_map:
                                 dep_path = module_map[prefix]
                                 if dep_path in python_files:
                                     dependencies[file_path_str].add(dep_path)
                                 break
                    else: # Relative import
                        current_dir = file_path.parent
                        # Go up 'level' directories (level=1 means current, level=2 means parent)
                        base_path = current_dir
                        for _ in range(level -1):
                            base_path = base_path.parent

                        # Try to resolve the relative module path
                        relative_module_parts = module_base.split('.')
                        target_path = base_path
                        if module_base: # If 'from .module import x'
                             for part in relative_module_parts:
                                 target_path = target_path / part

                        # Now check potential file/package paths based on this target
                        # This simplified version might miss complex relative imports
                        # Check if target_path itself (as __init__.py) exists
                        init_py = (target_path / '__init__.py').resolve()
                        if init_py.exists() and str(init_py) in python_files:
                            dependencies[file_path_str].add(str(init_py))
                        # Check if target_path.py exists
                        module_py = target_path.with_suffix('.py').resolve()
                        if module_py.exists() and str(module_py) in python_files:
                            dependencies[file_path_str].add(str(module_py))

                        # We could also try resolving the imported names (node.names)
                        # but let's keep dependency analysis high-level for now.

        except SyntaxError as e:
            print(f"Warning: Skipping import analysis in {file_path_str} due to SyntaxError: {e}")
        except Exception as e:
            print(f"Warning: Could not analyze imports in {file_path_str}: {e}")

    # Ensure dependencies only point to files within the initially provided 'files' list
    # (This should be handled by checking `dep_path in python_files` above)
    # Clean up dependencies: remove self-references
    for file in dependencies:
        dependencies[file].discard(file)

    return dependencies

# Keep get_module_prefixes as is
def get_module_prefixes(module_name: str) -> List[str]:
    """
    Generate all possible module prefixes for a given module name.
    For example, 'a.b.c' would return ['a.b.c', 'a.b', 'a']
    """
    parts = module_name.split('.')
    return ['.'.join(parts[:i]) for i in range(len(parts), 0, -1)]


# Keep generate_folder_tree as is
def generate_folder_tree(root_folder: str, included_files: List[str]) -> str:
    """Generate an ASCII folder tree representation, only showing directories and files that are included."""
    tree_output = []
    # Normalize included files to relative paths from the root folder for easier processing
    root_path = Path(root_folder).resolve()
    included_relative_paths = set()
    for f_abs in included_files:
        try:
            rel_path = Path(f_abs).resolve().relative_to(root_path)
            included_relative_paths.add(str(rel_path))
        except ValueError:
             # File is outside the root folder, might happen with multiple includes
             # For tree view, we only show things relative to the *main* root
             pass # Or log a warning

    # We need all directories that contain included files or other included directories
    included_dirs_rel = set()
    for rel_path_str in included_relative_paths:
        p = Path(rel_path_str)
        parent = p.parent
        while str(parent) != '.':
            included_dirs_rel.add(str(parent))
            parent = parent.parent
        if p.is_dir(): # If the path itself is a dir (though included_files should be files)
             included_dirs_rel.add(str(p))


    processed_dirs = set() # Avoid cycles and redundant processing

    def _generate_tree(current_dir_rel: str, prefix: str = ""):
        if current_dir_rel in processed_dirs:
            return
        processed_dirs.add(current_dir_rel)

        current_dir_abs = root_path / current_dir_rel
        dir_name = current_dir_abs.name if current_dir_rel != "." else "." # Handle root display name

        # Add the current directory to the output using appropriate prefix (later)
        # For now, collect children first

        entries = []
        try:
            for item in current_dir_abs.iterdir():
                item_rel_str = str(item.resolve().relative_to(root_path))
                if item.is_dir():
                    # Include dir if it's explicitly in included_dirs_rel OR contains included items
                    if item_rel_str in included_dirs_rel or any(
                        f.startswith(item_rel_str + os.sep) for f in included_relative_paths
                    ):
                       entries.append({'name': item.name, 'path': item_rel_str, 'is_dir': True})
                elif item.is_file():
                    if item_rel_str in included_relative_paths:
                        entries.append({'name': item.name, 'path': item_rel_str, 'is_dir': False})
        except (PermissionError, FileNotFoundError):
            pass # Skip inaccessible directories

        # Sort entries: directories first, then files, alphabetically
        entries.sort(key=lambda x: (not x['is_dir'], x['name']))

        # Now generate output for this level
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            tree_output.append(f"{prefix}{connector}{entry['name']}{'/' if entry['is_dir'] else ''}")

            if entry['is_dir']:
                new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                _generate_tree(entry['path'], new_prefix)


    # Start the recursion from the root directory representation "."
    tree_output.append(f"{root_folder}/") # Start with the root folder itself
    _generate_tree(".", prefix="│   ") # Use an initial prefix assuming root is not last


    # Quick fix for root display if only root is passed
    if len(tree_output) == 1 and tree_output[0] == f"{root_folder}/":
         # If no children were added, just show the root
         tree_output[0] = f"└── {root_folder}/" # Adjust prefix if it's the only thing
         # If files are directly in root, _generate_tree should handle them


    # Refine prefix for the first level items if they exist
    if len(tree_output) > 1:
        tree_output[0] = f"└── {root_folder}/" # Assume root is the end of its parent list
        # Need to adjust prefix logic inside _generate_tree or post-process
        # Let's stick to the simpler structure for now. ASCII trees can be tricky.


    return "\n".join(tree_output) # Return combined string


# Keep get_common_patterns as is, but call only if .py files are included
def get_common_patterns(files: List[str]) -> Dict[str, Any]:
    """Identify common design patterns in the codebase (Python focused)."""
    # ... (existing implementation) ...
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

            # Basic keyword/structure checks (can be improved)
            # Check for singleton pattern (simple heuristic)
            if ("instance = none" in content or "_instance = none" in content) and ("__new__" in content or " getinstance " in content):
                 patterns["singleton"].append(file_path)

            # Check for factory pattern
            if "factory" in file_basename_lower or ("def create_" in content and " return " in content) or ("def make_" in content and " return " in content):
                patterns["factory"].append(file_path)

            # Check for observer pattern
            if ("observer" in content or "listener" in content) and ("notify" in content or "update" in content or "addeventlistener" in content or "subscribe" in content):
                 patterns["observer"].append(file_path)

            # Check for decorator pattern (presence of @ syntax handled by Python itself)
            # Look for common decorator definition patterns
            if "def wrapper(" in content and "return wrapper" in content:
                 patterns["decorator"].append(file_path) # Might be too broad

            # Check for MVC components based on naming conventions
            if "model" in file_basename_lower or "models" in file_path.lower().split(os.sep):
                patterns["mvc_components"]["models"].append(file_path)
            if "view" in file_basename_lower or "views" in file_path.lower().split(os.sep) or "template" in file_basename_lower:
                patterns["mvc_components"]["views"].append(file_path)
            if "controller" in file_basename_lower or "controllers" in file_path.lower().split(os.sep) or "handler" in file_basename_lower or "routes" in file_basename_lower:
                patterns["mvc_components"]["controllers"].append(file_path)

        except Exception as e:
            # print(f"Warning: Could not analyze patterns in {file_path}: {e}") # Can be noisy
            continue # Ignore files that can't be read or processed

    # --- Clean up empty categories ---
    # Create a new dict to avoid modifying while iterating
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


# Keep find_key_files as is, but consider its Python focus
def find_key_files(files: List[str], dependencies: Dict[str, Set[str]]) -> List[str]:
    """Identify key files based on dependencies and naming conventions (Python focused)."""
    # ... (existing implementation) ...
     # Initialize scores for each file
    scores = {file: 0.0 for file in files} # Use float for potentially fractional scores

    # Track how many files depend on each file (dependents) - Python only for now
    python_files = {f for f in files if f.lower().endswith('.py')}
    dependent_count = {file: 0 for file in python_files}
    for file, deps in dependencies.items(): # dependencies should already be Python-only
        if file not in python_files: continue # Ensure source file is Python
        for dep in deps:
            if dep in dependent_count: # Target file must also be Python
                dependent_count[dep] += 1

    # Score by number of files that depend on this file (high impact)
    for file, count in dependent_count.items():
        scores[file] += count * 2.0

    # Score by file naming heuristics (more general)
    for file in files:
        p = Path(file)
        base_name = p.name.lower()
        parent_dir_name = p.parent.name.lower()

        # Core file names
        if any(core_name in base_name for core_name in ["main.", "app.", "core.", "__init__.py", "cli.", "server.", "manage.py"]):
            scores[file] += 5.0
        elif base_name == "settings.py" or base_name == "config.py":
             scores[file] += 4.0
        elif base_name.startswith("test_"):
             scores[file] -= 1.0 # Lower score for test files unless highly depended upon

        # Configuration and settings
        if any(config_name in base_name for config_name in ["config", "settings", "constant", "conf."]):
            scores[file] += 3.0

        # Base classes and abstract components
        if any(base_name_part in base_name for base_name_part in ["base.", "abstract", "interface", "factory"]):
            scores[file] += 2.0

        # Utilities and helpers
        if any(util_name in base_name for util_name in ["util", "helper", "common", "tool", "shared"]):
            scores[file] += 1.0

        # Score directories by importance
        if "src" == parent_dir_name: # Direct child of src
             scores[file] += 0.5
        if "core" in p.parent.parts:
             scores[file] += 1.0
        if "main" in p.parent.parts or "app" in p.parent.parts:
            scores[file] += 0.5

        # Score by file size (crude complexity measure)
        try:
            metadata = get_file_metadata(file)
            line_count = metadata.get("line_count", 0)
            if line_count > 0:
                 scores[file] += min(line_count / 100.0, 3.0)  # Cap at 3 points, less sensitive than /50

            # Bonus for significant files
            if line_count > 300:
                scores[file] += 1.0
            elif line_count < 10:
                 scores[file] -= 0.5 # Penalize very small files slightly
        except Exception:
            pass # Ignore if metadata fails

        # Score by extension - Python files are often central in Python projects
        if file.lower().endswith(".py"):
            scores[file] += 1.0
        elif file.lower().endswith((".md", ".txt", ".rst")):
             scores[file] += 0.1 # Documentation is useful context
        elif file.lower().endswith((".yaml", ".yml", ".json", ".toml")):
             scores[file] += 0.5 # Config files can be important

        # Examples and documentation are important but usually not "key" execution paths
        if "example" in file.lower() or "demo" in file.lower() or "doc" in file.lower():
            scores[file] += 0.2

    # Sort by score in descending order
    # Filter out files with zero or negative scores before sorting? Optional.
    key_files = sorted(files, key=lambda f: scores.get(f, 0.0), reverse=True)

    # Debugging info (optional, add a verbose flag?)
    # print(f"Top 5 key files with scores:")
    # for file in key_files[:5]:
    #     print(f"  {file}: {scores.get(file, 0.0):.1f} points")

    # Return top N files or percentage - make it configurable?
    # Let's stick to a reasonable number like top 5-10 or 20% capped at 20
    num_key_files = max(min(len(files) // 5, 20), min(5, len(files))) # 20% or 5, capped at 20
    return key_files[:num_key_files]


# --- New/Modified Core Logic ---

def parse_include_exclude_args(args: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Parses include/exclude arguments like 'py,js:src' or '*:temp'."""
    parsed = []
    if not args:
        return parsed

    for arg in args:
        if ':' not in arg:
            raise ValueError(f"Invalid include/exclude format: '{arg}'. Expected 'EXTS:PATH' or '*:PATTERN'.")

        exts_str, path_pattern = arg.split(':', 1)
        extensions = [ext.strip().lower() for ext in exts_str.split(',') if ext.strip()]

        # Normalize path pattern
        path_pattern = Path(path_pattern).as_posix() # Use forward slashes for consistency

        parsed.append({
            "extensions": extensions, # List of extensions, or ['*']
            "pattern": path_pattern # Path or pattern string
        })
    return parsed

def collect_files(sources: List[Dict[str, Any]], excludes: List[Dict[str, Any]]) -> Tuple[List[str], Set[str]]:
    """
    Finds files based on source definitions and applies exclusion rules.

    Args:
        sources: List of dicts, each with 'extensions' (list or ['*']) and 'root' (str).
        excludes: List of dicts, each with 'extensions' (list or ['*']) and 'pattern' (str).

    Returns:
        Tuple: (list of absolute file paths found, set of unique extensions found)
    """
    print("Collecting files...")
    all_found_files = set()
    all_extensions = set()
    project_root = Path.cwd().resolve() # Use CWD as the reference point

    for source in sources:
        root_path = Path(source['root']).resolve()
        extensions = source['extensions']
        print(f"  Scanning in: '{root_path}' for extensions: {extensions if extensions != ['*'] else 'all'}")

        # Decide which glob pattern to use
        glob_patterns = []
        if extensions == ['*']:
            # Glob all files recursively
            glob_patterns.append(str(root_path / '**' / '*'))
        else:
            for ext in extensions:
                 # Ensure extension starts with a dot if not already present
                 dot_ext = f".{ext}" if not ext.startswith('.') else ext
                 glob_patterns.append(str(root_path / '**' / f'*{dot_ext}'))
                 all_extensions.add(dot_ext) # Track requested extensions

        found_in_source = set()
        for pattern in glob_patterns:
            try:
                 # Use pathlib's rglob for recursive search
                 # Need to handle the non-extension specific case carefully
                 if pattern.endswith('*'): # Case for '*' extension
                     for item in root_path.rglob('*'):
                          if item.is_file():
                              found_in_source.add(str(item.resolve()))
                 else: # Specific extension
                     # Extract the base path and the extension pattern part
                     base_path_for_glob = Path(pattern).parent
                     ext_pattern = Path(pattern).name
                     for item in base_path_for_glob.rglob(ext_pattern):
                           if item.is_file():
                              found_in_source.add(str(item.resolve()))

            except Exception as e:
                 print(f"Warning: Error during globbing pattern '{pattern}': {e}")


        print(f"    Found {len(found_in_source)} potential files.")
        all_found_files.update(found_in_source)

    print(f"Total unique files found before exclusion: {len(all_found_files)}")

    # Apply exclusion rules
    excluded_files = set()
    if excludes:
        print("Applying exclusion rules...")
        # Prepare relative paths for matching
        relative_files_map = {
             str(Path(f).resolve().relative_to(project_root)): f
             for f in all_found_files
             if Path(f).resolve().is_relative_to(project_root) # Only exclude relative to project root
        }
        relative_file_paths = set(relative_files_map.keys())


        for rule in excludes:
            rule_exts = rule['extensions']
            rule_pattern = rule['pattern']
            print(f"  Excluding: extensions {rule_exts if rule_exts != ['*'] else 'any'} matching path pattern '*{rule_pattern}*'") # Match anywhere in path

            # Use fnmatch for pattern matching against relative paths
            pattern_to_match = f"*{rule_pattern}*" # Wrap pattern for contains check


            files_to_check = relative_file_paths
            # If rule has specific extensions, filter the files to check first
            if rule_exts != ['*']:
                dot_exts = {f".{e}" if not e.startswith('.') else e for e in rule_exts}
                files_to_check = {
                     rel_path for rel_path in relative_file_paths
                     if Path(rel_path).suffix.lower() in dot_exts
                }


            # Apply fnmatch
            matched_by_rule = {
                rel_path for rel_path in files_to_check
                if fnmatch.fnmatch(rel_path, pattern_to_match)
            }

            # Add the corresponding absolute paths to the excluded set
            for rel_path in matched_by_rule:
                excluded_files.add(relative_files_map[rel_path])
                # print(f"    Excluding: {relative_files_map[rel_path]}") # Verbose logging


    print(f"Excluded {len(excluded_files)} files.")
    final_files = sorted(list(all_found_files - excluded_files))

    # Determine actual extensions present in the final list
    final_extensions = {Path(f).suffix.lower() for f in final_files if Path(f).suffix}

    return final_files, final_extensions


def generate_markdown(
    files: List[str],
    analyzed_extensions: Set[str], # Use the actual extensions found
    output_path: str,
    root_folder_display: str = "." # How to display the root in summary/tree
) -> None:
    """Generate a comprehensive markdown document about the codebase."""
    print(f"Generating Markdown report at '{output_path}'...")
    # Only run Python-specific analysis if .py files are present
    has_python_files = any(f.lower().endswith('.py') for f in files)
    dependencies = {}
    patterns = {}
    if has_python_files:
        print("Analyzing Python dependencies...")
        dependencies = analyze_code_dependencies(files) # Pass all files, it filters internally
        print("Identifying common patterns...")
        patterns = get_common_patterns(files) # Pass all files, it filters internally
    else:
        print("Skipping Python-specific analysis (no .py files found).")


    print("Finding key files...")
    key_files = find_key_files(files, dependencies) # Pass all files, scorer handles types

    # Use the directory of the output file as the base for relative paths if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists


    with open(output_path, "w", encoding="utf-8") as md_file:
        # Write header
        md_file.write(f"# Code Repository Analysis\n\n")
        # Format timestamp for clarity
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        md_file.write(f"Generated on {timestamp}\n\n")


        # Write repository summary
        md_file.write("## Repository Summary\n\n")
        ext_list_str = ", ".join(sorted(list(analyzed_extensions))) if analyzed_extensions else "N/A"
        md_file.write(f"- **Extensions analyzed**: `{ext_list_str}`\n")
        md_file.write(f"- **Number of files analyzed**: {len(files)}\n")
        # Decide on root folder representation - maybe list all roots from sources?
        # For now, keep it simple.
        md_file.write(f"- **Primary analysis root (for tree)**: `{root_folder_display}`\n")

        total_lines = 0
        if files:
             try:
                # Calculate total lines safely
                total_lines = sum(get_file_metadata(f).get("line_count", 0) for f in files)
             except Exception as e:
                 print(f"Warning: Could not calculate total lines accurately: {e}")
                 total_lines = "N/A"
        else:
             total_lines = 0

        md_file.write(f"- **Total lines of code (approx)**: {total_lines}\n\n")

        # Generate and write folder tree relative to root_folder_display
        md_file.write("## Project Structure (Relative View)\n\n")
        md_file.write("```\n")
        # Pass absolute paths of files and the root display path
        try:
             # Ensure root_folder_display exists and is a directory for tree generation
             root_for_tree = Path(root_folder_display)
             if root_for_tree.is_dir():
                 # Pass absolute paths to generate_folder_tree
                 md_file.write(generate_folder_tree(str(root_for_tree.resolve()), files))
             else:
                  md_file.write(f"Cannot generate tree: '{root_folder_display}' is not a valid directory.")

        except Exception as tree_err:
             md_file.write(f"Error generating folder tree: {tree_err}")
        md_file.write("\n```\n\n")


        # --- Key Files Section ---
        md_file.write("## Key Files\n\n")
        if key_files:
             md_file.write("These files appear central based on dependencies, naming, and size:\n\n")
             # Use CWD as the base for relative paths in the report for consistency
             report_base_path = Path.cwd()
             for file_abs_path in key_files:
                  try:
                      rel_path = str(Path(file_abs_path).relative_to(report_base_path))
                  except ValueError:
                      rel_path = file_abs_path # Fallback to absolute if not relative to CWD

                  md_file.write(f"### {rel_path}\n\n")

                  metadata = get_file_metadata(file_abs_path)
                  md_file.write(f"- **Lines**: {metadata.get('line_count', 'N/A')}\n")
                  md_file.write(f"- **Size**: {metadata.get('size_bytes', 0) / 1024:.2f} KB\n")
                  md_file.write(f"- **Last modified**: {metadata.get('last_modified', 'Unknown')}\n")

                  # Dependency info (Python only)
                  dependent_files_rel = []
                  if has_python_files and file_abs_path in dependencies: # Check if file itself has deps analyzed
                       # Find which files depend *on this* key file
                      dependent_files_abs = [
                          f for f, deps in dependencies.items() if file_abs_path in deps
                      ]
                      dependent_files_rel = []
                      for dep_abs in dependent_files_abs:
                           try:
                               dependent_files_rel.append(str(Path(dep_abs).relative_to(report_base_path)))
                           except ValueError:
                                dependent_files_rel.append(dep_abs) # Fallback

                  if dependent_files_rel:
                      md_file.write(f"- **Used by**: {len(dependent_files_rel)} other Python file(s)\n") # maybe list top 3? e.g. `[:3]`


                  # Python component analysis
                  if file_abs_path.lower().endswith('.py'):
                       components = extract_python_components(file_abs_path)
                       if components.get("docstring"):
                           # Limit docstring length?
                           docstring_summary = (components["docstring"].strip().split('\n')[0])[:150] # First line, max 150 chars
                           md_file.write(f"\n**Description**: {docstring_summary}...\n")

                       if components.get("classes"):
                           md_file.write("\n**Classes**:\n")
                           for cls in components["classes"][:5]: # Limit displayed classes
                               md_file.write(f"- `{cls['name']}` ({len(cls['methods'])} methods)\n")
                           if len(components["classes"]) > 5:
                                md_file.write("- ... (and more)\n")


                       if components.get("functions"):
                           md_file.write("\n**Functions**:\n")
                           for func in components["functions"][:5]: # Limit displayed functions
                               md_file.write(f"- `{func['name']}(...)`\n") # Simplified signature
                           if len(components["functions"]) > 5:
                                md_file.write("- ... (and more)\n")

                  # File Content
                  md_file.write("\n**Content Snippet**:\n") # Changed from "Content" to avoid huge files
                  file_ext = Path(file_abs_path).suffix
                  lang_hint = file_ext.lstrip('.') if file_ext else ""
                  md_file.write(f"```{lang_hint}\n")

                  try:
                      with open(file_abs_path, "r", encoding="utf-8", errors='ignore') as code_file:
                          # Show first N lines (e.g., 50) as a snippet
                          snippet_lines = []
                          for i, line in enumerate(code_file):
                               if i >= 50:
                                   snippet_lines.append("...")
                                   break
                               snippet_lines.append(line.rstrip()) # Remove trailing newline for cleaner output
                          content_snippet = "\n".join(snippet_lines)
                          md_file.write(content_snippet)
                          if not content_snippet.endswith("\n"):
                               md_file.write("\n")
                  except Exception as e:
                      md_file.write(f"Error reading file content: {str(e)}\n")

                  md_file.write("```\n\n")
        else:
             md_file.write("No key files identified based on current criteria.\n\n")

        # --- Design Patterns Section ---
        if patterns:
             md_file.write("## Design Patterns (Python Heuristics)\n\n")
             md_file.write("Potential patterns identified based on naming and structure:\n\n")
             report_base_path = Path.cwd() # Base for relative paths

             for pattern_name, files_or_dict in patterns.items():
                 title = pattern_name.replace('_', ' ').title()
                 if isinstance(files_or_dict, list) and files_or_dict:
                     md_file.write(f"### {title} Pattern\n\n")
                     for f_abs in files_or_dict[:10]: # Limit displayed files per pattern
                          try:
                             rel_p = str(Path(f_abs).relative_to(report_base_path))
                          except ValueError:
                              rel_p = f_abs
                          md_file.write(f"- `{rel_p}`\n")
                     if len(files_or_dict) > 10: md_file.write("- ... (and more)\n")
                     md_file.write("\n")
                 elif isinstance(files_or_dict, dict): # e.g., MVC
                     has_content = any(sublist for sublist in files_or_dict.values())
                     if has_content:
                          md_file.write(f"### {title}\n\n")
                          for subpattern, subfiles in files_or_dict.items():
                              if subfiles:
                                  md_file.write(f"**{subpattern.title()}**:\n")
                                  for f_abs in subfiles[:5]: # Limit sub-pattern files
                                       try:
                                          rel_p = str(Path(f_abs).relative_to(report_base_path))
                                       except ValueError:
                                           rel_p = f_abs
                                       md_file.write(f"- `{rel_p}`\n")
                                  if len(subfiles) > 5: md_file.write("  - ... (and more)\n")
                                  md_file.write("\n")
             md_file.write("\n")
        elif has_python_files:
             md_file.write("## Design Patterns (Python Heuristics)\n\n")
             md_file.write("No common design patterns identified based on current heuristics.\n\n")


        # --- All Other Files Section ---
        md_file.write("## All Analyzed Files\n\n")
        other_files = [f for f in files if f not in key_files]

        if other_files:
             report_base_path = Path.cwd()
             for file_abs_path in other_files:
                  try:
                     rel_path = str(Path(file_abs_path).relative_to(report_base_path))
                  except ValueError:
                     rel_path = file_abs_path

                  md_file.write(f"### {rel_path}\n\n")

                  metadata = get_file_metadata(file_abs_path)
                  md_file.write(f"- **Lines**: {metadata.get('line_count', 'N/A')}\n")
                  md_file.write(f"- **Size**: {metadata.get('size_bytes', 0) / 1024:.2f} KB\n")
                  md_file.write(f"- **Last modified**: {metadata.get('last_modified', 'Unknown')}\n\n")

                  # Provide a content snippet for other files too
                  md_file.write("**Content Snippet**:\n")
                  file_ext = Path(file_abs_path).suffix
                  lang_hint = file_ext.lstrip('.') if file_ext else ""
                  md_file.write(f"```{lang_hint}\n")
                  try:
                      with open(file_abs_path, "r", encoding="utf-8", errors='ignore') as code_file:
                          snippet_lines = []
                          for i, line in enumerate(code_file):
                               if i >= 30: # Shorter snippet for non-key files
                                   snippet_lines.append("...")
                                   break
                               snippet_lines.append(line.rstrip())
                          content_snippet = "\n".join(snippet_lines)
                          md_file.write(content_snippet)
                          if not content_snippet.endswith("\n"):
                               md_file.write("\n")
                  except Exception as e:
                      md_file.write(f"Error reading file content: {str(e)}\n")
                  md_file.write("```\n\n")
        elif key_files:
             md_file.write("All analyzed files are listed in the 'Key Files' section.\n\n")
        else:
            md_file.write("No files were found matching the specified criteria.\n\n")

    print(f"Markdown report generated successfully at '{output_path}'")


def run_collection(
    include_args: Optional[List[str]],
    exclude_args: Optional[List[str]],
    output_arg: str,
    config_arg: Optional[str]
) -> None:
    """
    Main entry point for the code collection process, handling config and args.
    """
    # Defaults
    config_sources = []
    config_excludes = []
    config_output = None

    # 1. Load Config File (if provided)
    if config_arg:
        config_path = Path(config_arg)
        if config_path.is_file():
            print(f"Loading configuration from: {config_path}")
            try:
                with open(config_path, "rb") as f:
                    config_data = tomli.load(f)

                # Parse sources from config
                raw_sources = config_data.get('source', [])
                if not isinstance(raw_sources, list):
                     raise ValueError("Invalid config: 'source' must be an array of tables.")

                for src_table in raw_sources:
                    exts = src_table.get('exts', ['*']) # Default to all if not specified
                    root = src_table.get('root', '.')
                    exclude_patterns = src_table.get('exclude', []) # Excludes within a source block

                    if not isinstance(exts, list) or not all(isinstance(e, str) for e in exts):
                         raise ValueError(f"Invalid config: 'exts' must be a list of strings in source root '{root}'")
                    if not isinstance(root, str):
                         raise ValueError(f"Invalid config: 'root' must be a string in source.")
                    if not isinstance(exclude_patterns, list) or not all(isinstance(p, str) for p in exclude_patterns):
                          raise ValueError(f"Invalid config: 'exclude' must be a list of strings in source root '{root}'")


                    config_sources.append({
                        'root': Path(root).resolve(), # Store resolved path
                        'extensions': [e.lower().lstrip('.') for e in exts],
                    })
                    # Add source-specific excludes to the global excludes list
                    # Assume format '*:<pattern>' for simplicity from config's exclude list
                    for pattern in exclude_patterns:
                         config_excludes.append({'extensions': ['*'], 'pattern': Path(pattern).as_posix()})


                # Parse global output from config
                config_output = config_data.get('output')
                if config_output and not isinstance(config_output, str):
                    raise ValueError("Invalid config: 'output' must be a string.")

            except tomli.TOMLDecodeError as e:
                raise ValueError(f"Error parsing TOML config file '{config_path}': {e}")
            except FileNotFoundError:
                 raise ValueError(f"Config file not found: '{config_path}'")
        else:
            raise ValueError(f"Config file path is not a file: '{config_arg}'")

    # 2. Parse CLI arguments
    cli_includes = parse_include_exclude_args(include_args)
    cli_excludes = parse_include_exclude_args(exclude_args)
    cli_output = output_arg

    # 3. Combine sources: CLI overrides/appends config
    # If CLI includes are given, they replace config sources. Otherwise, use config sources.
    # If neither is given, default to '.py' in '.'
    final_sources = []
    if cli_includes:
        print("Using include sources from command line arguments.")
        final_sources = [{'root': Path(inc['pattern']).resolve(), 'extensions': inc['extensions']} for inc in cli_includes]
    elif config_sources:
        print("Using include sources from configuration file.")
        final_sources = config_sources # Already resolved paths
    else:
        print("No includes specified, defaulting to '.py' files in current directory.")
        final_sources = [{'root': Path('.').resolve(), 'extensions': ['py']}]

    # 4. Combine excludes: CLI appends to config excludes
    final_excludes = config_excludes + cli_excludes
    if final_excludes:
         print(f"Using {len(final_excludes)} exclusion rule(s).")


    # 5. Determine final output path: CLI > Config > Default
    final_output = cli_output if cli_output else config_output
    # Use default from argparse if cli_output is None/empty and config_output is None
    if not final_output:
         final_output = "repository_analysis.md" # Re-apply default if needed

    final_output_path = Path(final_output).resolve()
    print(f"Final output path: {final_output_path}")

    # 6. Collect files
    collected_files, actual_extensions = collect_files(final_sources, final_excludes)

    if not collected_files:
        print("Warning: No files found matching the specified criteria.")
        # Generate an empty/minimal report?
        # For now, let's allow generate_markdown to handle the empty list.
    else:
        print(f"Found {len(collected_files)} files to include in the report.")
        print(f"File extensions found: {', '.join(sorted(list(actual_extensions)))}")

    # 7. Generate Markdown
    # Use '.' as the display root for simplicity, could be made smarter
    generate_markdown(collected_files, actual_extensions, str(final_output_path), root_folder_display=".")


# Keep the standalone execution part for testing/direct use if needed
if __name__ == "__main__":
    import argparse

    # This argparse is now only for *direct* execution of code_collector.py
    parser = argparse.ArgumentParser(description="Analyze code repository (Standalone Execution)")
    parser.add_argument("-i", "--include", action="append", help="Include spec 'EXTS:FOLDER'")
    parser.add_argument("-e", "--exclude", action="append", help="Exclude spec '*:PATTERN'")
    parser.add_argument("-o", "--output", default="repository_analysis_standalone.md", help="Output markdown file")
    parser.add_argument("--config", help="Path to TOML config file")

    args = parser.parse_args()

    try:
        run_collection(
            include_args=args.include,
            exclude_args=args.exclude,
            output_arg=args.output,
            config_arg=args.config
        )
    except ValueError as e:
         print(f"Error: {e}")
         sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)
```

### src/pilot_rules/collector/__init__.py

- **Lines**: 97
- **Size**: 3.78 KB

**Content**:
```py
# src/pilot_rules/collector/__init__.py
"""
Code Collection and Analysis Sub-package.
Provides functionality to scan repositories, analyze code (primarily Python),
and generate Markdown summaries.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import necessary functions from sibling modules using relative imports
from .config import process_config_and_args
from .discovery import collect_files
from .analysis import analyze_code_dependencies, get_common_patterns, find_key_files
from .reporting import generate_markdown

def run_collection(
    include_args: Optional[List[str]],
    exclude_args: Optional[List[str]],
    output_arg: Optional[str], # Can be None if default is used by argparse
    config_arg: Optional[str]
) -> None:
    """
    Main entry point for the code collection process.

    Orchestrates configuration loading, file discovery, analysis, and report generation.
    """
    try:
        # 1. Process Configuration and Arguments
        # This resolves paths and merges CLI/config settings
        final_sources, final_excludes, final_output_path = process_config_and_args(
            include_args=include_args,
            exclude_args=exclude_args,
            output_arg=output_arg,
            config_arg=config_arg
        )

        # 2. Collect Files based on finalized sources and excludes
        # Uses glob.glob internally now
        collected_files, actual_extensions = collect_files(final_sources, final_excludes)

        if not collected_files:
            print("Warning: No files found matching the specified criteria.")
            # Generate an empty/minimal report? Or just exit?
            # Let generate_markdown handle the empty list for now.
        else:
            print(f"Found {len(collected_files)} files to include in the report.")
            print(f"File extensions found: {', '.join(sorted(list(actual_extensions)))}")


        # 3. Perform Analysis (Conditional based on files found)
        dependencies = {}
        patterns = {}
        key_files = []

        # Only run Python-specific analysis if .py files are present
        has_python_files = '.py' in actual_extensions
        if has_python_files:
            print("Analyzing Python dependencies...")
            dependencies = analyze_code_dependencies(collected_files)
            print("Identifying common patterns (Python)...")
            patterns = get_common_patterns(collected_files)
        else:
            print("Skipping Python-specific analysis (no .py files found).")

        # Find key files (uses heuristics applicable to various file types)
        if collected_files:
             print("Identifying key files...")
             key_files = find_key_files(collected_files, dependencies) # Pass all files


        # 4. Generate Markdown Report
        # Use '.' as the display root for simplicity in the tree view
        generate_markdown(
            files=collected_files,
            analyzed_extensions=actual_extensions,
            dependencies=dependencies,
            patterns=patterns,
            key_files=key_files,
            output_path=final_output_path,
            root_folder_display="." # Or derive from sources if needed
        )

    except ValueError as e:
         # Configuration or argument parsing errors
         print(f"[Collector Error] Configuration Error: {e}", file=sys.stderr)
         # Re-raise or exit? Re-raising lets the caller (main.py) handle exit status.
         raise # Or sys.exit(1)
    except Exception as e:
         # Catch-all for unexpected errors during collection/analysis/reporting
         print(f"[Collector Error] An unexpected error occurred: {e}", file=sys.stderr)
         import traceback
         traceback.print_exc()
         raise # Or sys.exit(1)


# __all__ = ['run_collection'] # Optionally define the public API
```

### src/pilot_rules/collector/discovery.py

- **Lines**: 126
- **Size**: 5.91 KB

**Content**:
```py
# src/pilot_rules/collector/discovery.py
import glob
import os
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

def collect_files(sources: List[Dict[str, Any]], excludes: List[Dict[str, Any]]) -> Tuple[List[str], Set[str]]:
    """
    Finds files based on source definitions (using glob.glob) and applies exclusion rules.

    Args:
        sources: List of dicts, each with 'root' (resolved Path) and 'extensions' (list or ['*']).
        excludes: List of dicts, each with 'extensions' (list or ['*']) and 'pattern' (str).

    Returns:
        Tuple: (list of absolute file paths found, set of unique extensions found (lowercase, with dot))
    """
    print("Collecting files...")
    all_found_files: Set[str] = set() # Store absolute paths as strings
    project_root = Path.cwd().resolve() # Use CWD as the reference point for excludes

    for source in sources:
        root_path: Path = source['root']
        extensions: List[str] = source['extensions'] # Already normalized (lowercase, no dot, or ['*'])
        print(f"  Scanning in: '{root_path}' for extensions: {extensions if extensions != ['*'] else 'all'}")

        if not root_path.is_dir():
            print(f"Warning: Source root '{root_path}' is not a directory. Skipping.")
            continue

        found_in_source: Set[str] = set()
        if extensions == ['*']:
            # Use glob.glob for all files recursively
            glob_pattern_str = str(root_path / '**' / '*')
            print(f"    Using glob pattern: {glob_pattern_str} (recursive)")
            try:
                # Use glob.glob with recursive=True
                for filepath_str in glob.glob(glob_pattern_str, recursive=True):
                     item = Path(filepath_str)
                     # Check if it's a file (glob might return directories matching pattern too)
                     if item.is_file():
                         # Add resolved absolute path as string
                         found_in_source.add(str(item.resolve()))
            except Exception as e:
                print(f"Warning: Error during globbing for '{glob_pattern_str}': {e}")
        else:
            # Specific extensions provided
            for ext in extensions:
                 # Construct pattern like '*.py'
                 pattern = f'*.{ext}'
                 glob_pattern_str = str(root_path / '**' / pattern)
                 print(f"    Using glob pattern: {glob_pattern_str} (recursive)")
                 try:
                    # Use glob.glob with recursive=True
                    for filepath_str in glob.glob(glob_pattern_str, recursive=True):
                         item = Path(filepath_str)
                         # Check if it's a file
                         if item.is_file():
                             # Add resolved absolute path as string
                             found_in_source.add(str(item.resolve()))
                 except Exception as e:
                     print(f"Warning: Error during globbing for '{glob_pattern_str}': {e}")

        print(f"    Found {len(found_in_source)} potential files in this source.")
        all_found_files.update(found_in_source)

    print(f"Total unique files found before exclusion: {len(all_found_files)}")

    # Apply exclusion rules
    excluded_files: Set[str] = set()
    if excludes:
        print("Applying exclusion rules...")
        # Create a map of relative paths (from project_root) to absolute paths
        # Only consider files that are within the project root for relative matching
        relative_files_map: Dict[str, str] = {}
        for abs_path_str in all_found_files:
             abs_path = Path(abs_path_str)
             try:
                  # Use POSIX paths for matching consistency
                  relative_path_str = abs_path.relative_to(project_root).as_posix()
                  relative_files_map[relative_path_str] = abs_path_str
             except ValueError:
                  # File is outside project root, cannot be excluded by relative pattern
                  pass

        relative_file_paths = set(relative_files_map.keys())

        for rule in excludes:
            rule_exts: List[str] = rule['extensions'] # Normalized (lowercase, no dot, or ['*'])
            rule_pattern: str = rule['pattern'] # Relative path pattern string
            print(f"  Excluding: extensions {rule_exts if rule_exts != ['*'] else 'any'} matching path pattern '*{rule_pattern}*'")

            # Use fnmatch for flexible pattern matching against relative paths
            # Wrap the pattern to check if the rule pattern exists anywhere in the path
            pattern_to_match = f"*{rule_pattern}*"

            files_to_check = relative_file_paths
            # If rule has specific extensions, filter the files to check first
            if rule_exts != ['*']:
                # Match against suffix (e.g., '.py')
                dot_exts = {f".{e}" for e in rule_exts}
                files_to_check = {
                     rel_path for rel_path in relative_file_paths
                     if Path(rel_path).suffix.lower() in dot_exts
                }

            # Apply fnmatch to the filtered relative paths
            matched_by_rule = {
                rel_path for rel_path in files_to_check
                if fnmatch.fnmatch(rel_path, pattern_to_match)
            }

            # Add the corresponding absolute paths to the excluded set
            for rel_path in matched_by_rule:
                if rel_path in relative_files_map:
                    excluded_files.add(relative_files_map[rel_path])
                    # print(f"    Excluding: {relative_files_map[rel_path]}") # Verbose logging

    print(f"Excluded {len(excluded_files)} files.")
    final_files = sorted(list(all_found_files - excluded_files))

    # Determine actual extensions present in the final list (lowercase, with dot)
    final_extensions = {Path(f).suffix.lower() for f in final_files if Path(f).suffix}

    return final_files, final_extensions
```

### src/pilot_rules/collector/reporting.py

- **Lines**: 281
- **Size**: 14.45 KB

**Content**:
```py
# src/pilot_rules/collector/reporting.py
import os
import datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Import functions from sibling modules
from .utils import get_file_metadata
from .analysis import extract_python_components # Import needed analysis functions


# --- Folder Tree Generation ---
# (generate_folder_tree function remains the same as the previous version)
def generate_folder_tree(root_folder_path: Path, included_files: List[str]) -> str:
    """Generate an ASCII folder tree representation for included files relative to a root."""
    tree_lines: List[str] = []
    included_files_set = {Path(f).resolve() for f in included_files} # Absolute paths

    # Store relative paths from the root_folder_path for display and structure building
    # We only include paths *under* the specified root_folder_path in the tree display
    included_relative_paths: Dict[Path, bool] = {} # Map relative path -> is_file
    all_parent_dirs: Set[Path] = set() # Set of relative directory paths

    for abs_path in included_files_set:
         try:
             rel_path = abs_path.relative_to(root_folder_path)
             included_relative_paths[rel_path] = True # Mark as file
             # Add all parent directories of this file
             parent = rel_path.parent
             while parent != Path('.'): # Stop before adding '.' itself
                  if parent not in included_relative_paths: # Avoid marking parent as file if dir listed later
                     included_relative_paths[parent] = False # Mark as directory
                  all_parent_dirs.add(parent)
                  parent = parent.parent
         except ValueError:
             # File is not under the root_folder_path, skip it in this tree view
             continue

    # Combine files and their necessary parent directories
    sorted_paths = sorted(included_relative_paths.keys(), key=lambda p: p.parts)

    # --- Tree building logic ---
    # Based on relative paths and depth
    tree_lines.append(f"{root_folder_path.name}/") # Start with the root dir name

    entries_by_parent: Dict[Path, List[Tuple[Path, bool]]] = {} # parent -> list of (child, is_file)
    for rel_path, is_file in included_relative_paths.items():
        parent = rel_path.parent
        if parent not in entries_by_parent:
             entries_by_parent[parent] = []
        entries_by_parent[parent].append((rel_path, is_file))

    # Sort children within each parent directory
    for parent in entries_by_parent:
         entries_by_parent[parent].sort(key=lambda item: (not item[1], item[0].parts)) # Dirs first, then alpha

    processed_paths = set() # To avoid duplicates if a dir is both parent and included

    def build_tree_recursive(parent_rel_path: Path, prefix: str):
        if parent_rel_path not in entries_by_parent:
            return

        children = entries_by_parent[parent_rel_path]
        for i, (child_rel_path, is_file) in enumerate(children):
            if child_rel_path in processed_paths:
                 continue

            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            entry_name = child_rel_path.name
            display_name = f"{entry_name}{'' if is_file else '/'}"
            tree_lines.append(f"{prefix}{connector}{display_name}")
            processed_paths.add(child_rel_path)

            if not is_file: # If it's a directory, recurse
                 new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                 build_tree_recursive(child_rel_path, new_prefix)

    # Start recursion from the root ('.') relative path
    build_tree_recursive(Path('.'), "")

    # Join lines, ensuring the root is handled correctly if empty
    if len(tree_lines) == 1 and not included_relative_paths: # Only root line, no files/dirs under it
         tree_lines[0] = f"└── {root_folder_path.name}/" # Adjust prefix for empty tree

    return "\n".join(tree_lines)


# --- Markdown Generation ---
def generate_markdown(
    files: List[str], # List of absolute paths
    analyzed_extensions: Set[str], # Set of actual extensions found (e.g., '.py', '.js')
    dependencies: Dict[str, Set[str]], # Python dependencies
    patterns: Dict[str, Any], # Detected patterns
    key_files: List[str], # List of absolute paths for key files
    output_path: Path,
    root_folder_display: str = "." # How to display the root in summary/tree
) -> None:
    """Generate a comprehensive markdown document about the codebase."""
    print(f"Generating Markdown report at '{output_path}'...")
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    report_base_path = Path.cwd() # Use CWD as the base for relative paths in the report

    has_python_files = '.py' in analyzed_extensions

    with open(output_path, "w", encoding="utf-8") as md_file:
        # --- Header ---
        md_file.write(f"# Code Repository Analysis\n\n")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # ms precision
        md_file.write(f"Generated on {timestamp}\n\n")

        # --- Repository Summary ---
        md_file.write("## Repository Summary\n\n")
        ext_list_str = ", ".join(sorted(list(analyzed_extensions))) if analyzed_extensions else "N/A"
        md_file.write(f"- **Extensions analyzed**: `{ext_list_str}`\n")
        md_file.write(f"- **Number of files analyzed**: {len(files)}\n")
        md_file.write(f"- **Analysis Root (for display)**: `{root_folder_display}`\n") # Indicate the main perspective

        total_lines = 0
        if files:
             try:
                 total_lines = sum(get_file_metadata(f).get("line_count", 0) for f in files)
             except Exception as e:
                 print(f"Warning: Could not calculate total lines accurately: {e}")
                 total_lines = "N/A"
        else:
             total_lines = 0
        md_file.write(f"- **Total lines of code (approx)**: {total_lines}\n\n")

        # --- Project Structure ---
        md_file.write("## Project Structure (Relative View)\n\n")
        md_file.write("```\n")
        try:
             root_for_tree = Path(root_folder_display).resolve()
             if root_for_tree.is_dir():
                 md_file.write(generate_folder_tree(root_for_tree, files))
             else:
                  print(f"Warning: Display root '{root_folder_display}' not found or not a directory, using CWD for tree.")
                  md_file.write(generate_folder_tree(report_base_path, files))
        except Exception as tree_err:
             print(f"Error generating folder tree: {tree_err}")
             md_file.write(f"Error generating folder tree: {tree_err}")
        md_file.write("\n```\n\n")

        # --- Key Files Section ---
        md_file.write("## Key Files\n\n")
        if key_files:
             md_file.write("These files appear central based on dependencies, naming, and size:\n\n")
             for file_abs_path in key_files:
                  try:
                      rel_path = str(Path(file_abs_path).relative_to(report_base_path))
                  except ValueError:
                      rel_path = file_abs_path # Fallback to absolute if not relative

                  md_file.write(f"### {rel_path}\n\n")
                  metadata = get_file_metadata(file_abs_path)
                  md_file.write(f"- **Lines**: {metadata.get('line_count', 'N/A')}\n")
                  md_file.write(f"- **Size**: {metadata.get('size_bytes', 0) / 1024:.2f} KB\n")
                  md_file.write(f"- **Last modified**: {metadata.get('last_modified', 'Unknown')}\n")

                  # Dependency info (Python only)
                  if has_python_files and file_abs_path in dependencies:
                      dependent_files_abs = {
                          f for f, deps in dependencies.items() if file_abs_path in deps
                      }
                      if dependent_files_abs:
                          md_file.write(f"- **Used by**: {len(dependent_files_abs)} other analyzed Python file(s)\n")

                  # Python component analysis
                  if file_abs_path.lower().endswith('.py'):
                       components = extract_python_components(file_abs_path) # Use imported function
                       if components.get("docstring"):
                           docstring_summary = (components["docstring"].strip().split('\n', 1)[0])[:150]
                           md_file.write(f"\n**Description**: {docstring_summary}{'...' if len(components['docstring']) > 150 else ''}\n")
                       if components.get("classes"):
                           md_file.write("\n**Classes**:\n")
                           for cls in components["classes"][:5]:
                               md_file.write(f"- `{cls['name']}` ({len(cls['methods'])} methods)\n")
                           if len(components["classes"]) > 5: md_file.write("- ... (and more)\n")
                       if components.get("functions"):
                           md_file.write("\n**Functions**:\n")
                           for func in components["functions"][:5]:
                               md_file.write(f"- `{func['name']}(...)`\n")
                           if len(components["functions"]) > 5: md_file.write("- ... (and more)\n")

                  # ==================================
                  # --- Include FULL File Content ---
                  md_file.write("\n**Content**:\n") # Changed from "Content Snippet"
                  file_ext = Path(file_abs_path).suffix.lower()
                  lang_hint = file_ext.lstrip('.') if file_ext else ""
                  md_file.write(f"```{lang_hint}\n")
                  try:
                      with open(file_abs_path, "r", encoding="utf-8", errors='ignore') as code_file:
                          # Read the entire file content
                          full_content = code_file.read()
                          md_file.write(full_content)
                          # Ensure a newline at the end of the code block if file doesn't have one
                          if not full_content.endswith("\n"):
                              md_file.write("\n")
                  except Exception as e:
                      md_file.write(f"Error reading file content: {str(e)}\n")
                  md_file.write("```\n\n")
                  # ==================================

        else:
             md_file.write("No key files identified based on current criteria.\n\n")


        # --- Design Patterns Section ---
        # (This section remains the same - it only lists file paths)
        if patterns:
             md_file.write("## Design Patterns (Python Heuristics)\n\n")
             md_file.write("Potential patterns identified based on naming and structure:\n\n")
             for pattern_name, files_or_dict in patterns.items():
                 title = pattern_name.replace('_', ' ').title()
                 if isinstance(files_or_dict, list) and files_or_dict:
                     md_file.write(f"### {title} Pattern\n\n")
                     for f_abs in files_or_dict[:10]:
                          try: rel_p = str(Path(f_abs).relative_to(report_base_path))
                          except ValueError: rel_p = f_abs
                          md_file.write(f"- `{rel_p}`\n")
                     if len(files_or_dict) > 10: md_file.write("- ... (and more)\n")
                     md_file.write("\n")
                 elif isinstance(files_or_dict, dict): # e.g., MVC
                     has_content = any(sublist for sublist in files_or_dict.values())
                     if has_content:
                          md_file.write(f"### {title}\n\n")
                          for subpattern, subfiles in files_or_dict.items():
                              if subfiles:
                                  md_file.write(f"**{subpattern.title()}**:\n")
                                  for f_abs in subfiles[:5]:
                                       try: rel_p = str(Path(f_abs).relative_to(report_base_path))
                                       except ValueError: rel_p = f_abs
                                       md_file.write(f"- `{rel_p}`\n")
                                  if len(subfiles) > 5: md_file.write("  - ... (and more)\n")
                                  md_file.write("\n")
             md_file.write("\n")
        elif has_python_files:
             md_file.write("## Design Patterns (Python Heuristics)\n\n")
             md_file.write("No common design patterns identified based on current heuristics.\n\n")


        # --- All Other Files Section ---
        md_file.write("## All Analyzed Files (excluding key files)\n\n")
        other_files = [f for f in files if f not in key_files]

        if other_files:
             for file_abs_path in other_files:
                  try: rel_path = str(Path(file_abs_path).relative_to(report_base_path))
                  except ValueError: rel_path = file_abs_path

                  md_file.write(f"### {rel_path}\n\n")
                  metadata = get_file_metadata(file_abs_path)
                  md_file.write(f"- **Lines**: {metadata.get('line_count', 'N/A')}\n")
                  md_file.write(f"- **Size**: {metadata.get('size_bytes', 0) / 1024:.2f} KB\n")

                  # ==================================
                  # --- Include FULL File Content ---
                  md_file.write("\n**Content**:\n") # Changed from "Content Snippet"
                  file_ext = Path(file_abs_path).suffix.lower()
                  lang_hint = file_ext.lstrip('.') if file_ext else ""
                  md_file.write(f"```{lang_hint}\n")
                  try:
                      with open(file_abs_path, "r", encoding="utf-8", errors='ignore') as code_file:
                          # Read the entire file content
                          full_content = code_file.read()
                          md_file.write(full_content)
                          # Ensure a newline at the end of the code block if file doesn't have one
                          if not full_content.endswith("\n"):
                               md_file.write("\n")
                  except Exception as e:
                      md_file.write(f"Error reading file content: {str(e)}\n")
                  md_file.write("```\n\n")
                  # ==================================
        elif key_files:
             md_file.write("All analyzed files were identified as key files and detailed above.\n\n")
        else:
            md_file.write("No files were found matching the specified criteria.\n\n")

    print(f"Markdown report generated successfully at '{output_path}'")
```

### src/pilot_rules/collector/utils.py

- **Lines**: 42
- **Size**: 1.67 KB

**Content**:
```py
# src/pilot_rules/collector/utils.py
import os
import datetime
from pathlib import Path
from typing import Dict, Any

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from a file."""
    metadata = {
        "path": file_path,
        "size_bytes": 0,
        "line_count": 0,
        "last_modified": "Unknown",
        "created": "Unknown",
    }

    try:
        p = Path(file_path)
        stats = p.stat()
        metadata["size_bytes"] = stats.st_size
        metadata["last_modified"] = datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        # ctime is platform dependent (creation on Windows, metadata change on Unix)
        # Use mtime as a reliable fallback for "created" if ctime is older than mtime
        ctime = stats.st_ctime
        mtime = stats.st_mtime
        best_ctime = ctime if ctime <= mtime else mtime # Heuristic
        metadata["created"] = datetime.datetime.fromtimestamp(best_ctime).strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Attempt to read as text, fallback for binary or encoding issues
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                metadata["line_count"] = len(content.splitlines())
        except (OSError, UnicodeDecodeError) as read_err:
            # Handle cases where reading might fail (binary file, permissions etc.)
            print(f"Warning: Could not read content/count lines for {file_path}: {read_err}")
            metadata["line_count"] = 0 # Indicate unreadable or binary

    except Exception as e:
        print(f"Warning: Could not get complete metadata for {file_path}: {e}")

    return metadata
```

