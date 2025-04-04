#!/usr/bin/env python3
"""
Code Repository Analyzer

Generates a comprehensive Markdown document of a code repository,
optimized for LLM consumption and understanding. Handles multiple file
types, exclusions, and configuration files.
"""

import os
import sys
import datetime
import ast
import fnmatch  # For wildcard path matching
import tomli  # For reading config file
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
        metadata["last_modified"] = datetime.datetime.fromtimestamp(
            stats.st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")
        # ctime is platform dependent (creation on Windows, metadata change on Unix)
        # Use mtime as a reliable fallback for "created" if ctime is older than mtime
        ctime = stats.st_ctime
        mtime = stats.st_mtime
        best_ctime = ctime if ctime <= mtime else mtime  # Heuristic
        metadata["created"] = datetime.datetime.fromtimestamp(best_ctime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                metadata["line_count"] = len(content.splitlines())
        except Exception as read_err:
            print(
                f"Warning: Could not read content/count lines for {file_path}: {read_err}"
            )
            metadata["line_count"] = 0  # Indicate unreadable/binary?

    except Exception as e:
        print(f"Warning: Could not get complete metadata for {file_path}: {e}")

    return metadata


# Keep extract_python_components as is, but call only for .py files
def extract_python_components(file_path: str) -> Dict[str, Any]:
    """Extract classes, functions, and imports from Python files."""
    # ... (existing implementation) ...
    components = {"classes": [], "functions": [], "imports": [], "docstring": None}

    # Ensure it's a python file before trying to parse
    if not file_path.lower().endswith(".py"):
        return components

    try:
        with open(file_path, "r", encoding="utf-8") as f:
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
                    "methods": [
                        m.name
                        for m in node.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                }
                components["classes"].append(class_info)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if it's truly top-level (not a method)
                # This check might be complex; let's list all for now and rely on context
                # if is_top_level_function(node, tree): # Simpler: List all functions found at top level of module body
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "args": [
                        arg.arg for arg in node.args.args if hasattr(arg, "arg")
                    ],  # Simplified arg extraction
                }
                components["functions"].append(func_info)

        # Extract all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    components["imports"].append(
                        alias.name
                    )  # Store the imported name/alias
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Handle relative imports representation
                relative_prefix = "." * node.level
                full_module_path = relative_prefix + module
                for alias in node.names:
                    # Store like 'from .module import name'
                    components["imports"].append(
                        f"from {full_module_path} import {alias.name}"
                    )

    except SyntaxError as e:
        print(
            f"Warning: Could not parse Python components in {file_path} due to SyntaxError: {e}"
        )
    except Exception as e:
        print(f"Warning: Could not parse Python components in {file_path}: {e}")

    return components


# Keep analyze_code_dependencies as is, but call only if .py files are included
def analyze_code_dependencies(files: List[str]) -> Dict[str, Set[str]]:
    """Analyze dependencies between Python files based on imports."""
    # ... (existing implementation) ...
    # Filter to only analyze python files within the provided list
    python_files = [f for f in files if f.lower().endswith(".py")]
    if not python_files:
        return {}  # No Python files to analyze

    dependencies = {file: set() for file in python_files}
    module_map = {}
    # Simplified module mapping - relies on relative paths from CWD or structured project
    project_root = Path.cwd()  # Assume CWD is project root for simplicity here

    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            # Attempt to create a module path relative to the project root
            relative_path = file_path.relative_to(project_root)
            parts = list(relative_path.parts)
            if parts[-1] == "__init__.py":
                parts.pop()  # Module is the directory name
                if not parts:
                    continue  # Skip root __init__.py mapping?
                module_name = ".".join(parts)
            elif parts[-1].endswith(".py"):
                parts[-1] = parts[-1][:-3]  # Remove .py
                module_name = ".".join(parts)
            else:
                continue  # Not a standard python module file

            if module_name:
                module_map[module_name] = str(
                    file_path
                )  # Map full module name to absolute path
                # Add shorter name if not conflicting? Risky. Stick to full paths.

        except ValueError:
            # File is outside the assumed project root, handle simple name mapping
            base_name = file_path.stem
            if base_name != "__init__" and base_name not in module_map:
                module_map[base_name] = str(file_path)

    # Now analyze imports in each Python file
    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
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
                                break  # Found the longest matching prefix

                elif isinstance(node, ast.ImportFrom):
                    level = node.level
                    module_base = node.module or ""

                    if level == 0:  # Absolute import
                        imported_module_str = module_base
                        for prefix in get_module_prefixes(imported_module_str):
                            if prefix in module_map:
                                dep_path = module_map[prefix]
                                if dep_path in python_files:
                                    dependencies[file_path_str].add(dep_path)
                                break
                    else:  # Relative import
                        current_dir = file_path.parent
                        # Go up 'level' directories (level=1 means current, level=2 means parent)
                        base_path = current_dir
                        for _ in range(level - 1):
                            base_path = base_path.parent

                        # Try to resolve the relative module path
                        relative_module_parts = module_base.split(".")
                        target_path = base_path
                        if module_base:  # If 'from .module import x'
                            for part in relative_module_parts:
                                target_path = target_path / part

                        # Now check potential file/package paths based on this target
                        # This simplified version might miss complex relative imports
                        # Check if target_path itself (as __init__.py) exists
                        init_py = (target_path / "__init__.py").resolve()
                        if init_py.exists() and str(init_py) in python_files:
                            dependencies[file_path_str].add(str(init_py))
                        # Check if target_path.py exists
                        module_py = target_path.with_suffix(".py").resolve()
                        if module_py.exists() and str(module_py) in python_files:
                            dependencies[file_path_str].add(str(module_py))

                        # We could also try resolving the imported names (node.names)
                        # but let's keep dependency analysis high-level for now.

        except SyntaxError as e:
            print(
                f"Warning: Skipping import analysis in {file_path_str} due to SyntaxError: {e}"
            )
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
    parts = module_name.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


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
            pass  # Or log a warning

    # We need all directories that contain included files or other included directories
    included_dirs_rel = set()
    for rel_path_str in included_relative_paths:
        p = Path(rel_path_str)
        parent = p.parent
        while str(parent) != ".":
            included_dirs_rel.add(str(parent))
            parent = parent.parent
        if (
            p.is_dir()
        ):  # If the path itself is a dir (though included_files should be files)
            included_dirs_rel.add(str(p))

    processed_dirs = set()  # Avoid cycles and redundant processing

    def _generate_tree(current_dir_rel: str, prefix: str = ""):
        if current_dir_rel in processed_dirs:
            return
        processed_dirs.add(current_dir_rel)

        current_dir_abs = root_path / current_dir_rel
        dir_name = (
            current_dir_abs.name if current_dir_rel != "." else "."
        )  # Handle root display name

        # Add the current directory to the output using appropriate prefix (later)
        # For now, collect children first

        entries = []
        try:
            for item in current_dir_abs.iterdir():
                item_rel_str = str(item.resolve().relative_to(root_path))
                if item.is_dir():
                    # Include dir if it's explicitly in included_dirs_rel OR contains included items
                    if item_rel_str in included_dirs_rel or any(
                        f.startswith(item_rel_str + os.sep)
                        for f in included_relative_paths
                    ):
                        entries.append(
                            {"name": item.name, "path": item_rel_str, "is_dir": True}
                        )
                elif item.is_file():
                    if item_rel_str in included_relative_paths:
                        entries.append(
                            {"name": item.name, "path": item_rel_str, "is_dir": False}
                        )
        except (PermissionError, FileNotFoundError):
            pass  # Skip inaccessible directories

        # Sort entries: directories first, then files, alphabetically
        entries.sort(key=lambda x: (not x["is_dir"], x["name"]))

        # Now generate output for this level
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            tree_output.append(
                f"{prefix}{connector}{entry['name']}{'/' if entry['is_dir'] else ''}"
            )

            if entry["is_dir"]:
                new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                _generate_tree(entry["path"], new_prefix)

    # Start the recursion from the root directory representation "."
    tree_output.append(f"{root_folder}/")  # Start with the root folder itself
    _generate_tree(
        ".", prefix="│   "
    )  # Use an initial prefix assuming root is not last

    # Quick fix for root display if only root is passed
    if len(tree_output) == 1 and tree_output[0] == f"{root_folder}/":
        # If no children were added, just show the root
        tree_output[0] = f"└── {root_folder}/"  # Adjust prefix if it's the only thing
        # If files are directly in root, _generate_tree should handle them

    # Refine prefix for the first level items if they exist
    if len(tree_output) > 1:
        tree_output[0] = (
            f"└── {root_folder}/"  # Assume root is the end of its parent list
        )
        # Need to adjust prefix logic inside _generate_tree or post-process
        # Let's stick to the simpler structure for now. ASCII trees can be tricky.

    return "\n".join(tree_output)  # Return combined string


# Keep get_common_patterns as is, but call only if .py files are included
def get_common_patterns(files: List[str]) -> Dict[str, Any]:
    """Identify common design patterns in the codebase (Python focused)."""
    # ... (existing implementation) ...
    patterns: Dict[str, Union[List[str], Dict[str, List[str]]]] = {
        "singleton": [],
        "factory": [],
        "observer": [],
        "decorator": [],
        "mvc_components": {"models": [], "views": [], "controllers": []},
    }
    python_files = [f for f in files if f.lower().endswith(".py")]

    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lower()  # Read content once
                file_basename_lower = os.path.basename(file_path).lower()

            # Basic keyword/structure checks (can be improved)
            # Check for singleton pattern (simple heuristic)
            if ("instance = none" in content or "_instance = none" in content) and (
                "__new__" in content or " getinstance " in content
            ):
                patterns["singleton"].append(file_path)

            # Check for factory pattern
            if (
                "factory" in file_basename_lower
                or ("def create_" in content and " return " in content)
                or ("def make_" in content and " return " in content)
            ):
                patterns["factory"].append(file_path)

            # Check for observer pattern
            if ("observer" in content or "listener" in content) and (
                "notify" in content
                or "update" in content
                or "addeventlistener" in content
                or "subscribe" in content
            ):
                patterns["observer"].append(file_path)

            # Check for decorator pattern (presence of @ syntax handled by Python itself)
            # Look for common decorator definition patterns
            if "def wrapper(" in content and "return wrapper" in content:
                patterns["decorator"].append(file_path)  # Might be too broad

            # Check for MVC components based on naming conventions
            if "model" in file_basename_lower or "models" in file_path.lower().split(
                os.sep
            ):
                patterns["mvc_components"]["models"].append(file_path)
            if (
                "view" in file_basename_lower
                or "views" in file_path.lower().split(os.sep)
                or "template" in file_basename_lower
            ):
                patterns["mvc_components"]["views"].append(file_path)
            if (
                "controller" in file_basename_lower
                or "controllers" in file_path.lower().split(os.sep)
                or "handler" in file_basename_lower
                or "routes" in file_basename_lower
            ):
                patterns["mvc_components"]["controllers"].append(file_path)

        except Exception:
            # print(f"Warning: Could not analyze patterns in {file_path}: {e}") # Can be noisy
            continue  # Ignore files that can't be read or processed

    # --- Clean up empty categories ---
    # Create a new dict to avoid modifying while iterating
    cleaned_patterns: Dict[str, Any] = {}
    for key, value in patterns.items():
        if isinstance(value, list):
            if value:  # Keep if list is not empty
                cleaned_patterns[key] = value
        elif isinstance(value, dict):
            # For nested dicts like MVC, keep the parent key if any child list is non-empty
            non_empty_sub_patterns = {
                subkey: sublist
                for subkey, sublist in value.items()
                if isinstance(sublist, list) and sublist
            }
            if non_empty_sub_patterns:  # Keep if dict has non-empty lists
                cleaned_patterns[key] = non_empty_sub_patterns

    return cleaned_patterns


# Keep find_key_files as is, but consider its Python focus
def find_key_files(files: List[str], dependencies: Dict[str, Set[str]]) -> List[str]:
    """Identify key files based on dependencies and naming conventions (Python focused)."""
    # ... (existing implementation) ...
    # Initialize scores for each file
    scores = {
        file: 0.0 for file in files
    }  # Use float for potentially fractional scores

    # Track how many files depend on each file (dependents) - Python only for now
    python_files = {f for f in files if f.lower().endswith(".py")}
    dependent_count = {file: 0 for file in python_files}
    for (
        file,
        deps,
    ) in dependencies.items():  # dependencies should already be Python-only
        if file not in python_files:
            continue  # Ensure source file is Python
        for dep in deps:
            if dep in dependent_count:  # Target file must also be Python
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
        if any(
            core_name in base_name
            for core_name in [
                "main.",
                "app.",
                "core.",
                "__init__.py",
                "cli.",
                "server.",
                "manage.py",
            ]
        ):
            scores[file] += 5.0
        elif base_name == "settings.py" or base_name == "config.py":
            scores[file] += 4.0
        elif base_name.startswith("test_"):
            scores[file] -= (
                1.0  # Lower score for test files unless highly depended upon
            )

        # Configuration and settings
        if any(
            config_name in base_name
            for config_name in ["config", "settings", "constant", "conf."]
        ):
            scores[file] += 3.0

        # Base classes and abstract components
        if any(
            base_name_part in base_name
            for base_name_part in ["base.", "abstract", "interface", "factory"]
        ):
            scores[file] += 2.0

        # Utilities and helpers
        if any(
            util_name in base_name
            for util_name in ["util", "helper", "common", "tool", "shared"]
        ):
            scores[file] += 1.0

        # Score directories by importance
        if "src" == parent_dir_name:  # Direct child of src
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
                scores[file] += min(
                    line_count / 100.0, 3.0
                )  # Cap at 3 points, less sensitive than /50

            # Bonus for significant files
            if line_count > 300:
                scores[file] += 1.0
            elif line_count < 10:
                scores[file] -= 0.5  # Penalize very small files slightly
        except Exception:
            pass  # Ignore if metadata fails

        # Score by extension - Python files are often central in Python projects
        if file.lower().endswith(".py"):
            scores[file] += 1.0
        elif file.lower().endswith((".md", ".txt", ".rst")):
            scores[file] += 0.1  # Documentation is useful context
        elif file.lower().endswith((".yaml", ".yml", ".json", ".toml")):
            scores[file] += 0.5  # Config files can be important

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
    num_key_files = max(
        min(len(files) // 5, 20), min(5, len(files))
    )  # 20% or 5, capped at 20
    return key_files[:num_key_files]


# --- New/Modified Core Logic ---


def parse_include_exclude_args(args: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Parses include/exclude arguments like 'py,js:src' or '*:temp'."""
    parsed = []
    if not args:
        return parsed

    for arg in args:
        if ":" not in arg:
            raise ValueError(
                f"Invalid include/exclude format: '{arg}'. Expected 'EXTS:PATH' or '*:PATTERN'."
            )

        exts_str, path_pattern = arg.split(":", 1)
        extensions = [ext.strip().lower() for ext in exts_str.split(",") if ext.strip()]

        # Normalize path pattern
        path_pattern = Path(
            path_pattern
        ).as_posix()  # Use forward slashes for consistency

        parsed.append(
            {
                "extensions": extensions,  # List of extensions, or ['*']
                "pattern": path_pattern,  # Path or pattern string
            }
        )
    return parsed


def collect_files(
    sources: List[Dict[str, Any]], excludes: List[Dict[str, Any]]
) -> Tuple[List[str], Set[str]]:
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
    project_root = Path.cwd().resolve()  # Use CWD as the reference point

    for source in sources:
        root_path = Path(source["root"]).resolve()
        extensions = source["extensions"]
        print(
            f"  Scanning in: '{root_path}' for extensions: {extensions if extensions != ['*'] else 'all'}"
        )

        # Decide which glob pattern to use
        glob_patterns = []
        if extensions == ["*"]:
            # Glob all files recursively
            glob_patterns.append(str(root_path / "**" / "*"))
        else:
            for ext in extensions:
                # Ensure extension starts with a dot if not already present
                dot_ext = f".{ext}" if not ext.startswith(".") else ext
                glob_patterns.append(str(root_path / "**" / f"*{dot_ext}"))
                all_extensions.add(dot_ext)  # Track requested extensions

        found_in_source = set()
        for pattern in glob_patterns:
            try:
                # Use pathlib's rglob for recursive search
                # Need to handle the non-extension specific case carefully
                if pattern.endswith("*"):  # Case for '*' extension
                    for item in root_path.rglob("*"):
                        if item.is_file():
                            found_in_source.add(str(item.resolve()))
                else:  # Specific extension
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
            if Path(f)
            .resolve()
            .is_relative_to(project_root)  # Only exclude relative to project root
        }
        relative_file_paths = set(relative_files_map.keys())

        for rule in excludes:
            rule_exts = rule["extensions"]
            rule_pattern = rule["pattern"]
            print(
                f"  Excluding: extensions {rule_exts if rule_exts != ['*'] else 'any'} matching path pattern '*{rule_pattern}*'"
            )  # Match anywhere in path

            # Use fnmatch for pattern matching against relative paths
            pattern_to_match = f"*{rule_pattern}*"  # Wrap pattern for contains check

            files_to_check = relative_file_paths
            # If rule has specific extensions, filter the files to check first
            if rule_exts != ["*"]:
                dot_exts = {f".{e}" if not e.startswith(".") else e for e in rule_exts}
                files_to_check = {
                    rel_path
                    for rel_path in relative_file_paths
                    if Path(rel_path).suffix.lower() in dot_exts
                }

            # Apply fnmatch
            matched_by_rule = {
                rel_path
                for rel_path in files_to_check
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
    analyzed_extensions: Set[str],  # Use the actual extensions found
    output_path: str,
    root_folder_display: str = ".",  # How to display the root in summary/tree
) -> None:
    """Generate a comprehensive markdown document about the codebase."""
    print(f"Generating Markdown report at '{output_path}'...")
    # Only run Python-specific analysis if .py files are present
    has_python_files = any(f.lower().endswith(".py") for f in files)
    dependencies = {}
    patterns = {}
    if has_python_files:
        print("Analyzing Python dependencies...")
        dependencies = analyze_code_dependencies(
            files
        )  # Pass all files, it filters internally
        print("Identifying common patterns...")
        patterns = get_common_patterns(files)  # Pass all files, it filters internally
    else:
        print("Skipping Python-specific analysis (no .py files found).")

    print("Finding key files...")
    key_files = find_key_files(
        files, dependencies
    )  # Pass all files, scorer handles types

    # Use the directory of the output file as the base for relative paths if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    with open(output_path, "w", encoding="utf-8") as md_file:
        # Write header
        md_file.write("# Code Repository Analysis\n\n")
        # Format timestamp for clarity
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        md_file.write(f"Generated on {timestamp}\n\n")

        # Write repository summary
        md_file.write("## Repository Summary\n\n")
        ext_list_str = (
            ", ".join(sorted(list(analyzed_extensions)))
            if analyzed_extensions
            else "N/A"
        )
        md_file.write(f"- **Extensions analyzed**: `{ext_list_str}`\n")
        md_file.write(f"- **Number of files analyzed**: {len(files)}\n")
        # Decide on root folder representation - maybe list all roots from sources?
        # For now, keep it simple.
        md_file.write(
            f"- **Primary analysis root (for tree)**: `{root_folder_display}`\n"
        )

        total_lines = 0
        if files:
            try:
                # Calculate total lines safely
                total_lines = sum(
                    get_file_metadata(f).get("line_count", 0) for f in files
                )
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
                md_file.write(
                    f"Cannot generate tree: '{root_folder_display}' is not a valid directory."
                )

        except Exception as tree_err:
            md_file.write(f"Error generating folder tree: {tree_err}")
        md_file.write("\n```\n\n")

        # --- Key Files Section ---
        md_file.write("## Key Files\n\n")
        if key_files:
            md_file.write(
                "These files appear central based on dependencies, naming, and size:\n\n"
            )
            # Use CWD as the base for relative paths in the report for consistency
            report_base_path = Path.cwd()
            for file_abs_path in key_files:
                try:
                    rel_path = str(Path(file_abs_path).relative_to(report_base_path))
                except ValueError:
                    rel_path = (
                        file_abs_path  # Fallback to absolute if not relative to CWD
                    )

                md_file.write(f"### {rel_path}\n\n")

                metadata = get_file_metadata(file_abs_path)
                md_file.write(f"- **Lines**: {metadata.get('line_count', 'N/A')}\n")
                md_file.write(
                    f"- **Size**: {metadata.get('size_bytes', 0) / 1024:.2f} KB\n"
                )
                md_file.write(
                    f"- **Last modified**: {metadata.get('last_modified', 'Unknown')}\n"
                )

                # Dependency info (Python only)
                dependent_files_rel = []
                if (
                    has_python_files and file_abs_path in dependencies
                ):  # Check if file itself has deps analyzed
                    # Find which files depend *on this* key file
                    dependent_files_abs = [
                        f for f, deps in dependencies.items() if file_abs_path in deps
                    ]
                    dependent_files_rel = []
                    for dep_abs in dependent_files_abs:
                        try:
                            dependent_files_rel.append(
                                str(Path(dep_abs).relative_to(report_base_path))
                            )
                        except ValueError:
                            dependent_files_rel.append(dep_abs)  # Fallback

                if dependent_files_rel:
                    md_file.write(
                        f"- **Used by**: {len(dependent_files_rel)} other Python file(s)\n"
                    )  # maybe list top 3? e.g. `[:3]`

                # Python component analysis
                if file_abs_path.lower().endswith(".py"):
                    components = extract_python_components(file_abs_path)
                    if components.get("docstring"):
                        # Limit docstring length?
                        docstring_summary = (
                            components["docstring"].strip().split("\n")[0]
                        )[:150]  # First line, max 150 chars
                        md_file.write(f"\n**Description**: {docstring_summary}...\n")

                    if components.get("classes"):
                        md_file.write("\n**Classes**:\n")
                        for cls in components["classes"][:5]:  # Limit displayed classes
                            md_file.write(
                                f"- `{cls['name']}` ({len(cls['methods'])} methods)\n"
                            )
                        if len(components["classes"]) > 5:
                            md_file.write("- ... (and more)\n")

                    if components.get("functions"):
                        md_file.write("\n**Functions**:\n")
                        for func in components["functions"][
                            :5
                        ]:  # Limit displayed functions
                            md_file.write(
                                f"- `{func['name']}(...)`\n"
                            )  # Simplified signature
                        if len(components["functions"]) > 5:
                            md_file.write("- ... (and more)\n")

                # File Content
                md_file.write(
                    "\n**Content Snippet**:\n"
                )  # Changed from "Content" to avoid huge files
                file_ext = Path(file_abs_path).suffix
                lang_hint = file_ext.lstrip(".") if file_ext else ""
                md_file.write(f"```{lang_hint}\n")

                try:
                    with open(
                        file_abs_path, "r", encoding="utf-8", errors="ignore"
                    ) as code_file:
                        # Show first N lines (e.g., 50) as a snippet
                        snippet_lines = []
                        for i, line in enumerate(code_file):
                            if i >= 50:
                                snippet_lines.append("...")
                                break
                            snippet_lines.append(
                                line.rstrip()
                            )  # Remove trailing newline for cleaner output
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
            md_file.write(
                "Potential patterns identified based on naming and structure:\n\n"
            )
            report_base_path = Path.cwd()  # Base for relative paths

            for pattern_name, files_or_dict in patterns.items():
                title = pattern_name.replace("_", " ").title()
                if isinstance(files_or_dict, list) and files_or_dict:
                    md_file.write(f"### {title} Pattern\n\n")
                    for f_abs in files_or_dict[
                        :10
                    ]:  # Limit displayed files per pattern
                        try:
                            rel_p = str(Path(f_abs).relative_to(report_base_path))
                        except ValueError:
                            rel_p = f_abs
                        md_file.write(f"- `{rel_p}`\n")
                    if len(files_or_dict) > 10:
                        md_file.write("- ... (and more)\n")
                    md_file.write("\n")
                elif isinstance(files_or_dict, dict):  # e.g., MVC
                    has_content = any(sublist for sublist in files_or_dict.values())
                    if has_content:
                        md_file.write(f"### {title}\n\n")
                        for subpattern, subfiles in files_or_dict.items():
                            if subfiles:
                                md_file.write(f"**{subpattern.title()}**:\n")
                                for f_abs in subfiles[:5]:  # Limit sub-pattern files
                                    try:
                                        rel_p = str(
                                            Path(f_abs).relative_to(report_base_path)
                                        )
                                    except ValueError:
                                        rel_p = f_abs
                                    md_file.write(f"- `{rel_p}`\n")
                                if len(subfiles) > 5:
                                    md_file.write("  - ... (and more)\n")
                                md_file.write("\n")
            md_file.write("\n")
        elif has_python_files:
            md_file.write("## Design Patterns (Python Heuristics)\n\n")
            md_file.write(
                "No common design patterns identified based on current heuristics.\n\n"
            )

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
                md_file.write(
                    f"- **Size**: {metadata.get('size_bytes', 0) / 1024:.2f} KB\n"
                )
                md_file.write(
                    f"- **Last modified**: {metadata.get('last_modified', 'Unknown')}\n\n"
                )

                # Provide a content snippet for other files too
                md_file.write("**Content Snippet**:\n")
                file_ext = Path(file_abs_path).suffix
                lang_hint = file_ext.lstrip(".") if file_ext else ""
                md_file.write(f"```{lang_hint}\n")
                try:
                    with open(
                        file_abs_path, "r", encoding="utf-8", errors="ignore"
                    ) as code_file:
                        snippet_lines = []
                        for i, line in enumerate(code_file):
                            if i >= 30:  # Shorter snippet for non-key files
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
            md_file.write(
                "All analyzed files are listed in the 'Key Files' section.\n\n"
            )
        else:
            md_file.write("No files were found matching the specified criteria.\n\n")

    print(f"Markdown report generated successfully at '{output_path}'")


def run_collection(
    include_args: Optional[List[str]],
    exclude_args: Optional[List[str]],
    output_arg: str,
    config_arg: Optional[str],
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
                raw_sources = config_data.get("source", [])
                if not isinstance(raw_sources, list):
                    raise ValueError(
                        "Invalid config: 'source' must be an array of tables."
                    )

                for src_table in raw_sources:
                    exts = src_table.get(
                        "exts", ["*"]
                    )  # Default to all if not specified
                    root = src_table.get("root", ".")
                    exclude_patterns = src_table.get(
                        "exclude", []
                    )  # Excludes within a source block

                    if not isinstance(exts, list) or not all(
                        isinstance(e, str) for e in exts
                    ):
                        raise ValueError(
                            f"Invalid config: 'exts' must be a list of strings in source root '{root}'"
                        )
                    if not isinstance(root, str):
                        raise ValueError(
                            "Invalid config: 'root' must be a string in source."
                        )
                    if not isinstance(exclude_patterns, list) or not all(
                        isinstance(p, str) for p in exclude_patterns
                    ):
                        raise ValueError(
                            f"Invalid config: 'exclude' must be a list of strings in source root '{root}'"
                        )

                    config_sources.append(
                        {
                            "root": Path(root).resolve(),  # Store resolved path
                            "extensions": [e.lower().lstrip(".") for e in exts],
                        }
                    )
                    # Add source-specific excludes to the global excludes list
                    # Assume format '*:<pattern>' for simplicity from config's exclude list
                    for pattern in exclude_patterns:
                        config_excludes.append(
                            {"extensions": ["*"], "pattern": Path(pattern).as_posix()}
                        )

                # Parse global output from config
                config_output = config_data.get("output")
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
        final_sources = [
            {"root": Path(inc["pattern"]).resolve(), "extensions": inc["extensions"]}
            for inc in cli_includes
        ]
    elif config_sources:
        print("Using include sources from configuration file.")
        final_sources = config_sources  # Already resolved paths
    else:
        print("No includes specified, defaulting to '.py' files in current directory.")
        final_sources = [{"root": Path(".").resolve(), "extensions": ["py"]}]

    # 4. Combine excludes: CLI appends to config excludes
    final_excludes = config_excludes + cli_excludes
    if final_excludes:
        print(f"Using {len(final_excludes)} exclusion rule(s).")

    # 5. Determine final output path: CLI > Config > Default
    final_output = cli_output if cli_output else config_output
    # Use default from argparse if cli_output is None/empty and config_output is None
    if not final_output:
        final_output = "repository_analysis.md"  # Re-apply default if needed

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
    generate_markdown(
        collected_files,
        actual_extensions,
        str(final_output_path),
        root_folder_display=".",
    )


# Keep the standalone execution part for testing/direct use if needed
if __name__ == "__main__":
    import argparse

    # This argparse is now only for *direct* execution of code_collector.py
    parser = argparse.ArgumentParser(
        description="Analyze code repository (Standalone Execution)"
    )
    parser.add_argument(
        "-i", "--include", action="append", help="Include spec 'EXTS:FOLDER'"
    )
    parser.add_argument(
        "-e", "--exclude", action="append", help="Exclude spec '*:PATTERN'"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="repository_analysis_standalone.md",
        help="Output markdown file",
    )
    parser.add_argument("--config", help="Path to TOML config file")

    args = parser.parse_args()

    try:
        run_collection(
            include_args=args.include,
            exclude_args=args.exclude,
            output_arg=args.output,
            config_arg=args.config,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
