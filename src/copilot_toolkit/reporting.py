# src/pilot_rules/collector/reporting.py
import datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Import functions from sibling modules
from .utils import (
    get_file_metadata,
    console,
    print_header,
    print_success,
    print_warning,
    print_error,
)
from .analysis import extract_python_components  # Import needed analysis functions
from ..model import Repository, ProjectFile, ProjectCodeFile


# --- Folder Tree Generation ---
def generate_folder_tree(root_folder_path: Path, included_files: List[str]) -> str:
    """Generate an ASCII folder tree representation for included files relative to a root."""
    tree_lines: List[str] = []
    included_files_set = {Path(f).resolve() for f in included_files}  # Absolute paths

    # Store relative paths from the root_folder_path for display and structure building
    # We only include paths *under* the specified root_folder_path in the tree display
    included_relative_paths: Dict[Path, bool] = {}  # Map relative path -> is_file
    all_parent_dirs: Set[Path] = set()  # Set of relative directory paths

    for abs_path in included_files_set:
        try:
            rel_path = abs_path.relative_to(root_folder_path)
            included_relative_paths[rel_path] = True  # Mark as file
            # Add all parent directories of this file
            parent = rel_path.parent
            while parent != Path("."):  # Stop before adding '.' itself
                if (
                    parent not in included_relative_paths
                ):  # Avoid marking parent as file if dir listed later
                    included_relative_paths[parent] = False  # Mark as directory
                all_parent_dirs.add(parent)
                parent = parent.parent
        except ValueError:
            # File is not under the root_folder_path, skip it in this tree view
            continue

    # Combine files and their necessary parent directories
    sorted_paths = sorted(included_relative_paths.keys(), key=lambda p: p.parts)

    # --- Tree building logic ---
    # Based on relative paths and depth
    tree_lines.append(f"{root_folder_path.name}/")  # Start with the root dir name

    entries_by_parent: Dict[
        Path, List[Tuple[Path, bool]]
    ] = {}  # parent -> list of (child, is_file)
    for rel_path, is_file in included_relative_paths.items():
        parent = rel_path.parent
        if parent not in entries_by_parent:
            entries_by_parent[parent] = []
        entries_by_parent[parent].append((rel_path, is_file))

    # Sort children within each parent directory
    for parent in entries_by_parent:
        entries_by_parent[parent].sort(
            key=lambda item: (not item[1], item[0].parts)
        )  # Dirs first, then alpha

    processed_paths = set()  # To avoid duplicates if a dir is both parent and included

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

            if not is_file:  # If it's a directory, recurse
                new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                build_tree_recursive(child_rel_path, new_prefix)

    # Start recursion from the root ('.') relative path
    build_tree_recursive(Path("."), "")

    # Join lines, ensuring the root is handled correctly if empty
    if (
        len(tree_lines) == 1 and not included_relative_paths
    ):  # Only root line, no files/dirs under it
        tree_lines[0] = f"└── {root_folder_path.name}/"  # Adjust prefix for empty tree

    return "\n".join(tree_lines)


# --- Repository Object Generation ---
def generate_repository(
    files: List[str],  # List of absolute paths
    analyzed_extensions: Set[str],  # Set of actual extensions found (e.g., '.py', '.js')
    dependencies: Dict[str, Set[str]],  # Python dependencies
    patterns: Dict[str, Any],  # Detected patterns
    key_files: List[str],  # List of absolute paths for key files
    repo_name: str = "Repository Analysis",
    root_folder_display: str = ".",  # How to display the root in summary/tree
) -> Repository:
    """Generate a Repository object with analyzed code structure and content."""
    print_header("Generating Repository Object", "green")
    report_base_path = Path.cwd()  # Use CWD as the base for relative paths in the report

    has_python_files = ".py" in analyzed_extensions

    # Generate statistics
    ext_list_str = ", ".join(sorted(list(analyzed_extensions))) if analyzed_extensions else "N/A"
    total_files = len(files)
    
    total_lines = 0
    if files:
        try:
            total_lines = sum(get_file_metadata(f).get("line_count", 0) for f in files)
        except Exception as e:
            print_warning(f"Could not calculate total lines accurately: {e}")
            total_lines = 0
    
    statistics = f"""
- Extensions analyzed: {ext_list_str}
- Number of files analyzed: {total_files}
- Total lines of code (approx): {total_lines}
"""

    # Process files to create ProjectFile objects
    project_files = []
    
    # First create a mapping of absolute paths to file_ids
    file_id_mapping = {}
    for i, file_abs_path in enumerate(files):
        try:
            rel_path = str(Path(file_abs_path).relative_to(report_base_path))
        except ValueError:
            rel_path = file_abs_path  # Fallback to absolute if not relative
        
        file_id = f"file_{i}"
        file_id_mapping[file_abs_path] = file_id
    
    # Now create ProjectFile objects with proper dependencies
    for file_abs_path in files:
        try:
            rel_path = str(Path(file_abs_path).relative_to(report_base_path))
        except ValueError:
            rel_path = file_abs_path  # Fallback to absolute if not relative
            
        metadata = get_file_metadata(file_abs_path)
        file_id = file_id_mapping[file_abs_path]
        
        try:
            with open(file_abs_path, "r", encoding="utf-8", errors="ignore") as code_file:
                content = code_file.read()
        except Exception as e:
            print_warning(f"Could not read file content for {rel_path}: {e}")
            content = f"Error reading file: {str(e)}"
        
        # Generate description based on file type
        description = f"File at {rel_path}"
        if file_abs_path.lower().endswith(".py"):
            components = extract_python_components(file_abs_path)
            if components.get("docstring"):
                docstring_summary = components["docstring"].strip().split("\n", 1)[0][:150]
                description = docstring_summary + ('...' if len(components["docstring"]) > 150 else '')
            
            # For Python files, create ProjectCodeFile with dependencies
            file_deps = []
            file_used_by = []
            
            # Find dependencies
            if has_python_files and file_abs_path in dependencies:
                file_deps = [file_id_mapping[dep] for dep in dependencies[file_abs_path] if dep in file_id_mapping]
                
                # Find files that depend on this file
                dependent_files_abs = {f for f, deps in dependencies.items() if file_abs_path in deps}
                file_used_by = [file_id_mapping[dep] for dep in dependent_files_abs if dep in file_id_mapping]
            
            project_file = ProjectCodeFile(
                file_id=file_id,
                description=description,
                file_path=rel_path,
                content=content,
                line_count=metadata.get('line_count', 0),
                dependencies=file_deps,
                used_by=file_used_by
            )
        else:
            # Regular ProjectFile for non-Python files
            project_file = ProjectFile(
                file_id=file_id,
                description=description,
                file_path=rel_path,
                content=content,
                line_count=metadata.get('line_count', 0)
            )
        
        project_files.append(project_file)
    
    # Create and return the Repository object
    repository = Repository(
        name=repo_name,
        statistics=statistics,
        project_files=project_files
    )
    
    print_success(f"Successfully generated Repository object with {len(project_files)} files")
    return repository 