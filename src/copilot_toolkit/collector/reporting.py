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