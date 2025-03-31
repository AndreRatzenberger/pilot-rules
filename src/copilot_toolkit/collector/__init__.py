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
from .utils import console, print_header, print_subheader, print_success, print_warning, print_error, print_file_stats

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
        print_header("Code Collection Process", "magenta")
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
            print_warning("No files found matching the specified criteria.")
            # Generate an empty/minimal report? Or just exit?
            # Let generate_markdown handle the empty list for now.
        else:
            print_success(f"Found [bold green]{len(collected_files)}[/bold green] files to include in the report.")
            ext_list = ", ".join(sorted(list(actual_extensions)))
            console.print(f"File extensions found: [cyan]{ext_list}[/cyan]")
            
            # Display file statistics in a nice table
            print_file_stats(collected_files, "Collection Statistics")


        # 3. Perform Analysis (Conditional based on files found)
        dependencies = {}
        patterns = {}
        key_files = []

        # Only run Python-specific analysis if .py files are present
        has_python_files = '.py' in actual_extensions
        if has_python_files:
            print_subheader("Analyzing Python Dependencies", "blue")
            dependencies = analyze_code_dependencies(collected_files)
            print_subheader("Identifying Code Patterns", "blue")
            patterns = get_common_patterns(collected_files)
        else:
            print_warning("Skipping Python-specific analysis (no .py files found).")

        # Find key files (uses heuristics applicable to various file types)
        if collected_files:
             # Note: find_key_files now has its own print_subheader call
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
         print_error(f"Configuration Error: {e}", 1)
    except Exception as e:
         # Catch-all for unexpected errors during collection/analysis/reporting
         print_error(f"An unexpected error occurred: {e}", 1)
         import traceback
         traceback.print_exc()


# __all__ = ['run_collection'] # Optionally define the public API