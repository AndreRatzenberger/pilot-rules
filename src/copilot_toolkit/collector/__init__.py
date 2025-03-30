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