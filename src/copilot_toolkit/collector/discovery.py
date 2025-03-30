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