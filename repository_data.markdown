# Repository Analysis

## Repository Statistics

- **Extensions analyzed**: .py
- **Number of files analyzed**: 14
- **Total lines of code (approx)**: 5519

## Project Files

### 1. src/copilot_toolkit/__init__.py

- **File ID**: file_0
- **Type**: Code File
- **Line Count**: 5
- **Description**: File at src/copilot_toolkit/__init__.py
- **Dependencies**:
  - file_10
- **Used By**: None

**Content**:
```
from .main import main as scaffolder_main


def main() -> None:
    scaffolder_main()

```

---

### 2. src/copilot_toolkit/agent.py

- **File ID**: file_1
- **Type**: Code File
- **Line Count**: 408
- **Description**: File at src/copilot_toolkit/agent.py
- **Dependencies**: None
- **Used By**: None

**Content**:
```
from pathlib import Path
from flock.core import FlockFactory, Flock
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from copilot_toolkit.model import OutputData, Project, TaskAndToDoItemList, Task, ToDoItem


# Create a console for rich output
console = Console()


def load_prompt(action: str, prompt_folder: str) -> str:
    """Load prompt from file."""
    prompt_path = f"{prompt_folder}/{action}.md"
    try:
        with open(prompt_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        console.print(f"[red]Error: Prompt file not found at '{prompt_path}'[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Error loading prompt from '{prompt_path}': {e}[/red]")
        raise


def extract_before_prompt(text: str) -> str:
    """
    Extract all text before '## Prompt' in a given string.
    
    Args:
        text: The input text to process
        
    Returns:
        The text content before '## Prompt' or an empty string if not found
    """
    if '## Prompt' in text:
        return text.split('## Prompt')[0].strip()
    return text.strip()


def extract_after_prompt(text: str) -> str:
    """
    Extract all text after '## Prompt' in a given string.
    
    Args:
        text: The input text to process
        
    Returns:
        The text content after '## Prompt' or an empty string if not found
    """
    if '## Prompt' in text:
        return text.split('## Prompt')[1].strip()
    return ""


def project_agent(
    action: str,
    user_instructions: str = "",
) -> Project:
    """
    Communicate with an LLM agent to perform a specified action.

    Args:
        action: The type of action to perform (e.g., "app", "specs")
        user_instructions: Additional instructions for the agent

    Returns:
        OutputData instance with the agent's response
    """

    MODEL = (
        "gemini/gemini-2.5-pro-exp-03-25"  # "groq/qwen-qwq-32b"    #"openai/gpt-4o" #
    )

    # load a file relative to the current file
    prompt_folder = Path(__file__).parent / "prompts"


    # Use a spinner for loading prompt files
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        load_task = progress.add_task("[blue]Loading prompt files...", total=None)

        try:
            
            prompt_file = load_prompt(action, prompt_folder)
            prompt_description = extract_before_prompt(prompt_file)
            prompt = extract_after_prompt(prompt_file)
      

            progress.update(
                load_task, description="[green]Prompts loaded successfully!"
            )
        except Exception as e:
            progress.update(load_task, description=f"[red]Error loading prompts: {e}")
            raise

     # Set up the agent with the prompt
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        setup_task = progress.add_task("[blue]Setting up agent...", total=None)

        try:
            # Initialize the Flock
            flock = Flock(model=MODEL, show_flock_banner=False)

            # Create the agent
            project_agent = FlockFactory.create_default_agent(
                name=f"project_agent",
                description=prompt_description,
                input="prompt: str, user_instructions: str",
                output="output: Project",
                max_tokens=64000,
                no_output=True,
            )

            # Add the agent to the Flock
            flock.add_agent(project_agent)
            progress.update(setup_task, description="[green]Agent setup complete!")
        except Exception as e:
            progress.update(setup_task, description=f"[red]Error setting up agent: {e}")
            raise


    with Progress(
        SpinnerColumn(), TextColumn("[bold yellow]{task.description}"), console=console
    ) as progress:
        agent_task = progress.add_task(
            f"[yellow]Running {action} agent (this may take a while)...", total=None
        )

        try:
            result = flock.run(
                start_agent=project_agent,
                input={
                    "prompt": prompt,
                    "user_instructions": user_instructions,
                },
            )
            progress.update(
                agent_task, description="[green]Agent completed successfully!"
            )
        except Exception as e:
            progress.update(
                agent_task, description=f"[red]Error during agent execution: {e}"
            )
            raise

    return result.output    


def task_agent(
    action: str,
    project_file: str,
    user_instructions: str = "",
) -> TaskAndToDoItemList:
    """
    Communicate with an LLM agent to perform a specified action.

    Args:
        action: The type of action to perform (e.g., "app", "specs")
        user_instructions: Additional instructions for the agent

    Returns:
        OutputData instance with the agent's response
    """

    MODEL = (
        "gemini/gemini-2.5-pro-exp-03-25"  # "groq/qwen-qwq-32b"    #"openai/gpt-4o" #
    )

    # load a file relative to the current file
    prompt_folder = Path(__file__).parent / "prompts"


    # Use a spinner for loading prompt files
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        load_task = progress.add_task("[blue]Loading prompt files...", total=None)

        try:
            
            project = Project.model_validate_json(Path(project_file).read_text())
            prompt_file = load_prompt(action, prompt_folder)
            prompt_description = extract_before_prompt(prompt_file)
            prompt = extract_after_prompt(prompt_file)
      

            progress.update(
                load_task, description="[green]Prompts loaded successfully!"
            )
        except Exception as e:
            progress.update(load_task, description=f"[red]Error loading prompts: {e}")
            raise

     # Set up the agent with the prompt
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        setup_task = progress.add_task("[blue]Setting up agent...", total=None)

        try:
            # Initialize the Flock
            flock = Flock(model=MODEL, show_flock_banner=False)

            # Create the agent
            project_agent = FlockFactory.create_default_agent(
                name=f"task_agent",
                description=prompt_description,
                input="prompt: str, project: Project, done_tasks: list[Task], current_files: list[ProjectFile], done_todo_items: list[ToDoItem], user_instructions: str",
                output="output: TaskAndToDoItemList",
                max_tokens=64000,
                no_output=True,
                write_to_file=True,
            )

            # Add the agent to the Flock
            flock.add_agent(project_agent)
            progress.update(setup_task, description="[green]Agent setup complete!")
        except Exception as e:
            progress.update(setup_task, description=f"[red]Error setting up agent: {e}")
            raise


    with Progress(
        SpinnerColumn(), TextColumn("[bold yellow]{task.description}"), console=console
    ) as progress:
        agent_task = progress.add_task(
            f"[yellow]Running {action} agent (this may take a while)...", total=None
        )

        try:
            result = flock.run(
                start_agent=project_agent,
                input={
                    "prompt": prompt,
                    "user_instructions": user_instructions,
                    "project": project,
                    "done_tasks": [],
                    "current_files": [],
                    "done_todo_items": [],
                },
            )
            progress.update(
                agent_task, description="[green]Agent completed successfully!"
            )
        except Exception as e:
            progress.update(
                agent_task, description=f"[red]Error during agent execution: {e}"
            )
            raise

    return result.output    


def speak_to_agent(
    action: str,
    input_data: str,
    user_instructions: str = "",
) -> OutputData:
    """
    Communicate with an LLM agent to perform a specified action.

    Args:
        action: The type of action to perform (e.g., "app", "specs")
        input_data: Either file path or raw input data (will be detected automatically)
        prompt_folder: Directory where prompt files are located
        user_instructions: Additional instructions for the agent

    Returns:
        OutputData instance with the agent's response
    """

    MODEL = (
        "gemini/gemini-2.5-pro-exp-03-25"  # "groq/qwen-qwq-32b"    #"openai/gpt-4o" #
    )

    # load a file relative to the current file
    prompt_folder = Path(__file__).parent / "prompts"

    # Show which model we're using
    console.print(f"[cyan]Using model:[/cyan] [bold magenta]{MODEL}[/bold magenta]")

    prompt_description = ""
    prompt = ""

    # Use a spinner for loading prompt files
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        load_task = progress.add_task("[blue]Loading prompt files...", total=None)

        try:
            
            prompt_file = load_prompt(action, prompt_folder)
            prompt_description = extract_before_prompt(prompt_file)
            prompt = extract_after_prompt(prompt_file)
      

            progress.update(
                load_task, description="[green]Prompts loaded successfully!"
            )
        except Exception as e:
            progress.update(load_task, description=f"[red]Error loading prompts: {e}")
            raise

    # Set up the agent with the prompt
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        setup_task = progress.add_task("[blue]Setting up agent...", total=None)

        try:
            # Initialize the Flock
            flock = Flock(model=MODEL, show_flock_banner=False)

            # Create the agent
            app_agent = FlockFactory.create_default_agent(
                name=f"{action}_agent",
                description=prompt_description,
                input="prompt: str, user_instructions: str, input_data: str",
                output="output: OutputData",
                max_tokens=64000,
                no_output=True,
            )

            # Add the agent to the Flock
            flock.add_agent(app_agent)
            progress.update(setup_task, description="[green]Agent setup complete!")
        except Exception as e:
            progress.update(setup_task, description=f"[red]Error setting up agent: {e}")
            raise


    # Load input data - dynamically determine if it's a file path
    input_content = input_data
    input_path = Path(input_data)
    
    # Check if input_data is a valid file path
    if input_path.is_file():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
        ) as progress:
            file_task = progress.add_task(
                f"[blue]Loading input from file: [cyan]{input_data}[/cyan]...",
                total=None,
            )

            try:
                with open(input_data, "r") as f:
                    input_content = f.read()
                file_size_kb = input_path.stat().st_size / 1024
                progress.update(
                    file_task,
                    description=f"[green]Input loaded successfully! ([cyan]{file_size_kb:.1f}[/cyan] KB)",
                )
            except Exception as e:
                progress.update(
                    file_task, description=f"[red]Error loading input file: {e}"
                )
                raise
    else:
        # If not a file, treat input_data as raw content
        console.print(f"[cyan]Using input as raw content...[/cyan]")

    # Call the agent with a progress spinner
    with Progress(
        SpinnerColumn(), TextColumn("[bold yellow]{task.description}"), console=console
    ) as progress:
        agent_task = progress.add_task(
            f"[yellow]Running {action} agent (this may take a while)...", total=None
        )

        try:
            result = flock.run(
                start_agent=app_agent,
                input={
                    "prompt": prompt,
                    "user_instructions": user_instructions,
                    "input_data": input_content,
                },
            )
            progress.update(
                agent_task, description="[green]Agent completed successfully!"
            )
        except Exception as e:
            progress.update(
                agent_task, description=f"[red]Error during agent execution: {e}"
            )
            raise

    # Show success message with panel
    console.print(
        Panel(
            f"[green]Successfully executed {action} agent",
            title="[bold green]Success[/bold green]",
            border_style="green",
        )
    )

    return result.output

```

---

### 3. src/copilot_toolkit/code_collector.py

- **File ID**: file_2
- **Type**: Code File
- **Line Count**: 1232
- **Description**: Code Repository Analyzer...
- **Dependencies**: None
- **Used By**: None

**Content**:
```
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

```

---

### 4. src/copilot_toolkit/collector/__init__.py

- **File ID**: file_3
- **Type**: Code File
- **Line Count**: 151
- **Description**: Code Collection and Analysis Sub-package....
- **Dependencies**:
  - file_8
  - file_5
  - file_6
  - file_4
  - file_11
  - file_9
- **Used By**: None

**Content**:
```
# src/pilot_rules/collector/__init__.py
"""
Code Collection and Analysis Sub-package.
Provides functionality to scan repositories, analyze code (primarily Python),
and generate Repository objects.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import necessary functions from sibling modules using relative imports
from .config import process_config_and_args
from .discovery import collect_files
from .analysis import analyze_code_dependencies, get_common_patterns, find_key_files
from .reporting import generate_repository
from .utils import (
    console,
    print_header,
    print_subheader,
    print_success,
    print_warning,
    print_error,
    print_file_stats,
)
from ..model import Repository


def run_collection(
    include_args: Optional[List[str]],
    exclude_args: Optional[List[str]],
    output_arg: Optional[str] = None,  # Kept for backward compatibility but used to save Repository as JSON
    config_arg: Optional[str] = None,
    repo_name: Optional[str] = None,
    calculate_metrics: bool = False,  # New parameter to control metrics calculation
) -> Repository:
    """
    Main entry point for the code collection process.

    Orchestrates configuration loading, file discovery, analysis, and Repository generation.
    
    Args:
        include_args: List of include patterns in format 'ext1,ext2:./folder'
        exclude_args: List of exclude patterns in format 'py:temp'
        output_arg: Path to output JSON file to save the repository (optional)
        config_arg: Path to optional TOML config file
        repo_name: Name for the repository (default is "Repository Analysis")
        calculate_metrics: Whether to calculate code quality metrics (default is False)
        
    Returns:
        Repository object with analyzed code data
    """
    try:
        # 1. Process Configuration and Arguments
        print_header("Code Collection Process", "magenta")
        final_sources, final_excludes, _ = process_config_and_args(
            include_args=include_args,
            exclude_args=exclude_args,
            output_arg=output_arg,
            config_arg=config_arg,
        )

        # Use provided repo_name or default
        repository_name = repo_name if repo_name else "Repository Analysis"

        # 2. Collect Files based on finalized sources and excludes
        collected_files, actual_extensions = collect_files(
            final_sources, final_excludes
        )

        if not collected_files:
            print_warning("No files found matching the specified criteria.")
            # Return minimal Repository with empty files list
            return Repository(
                name=repository_name,
                statistics="No files found matching the specified criteria.",
                project_files=[]
            )
        else:
            print_success(
                f"Found [bold green]{len(collected_files)}[/bold green] files to include in the analysis."
            )
            ext_list = ", ".join(sorted(list(actual_extensions)))
            console.print(f"File extensions found: [cyan]{ext_list}[/cyan]")

            # Display file statistics in a nice table
            print_file_stats(collected_files, "Collection Statistics")

        # 3. Perform Analysis (Conditional based on files found)
        dependencies = {}
        patterns = {}
        key_files = []

        # Only run Python-specific analysis if .py files are present
        has_python_files = ".py" in actual_extensions
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
            key_files = find_key_files(collected_files, dependencies)  # Pass all files

        # 4. Generate Repository Object
        repository = generate_repository(
            files=collected_files,
            analyzed_extensions=actual_extensions,
            dependencies=dependencies,
            patterns=patterns,
            key_files=key_files,
            repo_name=repository_name,
            root_folder_display=".",  # Or derive from sources if needed
            calculate_metrics=calculate_metrics,  # Pass the metrics flag
        )
        
        # 5. Save Repository as JSON if output_arg is provided
        if output_arg:
            import json
            from pathlib import Path
            
            try:
                # Convert repository to dict and save as JSON
                repo_dict = repository.dict()
                output_path = Path(output_arg)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(repo_dict, f, indent=2)
                print_success(f"Repository data saved to {output_path}")
            except Exception as e:
                print_error(f"Error saving repository data: {str(e)}")
        
        return repository

    except ValueError as e:
        # Configuration or argument parsing errors
        print_error(f"Configuration Error: {e}", 1)
        raise
    except Exception as e:
        # Catch-all for unexpected errors during collection/analysis/reporting
        print_error(f"An unexpected error occurred: {e}", 1)
        import traceback
        traceback.print_exc()
        raise

# Alias for backward compatibility
generate_repository_from_files = run_collection

```

---

### 5. src/copilot_toolkit/collector/analysis.py

- **File ID**: file_4
- **Type**: Code File
- **Line Count**: 414
- **Description**: File at src/copilot_toolkit/collector/analysis.py
- **Dependencies**:
  - file_9
- **Used By**:
  - file_8
  - file_3

**Content**:
```
# src/pilot_rules/collector/analysis.py
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Import utility function - use relative import within the package
from .utils import (
    print_warning,
    print_success,
    print_subheader,
)


# --- Python Component Extraction ---
def extract_python_components(file_path: str) -> Dict[str, Any]:
    """Extract classes, functions, and imports from Python files."""
    components = {"classes": [], "functions": [], "imports": [], "docstring": None}

    # Ensure it's a python file before trying to parse
    if not file_path.lower().endswith(".py"):
        return components  # Return empty structure for non-python files

    try:
        # Read with error handling for encoding issues
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        tree = ast.parse(content)

        # Extract module docstring
        components["docstring"] = ast.get_docstring(
            tree
        )  # Returns None if no docstring

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
                # We consider all functions directly under the module body as "top-level" here
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    # Simplified arg extraction (just names)
                    "args": [arg.arg for arg in node.args.args],
                }
                components["functions"].append(func_info)

        # Extract all imports (simplified representation)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Store 'import x' or 'import x as y'
                    components["imports"].append(
                        f"import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else "")
                    )
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

                components["imports"].append(
                    f"from {level_dots}{module_part} import {', '.join(imported_names)}"
                )

    except SyntaxError as e:
        print_warning(
            f"Could not parse Python components in [cyan]{file_path}[/cyan] due to SyntaxError: [red]{e}[/red]"
        )
    except Exception as e:
        print_warning(
            f"Could not parse Python components in [cyan]{file_path}[/cyan]: [red]{e}[/red]"
        )

    return components


# --- Dependency Analysis ---


def get_module_prefixes(module_name: str) -> List[str]:
    """
    Generate all possible module prefixes for a given module name.
    For example, 'a.b.c' would return ['a.b.c', 'a.b', 'a']
    """
    parts = module_name.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def analyze_code_dependencies(files: List[str]) -> Dict[str, Set[str]]:
    """Analyze dependencies between Python files based on imports."""
    # Filter to only analyze python files within the provided list
    python_files = {f for f in files if f.lower().endswith(".py")}
    if not python_files:
        return {}  # No Python files to analyze

    dependencies: Dict[str, Set[str]] = {file: set() for file in python_files}
    module_map: Dict[str, str] = {}  # Map potential module names to absolute file paths
    project_root = (
        Path.cwd().resolve()
    )  # Assume CWD is project root for relative imports

    # --- Build Module Map (heuristic) ---
    # Map files within the project to their potential Python module paths
    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            # Attempt to create a module path relative to the project root
            relative_path = file_path.relative_to(project_root)
            parts = list(relative_path.parts)
            module_name = None
            if parts[-1] == "__init__.py":
                module_parts = parts[:-1]
                if module_parts:  # Avoid mapping root __init__.py as empty string
                    module_name = ".".join(module_parts)
            elif parts[-1].endswith(".py"):
                module_parts = parts[:-1] + [parts[-1][:-3]]  # Remove .py
                module_name = ".".join(module_parts)

            if module_name:
                # console.print(f"[dim]Mapping module '{module_name}' to '{file_path_str}'[/dim]") # Debug
                module_map[module_name] = file_path_str

        except ValueError:
            # File is outside the assumed project root, less reliable mapping
            # Map only by filename stem if not already mapped? Risky.
            # console.print(f"[dim]Debug: File {file_path_str} is outside project root {project_root}[/dim]")
            pass

    # --- Analyze Imports in Each File ---
    for file_path_str in python_files:
        file_path = Path(file_path_str).resolve()
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
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
                                if (
                                    target_file in python_files
                                    and target_file != file_path_str
                                ):
                                    dependencies[file_path_str].add(target_file)
                                break  # Found the longest matching prefix

                # Handle 'from x import y' or 'from .x import y'
                elif isinstance(node, ast.ImportFrom):
                    level = node.level
                    module_base = node.module or ""

                    if level == 0:  # Absolute import: 'from package import module'
                        imported_module_str = module_base
                        for prefix in get_module_prefixes(imported_module_str):
                            if prefix in module_map:
                                target_file = module_map[prefix]
                                if (
                                    target_file in python_files
                                    and target_file != file_path_str
                                ):
                                    dependencies[file_path_str].add(target_file)
                                break
                    else:  # Relative import: 'from . import x', 'from ..y import z'
                        current_dir = file_path.parent
                        base_path = current_dir
                        # Navigate up for '..' (level 2 means one level up, etc.)
                        for _ in range(level - 1):
                            base_path = base_path.parent

                        # Try to resolve the relative path
                        relative_module_parts = (
                            module_base.split(".") if module_base else []
                        )
                        target_path_base = base_path
                        for part in relative_module_parts:
                            target_path_base = target_path_base / part

                        # Check if the resolved path corresponds to a known file/module
                        # Check 1: Is it a directory with __init__.py?
                        init_py_path = (target_path_base / "__init__.py").resolve()
                        init_py_str = str(init_py_path)
                        if init_py_str in python_files and init_py_str != file_path_str:
                            dependencies[file_path_str].add(init_py_str)
                            target_file = init_py_str  # Mark as found

                        # Check 2: Is it a .py file directly?
                        module_py_path = target_path_base.with_suffix(".py").resolve()
                        module_py_str = str(module_py_path)
                        if (
                            not target_file
                            and module_py_str in python_files
                            and module_py_str != file_path_str
                        ):
                            dependencies[file_path_str].add(module_py_str)
                            target_file = module_py_str

                        # Note: This relative import resolution is basic and might miss complex cases.
                        # We are primarily checking if the base module path (e.g., `.`, `..utils`) exists.

        except SyntaxError as e:
            print_warning(
                f"Skipping import analysis in [cyan]{file_path_str}[/cyan] due to SyntaxError: [red]{e}[/red]"
            )
        except Exception as e:
            print_warning(
                f"Could not analyze imports in [cyan]{file_path_str}[/cyan]: [red]{e}[/red]"
            )

    return dependencies


# --- Pattern Detection ---


def get_common_patterns(files: List[str]) -> Dict[str, Any]:
    """Identify common code patterns across the repository."""
    patterns = {"python_patterns": {}}

    # Get all Python files
    python_files = [f for f in files if f.lower().endswith(".py")]
    if not python_files:
        return patterns  # No Python files to analyze

    # --- Python Import Patterns ---
    all_imports: Dict[str, int] = {}
    file_imports: Dict[str, List[str]] = {}

    # Basic frameworks imports to check for
    frameworks = {
        "Django": ["django", "django.db", "django.http", "django.urls", "django.views"],
        "Flask": ["flask", "flask_restful", "flask_sqlalchemy"],
        "FastAPI": ["fastapi"],
        "SQLAlchemy": ["sqlalchemy"],
        "PyTorch": ["torch"],
        "TensorFlow": ["tensorflow", "tf"],
        "Pandas": ["pandas"],
        "Numpy": ["numpy", "np"],
        "Pytest": ["pytest"],
        "Unittest": ["unittest"],
    }

    # Track framework detections
    framework_evidence: Dict[str, str] = {}

    for file_path in python_files:
        # Extract components including imports
        components = extract_python_components(file_path)
        imports = components.get("imports", [])

        # Store all the raw imports
        file_imports[file_path] = imports

        # Process each import line
        for imp in imports:
            # Normalize import line by removing "as X" aliases
            # This helps count semantically identical imports
            base_import = imp.split(" as ")[0].strip()
            all_imports[base_import] = all_imports.get(base_import, 0) + 1

            # Check for framework indicators
            for framework, indicators in frameworks.items():
                for indicator in indicators:
                    if indicator in imp.split()[1]:  # Check the module part
                        if framework not in framework_evidence:
                            framework_evidence[framework] = (
                                f"Found import '{imp}' in {Path(file_path).name}"
                            )
                        break

    # Sort by frequency
    common_imports = sorted(all_imports.items(), key=lambda x: x[1], reverse=True)
    patterns["python_patterns"]["common_imports"] = common_imports

    # Add framework detections if any
    if framework_evidence:
        patterns["python_patterns"]["framework_patterns"] = framework_evidence

    # Important: try/except to avoid failures during pattern detection
    # This is analysis code, shouldn't crash the report generation
    try:
        # Common file patterns based on naming conventions
        repository_patterns = {}
        # ... extend with more pattern detection as needed...

        # Add to patterns dict
        patterns.update(repository_patterns)
    except Exception:
        # Less prominent warning since this is enhancement, not core functionality
        # print(f"Warning: Could not analyze patterns in {file_path}: {e}") # Can be noisy
        pass

    return patterns


# --- Key File Identification ---


def find_key_files(files: List[str], dependencies: Dict[str, Set[str]]) -> List[str]:
    """
    Identify key files in the repository based on several heuristic factors.
    - Dependency count (how many files depend on this one)
    - Naming convention (e.g., main.py, __init__.py)
    - File size and location
    """
    print_subheader("Scoring files to identify key ones", "cyan")

    if not files:
        return []

    # 1. Prepare scoring dict
    scores: Dict[str, float] = {file: 0.0 for file in files}

    # 2. Score based on file naming and key locations
    key_names = ["main", "app", "core", "index", "server", "engine", "controller"]
    for file in files:
        file_path = Path(file)
        filename_stem = file_path.stem.lower()

        # Key names in filename get points
        for key in key_names:
            if key == filename_stem:
                scores[file] += 5.0  # Exact match
            elif key in filename_stem:
                scores[file] += 2.0  # Partial match

        # Special files
        if filename_stem == "__init__":
            scores[file] += 1.0
        if filename_stem == "__main__":
            scores[file] += 3.0

        # Files in root directories are often important
        try:
            rel_path = file_path.relative_to(Path.cwd())
            depth = len(rel_path.parts)
            if depth <= 2:  # In root or direct subdirectory
                scores[file] += 3.0 / depth  # More points for less depth
        except ValueError:
            # File outside cwd, skip this bonus
            pass

        # Size can indicate importance (within reason)
        try:
            size = file_path.stat().st_size
            # Log scale to avoid over-prioritizing large files
            if size > 0:
                import math

                size_score = min(3.0, math.log(size) / 3)
                scores[file] += size_score
        except OSError:
            pass

    # 3. Dependency analysis (Python)
    # Calculate how many files depend on each file (reversed dependency graph)
    dependents: Dict[str, Set[str]] = {file: set() for file in files}
    for source, targets in dependencies.items():
        for target in targets:
            if target in dependents:
                dependents[target].add(source)

    # Score based on dependent count (files that import this file)
    for file, deps in dependents.items():
        count = len(deps)
        if count > 0:
            # More weight for dependencies
            scores[file] += count * 2.0
            # console.print(f"  Score bump (deps): {Path(file).name} +{count * 2.0} (depended by {count})")

    # 4. Select top files based on scores
    # Calculate a reasonable number based on repository size
    num_key_files = min(
        10, max(3, int(len(files) * 0.1))
    )  # 10% but at least 3, at most 10

    # Sort files by score (descending) and select top N
    top_files = sorted(files, key=lambda f: scores.get(f, 0), reverse=True)[
        :num_key_files
    ]

    print_success(f"Selected top {num_key_files} files as key files.")

    # Debug info (commented out in production)
    # for i, f in enumerate(top_files):
    #      console.print(f"  {i+1}. {Path(f).name}: {scores.get(f, 0.0):.2f}")

    return top_files

```

---

### 6. src/copilot_toolkit/collector/config.py

- **File ID**: file_5
- **Type**: Code File
- **Line Count**: 243
- **Description**: File at src/copilot_toolkit/collector/config.py
- **Dependencies**:
  - file_9
- **Used By**:
  - file_3

**Content**:
```
# src/pilot_rules/collector/config.py
import tomli
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .utils import (
    console,
    print_subheader,
)

DEFAULT_OUTPUT_FILENAME = "repository_analysis.md"
DEFAULT_INCLUDE_SPEC = "py:."  # Default to python files in current dir


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
        extensions = [
            ext.strip().lower().lstrip(".")
            for ext in exts_str.split(",")
            if ext.strip()
        ]
        if not extensions:
            raise ValueError(f"No extensions specified in '{arg}'. Use '*' for all.")

        # Use '*' as a special marker for all extensions
        if "*" in extensions:
            extensions = ["*"]

        # Normalize path pattern to use forward slashes for consistency
        # Keep it relative for now, resolve later if needed
        path_pattern = Path(path_pattern).as_posix()

        parsed.append(
            {
                "extensions": extensions,  # List of extensions (lowercase, no dot), or ['*']
                "pattern": path_pattern,  # Path or pattern string (relative or absolute)
            }
        )
    return parsed


def load_config_from_toml(
    config_path: Path,
) -> Tuple[List[Dict], List[Dict], Optional[str]]:
    """Loads sources, excludes, and output path from a TOML file."""
    config_sources = []
    config_excludes = []
    config_output = None

    print_subheader(f"Loading configuration from: [cyan]{config_path}[/cyan]")
    try:
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)

        # --- Parse sources ---
        raw_sources = config_data.get("source", [])
        if not isinstance(raw_sources, list):
            raise ValueError("Invalid config: 'source' must be an array of tables.")

        for i, src_table in enumerate(raw_sources):
            if not isinstance(src_table, dict):
                raise ValueError(
                    f"Invalid config: Item {i} in 'source' array is not a table."
                )

            exts = src_table.get("exts", ["*"])  # Default to all if not specified
            root = src_table.get("root", ".")
            exclude_patterns = src_table.get(
                "exclude", []
            )  # Excludes within a source block

            if not isinstance(exts, list) or not all(isinstance(e, str) for e in exts):
                raise ValueError(
                    f"Invalid config: 'exts' must be a list of strings in source #{i + 1}"
                )
            if not isinstance(root, str):
                raise ValueError(
                    f"Invalid config: 'root' must be a string in source #{i + 1}."
                )
            if not isinstance(exclude_patterns, list) or not all(
                isinstance(p, str) for p in exclude_patterns
            ):
                raise ValueError(
                    f"Invalid config: 'exclude' must be a list of strings in source #{i + 1}"
                )

            # Normalize extensions: lowercase, no leading dot
            normalized_exts = [e.lower().lstrip(".") for e in exts]
            if "*" in normalized_exts:
                normalized_exts = ["*"]  # Treat ['*'] as the 'all' marker

            # Store source config
            config_sources.append(
                {
                    "root": root,  # Keep relative for now, resolve later
                    "extensions": normalized_exts,
                }
            )

            # Add source-specific excludes to the global excludes list
            # Assume format '*:<pattern>' for excludes defined within a source block
            for pattern in exclude_patterns:
                config_excludes.append(
                    {"extensions": ["*"], "pattern": Path(pattern).as_posix()}
                )

        # --- Parse global output ---
        config_output = config_data.get("output")
        if config_output and not isinstance(config_output, str):
            raise ValueError("Invalid config: 'output' must be a string.")

        # --- Parse global excludes (optional top-level section) ---
        raw_global_excludes = config_data.get("exclude", [])
        if not isinstance(raw_global_excludes, list):
            raise ValueError("Invalid config: Top-level 'exclude' must be an array.")
        for i, ex_table in enumerate(raw_global_excludes):
            if not isinstance(ex_table, dict):
                raise ValueError(
                    f"Invalid config: Item {i} in top-level 'exclude' array is not a table."
                )
            exts = ex_table.get("exts", ["*"])
            pattern = ex_table.get("pattern")
            if pattern is None:
                raise ValueError(
                    f"Invalid config: 'pattern' missing in top-level exclude #{i + 1}"
                )
            if not isinstance(pattern, str):
                raise ValueError(
                    f"Invalid config: 'pattern' must be a string in top-level exclude #{i + 1}"
                )
            if not isinstance(exts, list) or not all(isinstance(e, str) for e in exts):
                raise ValueError(
                    f"Invalid config: 'exts' must be a list of strings in top-level exclude #{i + 1}"
                )

            normalized_exts = [e.lower().lstrip(".") for e in exts]
            if "*" in normalized_exts:
                normalized_exts = ["*"]

            config_excludes.append(
                {"extensions": normalized_exts, "pattern": Path(pattern).as_posix()}
            )

    except tomli.TOMLDecodeError as e:
        raise ValueError(f"Error parsing TOML config file '{config_path}': {e}")
    except FileNotFoundError:
        raise ValueError(f"Config file not found: '{config_path}'")

    return config_sources, config_excludes, config_output


def process_config_and_args(
    include_args: Optional[List[str]],
    exclude_args: Optional[List[str]],
    output_arg: Optional[str],  # Output from CLI args might be None if default used
    config_arg: Optional[str],
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
            config_sources, config_excludes, config_output = load_config_from_toml(
                config_path
            )
        else:
            # Argparse should handle file existence, but double-check
            raise ValueError(
                f"Config file path specified but not found or not a file: '{config_arg}'"
            )

    # 2. Parse CLI arguments
    cli_includes = parse_include_exclude_args(include_args)
    cli_excludes = parse_include_exclude_args(exclude_args)
    # Use output_arg directly (it incorporates the argparse default if not provided)
    cli_output = output_arg if output_arg else DEFAULT_OUTPUT_FILENAME

    # 3. Combine sources: CLI overrides config sources entirely if provided.
    final_sources_specs = []
    if cli_includes:
        console.print("[cyan]Using include sources from command line arguments.[/cyan]")
        final_sources_specs = cli_includes  # Use CLI specs directly
    elif config_sources:
        console.print("[cyan]Using include sources from configuration file.[/cyan]")
        final_sources_specs = config_sources  # Use config specs
    else:
        console.print(
            f"[yellow]No includes specified via CLI or config, defaulting to '[bold]{DEFAULT_INCLUDE_SPEC}[/bold]'.[/yellow]"
        )
        final_sources_specs = parse_include_exclude_args([DEFAULT_INCLUDE_SPEC])

    # 4. Combine excludes: CLI appends to config excludes
    # Exclude patterns remain relative path strings for fnmatch
    final_excludes = config_excludes + cli_excludes
    if final_excludes:
        console.print(
            f"Applying [bold yellow]{len(final_excludes)}[/bold yellow] exclusion rule(s)."
        )

    # 5. Determine final output path: CLI > Config > Default
    # cli_output already incorporates the default if needed
    final_output_str = cli_output if cli_output else config_output
    if not final_output_str:  # Should not happen if argparse default is set
        final_output_str = DEFAULT_OUTPUT_FILENAME

    # Resolve output path relative to CWD
    final_output_path = Path(final_output_str).resolve()
    console.print(f"Final output path: [bold green]{final_output_path}[/bold green]")

    # 6. Resolve source roots relative to CWD *after* deciding which specs to use
    resolved_final_sources = []
    for spec in final_sources_specs:
        # spec['pattern'] here is the root directory from include args or config
        resolved_root = Path(spec["pattern"]).resolve()
        resolved_final_sources.append(
            {
                "root": resolved_root,
                "extensions": spec["extensions"],  # Keep normalized extensions
            }
        )

    return resolved_final_sources, final_excludes, final_output_path

```

---

### 7. src/copilot_toolkit/collector/discovery.py

- **File ID**: file_6
- **Type**: Code File
- **Line Count**: 196
- **Description**: File at src/copilot_toolkit/collector/discovery.py
- **Dependencies**: None
- **Used By**:
  - file_3

**Content**:
```
# src/pilot_rules/collector/discovery.py
import glob
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Create a console for rich output
console = Console()


def collect_files(
    sources: List[Dict[str, Any]], excludes: List[Dict[str, Any]]
) -> Tuple[List[str], Set[str]]:
    """
    Finds files based on source definitions (using glob.glob) and applies exclusion rules.

    Args:
        sources: List of dicts, each with 'root' (resolved Path) and 'extensions' (list or ['*']).
        excludes: List of dicts, each with 'extensions' (list or ['*']) and 'pattern' (str).

    Returns:
        Tuple: (list of absolute file paths found, set of unique extensions found (lowercase, with dot))
    """
    console.print(
        Panel.fit("[bold blue]Collecting Files[/bold blue]", border_style="blue")
    )
    all_found_files: Set[str] = set()  # Store absolute paths as strings
    project_root = Path.cwd().resolve()  # Use CWD as the reference point for excludes

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        source_task = progress.add_task(
            "[yellow]Processing sources...", total=len(sources)
        )

        for source in sources:
            root_path: Path = source["root"]
            extensions: List[str] = source[
                "extensions"
            ]  # Already normalized (lowercase, no dot, or ['*'])

            ext_display = extensions if extensions != ["*"] else "all"
            progress.update(
                source_task,
                description=f"[yellow]Scanning: [cyan]{root_path}[/cyan] for [green]{ext_display}[/green]",
            )

            if not root_path.is_dir():
                console.print(
                    f"[yellow]⚠ Warning:[/yellow] Source root '[cyan]{root_path}[/cyan]' is not a directory. Skipping."
                )
                progress.update(source_task, advance=1)
                continue

            found_in_source: Set[str] = set()
            if extensions == ["*"]:
                # Use glob.glob for all files recursively
                glob_pattern_str = str(root_path / "**" / "*")
                try:
                    # Use glob.glob with recursive=True
                    for filepath_str in glob.glob(glob_pattern_str, recursive=True):
                        item = Path(filepath_str)
                        # Check if it's a file (glob might return directories matching pattern too)
                        if item.is_file():
                            # Add resolved absolute path as string
                            found_in_source.add(str(item.resolve()))
                except Exception as e:
                    console.print(
                        f"[yellow]⚠ Warning:[/yellow] Error during globbing for '[cyan]{glob_pattern_str}[/cyan]': [red]{e}[/red]"
                    )
            else:
                # Specific extensions provided
                for ext in extensions:
                    # Construct pattern like '*.py'
                    pattern = f"*.{ext}"
                    glob_pattern_str = str(root_path / "**" / pattern)
                    try:
                        # Use glob.glob with recursive=True
                        for filepath_str in glob.glob(glob_pattern_str, recursive=True):
                            item = Path(filepath_str)
                            # Check if it's a file
                            if item.is_file():
                                # Add resolved absolute path as string
                                found_in_source.add(str(item.resolve()))
                    except Exception as e:
                        console.print(
                            f"[yellow]⚠ Warning:[/yellow] Error during globbing for '[cyan]{glob_pattern_str}[/cyan]': [red]{e}[/red]"
                        )

            console.print(
                f"  Found [green]{len(found_in_source)}[/green] potential files in this source."
            )
            all_found_files.update(found_in_source)
            progress.update(source_task, advance=1)

    console.print(
        f"Total unique files found before exclusion: [bold green]{len(all_found_files)}[/bold green]"
    )

    # Apply exclusion rules
    excluded_files: Set[str] = set()
    if excludes:
        console.print(
            Panel.fit(
                "[bold yellow]Applying Exclusion Rules[/bold yellow]",
                border_style="yellow",
            )
        )
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

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            exclude_task = progress.add_task(
                "[yellow]Processing exclusion rules...", total=len(excludes)
            )

            for rule in excludes:
                rule_exts: List[str] = rule[
                    "extensions"
                ]  # Normalized (lowercase, no dot, or ['*'])
                rule_pattern: str = rule["pattern"]  # Relative path pattern string

                ext_display = rule_exts if rule_exts != ["*"] else "any"
                progress.update(
                    exclude_task,
                    description=f"[yellow]Excluding: [green]{ext_display}[/green] matching [cyan]*{rule_pattern}*[/cyan]",
                )

                # Use fnmatch for flexible pattern matching against relative paths
                # Wrap the pattern to check if the rule pattern exists anywhere in the path
                pattern_to_match = f"*{rule_pattern}*"

                files_to_check = relative_file_paths
                # If rule has specific extensions, filter the files to check first
                if rule_exts != ["*"]:
                    # Match against suffix (e.g., '.py')
                    dot_exts = {f".{e}" for e in rule_exts}
                    files_to_check = {
                        rel_path
                        for rel_path in relative_file_paths
                        if Path(rel_path).suffix.lower() in dot_exts
                    }

                # Apply fnmatch to the filtered relative paths
                matched_by_rule = {
                    rel_path
                    for rel_path in files_to_check
                    if fnmatch.fnmatch(rel_path, pattern_to_match)
                }

                # Add the corresponding absolute paths to the excluded set
                for rel_path in matched_by_rule:
                    if rel_path in relative_files_map:
                        excluded_files.add(relative_files_map[rel_path])
                        # Verbose logging disabled by default
                        # console.print(f"    Excluding: [dim]{relative_files_map[rel_path]}[/dim]")

                progress.update(exclude_task, advance=1)

    console.print(f"Excluded [bold yellow]{len(excluded_files)}[/bold yellow] files.")
    final_files = sorted(list(all_found_files - excluded_files))

    # Determine actual extensions present in the final list (lowercase, with dot)
    final_extensions = {Path(f).suffix.lower() for f in final_files if Path(f).suffix}

    console.print(
        f"[bold green]✓[/bold green] Collection complete! Found [bold green]{len(final_files)}[/bold green] files with [bold cyan]{len(final_extensions)}[/bold cyan] unique extensions."
    )

    return final_files, final_extensions

```

---

### 8. src/copilot_toolkit/collector/metrics.py

- **File ID**: file_7
- **Type**: Code File
- **Line Count**: 231
- **Description**: Code quality metrics calculations for the code collector.
- **Dependencies**:
  - file_9
- **Used By**:
  - file_8

**Content**:
```
"""
Code quality metrics calculations for the code collector.
Provides functions to analyze code complexity, maintainability and other quality metrics.
"""

import os
import ast
from pathlib import Path
from typing import Dict, Any, Optional, List

import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import radon.raw as radon_raw
from radon.visitors import ComplexityVisitor

from .utils import print_warning, print_subheader


def calculate_python_metrics(file_path: str) -> Dict[str, Any]:
    """
    Calculate code quality metrics for a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        Dictionary containing various code quality metrics
    """
    metrics = {
        "cyclomatic_complexity": None,
        "maintainability_index": None,
        "raw_metrics": None,
        "code_smells": [],
        "complexity_by_function": []
    }
    
    if not file_path.lower().endswith('.py'):
        return metrics
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        # Calculate raw metrics (lines of code, comments, etc.)
        raw_metrics = radon_raw.analyze(code)
        metrics["raw_metrics"] = {
            "loc": raw_metrics.loc,  # Lines of code (excluding comments)
            "lloc": raw_metrics.lloc,  # Logical lines of code
            "sloc": raw_metrics.sloc,  # Source lines of code 
            "comments": raw_metrics.comments,  # Number of comments
            "multi": raw_metrics.multi,  # Number of multi-line strings
            "blank": raw_metrics.blank,  # Number of blank lines
            "comment_ratio": raw_metrics.comments / raw_metrics.loc if raw_metrics.loc > 0 else 0
        }
        
        # Calculate maintainability index (0-100, higher is better)
        mi = radon_metrics.mi_visit(code, multi=True)
        metrics["maintainability_index"] = {
            "value": mi,
            "rank": _get_maintainability_rank(mi)
        }
        
        # Calculate cyclomatic complexity
        try:
            # Get complexity for the entire file
            complexity = ComplexityVisitor.from_code(code)
            if complexity.total_complexity is not None:
                # Calculate average complexity manually if needed
                total_complexity = complexity.total_complexity
                num_functions = len(complexity.functions) if complexity.functions else 1
                avg_complexity = total_complexity / num_functions if num_functions > 0 else total_complexity
                
                metrics["cyclomatic_complexity"] = {
                    "total": total_complexity,
                    "average": avg_complexity,
                    "rank": _get_complexity_rank(avg_complexity)
                }
            
            # Get complexity for each function/method
            for item in complexity.functions:
                metrics["complexity_by_function"].append({
                    "name": item.name,
                    "line_number": item.lineno,
                    "complexity": item.complexity,
                    "rank": _get_complexity_rank(item.complexity)
                })
                
                # Identify code smells based on complexity
                if item.complexity > 10:
                    metrics["code_smells"].append({
                        "type": "high_complexity",
                        "location": f"{item.name} (line {item.lineno})",
                        "description": f"Function has high cyclomatic complexity ({item.complexity})",
                        "suggestion": "Consider refactoring into smaller functions"
                    })
        except SyntaxError:
            # Fall back to simpler analysis if visitor fails
            pass
        
        # Check for additional code smells
        metrics["code_smells"].extend(_detect_code_smells(code, file_path))
        
    except Exception as e:
        print_warning(f"Could not analyze metrics in {file_path}: {str(e)}")
    
    return metrics


def _get_complexity_rank(complexity: float) -> str:
    """
    Convert a cyclomatic complexity score to a letter rank.
    
    Args:
        complexity: The cyclomatic complexity value
    
    Returns:
        Letter rank from A (best) to F (worst)
    """
    if complexity <= 5:
        return "A"  # Low - good
    elif complexity <= 10:
        return "B"  # Medium - acceptable
    elif complexity <= 20:
        return "C"  # High - concerning
    elif complexity <= 30:
        return "D"  # Very high - problematic
    else:
        return "F"  # Extremely high - needs immediate refactoring


def _get_maintainability_rank(mi_value: float) -> str:
    """
    Convert a maintainability index to a letter rank.
    
    Args:
        mi_value: The maintainability index value
    
    Returns:
        Letter rank from A (best) to F (worst)
    """
    if mi_value >= 85:
        return "A"  # Highly maintainable
    elif mi_value >= 65:
        return "B"  # Maintainable
    elif mi_value >= 40:
        return "C"  # Moderately maintainable
    elif mi_value >= 25:
        return "D"  # Difficult to maintain
    else:
        return "F"  # Very difficult to maintain


def _detect_code_smells(code: str, file_path: str) -> List[Dict[str, str]]:
    """
    Detect common code smells in Python code.
    
    Args:
        code: Python code as a string
        file_path: Path to the Python file (for reporting)
    
    Returns:
        List of detected code smells with descriptions
    """
    smells = []
    
    try:
        tree = ast.parse(code)
        
        # Check for long functions (by line count)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    smells.append({
                        "type": "long_function",
                        "location": f"{node.name} (line {node.lineno})",
                        "description": f"Function is too long ({func_lines} lines)",
                        "suggestion": "Consider breaking into smaller functions"
                    })
                
                # Check for too many parameters
                if len(node.args.args) > 7:  # Including self for methods
                    smells.append({
                        "type": "too_many_parameters",
                        "location": f"{node.name} (line {node.lineno})",
                        "description": f"Function has too many parameters ({len(node.args.args)})",
                        "suggestion": "Consider using a class or data objects to group parameters"
                    })
                    
        # Check for too many imports (module level)
        import_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += len(node.names)
        
        if import_count > 15:
            smells.append({
                "type": "too_many_imports",
                "location": f"{Path(file_path).name}",
                "description": f"Module has too many imports ({import_count})",
                "suggestion": "Consider refactoring to reduce dependencies"
            })
            
    except Exception:
        # Silently fail for syntax errors or other parsing issues
        pass
    
    return smells


def calculate_file_metrics(file_path: str) -> Dict[str, Any]:
    """
    Calculate appropriate metrics based on file type.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary of metrics appropriate for the file type
    """
    metrics = {}
    
    # Handle Python files
    if file_path.lower().endswith('.py'):
        metrics = calculate_python_metrics(file_path)
    
    # TODO: Add support for other languages (JavaScript, etc.)
    # elif file_path.lower().endswith('.js'):
    #     metrics = calculate_javascript_metrics(file_path)
    
    return metrics 
```

---

### 9. src/copilot_toolkit/collector/reporting.py

- **File ID**: file_8
- **Type**: Code File
- **Line Count**: 566
- **Description**: File at src/copilot_toolkit/collector/reporting.py
- **Dependencies**:
  - file_7
  - file_4
  - file_11
  - file_9
- **Used By**:
  - file_3

**Content**:
```
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
    print_subheader,
)
from .analysis import extract_python_components  # Import needed analysis functions
from .metrics import calculate_file_metrics  # Import the new metrics module
from ..model import Repository, ProjectFile, ProjectCodeFile


# --- Folder Tree Generation ---
# (generate_folder_tree function remains the same as the previous version)
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


# --- Markdown Generation ---
def generate_markdown(
    files: List[str],  # List of absolute paths
    analyzed_extensions: Set[
        str
    ],  # Set of actual extensions found (e.g., '.py', '.js')
    dependencies: Dict[str, Set[str]],  # Python dependencies
    patterns: Dict[str, Any],  # Detected patterns
    key_files: List[str],  # List of absolute paths for key files
    output_path: Path,
    root_folder_display: str = ".",  # How to display the root in summary/tree
) -> None:
    """Generate a comprehensive markdown document about the codebase."""
    print_header("Generating Report", "green")
    console.print(f"Output file: [cyan]{output_path}[/cyan]")
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    report_base_path = (
        Path.cwd()
    )  # Use CWD as the base for relative paths in the report

    has_python_files = ".py" in analyzed_extensions

    with open(output_path, "w", encoding="utf-8") as md_file:
        # --- Header ---
        md_file.write("# Code Repository Analysis\n\n")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # ms precision
        md_file.write(f"Generated on {timestamp}\n\n")

        # --- Repository Summary ---
        md_file.write("## Repository Summary\n\n")
        ext_list_str = (
            ", ".join(sorted(list(analyzed_extensions)))
            if analyzed_extensions
            else "N/A"
        )
        md_file.write(f"- **Extensions analyzed**: `{ext_list_str}`\n")
        md_file.write(f"- **Number of files analyzed**: {len(files)}\n")
        md_file.write(
            f"- **Analysis Root (for display)**: `{root_folder_display}`\n"
        )  # Indicate the main perspective

        total_lines = 0
        if files:
            try:
                total_lines = sum(
                    get_file_metadata(f).get("line_count", 0) for f in files
                )
            except Exception as e:
                print_warning(f"Could not calculate total lines accurately: {e}")
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
                print_warning(
                    f"Display root '{root_folder_display}' not found or not a directory, using CWD for tree."
                )
                md_file.write(generate_folder_tree(report_base_path, files))
        except Exception as tree_err:
            print_error(f"Error generating folder tree: {tree_err}")
            md_file.write(f"Error generating folder tree: {tree_err}")
        md_file.write("\n```\n\n")

        # --- Key Files Section ---
        md_file.write("## Key Files\n\n")
        if key_files:
            md_file.write(
                "These files appear central based on dependencies, naming, and size:\n\n"
            )
            for file_abs_path in key_files:
                try:
                    rel_path = str(Path(file_abs_path).relative_to(report_base_path))
                except ValueError:
                    rel_path = file_abs_path  # Fallback to absolute if not relative

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
                if has_python_files and file_abs_path in dependencies:
                    dependent_files_abs = {
                        f for f, deps in dependencies.items() if file_abs_path in deps
                    }
                    if dependent_files_abs:
                        md_file.write(
                            f"- **Used by**: {len(dependent_files_abs)} other analyzed Python file(s)\n"
                        )

                # Python component analysis
                if file_abs_path.lower().endswith(".py"):
                    components = extract_python_components(
                        file_abs_path
                    )  # Use imported function
                    if components.get("docstring"):
                        docstring_summary = (
                            components["docstring"].strip().split("\n", 1)[0]
                        )[:150]
                        md_file.write(
                            f"\n**Description**: {docstring_summary}{'...' if len(components['docstring']) > 150 else ''}\n"
                        )
                    if components.get("classes"):
                        md_file.write("\n**Classes**:\n")
                        for cls in components["classes"][:5]:
                            md_file.write(
                                f"- `{cls['name']}` ({len(cls['methods'])} methods)\n"
                            )
                        if len(components["classes"]) > 5:
                            md_file.write("- ... (and more)\n")
                    if components.get("functions"):
                        md_file.write("\n**Functions**:\n")
                        for func in components["functions"][:5]:
                            md_file.write(f"- `{func['name']}(...)`\n")
                        if len(components["functions"]) > 5:
                            md_file.write("- ... (and more)\n")

                # ==================================
                # --- Include FULL File Content ---
                md_file.write("\n**Content**:\n")  # Changed from "Content Snippet"
                file_ext = Path(file_abs_path).suffix.lower()
                lang_hint = file_ext.lstrip(".") if file_ext else ""
                md_file.write(f"```{lang_hint}\n")
                try:
                    with open(
                        file_abs_path, "r", encoding="utf-8", errors="ignore"
                    ) as code_file:
                        # Read the entire file content
                        full_content = code_file.read()
                        md_file.write(full_content)
                        # Ensure a newline at the end of the code block if file doesn't have one
                        if not full_content.endswith("\n"):
                            md_file.write("\n")
                except Exception as e:
                    md_file.write(f"Error reading file: {e}\n")
                md_file.write("```\n\n")

        # --- Other Markdown Files Section ---
        md_file.write("## Other Files\n\n")
        md_file.write("This section includes content of all other analyzed files that aren't in the key files list.\n\n")
        
        # Filter out key files
        other_files = [f for f in files if f not in key_files]
        
        if other_files:
            for file_abs_path in other_files:
                try:
                    rel_path = str(Path(file_abs_path).relative_to(report_base_path))
                except ValueError:
                    rel_path = file_abs_path  # Fallback to absolute if not relative

                md_file.write(f"### {rel_path}\n\n")
                metadata = get_file_metadata(file_abs_path)
                md_file.write(f"- **Lines**: {metadata.get('line_count', 'N/A')}\n")
                md_file.write(
                    f"- **Size**: {metadata.get('size_bytes', 0) / 1024:.2f} KB\n"
                )
                md_file.write(
                    f"- **Last modified**: {metadata.get('last_modified', 'Unknown')}\n"
                )

                # Include full file content
                md_file.write("\n**Content**:\n")
                file_ext = Path(file_abs_path).suffix.lower()
                lang_hint = file_ext.lstrip(".") if file_ext else ""
                md_file.write(f"```{lang_hint}\n")
                try:
                    with open(
                        file_abs_path, "r", encoding="utf-8", errors="ignore"
                    ) as code_file:
                        # Read the entire file content
                        full_content = code_file.read()
                        md_file.write(full_content)
                        # Ensure a newline at the end of the code block if file doesn't have one
                        if not full_content.endswith("\n"):
                            md_file.write("\n")
                except Exception as e:
                    md_file.write(f"Error reading file: {e}\n")
                md_file.write("```\n\n")
        else:
            md_file.write("No additional files found.\n\n")

        # --- Python Dependency Analysis (if applicable) ---
        if has_python_files and dependencies:
            md_file.write("## Python Dependencies\n\n")
            md_file.write(
                "This section shows Python modules and their dependencies within the project.\n\n"
            )

            dep_count = sum(len(deps) for deps in dependencies.values())
            if dep_count > 0:
                md_file.write("### Internal Dependencies\n\n")
                md_file.write("```mermaid\ngraph TD;\n")
                # Generate mermaid.js compatible graph nodes and edges
                node_ids = {}
                for i, file_path in enumerate(dependencies.keys()):
                    try:
                        rel_path = str(Path(file_path).relative_to(report_base_path))
                    except ValueError:
                        rel_path = str(
                            Path(file_path).name
                        )  # Just use filename if not relative
                    node_id = f"F{i}"
                    node_ids[file_path] = node_id
                    # Escape any problematic characters in label
                    label = rel_path.replace('"', '\\"')
                    md_file.write(f'    {node_id}["{label}"];\n')

                # Add edges for dependencies
                for file_path, deps in dependencies.items():
                    if not deps:
                        continue
                    source_id = node_ids[file_path]
                    for dep in deps:
                        if dep in node_ids:  # Ensure dep is in our analyzed files
                            target_id = node_ids[dep]
                            md_file.write(f"    {source_id} --> {target_id};\n")
                md_file.write("```\n\n")

                # Add plain text dependency list as fallback
                md_file.write("### Dependency List (Plain Text)\n\n")
                for file_path, deps in dependencies.items():
                    if not deps:
                        continue  # Skip files with no dependencies
                    try:
                        rel_path = str(Path(file_path).relative_to(report_base_path))
                    except ValueError:
                        rel_path = file_path
                    md_file.write(f"- **{rel_path}** depends on:\n")
                    for dep in sorted(deps):
                        try:
                            dep_rel = str(Path(dep).relative_to(report_base_path))
                        except ValueError:
                            dep_rel = dep
                        md_file.write(f"  - {dep_rel}\n")

        # --- Common Code Patterns ---
        if patterns and patterns.get("python_patterns"):
            py_patterns = patterns["python_patterns"]
            if py_patterns:
                md_file.write("## Common Code Patterns\n\n")
                md_file.write("### Python Patterns\n\n")

                if py_patterns.get("common_imports"):
                    md_file.write("#### Common Imports\n\n")
                    for imp, count in py_patterns["common_imports"][:10]:
                        md_file.write(f"- `{imp}` ({count} files)\n")
                    if len(py_patterns["common_imports"]) > 10:
                        md_file.write("- *(and more...)*\n")
                    md_file.write("\n")

                if py_patterns.get("framework_patterns"):
                    md_file.write("#### Framework Detection\n\n")
                    for framework, evidence in py_patterns[
                        "framework_patterns"
                    ].items():
                        md_file.write(f"- **{framework}**: {evidence}\n")
                    md_file.write("\n")

    # Final success message
    print_success(f"Markdown report generated successfully at '{output_path}'")


# --- Repository Object Generation ---
def generate_repository(
    files: List[str],  # List of absolute paths
    analyzed_extensions: Set[str],  # Set of actual extensions found (e.g., '.py', '.js')
    dependencies: Dict[str, Set[str]],  # Python dependencies
    patterns: Dict[str, Any],  # Detected patterns
    key_files: List[str],  # List of absolute paths for key files
    repo_name: str = "Repository Analysis",
    root_folder_display: str = ".",  # How to display the root in summary/tree
    calculate_metrics: bool = False,  # New parameter to control metrics calculation
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
    
    # Pre-calculate metrics for Python files to include in statistics, but only if metrics are enabled
    python_files = [f for f in files if f.lower().endswith('.py')]
    metrics_by_file = {}
    
    if calculate_metrics and python_files:
        print_subheader("Calculating Code Quality Metrics", "blue")
        
        # Track overall metrics
        total_complexity = 0
        files_with_complexity = 0
        total_maintainability = 0
        files_with_maintainability = 0
        complexity_ranks = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        maintainability_ranks = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        total_code_smells = 0
        
        # Calculate metrics for each Python file
        for file_path in python_files:
            metrics = calculate_file_metrics(file_path)
            metrics_by_file[file_path] = metrics
            
            # Aggregate statistics
            cc = metrics.get("cyclomatic_complexity", {})
            if cc and "total" in cc:
                total_complexity += cc["total"]
                files_with_complexity += 1
                if "rank" in cc:
                    complexity_ranks[cc["rank"]] = complexity_ranks.get(cc["rank"], 0) + 1
            
            mi = metrics.get("maintainability_index", {})
            if mi and "value" in mi:
                total_maintainability += mi["value"]
                files_with_maintainability += 1
                if "rank" in mi:
                    maintainability_ranks[mi["rank"]] = maintainability_ranks.get(mi["rank"], 0) + 1
            
            smells = metrics.get("code_smells", [])
            total_code_smells += len(smells)
        
        # Add code quality metrics to statistics
        avg_complexity = total_complexity / files_with_complexity if files_with_complexity > 0 else 0
        avg_maintainability = total_maintainability / files_with_maintainability if files_with_maintainability > 0 else 0
        
        complexity_distribution = ", ".join([f"{rank}: {count}" for rank, count in complexity_ranks.items() if count > 0])
        maintainability_distribution = ", ".join([f"{rank}: {count}" for rank, count in maintainability_ranks.items() if count > 0])
        
        quality_stats = f"""
- Average cyclomatic complexity: {avg_complexity:.2f}
- Complexity distribution: {complexity_distribution}
- Average maintainability index: {avg_maintainability:.2f}
- Maintainability distribution: {maintainability_distribution}
- Total code smells detected: {total_code_smells}
"""
        
        # Add code quality metrics to main statistics if metrics were calculated
        if files_with_complexity > 0 or files_with_maintainability > 0:
            statistics += quality_stats

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
            
            # Use pre-calculated metrics if available, otherwise calculate them now
            complexity_metrics = {}
            if calculate_metrics:
                complexity_metrics = metrics_by_file.get(file_abs_path, {})
                if not complexity_metrics:
                    complexity_metrics = calculate_file_metrics(file_abs_path)
            
            project_file = ProjectCodeFile(
                file_id=file_id,
                description=description,
                file_path=rel_path,
                content=content,
                line_count=metadata.get('line_count', 0),
                dependencies=file_deps,
                used_by=file_used_by,
                complexity_metrics=complexity_metrics
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

```

---

### 10. src/copilot_toolkit/collector/utils.py

- **File ID**: file_9
- **Type**: Code File
- **Line Count**: 133
- **Description**: File at src/copilot_toolkit/collector/utils.py
- **Dependencies**: None
- **Used By**:
  - file_8
  - file_5
  - file_7
  - file_4
  - file_3

**Content**:
```
# src/pilot_rules/collector/utils.py
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import box

# Create a shared console instance for consistent styling
console = Console()


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
            # Attempt to read as text, fallback for binary or encoding issues
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                metadata["line_count"] = len(content.splitlines())
        except (OSError, UnicodeDecodeError) as read_err:
            # Handle cases where reading might fail (binary file, permissions etc.)
            console.print(
                f"[yellow]⚠ Warning:[/yellow] Could not read content/count lines for [cyan]{file_path}[/cyan]: [red]{read_err}[/red]"
            )
            metadata["line_count"] = 0  # Indicate unreadable or binary

    except Exception as e:
        console.print(
            f"[yellow]⚠ Warning:[/yellow] Could not get complete metadata for [cyan]{file_path}[/cyan]: [red]{e}[/red]"
        )

    return metadata


# --- Rich Formatting Utilities ---


def print_header(title: str, style: str = "blue") -> None:
    """Print a styled header with a panel."""
    console.rule()
    console.print(
        Panel.fit(f"[bold {style}]{title}[/bold {style}]", border_style=style)
    )


def print_subheader(title: str, style: str = "cyan") -> None:
    """Print a styled subheader."""
    console.print(f"[bold {style}]== {title} ==[/bold {style}]")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str, exit_code: Optional[int] = None) -> None:
    """Print an error message and optionally exit."""
    console.print(f"[bold red]✗ ERROR:[/bold red] {message}")
    if exit_code is not None:
        exit(exit_code)


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠ Warning:[/yellow] {message}")


def create_progress() -> Progress:
    """Create a standardized progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TextColumn("[bold]{task.completed}/{task.total}"),
        console=console,
    )


def create_task_table(title: str) -> Table:
    """Create a standardized table for displaying task information."""
    table = Table(
        title=title, show_header=True, header_style="bold cyan", box=box.ROUNDED
    )
    return table


def print_file_stats(files: List[str], title: str = "File Statistics") -> None:
    """Print statistics about a list of files."""
    if not files:
        console.print("[yellow]No files found to display statistics.[/yellow]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green")

    extensions = {Path(f).suffix.lower() for f in files if Path(f).suffix}
    total_size = sum(get_file_metadata(f).get("size_bytes", 0) for f in files)
    total_lines = sum(get_file_metadata(f).get("line_count", 0) for f in files)

    table.add_row("Total Files", str(len(files)))
    table.add_row("Total Size", f"{total_size / 1024:.2f} KB")
    table.add_row("Total Lines", str(total_lines))
    table.add_row("Extensions", ", ".join(sorted(extensions)) if extensions else "None")

    console.print(table)

```

---

### 11. src/copilot_toolkit/main.py

- **File ID**: file_10
- **Type**: Code File
- **Line Count**: 1085
- **Description**: File at src/copilot_toolkit/main.py
- **Dependencies**: None
- **Used By**:
  - file_0

**Content**:
```
# src/pilot_rules/main.py
import json
import os
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Third-party imports
from dotenv import set_key
import tomli
import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from copilot_toolkit import collector
from copilot_toolkit.agent import project_agent, speak_to_agent, task_agent
from copilot_toolkit.collector.utils import (
    print_header,
    print_success,
    print_warning,
    print_error,
)
from copilot_toolkit.utils.cli_helper import init_console

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
        while current_dir != current_dir.parent:  # Stop at root directory '/'
            pyproject_path = current_dir / "pyproject.toml"
            if pyproject_path.exists():
                # print(f"DEBUG: Found pyproject.toml at {pyproject_path}") # Debug print
                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomli.load(f)
                version = pyproject_data.get("project", {}).get("version", "0.0.0")
                if version != "0.0.0":
                    return version
                # If version is placeholder, keep searching upwards
            current_dir = current_dir.parent

        # If not found after searching upwards
        # print("DEBUG: pyproject.toml with version not found.") # Debug print
        return "0.0.0"  # Fallback if not found
    except Exception:
        # print(f"DEBUG: Error getting version: {e}") # Debug print
        import traceback

        traceback.print_exc()  # Print error during dev
        return "0.0.0"  # Fallback on error


def display_guide(guide_path: Path, console: Console) -> None:
    """
    Display the markdown guide using rich formatting.

    Args:
        guide_path: Path to the markdown guide file.
        console: The Rich console instance to use for output.
    """
    if not guide_path.is_file():
        print_error(f"Guide file not found at '{guide_path}'")
        return

    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        markdown = Markdown(markdown_content)
        console.print("\n")
        console.rule("[bold blue]Getting Started Guide")
        console.print("\n")
        console.print(markdown)
        console.print("\n")
        console.rule("[bold blue]End of Guide")

    except Exception as e:
        print_error(f"Error displaying guide '{guide_path}': {str(e)}")


def copy_template(
    template_type: str, root_dir: Path, console: Console
) -> Optional[Path]:
    """
    Copy template files based on the specified type ('cursor' or 'copilot').

    Args:
        template_type: Either 'cursor' or 'copilot'.
        root_dir: The root directory (usually CWD) where to copy the templates.
        console: The Rich console instance to use for output.

    Returns:
        Path to the relevant guide file if successful, None otherwise.
    """
    package_dir = Path(__file__).parent  # Directory where main.py is located
    templates_dir = package_dir / "templates"
    guides_dir = package_dir / "guides"

    source_dir: Optional[Path] = None
    target_dir: Optional[Path] = None
    guide_file: Optional[Path] = None

    if template_type == "cursor":
        source_dir = templates_dir / "cursor"
        target_dir = root_dir / ".cursor"  # Target is relative to root_dir (CWD)
        guide_file = guides_dir / "cursor.md"
    elif template_type == "copilot":
        source_dir = templates_dir / "github"
        target_dir = root_dir / ".github"  # Target is relative to root_dir (CWD)
        guide_file = guides_dir / "copilot.md"
    else:
        # This case should not be reached due to argparse mutual exclusion
        print_error(f"Internal Error: Unknown template type '{template_type}'")
        return None

    if not source_dir or not source_dir.is_dir():
        print_error(f"Template source directory not found: '{source_dir}'")
        return None
    if not guide_file or not guide_file.is_file():
        print_warning(f"Guide file not found: '{guide_file}'")
        # Decide whether to proceed without a guide or stop
        # return None # Stop if guide is essential

    # Create target directory if it doesn't exist
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print_error(f"Could not create target directory '{target_dir}': {e}")
        return None

    # Copy the contents
    print_header(f"Setting up {template_type.title()} Templates", "cyan")
    console.print(f"Target directory: [yellow]{target_dir}[/yellow]")

    # Use a spinner for copying files
    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]Copying {template_type} templates...", total=None
        )

        try:
            for item in source_dir.iterdir():
                target_path = target_dir / item.name
                progress.update(
                    task, description=f"[cyan]Copying [bold]{item.name}[/bold]..."
                )
                if item.is_file():
                    shutil.copy2(item, target_path)
                elif item.is_dir():
                    shutil.copytree(item, target_path, dirs_exist_ok=True)

            progress.update(task, description="[green]Copy completed successfully!")
            print_success(
                f"Successfully copied {template_type} templates to {target_dir}"
            )
            return guide_file  # Return path to guide file on success
        except Exception as e:
            progress.update(task, description=f"[red]Error copying files: {e}")
            print_error(
                f"Error copying templates from '{source_dir}' to '{target_dir}': {e}"
            )
            return None


# --- Main Application Logic ---


def run_interactive_mode(console: Console) -> None:
    """
    Run the application in interactive mode, allowing the user to select
    actions and parameters through questionary prompts.
    
    Args:
        console: The Rich console instance to use for output.
    """
    print_header("Interactive Mode", "blue")
    console.print("[bold]Welcome to the Copilot Toolkit Interactive Mode[/bold]\n")
    
    while True:
        # Main action selection
        action = questionary.select(
            "Select an action to perform:",
            choices=[
                "collect - Collect code from the repository",
                "cursor - Scaffold Cursor templates",
                "copilot - Scaffold Copilot templates",
                questionary.Separator(),
                "project - Create a project with user stories and tasks",
                "task - Create tasks for the next user story",
                "specs - Create a project specification",
                "app - Create a standalone webapp based on some data",
                questionary.Separator(),
                "set_key - Set the API key for the agent",
                "exit - Exit interactive mode"
            ]
        ).ask()
        
        if action is None or action.startswith("exit"):
            print_success("Exiting interactive mode")
            return
        
        # Extract the action type
        action_type = action.split(" - ")[0]
        
        # Handle different action types
        if action_type == "collect":
            run_interactive_collect(console)
        elif action_type == "specs":
            run_interactive_specs(console)
        elif action_type == "app":
            run_interactive_app(console)
        elif action_type == "project":
            run_interactive_project(console)
        elif action_type == "task":
            run_interactive_task(console)
        elif action_type == "cursor":
            # Scaffold cursor templates
            scaffold_root_dir = Path.cwd()
            guide_file = copy_template("cursor", scaffold_root_dir, console)
            if guide_file:
                if questionary.confirm("Would you like to view the guide?").ask():
                    display_guide(guide_file, console)
        elif action_type == "copilot":
            # Scaffold copilot templates
            scaffold_root_dir = Path.cwd()
            guide_file = copy_template("copilot", scaffold_root_dir, console)
            if guide_file:
                if questionary.confirm("Would you like to view the guide?").ask():
                    display_guide(guide_file, console)
        elif action_type == "set_key":
            key = questionary.password("Enter your API key:").ask()
            if key:
                try:
                    set_key(".env", "GEMINI_API_KEY", key)
                    print_success("API key set successfully in .env file")
                except Exception as e:
                    print_error(f"Error setting API key: {e}")
        
        # Ask if the user wants to continue or exit
        if not questionary.confirm("Would you like to perform another action?", default=True).ask():
            print_success("Exiting interactive mode")
            return
        
        console.print("\n" + "-" * 80 + "\n")  # Separator between actions

def run_interactive_collect(console: Console) -> None:
    """
    Run the code collection in interactive mode.
    
    Args:
        console: The Rich console instance to use for output
    """
    print_header("Code Collection", "cyan")
    
    # Get include paths
    include_args: List[str] = []
    while True:
        include_path = questionary.text(
            "Enter files to include (format: 'ext1,ext2:./folder' or '*:.') or leave empty to finish:",
            default="py:."
        ).ask()
        
        if not include_path:
            # If no includes provided and list is empty, add default
            if not include_args:
                include_args.append("py:.")
            break
        
        include_args.append(include_path)
        
    # Get exclude paths
    exclude_args: List[str] = []
    while True:
        exclude_path = questionary.text(
            "Enter paths to exclude (format: 'py:temp' or '*:node_modules') or leave empty to finish:"
        ).ask()
        
        if not exclude_path:
            break
            
        exclude_args.append(exclude_path)
    
    # Get repo name
    repo_name = questionary.text(
        "Name for the repository (leave empty for default 'Repository Analysis'):"
    ).ask()
    
    if not repo_name:
        repo_name = None
    
    # Get config file
    config_arg = questionary.text(
        "Path to a .toml configuration file (leave empty for none):"
    ).ask()
    
    if not config_arg:
        config_arg = None
    
    # Ask about metrics calculation
    calculate_metrics = questionary.confirm(
        "Would you like to calculate code quality metrics?",
        default=False
    ).ask()
    
    # Confirm the selections
    console.print("\n[bold]Collection Configuration:[/bold]")
    console.print(f"[cyan]Include paths:[/cyan] {include_args}")
    console.print(f"[cyan]Exclude paths:[/cyan] {exclude_args}")
    console.print(f"[cyan]Repository name:[/cyan] {repo_name or 'Repository Analysis'}")
    console.print(f"[cyan]Config file:[/cyan] {config_arg or 'None'}")
    console.print(f"[cyan]Calculate metrics:[/cyan] {'Yes' if calculate_metrics else 'No'}")
    
    if questionary.confirm("Proceed with collection?").ask():
        try:
            repository = collector.run_collection(
                include_args=include_args,
                exclude_args=exclude_args,
                repo_name=repo_name,
                config_arg=config_arg,
                calculate_metrics=calculate_metrics,
            )
            # Display repository using rich rendering methods
            repository.render_summary(console)
            repository.render_files(console)
            print_success("Repository object generated successfully")
            
            # Ask if user wants to save the repository to a file
            if questionary.confirm("Would you like to save the repository data to a file?").ask():
                from pathlib import Path
                
                # Ask for the export format
                export_format = questionary.select(
                    "Select export format:",
                    choices=["json", "markdown"]
                ).ask()
                
                output_path = questionary.text(
                    f"Enter the output file path (e.g., 'repository_data.{export_format}'):",
                    default=f"repository_data.{export_format}"
                ).ask()
                
                # Use the appropriate save method based on format
                if export_format == "json":
                    repository.save_to_json(output_path, console)
                else:
                    repository.save_to_markdown(output_path, console)
                    
        except Exception as e:
            print_error(f"Error during collection: {str(e)}")

def run_interactive_project(console: Console) -> None:
    """
    Run the project creation in interactive mode.
    """
    print_header("Project Creation", "green")

    # Get user instructions
    user_instructions = questionary.text(
        "Enter additional instructions for the agent (leave empty for none):"
    ).ask()
    
    if not user_instructions:
        user_instructions = ""
    
    # Confirm the selections
    console.print("\n[bold]Project Creation Configuration:[/bold]")
    console.print(f"[green]User instructions:[/green] {user_instructions or 'None'}")

    # Get output path
    output_path = questionary.path(
        "Enter the output path (leave empty for default):"
    ).ask()
    
    if not output_path:
        output_path = "project.json"

    if not output_path.endswith(".json"):
        output_path = output_path + ".json"

    # to real path
    output_path = Path(output_path).resolve()

    if questionary.confirm("Proceed with app creation?").ask():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[magenta]Creating app...", total=None)
            
            try:
                output = project_agent(
                    action="project", 
                    user_instructions=user_instructions,
                )
                progress.update(
                    task, description="[green]App created successfully!"
                )
                
                # Use the rendering methods instead of direct printing
                console.print("\n")
                output.render_summary(console)
                json_str = output.model_dump_json()
                # write json to output_path
                # create output_path if it doesn't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(json_str)
                #output.render_output_files(console)
                print_success("App creation process completed")
            except Exception as e:
                progress.update(task, description=f"[red]Error creating app: {e}")
                print_error(f"Error during app creation: {str(e)}")


def run_interactive_task(console: Console) -> None:
    """
    Run the project creation in interactive mode.
    """
    print_header("Project Implementation", "green")

    # Get user instructions
    project_file = questionary.path(
        "Enter the path to the project file:"
    ).ask()
    
    if not project_file:
        print_error("Project file is required")
        return

    # Get user instructions
    user_instructions = questionary.text(
        "Enter additional instructions for the agent (leave empty for none):"
    ).ask()
    
    if not user_instructions:
        user_instructions = ""
    
    # Confirm the selections
    console.print("\n[bold]Project Creation Configuration:[/bold]")
    console.print(f"[green]User instructions:[/green] {user_instructions or 'None'}")

    # Get output path
    output_path = questionary.path(
        "Enter the output folder (leave empty for default):"
    ).ask()
    
    if not output_path:
        output_path =".project/tasks"



    # to real path
    output_path = Path(output_path).resolve()

    if questionary.confirm("Proceed with app creation?").ask():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[magenta]Creating app...", total=None)
            
            try:
                output = task_agent(
                    action="task",
                    project_file=project_file,
                    user_instructions=user_instructions,
                )
                progress.update(
                    task, description="[green]App created successfully!"
                )
                
                # Use the rendering methods instead of direct printing
                console.print("\n")
             
                # write json to output_path
                # create output_path if it doesn't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # write output to output_path
                with open(output_path, "w") as f:
                    f.write(output.model_dump_json())

                #output.render_output_files(console)
                print_success("App creation process completed")
            except Exception as e:
                progress.update(task, description=f"[red]Error creating app: {e}")
                print_error(f"Error during app creation: {str(e)}")
    
    



def run_interactive_specs(console: Console) -> None:
    """
    Run the project specifications generation in interactive mode.
    
    Args:
        console: The Rich console instance to use for output.
    """
    print_header("Project Specifications Generation", "yellow")
    
    # Get input file or folder
    input_path = questionary.path(
        "Enter the path to the input file or folder:"
    ).ask()
    
    if not input_path:
        print_error("Input path is required")
        return
    
    # Get output path
    output_path = questionary.path(
        "Enter the output path (leave empty for default):"
    ).ask()
    
    if not output_path:
        output_path = None

    
    
    # Get user instructions
    user_instructions = questionary.text(
        "Enter additional instructions for the agent (leave empty for none):"
    ).ask()
    
    if not user_instructions:
        user_instructions = ""
    
    # Confirm the selections
    console.print("\n[bold]Specifications Configuration:[/bold]")
    console.print(f"[yellow]Input path:[/yellow] {input_path}")
    console.print(f"[yellow]Output path:[/yellow] {output_path or 'None'}")
    console.print(f"[yellow]User instructions:[/yellow] {user_instructions or 'None'}")
    
    if questionary.confirm("Proceed with specifications generation?").ask():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            console=console
        ) as progress:
            collect_task = progress.add_task(
                "[yellow]Processing input...", total=None
            )
            
            try:
                # If folder, run collection first to generate Repository object
                if os.path.isdir(input_path):
                    # Use the repository generation function
                    repository = collector.run_collection(
                        include_args=[f"py:./{input_path}"],
                        exclude_args=[],
                        config_arg=None,
                        calculate_metrics=False,  # Don't calculate metrics for specs generation
                    )
                    progress.update(
                        collect_task,
                        description="[green]Repository data collected!",
                    )
                    
                    # Display the repository summary and files
                    console.print("\n[bold]Repository Analysis:[/bold]")
                    repository.render_summary(console)
                    
                    # Ask if the user wants to see the files details
                    if questionary.confirm("Would you like to view the repository file details?", default=False).ask():
                        repository.render_files(console)
                    
                    # Now generate specs using the repository object directly
                    generate_task = progress.add_task(
                        "[yellow]Generating specifications...", total=None
                    )
                    
                    # Convert repository to JSON string for the agent to use
                    import json
                    repository_json = json.dumps(repository.dict())
                    
                    # Use the repository object with the agent
                    output = speak_to_agent(
                        action="specs", 
                        input_data=repository_json, 
                        user_instructions=user_instructions,
                    )
                    
                    progress.update(
                        generate_task,
                        description="[green]Specifications generated successfully!",
                    )
                
                # If file, use it directly (assuming it's a JSON repository data file)
                elif os.path.isfile(input_path):
                    generate_task = progress.add_task(
                        "[yellow]Generating specifications from file...", total=None
                    )
                    
                    # We assume JSON files contain repository data
                    
                    # If it's a repository JSON file, display it first
                    try:
                        import json
                        with open(input_path, 'r') as file:
                            repo_data = json.load(file)
                            from copilot_toolkit.model import Repository
                            repository = Repository.parse_obj(repo_data)
                            
                            # Display the repository summary
                            console.print("\n[bold]Repository from file:[/bold]")
                            repository.render_summary(console)
                            
                            # Ask if the user wants to see the files details
                            if questionary.confirm("Would you like to view the repository file details?", default=False).ask():
                                repository.render_files(console)
                    except Exception as e:
                        print_warning(f"Could not parse repository data from file: {e}")
                    
                    output = speak_to_agent(
                        action="specs", 
                        input_data=input_path, 
                        user_instructions=user_instructions,
                    )
                    progress.update(
                        generate_task,
                        description="[green]Specifications generated successfully!",
                    )
                else:
                    progress.update(collect_task, description="[red]Invalid input path")
                    raise ValueError(
                        f"Input path is neither a file nor a directory: {input_path}"
                    )
                
                # Display results
                console.print("\n")
                output.render_summary(console)
                output.render_output_files(console, output_path)
                
                print_success("Specification generation completed")
            except Exception as e:
                progress.update(
                    collect_task,
                    description=f"[red]Error: {e}",
                )
                print_error(f"Error during specifications generation: {str(e)}")

def run_interactive_app(console: Console) -> None:
    """
    Run the app creation in interactive mode.
    
    Args:
        console: The Rich console instance to use for output.
    """
    print_header("App Creation", "magenta")
    
    # Get input file
    input_file = questionary.path(
        "Enter the path to the input file:"
    ).ask()
    
    if not input_file:
        print_error("Input file is required")
        return
    
    # Get output path
    output_path = questionary.path(
        "Enter the output path (leave empty for default):"
    ).ask()
    
    if not output_path:
        output_path = None

   

    # Get user instructions
    user_instructions = questionary.text(
        "Enter additional instructions for the agent (leave empty for none):"
    ).ask()
    
    if not user_instructions:
        user_instructions = ""
    
    # Confirm the selections
    console.print("\n[bold]App Creation Configuration:[/bold]")
    console.print(f"[magenta]Input file:[/magenta] {input_file}")
    console.print(f"[magenta]Output path:[/magenta] {output_path or 'Default'}")
    console.print(f"[magenta]User instructions:[/magenta] {user_instructions or 'None'}")
    
    if questionary.confirm("Proceed with app creation?").ask():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[magenta]Creating app...", total=None)
            
            try:
                output = speak_to_agent(
                    action="app", 
                    input_data=input_file, 
                    user_instructions=user_instructions,
                )
                progress.update(
                    task, description="[green]App created successfully!"
                )
                
                # Use the rendering methods instead of direct printing
                console.print("\n")
                output.render_summary(console)
                output.render_output_files(console, output_path)
                print_success("App creation process completed")
            except Exception as e:
                progress.update(task, description=f"[red]Error creating app: {e}")
                print_error(f"Error during app creation: {str(e)}")

def main():
    """
    Entry point for the pilot-rules CLI application.
    Handles argument parsing and delegates tasks to scaffolding or collection modules.
    """
    console = Console()
    console.clear()

    init_console()

    parser = argparse.ArgumentParser(
        description="Manage Pilot Rules templates or collect code for analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Mutually Exclusive Actions ---
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--cursor", action="store_true", help="Scaffold Cursor templates (.cursor)"
    )
    action_group.add_argument(
        "--copilot", action="store_true", help="Scaffold Copilot templates (.github)"
    )
    action_group.add_argument(
        "--collect", action="store_true", help="Collect code from the repository"
    )
    action_group.add_argument(
        "--app",
        action="store_true",
        help="Create a standalone webapp based on some data",
    )
    action_group.add_argument(
        "--prompt", action="store_true", help="Prompt an agent to do something"
    )
    action_group.add_argument("--build", action="store_true", help="Build the project")
    action_group.add_argument("--clean", action="store_true", help="Clean the project")
    action_group.add_argument(
        "--init", action="store_true", help="Initialize a new project"
    )
    action_group.add_argument(
        "--interactive", action="store_true", help="Interactive mode"
    )
    action_group.add_argument(
        "--specs", action="store_true", help="Create a project specification"
    )
    action_group.add_argument(
        "--project", action="store_true", help="Create a project"
    )
    action_group.add_argument(
        "--set_key", metavar="KEY", help="Set the API key for the agent"
    )

    argument_group = parser.add_argument_group(
        "Additional Options"
    )
    argument_group.add_argument(
        "--user_instructions",
        help="Additional instructions to pass to the agent"
    )
    argument_group.add_argument(
        "--prompts",
        help="Path to the folder containing prompt files (default: 'prompts')"
    )
    # --- Options for Code Collection ---
    collect_group = parser.add_argument_group(
        "Code Collection Options (used with --collect)"
    )
    collect_group.add_argument(
        "--include",
        action="append",
        metavar="EXTS:FOLDER",
        help="Specify files to include. Format: 'ext1,ext2:./folder' or '*:.'."
        " Can be used multiple times. Default: 'py:.' if no includes provided.",
    )
    collect_group.add_argument(
        "--exclude",
        action="append",
        metavar="EXTS_OR_*:PATTERN",
        help="Specify path patterns to exclude. Format: 'py:temp' or '*:node_modules'."
        " '*' matches any extension. Can be used multiple times.",
    )
    collect_group.add_argument(
        "--output",
        default=None,
        metavar="FILEPATH",
        help="Path to save the output JSON file with repository data",
    )
    collect_group.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Format to save the repository data (json or markdown)",
    )
    collect_group.add_argument(
        "--input",
        default=None,
        metavar="FILEPATH",
        help="Path to the input file or folder",
    )
    
    collect_group.add_argument(
        "--config",
        metavar="TOML_FILE",
        help="Path to a .toml configuration file for collection settings.",
    )
    
    collect_group.add_argument(
        "--repo-name",
        metavar="NAME",
        help="Name for the repository",
    )

    collect_group.add_argument(
        "--metrics",
        action="store_true",
        help="Calculate code quality metrics (cyclomatic complexity, maintainability, etc.)",
    )

    args = parser.parse_args()

    # Root directory for scaffolding is the current working directory
    scaffold_root_dir = Path.cwd()
    guide_file_to_display: Optional[Path] = None

    try:
        if args.interactive:
            run_interactive_mode(console)
        elif args.collect:
            print_header("Code Collection Mode", "cyan")
            
            # Generate a Repository object
            repository = collector.run_collection(
                include_args=args.include,
                exclude_args=args.exclude,
                output_arg=args.output,
                config_arg=args.config,
                repo_name=args.repo_name,
                calculate_metrics=args.metrics,
            )
            
            # Display repository using rich rendering methods
            repository.render_summary(console)
            repository.render_files(console)
            
            # Save repository data if output path is specified
            if args.output:
                if args.format == "json":
                    repository.save_to_json(args.output, console)
                else:
                    repository.save_to_markdown(args.output, console)
            
            print_success("Repository analysis completed successfully")

        elif args.cursor:
            guide_file_to_display = copy_template("cursor", scaffold_root_dir, console)
            # Success/Error messages printed within copy_template

        elif args.copilot:
            guide_file_to_display = copy_template("copilot", scaffold_root_dir, console)
            # Success/Error messages printed within copy_template

        elif args.project:
            print_header("Project Creation Mode", "green")

            # Set defaults for prompts and user_instructions if not provided
            user_instructions = args.user_instructions if args.user_instructions else ""
            output_path = Path(args.output).resolve() if args.output else None

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[green]Creating project...", total=None)
                try:
                    output = project_agent(
                        action="project", 
                        user_instructions=user_instructions,
                    )
                    progress.update(
                        task, description="[green]Project created successfully!"
                    )

                    # Use the rendering methods instead of direct printing
                    console.print("\n")
                    output.render_summary(console)
                    output.render_summary(console)
                    json_str = output.model_dump_json()
                    # write json to output_path
                    # create output_path if it doesn't exist
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        f.write(json_str)
                    #output.render_output_files(console)
                    print_success("App creation process completed")
                    print_success("App creation process completed")
                except Exception as e:
                    progress.update(task, description=f"[red]Error creating app: {e}")
                    raise

        elif args.app:
            print_header("App Creation Mode", "magenta")

            # Set defaults for prompts and user_instructions if not provided
            user_instructions = args.user_instructions if args.user_instructions else ""

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold magenta]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[magenta]Creating app...", total=None)
                try:
                    output = speak_to_agent(
                        action="app", 
                        input_data=args.input, 
                        user_instructions=user_instructions,
                    )
                    progress.update(
                        task, description="[green]App created successfully!"
                    )

                    # Use the rendering methods instead of direct printing
                    console.print("\n")
                    output.render_summary(console)
                    output.render_output_files(console)
                    print_success("App creation process completed")
                except Exception as e:
                    progress.update(task, description=f"[red]Error creating app: {e}")
                    raise

        elif args.specs:
            file_or_folder = args.input
            print_header("Project Specifications Generation", "yellow")

            # Set defaults for prompts and user_instructions if not provided
            user_instructions = args.user_instructions if args.user_instructions else ""

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold yellow]{task.description}"),
                console=console,
            ) as progress:
                collect_task = progress.add_task(
                    "[yellow]Collecting repository data...", total=None
                )

                # If folder, run collection first
                if os.path.isdir(file_or_folder):
                    try:
                        repository = collector.run_collection(
                            include_args=[f"py:./{file_or_folder}"],
                            exclude_args=[],
                            config_arg=None,
                            calculate_metrics=True, 
                        )
                        progress.update(
                            collect_task,
                            description="[green]Repository data collected!",
                        )

                        # Display repository
                        console.print("\n")
                        repository.render_summary(console)

                        # Now generate specs from the repository
                        generate_task = progress.add_task(
                            "[yellow]Generating specifications...", total=None
                        )
                        
                        # Convert repository to JSON
                        import json
                        repository_json = json.dumps(repository.dict())
                        
                        output = speak_to_agent(
                            action="specs", 
                            input_data=repository_json, 
                            user_instructions=user_instructions,
                        )
                        progress.update(
                            generate_task,
                            description="[green]Specifications generated successfully!",
                        )
                    except Exception as e:
                        progress.update(
                            collect_task,
                            description=f"[red]Error during collection: {e}",
                        )
                        raise

                # If file, use it directly as repository JSON
                elif os.path.isfile(file_or_folder):
                    try:
                        generate_task = progress.add_task(
                            "[yellow]Generating specifications from file...", total=None
                        )
                        
                        output = speak_to_agent(
                            action="specs", 
                            input_data=file_or_folder, 
                            user_instructions=user_instructions,
                        )
                        progress.update(
                            generate_task,
                            description="[green]Specifications generated successfully!",
                        )
                    except Exception as e:
                        progress.update(
                            generate_task,
                            description=f"[red]Error generating specifications: {e}",
                        )
                        raise
                else:
                    progress.update(collect_task, description="[red]Invalid input path")
                    raise ValueError(
                        f"Input path is neither a file nor a directory: {file_or_folder}"
                    )

            # Display results using the rendering methods
            console.print("\n")
            output.render_summary(console)
            output.render_output_files(console)

            print_success("Specification generation completed")

        elif args.set_key:
            print_header("Setting API Key", "green")
            try:
                set_key(".env", "GEMINI_API_KEY", args.set_key)
                print_success("API key set successfully in .env file")
            except Exception as e:
                print_error(f"Error setting API key: {e}")

        # Display guide only if scaffolding was successful and returned a guide path
        if guide_file_to_display:
            display_guide(guide_file_to_display, console)

    except FileNotFoundError as e:
        # Should primarily be caught within helpers now, but keep as fallback
        print_error(f"Required file or directory not found: {str(e)}")
        exit(1)
    except ValueError as e:  # Catch config errors propagated from collector
        print_error(f"Configuration Error: {str(e)}")
        exit(1)
    except Exception as e:
        # Catch-all for unexpected errors in main logic or propagated from helpers/collector
        print_error(f"An unexpected error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)


# --- Standard Python entry point check ---
if __name__ == "__main__":
    main()

```

---

### 12. src/copilot_toolkit/model.py

- **File ID**: file_11
- **Type**: Code File
- **Line Count**: 585
- **Description**: File at src/copilot_toolkit/model.py
- **Dependencies**: None
- **Used By**:
  - file_8
  - file_3

**Content**:
```
from typing import Any, Literal, Optional
from flock.core.flock_registry import flock_type
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from pathlib import Path


@flock_type
class OutputData(BaseModel):
    name: str = Field(..., description="Name of the output")
    description: str = Field(
        ...,
        description="High level description of the data and functionality of the app, as well as design decisions. In beautiful markdown.",
    )
    output_dictionary_definition: str = Field(
        ..., description="Explanation of the output dictionary and the data it contains"
    )
    output: dict[str, Any] = Field(
        ...,
        description="The output dictionary. Usually a dictionary with keys equals paths to files, and values equal the content of the files.",
    )

    # beautiful rendering of the output
    def render_summary(self, console: Console):
        console.print("\n")
        console.rule(f"[bold blue]{self.name}")
        console.print("\n")
        console.print(self.description)
        console.print("\n")
        console.print(self.output_dictionary_definition)

    def render_output_files(self, console: Console, output_prefix: str = ".project/"):
        """
        Renders the output files in a beautiful table format.

        Args:
            console: The Rich console instance to use for output
            output_prefix: Optional prefix to prepend to file paths (defaults to '.project/')
        """
        console.print("\n")
        console.rule("[bold cyan]Output Files")
        console.print("\n")

        # Create a nice table to display the files
        files_table = Table(title="Generated Files", box=box.ROUNDED)
        files_table.add_column("File Path", style="cyan")
        files_table.add_column("Status", style="green")
        files_table.add_column("Size", style="magenta")

        file_count = 0

        # Process each output file
        for file_path, content in self.output.items():
            file_path_with_prefix = f"{output_prefix}{file_path}"
            # if is directory create and skip
            if Path(file_path_with_prefix).is_dir():
                Path(file_path_with_prefix).mkdir(parents=True, exist_ok=True)
                continue

            file_count += 1

            # Calculate the content size
            content_size = len(content) if content else 0
            size_display = (
                f"{content_size / 1024:.2f} KB"
                if content_size > 1024
                else f"{content_size} bytes"
            )

            # Create entry in the table
            try:
                # Create the directory if it doesn't exist
                Path(file_path_with_prefix).parent.mkdir(parents=True, exist_ok=True)

                # Write the file content
                with open(file_path_with_prefix, "w", encoding="utf-8") as f:
                    f.write(content)

                files_table.add_row(
                    file_path_with_prefix, "[green]✓ Created[/green]", size_display
                )
            except Exception as e:
                files_table.add_row(
                    file_path_with_prefix, f"[red]✗ Error: {str(e)}[/red]", size_display
                )

        if file_count > 0:
            console.print(files_table)
            console.print(
                f"\n[green]Successfully generated {file_count} files.[/green]\n"
            )
        else:
            console.print("[yellow]No output files were generated.[/yellow]\n")

        console.rule("[bold cyan]End of Output")

class ProjectFile(BaseModel):
    file_id: str = Field(..., description="Unique identifier for the file")
    description: str = Field(..., description="Description of the file")
    file_path: str = Field(..., description="Path to the file")
    content: str = Field(..., description="Content of the file")
    line_count: int = Field(..., description="Number of lines in the file")

class ProjectCodeFile(ProjectFile):
    dependencies: list[str] = Field(..., description="List of file ids that must be created before this one")
    used_by: list[str] = Field(..., description="List of file ids that depend on this one")
    complexity_metrics: dict[str, Any] = Field(default_factory=dict, description="Code quality and complexity metrics")


@flock_type
class Repository(BaseModel):
    name: str = Field(..., description="Name of the repository")
    statistics: str = Field(..., description="Statistics of the repository")
    project_files: list[ProjectFile | ProjectCodeFile] = Field(..., description="Output data of the repository")

    def render_summary(self, console: Console) -> None:
        """
        Render a summary of the repository in a beautiful format.
        
        Args:
            console: The Rich console instance to use for output
        """
        console.print("\n")
        console.rule(f"[bold blue]{self.name}")
        console.print("\n")
        
        # Create a table for statistics
        stats_table = Table(title="Repository Statistics", box=box.ROUNDED)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="green")
        
        # Parse statistics string into individual items
        for stat_line in self.statistics.strip().split('\n'):
            if stat_line and '-' in stat_line:
                key, value = stat_line.split(':', 1) if ':' in stat_line else stat_line.split('-', 1)
                stats_table.add_row(key.strip('- '), value.strip())
        
        console.print(stats_table)
        console.print("\n")
    
    def render_files(self, console: Console, max_files: int = 20) -> None:
        """
        Render the repository files in a beautiful table format.
        
        Args:
            console: The Rich console instance to use for output
            max_files: Maximum number of files to display (default: 20)
        """
        console.print("\n")
        console.rule("[bold cyan]Repository Files")
        console.print("\n")
        
        # Create a table for files
        files_table = Table(title="Project Files", box=box.ROUNDED)
        files_table.add_column("File ID", style="cyan")
        files_table.add_column("Path", style="magenta")
        files_table.add_column("Lines", style="green")
        files_table.add_column("Type", style="yellow")
        files_table.add_column("Complexity", style="red")
        files_table.add_column("Maintainability", style="blue")
        
        # Add files to the table (limited to max_files)
        for file in self.project_files[:max_files]:
            file_type = "Code File" if isinstance(file, ProjectCodeFile) else "File"
            
            complexity = "-"
            maintainability = "-"
            if isinstance(file, ProjectCodeFile) and file.complexity_metrics:
                cc = file.complexity_metrics.get("cyclomatic_complexity", {})
                mi = file.complexity_metrics.get("maintainability_index", {})
                
                if cc and "rank" in cc:
                    complexity = f"{cc.get('total', '?')} ({cc.get('rank', '?')})"
                
                if mi and "rank" in mi:
                    maintainability = f"{int(mi.get('value', 0))} ({mi.get('rank', '?')})"
            
            files_table.add_row(
                file.file_id,
                file.file_path,
                str(file.line_count),
                file_type,
                complexity,
                maintainability
            )
        
        if len(self.project_files) > max_files:
            files_table.add_row(
                "...",
                f"[yellow]And {len(self.project_files) - max_files} more files...[/yellow]",
                "",
                "",
                "",
                ""
            )
        
        console.print(files_table)
        
        # Dependency information (only if there are ProjectCodeFiles)
        code_files = [f for f in self.project_files if isinstance(f, ProjectCodeFile)]
        if code_files:
            console.print("\n")
            console.rule("[bold cyan]File Dependencies")
            console.print("\n")
            
            deps_table = Table(title="Dependencies Between Files", box=box.ROUNDED)
            deps_table.add_column("File", style="cyan")
            deps_table.add_column("Depends On", style="magenta")
            deps_table.add_column("Used By", style="green")
            
            for file in code_files[:max(10, max_files // 2)]:  # Show fewer files in dependency table
                # Get actual file paths instead of IDs for better readability
                depends_on_paths = []
                for dep_id in file.dependencies:
                    for f in self.project_files:
                        if f.file_id == dep_id:
                            depends_on_paths.append(f.file_path)
                            break
                
                used_by_paths = []
                for used_id in file.used_by:
                    for f in self.project_files:
                        if f.file_id == used_id:
                            used_by_paths.append(f.file_path)
                            break
                
                deps_table.add_row(
                    file.file_path,
                    "\n".join(depends_on_paths[:5]) + ("\n..." if len(depends_on_paths) > 5 else ""),
                    "\n".join(used_by_paths[:5]) + ("\n..." if len(used_by_paths) > 5 else "")
                )
            
            if len(code_files) > max(10, max_files // 2):
                deps_table.add_row(
                    "[yellow]And more files...[/yellow]",
                    "",
                    ""
                )
                
            console.print(deps_table)
            
            # New Code Metrics Table
            files_with_metrics = [f for f in code_files if isinstance(f, ProjectCodeFile) and f.complexity_metrics]
            if files_with_metrics:
                console.print("\n")
                console.rule("[bold cyan]Code Quality Metrics")
                console.print("\n")
                
                metrics_table = Table(title="Code Complexity and Quality", box=box.ROUNDED)
                metrics_table.add_column("File", style="cyan")
                metrics_table.add_column("Cyclomatic Complexity", style="red")
                metrics_table.add_column("Maintainability", style="blue")
                metrics_table.add_column("Code Smells", style="yellow")
                
                for file in files_with_metrics[:max(10, max_files // 2)]:
                    metrics = file.complexity_metrics
                    cc = metrics.get("cyclomatic_complexity", {})
                    mi = metrics.get("maintainability_index", {})
                    smells = metrics.get("code_smells", [])
                    
                    cc_display = "[grey]-[/grey]"
                    if cc:
                        rank = cc.get("rank", "?")
                        rank_color = {
                            "A": "green", "B": "green",
                            "C": "yellow", "D": "red",
                            "F": "red bold"
                        }.get(rank, "white")
                        cc_display = f"Total: {cc.get('total', '?')} | Avg: {cc.get('average', '?')} | Rank: [{rank_color}]{rank}[/{rank_color}]"
                    
                    mi_display = "[grey]-[/grey]"
                    if mi:
                        rank = mi.get("rank", "?")
                        rank_color = {
                            "A": "green", "B": "green",
                            "C": "yellow", "D": "red",
                            "F": "red bold"
                        }.get(rank, "white")
                        mi_display = f"Index: {int(mi.get('value', 0))} | Rank: [{rank_color}]{rank}[/{rank_color}]"
                    
                    smells_display = "[grey]None detected[/grey]"
                    if smells:
                        smells_list = []
                        for i, smell in enumerate(smells[:3]):  # Show up to 3 smells
                            smells_list.append(f"{smell['type']} in {smell['location']}")
                        if len(smells) > 3:
                            smells_list.append(f"...and {len(smells) - 3} more")
                        smells_display = "\n".join(smells_list)
                    
                    metrics_table.add_row(
                        file.file_path,
                        cc_display,
                        mi_display,
                        smells_display
                    )
                
                if len(files_with_metrics) > max(10, max_files // 2):
                    metrics_table.add_row(
                        "[yellow]And more files...[/yellow]",
                        "",
                        "",
                        ""
                    )
                    
                console.print(metrics_table)
        
        console.rule("[bold cyan]End of Repository")
    
    def save_to_json(self, output_path: str, console: Optional[Console] = None) -> bool:
        """
        Save the repository data to a JSON file.
        
        Args:
            output_path: The path where to save the JSON file
            console: Optional Rich console for output messages
            
        Returns:
            True if successful, False otherwise
        """
        import json
        from pathlib import Path
        
        try:
            # Convert repository to dict and save as JSON
            repo_dict = self.dict()
            file_path = Path(output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(repo_dict, f, indent=2)
            
            if console:
                console.print(f"\n[green]Repository data saved to {output_path}[/green]")
            
            return True
        except Exception as e:
            if console:
                console.print(f"\n[red]Error saving repository data: {str(e)}[/red]")
            return False
            
    def save_to_markdown(self, output_path: str, console: Optional[Console] = None) -> bool:
        """
        Save the repository data to a Markdown file.
        This exports the same data as save_to_json but in Markdown format.
        
        Args:
            output_path: The path where to save the Markdown file
            console: Optional Rich console for output messages
            
        Returns:
            True if successful, False otherwise
        """
        from pathlib import Path
        
        try:
            file_path = Path(output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as md_file:
                # Write repository header and basic info
                md_file.write(f"# {self.name}\n\n")
                
                # Write statistics section
                md_file.write("## Repository Statistics\n\n")
                for stat_line in self.statistics.strip().split('\n'):
                    if stat_line and '-' in stat_line:
                        if ':' in stat_line:
                            key, value = stat_line.split(':', 1)
                        else:
                            key, value = stat_line.split('-', 1)
                        md_file.write(f"- **{key.strip('- ')}**: {value.strip()}\n")
                md_file.write("\n")
                
                # Write files section with complete data for each file
                md_file.write("## Project Files\n\n")
                
                for i, file in enumerate(self.project_files):
                    # Make a header for each file
                    md_file.write(f"### {i+1}. {file.file_path}\n\n")
                    md_file.write(f"- **File ID**: {file.file_id}\n")
                    md_file.write(f"- **Type**: {'Code File' if isinstance(file, ProjectCodeFile) else 'File'}\n")
                    md_file.write(f"- **Line Count**: {file.line_count}\n")
                    md_file.write(f"- **Description**: {file.description}\n")
                    
                    # If it's a code file, include additional information
                    if isinstance(file, ProjectCodeFile):
                        # Dependencies
                        if file.dependencies:
                            md_file.write("- **Dependencies**:\n")
                            for dep_id in file.dependencies:
                                md_file.write(f"  - {dep_id}\n")
                        else:
                            md_file.write("- **Dependencies**: None\n")
                        
                        # Used by
                        if file.used_by:
                            md_file.write("- **Used By**:\n")
                            for used_id in file.used_by:
                                md_file.write(f"  - {used_id}\n")
                        else:
                            md_file.write("- **Used By**: None\n")
                        
                        # Complexity metrics
                        if file.complexity_metrics:
                            md_file.write("- **Complexity Metrics**:\n")
                            
                            # Cyclomatic complexity
                            if "cyclomatic_complexity" in file.complexity_metrics:
                                cc = file.complexity_metrics["cyclomatic_complexity"]
                                md_file.write("  - **Cyclomatic Complexity**:\n")
                                for cc_key, cc_value in cc.items():
                                    md_file.write(f"    - {cc_key}: {cc_value}\n")
                            
                            # Maintainability index
                            if "maintainability_index" in file.complexity_metrics:
                                mi = file.complexity_metrics["maintainability_index"]
                                md_file.write("  - **Maintainability Index**:\n")
                                for mi_key, mi_value in mi.items():
                                    md_file.write(f"    - {mi_key}: {mi_value}\n")
                            
                            # Code smells
                            if "code_smells" in file.complexity_metrics and file.complexity_metrics["code_smells"]:
                                smells = file.complexity_metrics["code_smells"]
                                md_file.write("  - **Code Smells**:\n")
                                for smell in smells:
                                    md_file.write(f"    - Type: {smell.get('type', 'Unknown')}, Location: {smell.get('location', 'Unknown')}\n")
                                    if "details" in smell:
                                        md_file.write(f"      Details: {smell['details']}\n")
                    
                    # Include the complete file content
                    md_file.write("\n**Content**:\n")
                    md_file.write("```\n")
                    md_file.write(file.content)
                    md_file.write("\n```\n\n")
                    
                    # Add a separator between files
                    md_file.write("---\n\n")
                
                # Add overall statistics as a summary at the end
                md_file.write("## Summary\n\n")
                md_file.write(f"- **Total Files**: {len(self.project_files)}\n")
                md_file.write(f"- **Code Files**: {len([f for f in self.project_files if isinstance(f, ProjectCodeFile)])}\n")
                md_file.write(f"- **Regular Files**: {len([f for f in self.project_files if not isinstance(f, ProjectCodeFile)])}\n")
                
                # Calculate total lines of code
                total_lines = sum(f.line_count for f in self.project_files)
                md_file.write(f"- **Total Lines of Code**: {total_lines}\n")
            
            if console:
                console.print(f"\n[green]Repository data saved to {output_path}[/green]")
            
            return True
        except Exception as e:
            if console:
                console.print(f"\n[red]Error saving repository data to Markdown: {str(e)}[/red]")
            import traceback
            if console:
                console.print(f"\n[red]{traceback.format_exc()}[/red]")
            return False

@flock_type
class UserStory(BaseModel):
    user_story_id: str = Field(..., description="Unique identifier for the user story")
    status: Literal["active", "created", "done"] = Field(..., description="Status of the user story")
    description: str = Field(..., description="Description of the user story")
    definition_of_done: list[str] = Field(..., description="List of criteria for the user story to be considered done")
    tasks: list[str] = Field(..., description="List of task ids that are part of this user story")
    story_points: int = Field(..., description="Number of story points for the user story")
    dependencies: list[str] = Field(..., description="List of user story ids that must be completed before this one")
    used_by: list[str] = Field(..., description="List of user story ids that depend on this one")

@flock_type
class Task(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    status: Literal["active", "created", "done"] = Field(..., description="Status of the task")
    acceptance_criteria: list[str] = Field(..., description="List of acceptance criteria for the task")
    description: str = Field(..., description="Description of the task")
    estimated_lines_of_code: int = Field(..., description="Estimated number of lines of code for the task")
    dependencies: list[str] = Field(..., description="List of task ids that must be completed before this one")
    used_by: list[str] = Field(..., description="List of task ids that depend on this one")

@flock_type
class ToDoItem(BaseModel):
    todo_id: str = Field(..., description="Unique identifier for the todo item")
    user_story_id: str = Field(..., description="Unique identifier for the user story")
    task_id: str = Field(..., description="Unique identifier for the task")
    cli_command_linux: str | None = Field(..., description="valid CLI command to be executed on linux")
    cli_command_windows: str | None = Field(..., description="valid CLI command to be executed on windows")
    cli_command_macos: str | None = Field(..., description="valid CLI command to be executed on macos")
    file_content: str | None = Field(..., description="Complete content of the file if action is create_file or update_file")
    description: str = Field(..., description="Description and/or reasoning of the todo item")

@flock_type
class TaskAndToDoItemList(BaseModel):
    tasks: list[Task] = Field(..., description="List of tasks")
    todo_items: list[ToDoItem] = Field(..., description="List of todo items")
    



@flock_type
class Project(BaseModel):
    name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Description of the project")
    implementation_plan: str = Field(..., description="High Level Implementation plan for the project in beautiful markdown")
    readme: str = Field(..., description="README.md file for the project in beautiful markdown")
    requirements: list[str] = Field(..., description="List of feature requirements for the project")
    tech_stack: list[str] = Field(..., description="List of technologies used in the project")
    user_stories: list[UserStory] | None = Field(..., description="List of user stories for the project")
    # tasks: list[Task]| None = Field(..., description="List of tasks for the project")
    # project_files: list[ProjectFile | ProjectCodeFile] = Field(..., description="Output data of the project")

    def render_summary(self, console: Console) -> None:
        """
        Render a summary of the project in a beautiful format.
        
        Args:
            console: The Rich console instance to use for output
        """
        console.print("\n")
        console.rule(f"[bold blue]{self.name}")
        console.print("\n")
        console.print(self.description)
        console.print("\n")

        console.print(Markdown(self.implementation_plan))
        console.print("\n")
        console.print(Markdown(self.readme))
        console.print("\n")
        
        # Create a table for requirements
        req_table = Table(title="Project Requirements", box=box.ROUNDED)
        req_table.add_column("Requirement", style="cyan")
        
        for req in self.requirements:
            req_table.add_row(req)
            
        console.print(req_table)
        console.print("\n")
        
        # Create a table for tech stack
        tech_table = Table(title="Technology Stack", box=box.ROUNDED)
        tech_table.add_column("Technology", style="green")
        
        for tech in self.tech_stack:
            tech_table.add_row(tech)
            
        console.print(tech_table)
        
        # Summary of user stories and tasks
        if self.user_stories:
            console.print("\n")
            console.rule("[bold cyan]User Stories")
            console.print(f"\n[bold]Total User Stories:[/bold] {len(self.user_stories)}")
            for user_story in self.user_stories:
                console.print(f"\n[bold]User Story:[/bold] {user_story.user_story_id}")
                console.print(f"[bold]Description:[/bold] {user_story.description}")
                console.print(f"[bold]Definition of Done:[/bold] {user_story.definition_of_done}")
                #console.print(f"[bold]Tasks:[/bold] {user_story.tasks}")
                console.print(f"[bold]Story Points:[/bold] {user_story.story_points}")
        
        # if self.tasks:
        #     console.print("\n")
        #     console.rule("[bold cyan]Tasks")
        #     console.print(f"\n[bold]Total Tasks:[/bold] {len(self.tasks)}")
        #     for task in self.tasks:
        #         console.print(f"\n[bold]Task:[/bold] {task.task_id}")
        #         console.print(f"[bold]Description:[/bold] {task.description}")
        #         console.print(f"[bold]Acceptance Criteria:[/bold] {task.acceptance_criteria}")
        #         console.print(f"[bold]Estimated Lines of Code:[/bold] {task.estimated_lines_of_code}")
        #         console.print(f"[bold]Dependencies:[/bold] {task.dependencies}")
        #         console.print(f"[bold]Used By:[/bold] {task.used_by}")
        
        # # Summary of files
        # console.print("\n")
        # console.rule("[bold cyan]Files")
        # console.print(f"\n[bold]Total Files:[/bold] {len(self.project_files)}")
        
        # console.rule("[bold cyan]End of Project Summary")


    

```

---

### 13. src/copilot_toolkit/reporting.py

- **File ID**: file_12
- **Type**: Code File
- **Line Count**: 217
- **Description**: File at src/copilot_toolkit/reporting.py
- **Dependencies**: None
- **Used By**: None

**Content**:
```
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
```

---

### 14. src/copilot_toolkit/utils/cli_helper.py

- **File ID**: file_13
- **Type**: Code File
- **Line Count**: 53
- **Description**: File at src/copilot_toolkit/utils/cli_helper.py
- **Dependencies**: None
- **Used By**: None

**Content**:
```
from importlib.metadata import PackageNotFoundError, version

from rich.console import Console
from rich.syntax import Text

try:
    __version__ = version("copilot-toolkit")
except PackageNotFoundError:
    __version__ = "0.1.0"

console = Console()


def init_console(clear_screen: bool = True):
    """Display the Flock banner."""
    banner_text = Text(
        """
  ___             _  _       _     _____            _  _    _  _
 / __| ___  _ __ (_)| | ___ | |_  |_   _| ___  ___ | || |__(_)| |_
| (__ / _ \| '_ \| || |/ _ \|  _|   | |  / _ \/ _ \| || / /| ||  _|
 \___|\___/| .__/|_||_|\___/ \__|   |_|  \___/\___/|_||_\_\|_| \__|
           |_|

""",
        justify="center",
        style="bold orange3",
    )
    if clear_screen:
        console.clear()
    console.print(banner_text)

    console.print(
        f"v{__version__} - [bold]white duck GmbH[/] - [cyan]https://whiteduck.de[/]\n"
    )


def display_banner_no_version():
    """Display the Flock banner."""
    banner_text = Text(
        """
🦆    🐓     🐤     🐧
╭━━━━━━━━━━━━━━━━━━━━━━━━╮
│ ▒█▀▀▀ █░░ █▀▀█ █▀▀ █░█ │
│ ▒█▀▀▀ █░░ █░░█ █░░ █▀▄ │
│ ▒█░░░ ▀▀▀ ▀▀▀▀ ▀▀▀ ▀░▀ │
╰━━━━━━━━━━━━━━━━━━━━━━━━╯
🦆     🐤    🐧     🐓
""",
        justify="center",
        style="bold orange3",
    )
    console.print(banner_text)
    console.print("[bold]white duck GmbH[/] - [cyan]https://whiteduck.de[/]\n")

```

---

## Summary

- **Total Files**: 14
- **Code Files**: 14
- **Regular Files**: 0
- **Total Lines of Code**: 5519
