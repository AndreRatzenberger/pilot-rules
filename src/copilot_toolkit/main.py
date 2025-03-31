# src/pilot_rules/main.py
import os
import shutil
import argparse
import sys
from pathlib import Path
from typing import Optional

# Third-party imports
from dotenv import set_key
import tomli
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.table import Table
from rich.layout import Layout
from rich.live import Live

from copilot_toolkit import collector
from copilot_toolkit.agent import speak_to_agent
from copilot_toolkit.collector.utils import print_header, print_subheader, print_success, print_warning, print_error
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
        while current_dir != current_dir.parent: # Stop at root directory '/'
            pyproject_path = current_dir / "pyproject.toml"
            if pyproject_path.exists():
                #print(f"DEBUG: Found pyproject.toml at {pyproject_path}") # Debug print
                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomli.load(f)
                version = pyproject_data.get("project", {}).get("version", "0.0.0")
                if version != "0.0.0":
                    return version
                # If version is placeholder, keep searching upwards
            current_dir = current_dir.parent

        # If not found after searching upwards
        #print("DEBUG: pyproject.toml with version not found.") # Debug print
        return "0.0.0" # Fallback if not found
    except Exception as e:
        #print(f"DEBUG: Error getting version: {e}") # Debug print
        import traceback
        traceback.print_exc() # Print error during dev
        return "0.0.0" # Fallback on error


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
        print_error(f"Error displaying guide '{guide_path}': {str(e)}")


def copy_template(template_type: str, root_dir: Path, console: Console) -> Optional[Path]:
    """
    Copy template files based on the specified type ('cursor' or 'copilot').

    Args:
        template_type: Either 'cursor' or 'copilot'.
        root_dir: The root directory (usually CWD) where to copy the templates.
        console: The Rich console instance to use for output.

    Returns:
        Path to the relevant guide file if successful, None otherwise.
    """
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
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Copying {template_type} templates...", total=None)
        
        try:
            for item in source_dir.iterdir():
                target_path = target_dir / item.name
                progress.update(task, description=f"[cyan]Copying [bold]{item.name}[/bold]...")
                if item.is_file():
                    shutil.copy2(item, target_path)
                elif item.is_dir():
                    shutil.copytree(item, target_path, dirs_exist_ok=True)
            
            progress.update(task, description="[green]Copy completed successfully!")
            print_success(f"Successfully copied {template_type} templates to {target_dir}")
            return guide_file # Return path to guide file on success
        except Exception as e:
            progress.update(task, description=f"[red]Error copying files: {e}")
            print_error(f"Error copying templates from '{source_dir}' to '{target_dir}': {e}")
            return None


# --- Main Application Logic ---

def main():
    """
    Entry point for the pilot-rules CLI application.
    Handles argument parsing and delegates tasks to scaffolding or collection modules.
    """
    console = Console()
    console.clear()
    #version = get_version()
    
    # Create a fancy header
    # header_panel = Panel(
    #     f"[bold blue]Pilot Rules v{version}[/bold blue]\n[cyan]www.whiteduck.de[/cyan]",
    #     box=box.ROUNDED,
    #     border_style="blue",
    #     title="[yellow]CLI Tool[/yellow]",
    #     subtitle="[yellow]Powered by LLMs[/yellow]"
    # )
    # console.print(header_panel)
    
    init_console()

    parser = argparse.ArgumentParser(
        description="Manage Pilot Rules templates or collect code for analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Mutually Exclusive Actions ---
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--cursor", action="store_true", help="Scaffold Cursor templates (.cursor)")
    action_group.add_argument("--copilot", action="store_true", help="Scaffold Copilot templates (.github)")
    action_group.add_argument("--collect", action="store_true", help="Collect code from the repository")
    action_group.add_argument("--app", action="store_true", help="Create a standalone webapp based on some data")
    action_group.add_argument("--prompt", action="store_true", help="Prompt an agent to do something")
    action_group.add_argument("--build", action="store_true", help="Build the project")
    action_group.add_argument("--clean", action="store_true", help="Clean the project")
    action_group.add_argument("--init", action="store_true", help="Initialize a new project")
    action_group.add_argument("--interactive", action="store_true", help="Interactive mode")
    action_group.add_argument("--specs", action="store_true", help="Create a project specification")
    action_group.add_argument(
        "--set_key",
        metavar="KEY",
        help="Set the API key for the agent"
    )

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
        "--input",
        default=None, # Default is handled inside the collector logic now
        metavar="FILEPATH",
        help=f"Path to the input file or folder"
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
            print_header("Code Collection Mode", "cyan")
            # --- Call the refactored collector function ---
            # Errors within the collector (ValueError, etc.) will be caught below
            collector.run_collection(
                include_args=args.include,
                exclude_args=args.exclude,
                output_arg=args.output, # Pass CLI arg (can be None)
                config_arg=args.config
            )
            print_success("Code collection process completed successfully")

        elif args.cursor:
            guide_file_to_display = copy_template("cursor", scaffold_root_dir, console)
            # Success/Error messages printed within copy_template

        elif args.copilot:
            guide_file_to_display = copy_template("copilot", scaffold_root_dir, console)
            # Success/Error messages printed within copy_template

        elif args.app:
            print_header("App Creation Mode", "magenta")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold magenta]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[magenta]Creating app...", total=None)
                try:
                    output = speak_to_agent("app", args.input, True)
                    progress.update(task, description="[green]App created successfully!")
                    
                    # Use the new rendering methods instead of direct printing
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
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold yellow]{task.description}"),
                console=console
            ) as progress:
                collect_task = progress.add_task("[yellow]Collecting repository data...", total=None)
                
                # If folder, run collection first
                if os.path.isdir(file_or_folder):
                    try:
                        collector.run_collection(
                            include_args=[f"py,md:./{file_or_folder}"],
                            exclude_args=[],
                            output_arg=None,
                            config_arg=None
                        )
                        progress.update(collect_task, description="[green]Repository data collected!")
                        
                        # Now generate specs from the analysis
                        generate_task = progress.add_task("[yellow]Generating specifications...", total=None)
                        output = speak_to_agent("specs", "repository_analysis.md", True)
                        progress.update(generate_task, description="[green]Specifications generated successfully!")
                    except Exception as e:
                        progress.update(collect_task, description=f"[red]Error during collection: {e}")
                        raise
                
                # If file, use it directly
                elif os.path.isfile(file_or_folder):
                    try:
                        generate_task = progress.add_task("[yellow]Generating specifications from file...", total=None)
                        output = speak_to_agent("specs", file_or_folder, True)
                        progress.update(generate_task, description="[green]Specifications generated successfully!")
                    except Exception as e:
                        progress.update(generate_task, description=f"[red]Error generating specifications: {e}")
                        raise
                else:
                    progress.update(collect_task, description="[red]Invalid input path")
                    raise ValueError(f"Input path is neither a file nor a directory: {file_or_folder}")
            
            # Display results using the new rendering methods
            console.print("\n")
            output.render_summary(console)
            
            # Write output files and display them
            if hasattr(output, "output") and isinstance(output.output, dict):
                files_table = Table(title="Writing Output Files", box=box.ROUNDED)
                files_table.add_column("File Path", style="cyan")
                files_table.add_column("Status", style="green")
                
                for key, value in output.output.items():
                    try:
                        # Create the directory if it doesn't exist
                        Path(key).parent.mkdir(parents=True, exist_ok=True)
                        with open(key, "w") as f:
                            f.write(value)
                        files_table.add_row(key, "[green]✓ Created[/green]")
                    except Exception as e:
                        files_table.add_row(key, f"[red]✗ Error: {e}[/red]")
                
                console.print(files_table)
            
            print_success("Specification generation completed")

        elif args.set_key:
            print_header("Setting API Key", "green")
            try:
                set_key('.env', 'GEMINI_API_KEY', args.set_key)
                print_success(f"API key set successfully in .env file")
            except Exception as e:
                print_error(f"Error setting API key: {e}")

        # Display guide only if scaffolding was successful and returned a guide path
        if guide_file_to_display:
            display_guide(guide_file_to_display, console)

    except FileNotFoundError as e:
         # Should primarily be caught within helpers now, but keep as fallback
         print_error(f"Required file or directory not found: {str(e)}")
         exit(1)
    except ValueError as e: # Catch config errors propagated from collector
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