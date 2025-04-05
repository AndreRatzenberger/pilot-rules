# src/pilot_rules/main.py
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
from copilot_toolkit.agent import speak_to_agent
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
        console: The Rich console instance to use for output.
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
    
    # Confirm the selections
    console.print("\n[bold]Collection Configuration:[/bold]")
    console.print(f"[cyan]Include paths:[/cyan] {include_args}")
    console.print(f"[cyan]Exclude paths:[/cyan] {exclude_args}")
    console.print(f"[cyan]Repository name:[/cyan] {repo_name or 'Repository Analysis'}")
    console.print(f"[cyan]Config file:[/cyan] {config_arg or 'None'}")
    
    if questionary.confirm("Proceed with collection?").ask():
        try:
            repository = collector.run_collection(
                include_args=include_args,
                exclude_args=exclude_args,
                repo_name=repo_name,
                config_arg=config_arg,
            )
            # Display repository using rich rendering methods
            repository.render_summary(console)
            repository.render_files(console)
            print_success("Repository object generated successfully")
            
            # Ask if user wants to save the repository to a file
            if questionary.confirm("Would you like to save the repository data to a file?").ask():
                from pathlib import Path
                
                output_path = questionary.text(
                    "Enter the output file path (e.g., 'repository_data.json'):",
                    default="repository_data.json"
                ).ask()
                
                # Use the save_to_json method
                repository.save_to_json(output_path, console)
        except Exception as e:
            print_error(f"Error during collection: {str(e)}")

def run_interactive_specs(console: Console) -> None:
    """
    Run the project specifications generation in interactive mode.
    
    Args:
        console: The Rich console instance to use for output.
    """
    print_header("Project Specifications Generation", "yellow")
    
    # Get input file or folder
    input_path = questionary.text(
        "Enter the path to the input file or folder:"
    ).ask()
    
    if not input_path:
        print_error("Input path is required")
        return
    
    # Get prompt folder path
    prompts = questionary.text(
        "Enter the prompt folder path (leave empty for default 'prompts'):"
    ).ask()
    
    if not prompts:
        prompts = "prompts"
    
    # Get user instructions
    user_instructions = questionary.text(
        "Enter additional instructions for the agent (leave empty for none):"
    ).ask()
    
    if not user_instructions:
        user_instructions = ""
    
    # Confirm the selections
    console.print("\n[bold]Specifications Configuration:[/bold]")
    console.print(f"[yellow]Input path:[/yellow] {input_path}")
    console.print(f"[yellow]Prompt folder:[/yellow] {prompts}")
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
                        prompt_folder=prompts, 
                        user_instructions=user_instructions,
                        input_type="repository"  # Add a flag to indicate this is a repository object
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
                    input_type = "repository" 
                    
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
                        prompt_folder=prompts, 
                        user_instructions=user_instructions,
                        input_type=input_type
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
                output.render_output_files(console)
                
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
    input_file = questionary.text(
        "Enter the path to the input file:"
    ).ask()
    
    if not input_file:
        print_error("Input file is required")
        return
    
    # Get output path
    output_path = questionary.text(
        "Enter the output path (leave empty for default):"
    ).ask()
    
    if not output_path:
        output_path = None
    
    # Get prompt folder path
    prompts = questionary.text(
        "Enter the prompt folder path (leave empty for default 'prompts'):"
    ).ask()
    
    if not prompts:
        prompts = "prompts"
    
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
    console.print(f"[magenta]Prompt folder:[/magenta] {prompts}")
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
                    prompt_folder=prompts, 
                    user_instructions=user_instructions
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
            )
            
            # Display repository using rich rendering methods
            repository.render_summary(console)
            repository.render_files(console)
            print_success("Repository analysis completed successfully")

        elif args.cursor:
            guide_file_to_display = copy_template("cursor", scaffold_root_dir, console)
            # Success/Error messages printed within copy_template

        elif args.copilot:
            guide_file_to_display = copy_template("copilot", scaffold_root_dir, console)
            # Success/Error messages printed within copy_template

        elif args.app:
            print_header("App Creation Mode", "magenta")

            # Set defaults for prompts and user_instructions if not provided
            prompt_folder = args.prompts if args.prompts else "prompts"
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
                        prompt_folder=prompt_folder, 
                        user_instructions=user_instructions
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
            prompt_folder = args.prompts if args.prompts else "prompts"
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
                            prompt_folder=prompt_folder, 
                            user_instructions=user_instructions,
                            input_type="repository"
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
                            prompt_folder=prompt_folder,
                            user_instructions=user_instructions,
                            input_type="repository"
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
