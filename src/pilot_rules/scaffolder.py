import os
import shutil
import argparse
import tomli
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

def get_version() -> str:
    """
    Get the current version from pyproject.toml.
    
    Returns:
        str: The current version number
    """
    try:
        # Find pyproject.toml by looking in parent directories
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:  # Stop at root directory
            pyproject_path = current_dir / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomli.load(f)
                return pyproject_data["project"]["version"]
            current_dir = current_dir.parent
            
        # If we get here, we couldn't find pyproject.toml
        return "unknown"
    except Exception:
        return "unknown"

def display_guide(guide_path: Path) -> None:
    """
    Display the markdown guide using rich formatting.
    
    Args:
        guide_path: Path to the markdown guide file
    """
    console = Console()
    
    try:
        with open(guide_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
            
        markdown = Markdown(markdown_content)
        console.print("\n")  # Add some spacing
        console.rule("[bold blue]Getting Started Guide")
        console.print("\n")  # Add some spacing
        console.print(markdown)
        console.print("\n")  # Add some spacing
        console.rule("[bold blue]End of Guide")
        
    except Exception as e:
        console.print(f"[red]Error displaying guide: {str(e)}[/red]")

def copy_template(template_type: str, root_dir: Path) -> None:
    """
    Copy template files based on the specified type.
    
    Args:
        template_type: Either 'cursor' or 'copilot'
        root_dir: The root directory where to copy the templates
    """
    # Get the directory where the package is installed
    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates"
    guides_dir = package_dir / "guides"
    
    if template_type == "cursor":
        source_dir = templates_dir / "cursor"
        target_dir = root_dir / ".cursor"
        guide_file = guides_dir / "cursor.md"
    elif template_type == "copilot":
        source_dir = templates_dir / "github"
        target_dir = root_dir / ".github"
        guide_file = guides_dir / "copilot.md"
    else:
        raise ValueError(f"Unknown template type: {template_type}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {source_dir}")

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy the contents
    for item in source_dir.glob("*"):
        if item.is_file():
            shutil.copy2(item, target_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
            
    return guide_file

def main():
    console = Console()
    
    # Clear the console
    console.clear()
    
    # Print header information
    version = get_version()
    console.print(f"[bold blue]Pilot Rules v{version}[/bold blue]")
    console.print("[blue]www.whiteduck.de[/blue]")
    console.print()  # Empty line
    
    parser = argparse.ArgumentParser(description="Copy template files for Cursor or Copilot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cursor", action="store_true", help="Copy Cursor templates")
    group.add_argument("--copilot", action="store_true", help="Copy Copilot templates")
    
    args = parser.parse_args()
    
    # Use current working directory as root
    root_dir = Path.cwd()
    
    try:
        if args.cursor:
            guide_file = copy_template("cursor", root_dir)
            console.print(f"[green]Successfully copied Cursor templates to {root_dir / '.cursor'}[/green]")
        elif args.copilot:
            guide_file = copy_template("copilot", root_dir)
            console.print(f"[green]Successfully copied Copilot templates to {root_dir / '.github'}[/green]")
            
        # Display the appropriate guide
        display_guide(guide_file)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        exit(1)

if __name__ == "__main__":
    main()
