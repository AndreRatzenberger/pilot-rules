import os
import shutil
import argparse
from pathlib import Path

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
    
    if template_type == "cursor":
        source_dir = templates_dir / "cursor"
        target_dir = root_dir / ".cursor"
    elif template_type == "copilot":
        source_dir = templates_dir / "github"
        target_dir = root_dir / ".github"
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

def main():
    parser = argparse.ArgumentParser(description="Copy template files for Cursor or Copilot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cursor", action="store_true", help="Copy Cursor templates")
    group.add_argument("--copilot", action="store_true", help="Copy Copilot templates")
    
    args = parser.parse_args()
    
    # Use current working directory as root
    root_dir = Path.cwd()
    
    try:
        if args.cursor:
            copy_template("cursor", root_dir)
            print(f"Successfully copied Cursor templates to {root_dir / '.cursor'}")
        elif args.copilot:
            copy_template("copilot", root_dir)
            print(f"Successfully copied Copilot templates to {root_dir / '.github'}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
