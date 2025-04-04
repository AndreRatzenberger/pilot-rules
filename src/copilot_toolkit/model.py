from typing import Any
from flock.core.flock_registry import flock_type
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
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
