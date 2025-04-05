from typing import Any, Literal, Optional
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

class UserStory(BaseModel):
    user_story_id: str = Field(..., description="Unique identifier for the user story")
    description: str = Field(..., description="Description of the user story")
    definition_of_done: list[str] = Field(..., description="List of criteria for the user story to be considered done")
    tasks: list[str] = Field(..., description="List of task ids that are part of this user story")
    story_points: int = Field(..., description="Number of story points for the user story")
    dependencies: list[str] = Field(..., description="List of user story ids that must be completed before this one")
    used_by: list[str] = Field(..., description="List of user story ids that depend on this one")

class Task(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    acceptance_criteria: list[str] = Field(..., description="List of acceptance criteria for the task")
    description: str = Field(..., description="Description of the task")
    estimated_lines_of_code: int = Field(..., description="Estimated number of lines of code for the task")
    dependencies: list[str] = Field(..., description="List of task ids that must be completed before this one")
    used_by: list[str] = Field(..., description="List of task ids that depend on this one")
    file_actions: list["FileAction"] = Field(..., description="List of file actions for the task")

class FileAction(BaseModel):
    file_id: str = Field(..., description="Unique identifier for the file")
    status: Literal["planned", "created", "done"] = Field(..., description="Status of the file")
    action: Literal["create", "update", "delete"] = Field(..., description="Action to be taken on the file")
    description: str = Field(..., description="Description of the action")



@flock_type
class Project(BaseModel):
    name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Description of the project")
    requirements: list[str] = Field(..., description="List of requirements for the project")
    tech_stack: list[str] = Field(..., description="List of technologies used in the project")
    user_stories: list[UserStory] | None = Field(..., description="List of user stories for the project")
    tasks: list[Task]| None = Field(..., description="List of tasks for the project")
    project_files: list[ProjectFile | ProjectCodeFile] = Field(..., description="Output data of the project")


    
