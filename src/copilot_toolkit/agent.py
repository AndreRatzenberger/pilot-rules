import json
import os
from typing import Any, List, Optional, Tuple
from pathlib import Path
import box
from pydantic import Field, BaseModel
from flock.core import FlockFactory, Flock
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn



class OutputModel(BaseModel):
    name: str = Field(..., description="Name of the output")
    description: str = Field(..., description="High level description of the data and functionality of the app, as well as design decisions. In beautiful markdown.")
    output_dictionary_definition: str = Field(..., description="Explanation of the output dictionary and the data it contains")
    output: dict[str, Any] = Field(..., description="The output dictionary. Usually a dictionary with keys equals paths to files, and values equal the content of the files.")
    
    def render_summary(self, console: Console) -> None:
        """
        Render a pretty summary of the output to the console
        """
        # Show project name with a panel
        console.print(
            Panel(
                f"[bold cyan]{self.name}[/bold cyan]",
                title="[bold blue]Project[/bold blue]",
                border_style="blue"
            )
        )
        
        # Show dictionary definition
        console.print(
            Panel(
                f"[yellow]{self.output_dictionary_definition}[/yellow]",
                title="[bold yellow]Output Structure[/bold yellow]",
                border_style="yellow"
            )
        )
        
        # Render the markdown description
        console.print("\n[bold magenta]Project Description:[/bold magenta]")
        console.print(Markdown(self.description))
        
    def render_output_files(self, console: Console) -> None:
        """
        Render a table showing all the generated output files
        """
        if not self.output:
            console.print("[yellow]No output files generated[/yellow]")
            return
            
        # Group files by type
        file_groups: dict[str, List[Tuple[str, str]]] = {}
        for path, content in self.output.items():
            file_type = path.split('.')[-1] if '.' in path else 'other'
            if file_type not in file_groups:
                file_groups[file_type] = []
            file_groups[file_type].append((path, content))
        
        # Create a table for the files
        files_table = Table(
            title="Generated Output Files",
            box=box.ROUNDED,
            title_style="bold blue",
            header_style="bold cyan"
        )
        files_table.add_column("File Type", style="green")
        files_table.add_column("Path", style="cyan")
        files_table.add_column("Size", style="magenta", justify="right")
        
        for file_type, files in sorted(file_groups.items()):
            for path, content in sorted(files, key=lambda x: x[0]):
                content_size = len(content)
                size_str = f"{content_size} bytes"
                if content_size > 1024:
                    size_str = f"{content_size / 1024:.1f} KB"
                
                files_table.add_row(
                    file_type,
                    path,
                    size_str
                )
        
        console.print(files_table)

# Create a console for rich output
console = Console()

def load_prompt(action: str) -> str:
    """Load prompt from file."""
    prompt_path = f"prompts/{action}.md"
    try:
        with open(prompt_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        console.print(f"[red]Error: Prompt file not found at '{prompt_path}'[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Error loading prompt from '{prompt_path}': {e}[/red]")
        raise

def speak_to_agent(action: str, input_data: str, input_data_is_file: bool = False) -> OutputModel:
    """
    Communicate with an LLM agent to perform a specified action.
    
    Args:
        action: The type of action to perform (e.g., "app", "specs")
        input_data: Either file path or raw input data
        input_data_is_file: Whether input_data is a file path
        
    Returns:
        OutputModel instance with the agent's response
    """
    MODEL = "gemini/gemini-2.5-pro-exp-03-25" #"groq/qwen-qwq-32b"    #"openai/gpt-4o" # 
    
    # Show which model we're using
    console.print(f"[cyan]Using model:[/cyan] [bold magenta]{MODEL}[/bold magenta]")
    
    prompt = ""
    prompt_definition = ""
    
    # Use a spinner for loading prompt files
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:
        load_task = progress.add_task("[blue]Loading prompt files...", total=None)
        
        try:
            if action == "app":
                prompt = load_prompt("app")
                prompt_definition = load_prompt("app.def")
            elif action == "specs":
                prompt = load_prompt("specs")
                prompt_definition = load_prompt("specs.def")
            else:
                prompt = load_prompt(action)
                prompt_definition = load_prompt(f"{action}.def")
                
            progress.update(load_task, description="[green]Prompts loaded successfully!")
        except Exception as e:
            progress.update(load_task, description=f"[red]Error loading prompts: {e}")
            raise

    # Set up the agent with the prompt
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:
        setup_task = progress.add_task("[blue]Setting up agent...", total=None)
        
        try:
            # Initialize the Flock
            flock = Flock(model=MODEL)
            
            # Create the agent
            app_agent = FlockFactory.create_default_agent(
                name=f"{action}_agent",
                description=prompt,
                input="prompt: str, prompt_definition: str, input_data: str",
                output="output: OutputModel",
                max_tokens=60000
            )
            
            # Add the agent to the Flock
            flock.add_agent(app_agent)
            progress.update(setup_task, description="[green]Agent setup complete!")
        except Exception as e:
            progress.update(setup_task, description=f"[red]Error setting up agent: {e}")
            raise

    # Load input data
    input_content = input_data
    if input_data_is_file:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        ) as progress:
            file_task = progress.add_task(f"[blue]Loading input from file: [cyan]{input_data}[/cyan]...", total=None)
            
            try:
                with open(input_data, 'r') as f:
                    input_content = f.read()
                file_size_kb = Path(input_data).stat().st_size / 1024
                progress.update(file_task, description=f"[green]Input loaded successfully! ([cyan]{file_size_kb:.1f}[/cyan] KB)")
            except Exception as e:
                progress.update(file_task, description=f"[red]Error loading input file: {e}")
                raise

    # Call the agent with a progress spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]{task.description}"),
        console=console
    ) as progress:
        agent_task = progress.add_task(f"[yellow]Running {action} agent (this may take a while)...", total=None)
        
        try:
            result = flock.run(
                start_agent=app_agent, 
                input={
                    'prompt': prompt, 
                    'prompt_definition': prompt_definition, 
                    'input_data': input_content
                }
            )
            progress.update(agent_task, description="[green]Agent completed successfully!")
        except Exception as e:
            progress.update(agent_task, description=f"[red]Error during agent execution: {e}")
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
