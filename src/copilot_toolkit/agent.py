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
