from pathlib import Path
from flock.core import FlockFactory, Flock
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from copilot_toolkit.model import OutputData


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


def speak_to_agent(
    action: str,
    input_data: str,
    input_data_is_file: bool = False,
    prompt_folder: str = "prompts",
) -> OutputData:
    """
    Communicate with an LLM agent to perform a specified action.

    Args:
        action: The type of action to perform (e.g., "app", "specs")
        input_data: Either file path or raw input data
        input_data_is_file: Whether input_data is a file path
        prompt_folder: Directory where prompt files are located

    Returns:
        OutputData instance with the agent's response
    """

    MODEL = (
        "gemini/gemini-2.5-pro-exp-03-25"  # "groq/qwen-qwq-32b"    #"openai/gpt-4o" #
    )

    # Show which model we're using
    console.print(f"[cyan]Using model:[/cyan] [bold magenta]{MODEL}[/bold magenta]")

    prompt = ""
    prompt_definition = ""

    # Use a spinner for loading prompt files
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        load_task = progress.add_task("[blue]Loading prompt files...", total=None)

        try:
            if action == "app":
                prompt = load_prompt("app", prompt_folder)
                prompt_definition = load_prompt("app.def", prompt_folder)
            elif action == "specs":
                prompt = load_prompt("specs", prompt_folder)
                prompt_definition = load_prompt("specs.def", prompt_folder)
            else:
                prompt = load_prompt(action)
                prompt_definition = load_prompt(f"{action}.def", prompt_folder)

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
                description=prompt,
                input="prompt: str, prompt_definition: str, input_data: str",
                output="output: OutputData | The output dictionary. Usually a dictionary with keys equals paths to files, and values equal the content of the files.",
                max_tokens=60000,
                no_output=True,
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
            console=console,
        ) as progress:
            file_task = progress.add_task(
                f"[blue]Loading input from file: [cyan]{input_data}[/cyan]...",
                total=None,
            )

            try:
                with open(input_data, "r") as f:
                    input_content = f.read()
                file_size_kb = Path(input_data).stat().st_size / 1024
                progress.update(
                    file_task,
                    description=f"[green]Input loaded successfully! ([cyan]{file_size_kb:.1f}[/cyan] KB)",
                )
            except Exception as e:
                progress.update(
                    file_task, description=f"[red]Error loading input file: {e}"
                )
                raise

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
                    "prompt_definition": prompt_definition,
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
