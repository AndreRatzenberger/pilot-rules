from importlib.metadata import PackageNotFoundError, version

from rich.console import Console
from rich.syntax import Text
from art import *

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
