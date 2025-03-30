# Copilot-Toolkit 🚀 A toolkit for coding agents 🚀 

Get the most out of GitHub Copilot, Cursor and Co. Powered by flock.

This repository contains a collection of tools and configuration like custom rules for AI-assisted development that significantly enhance your productivity and code quality. These tools provide structure, automation, and consistency to your development process, regardless of which AI assistant you use.

Expand the abilities of GitHub Copilot, Cursor and other LLM-based development agents.
For example with the help of custom rules enable Copilot to follow project management best practices which will improve the quality of the code and the project tenfold.

Do custom code analysis and share the result with your coding agents, or create automatic documentations.


## Quick Setup 🛠️ - No Installation

Requirements: [`uv`](https://github.com/astral-sh/uv)

You can use everything `copilot-toolkit` has to offer without installing anything else — just make sure `uv` is available on your system.


## Non-agentic Features

### Scaffold Rules

```bash
# For Cursor AI
uvx copilot-toolkit --cursor

# For GitHub Copilot
uvx copilot-toolkit --copilot
```


### Collect Code

```bash
# Collect all .py files in the current directory and subdirectories
# Writes output to code.md in the current directory

uvx copilot-toolkit --collect

# which is the same as
uvx copilot-toolkit --collect \
  --include "py:." \
  --output repository_analysis.md
```

```bash
# Collect multiple file types from multiple roots
# Excludes paths containing 'external'
# syntax
# <file extensions>:<folder>
# *:. 
# all files in all folders
# Writes output to .project/code.md
uvx copilot-toolkit --collect \
  --include "py,js:." \
  --exclude "*:external" \
  --include "md:docs" \
  --output .project/code.md
```

Instead of passing everything via CLI, you can define your sources and output in a simple `.toml` file.

Example `copilot-toolkit.toml`:

```toml
[[source]]
exts = ["py", "js"]
root = "."
exclude = ["external"]

[[source]]
exts = ["md"]
root = "docs"

output = ".project/code.md"
```

Then run:

```bash
uvx copilot-toolkit --collect --config copilot-toolkit.toml
```

### Clean

```bash
# Clean all python cache folders

uvx copilot-toolkit --clean
```

### Init

```bash
# Init a basic project with some sensical defaults

uvx copilot-toolkit --init
```


### Build

```bash
# Builds the project and installs it as editable package

uvx copilot-toolkit --build
```



## Agentic Features

Agentic Features are abusing making use of Gemini's immense context window.
Paired with the improved abilities of Gemini 2.5 results in tools that were not possible ever before.



```bash
# define your google ai studio key
uvx copilot-toolkit --set_key xxx
```

### Create a project specification

```bash
# create specifications based on the current project
uvx copilot-toolkit --specs --output .project/

uvx copilot-toolkit --prompt specs --def specs.def --input . --output .project/
```

### App-ify data

```bash
# create a standalone webapp based on some data
uvx copilot-toolkit --app --input data.json --output app/

uvx copilot-toolkit --prompt app --input data.json --output app/
```



---

Last but not least there is an interactive mode that let's you chat with an agent, and the agent figures out what you want to do.

```bash
uvx copilot-toolkit --interactive
```


## License 📜

MIT License - See [LICENSE](LICENSE) for details.

---

"If you want to build a ship, don't drum up people to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea." - Antoine de Saint-Exupéry

_In the same way, effective AI systems don't just execute code, but operate within a framework of principles and specifications that guide them toward building solutions that fulfill the true vision of what we seek to create._

_Let your coding agents work *with* your rules — not against them._
