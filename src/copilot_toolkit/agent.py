import json
from typing import Any, Optional

from flock.core import FlockFactory, Flock

from copilot_toolkit.model import OutputModel



def load_prompt(action: str) -> str:
    with open(f"prompts/{action}.md", "r") as f:
        return f.read()

    
    
def speak_to_agent(action: str, input_data: str, input_data_is_file: bool = False) -> OutputModel:
    MODEL = "gemini/gemini-2.5-pro-exp-03-25" #"groq/qwen-qwq-32b"    #"openai/gpt-4o" # 
    flock = Flock(model=MODEL)

    if action == "app":
        prompt = load_prompt("app")
        prompt_definition = load_prompt("app.def")

    app_agent = FlockFactory.create_default_agent(name=f"{action}_agent",
                                                description=prompt,
                                                input="prompt: str, prompt_definition: str, input_data: str",
                                                output="output: OutputModel",
                                                max_tokens=60000)

    flock.add_agent(app_agent)


    # Load the input data as string 
    if input_data_is_file:
        with open(input_data, 'r') as f:
            input_data = f.read()

    result = flock.run(start_agent=app_agent, input={'prompt': prompt, 'prompt_definition': prompt_definition, 'input_data': input_data}) 
    return result.output
