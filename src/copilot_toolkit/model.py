from typing import Any
from pydantic import BaseModel, Field


class OutputModel(BaseModel):
    name: str = Field(..., description="Name of the output")
    description: str = Field(..., description="High level description of the data and functionality of the app, as well as design decisions. In beautiful markdown.")
    output_dictionary_definition: str = Field(..., description="Explanation of the output dictionary and the data it contains")
    output: dict[str, Any] = Field(..., description="The output dictionary. Usually a dictionary with keys equals paths to files, and values equal the content of the files.")
    