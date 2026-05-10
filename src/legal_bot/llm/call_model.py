import os
from pydantic import BaseModel, ValidationError
from typing import Type
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def call_model(model: str, prompt: str, structure: Type[BaseModel]) -> BaseModel:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    try:
        if structure:
            response = client.responses.parse(
                model=model,
                input=prompt,
                text_format=structure
            )

            if response.output_parsed is None:
                raise ValueError("Model returned no parsable structured output (output_parsed is None).")
            return response.output_parsed
        else:
            response = client.responses.parse(
                model=model,
                input=prompt
            )

            return response.output_text

    except ValidationError as ve:
        raise ValueError(f"Model output failed schema validation: {ve}") from ve

    except Exception as e:
        raise RuntimeError(f"Model call failed: {e}") from e
