from pathlib import Path
import ast
from pathlib import Path
import re
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ValidationError
from typing import Any, List, Type
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class PromptManager:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]

    def load(self, prompt_file: str) -> str:
        path = self.base_dir / "prompts" / prompt_file
        return path.read_text(encoding="utf-8")

    def render(self, prompt_file: str, **kwargs) -> str:
        template = self.load(prompt_file)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            missing = e.args[0]
            raise ValueError(
                f"Missing template variable '{missing}' for prompt '{prompt_file}'. "
                f"Provided keys: {sorted(kwargs.keys())}"
            ) from e