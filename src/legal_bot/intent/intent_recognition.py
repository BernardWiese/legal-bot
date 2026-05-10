from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from legal_bot.prompts.prompt_manager import PromptManager
from legal_bot.llm.call_model import call_model

class Intent(str, Enum):
    explain_section = "explain_section"
    fallback_response = "fallback_response"

class IntentStructure(BaseModel):
    intent: Intent = Field(..., description="User intent category for routing.")
    clarify: bool = Field(..., description="True if a clarifying question is needed before retrieval/answering.")
    clarifying_question: Optional[str] = Field(None, description="If clarify=true, ask this question to disambiguate the user's intent.")
    section_refs: Optional[List[str]] = Field(None, description="Section identifiers referenced, e.g. ['164', '169', '200', '164A'].")

def intent_logic(user_message: str, prompt_name: str):
    pm = PromptManager()

    prompt = pm.render(
        prompt_file=f"intent/{prompt_name}.prompt",
        user_message=user_message
    )

    response = call_model(model="gpt-4o-mini", prompt=prompt, structure=IntentStructure)

    return response
