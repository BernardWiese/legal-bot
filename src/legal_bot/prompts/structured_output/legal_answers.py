from typing import List
from pydantic import BaseModel, Field

class Citations(BaseModel):
    document_name: str
    chapter: str
    part: str
    section: str
    section_title: str

class SectionExplanation(BaseModel):
    section_number: str
    section_title: str
    summary: str
    extract: str    

class Structure(BaseModel):
    section_explanations: List[SectionExplanation] = Field(..., description="List of sections and their summaries and direct extracts.")
    citations: List[Citations] = Field(..., description="List of sections referenced in the model's answer.")

# --- CITATION FORMAT ---
# Every legal statement MUST include a citation.

# Use this format:
# [<document_name>, Section <section>(<subsection>) – <section_title>]

# Example:
# [Consumer Protection Act, Section 2(1) – Related and inter-related persons]

# If citing multiple:
# [Consumer Protection Act: Section 2(1); Section 2(2)]
