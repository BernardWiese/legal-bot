import os
from openai import OpenAI
import json
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from utils import search

load_dotenv()

from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]  # if app.py is in streamlit/, this points to repo root

@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    vector_index_path = BASE_DIR / "data" / "document_vectors" / "individual" / "companies_act_v1.index"
    document_chunks_path = BASE_DIR / "data" / "document_chunks" / "companies_act_section_chunks.jsonl"

    # Helpful error message instead of crashing deep inside FAISS
    if not vector_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {vector_index_path} (did you commit/upload it?)")

    if not document_chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at: {document_chunks_path} (did you commit/upload it?)")

    vector_index = faiss.read_index(str(vector_index_path))
    document_chunks = [json.loads(line) for line in document_chunks_path.open("r", encoding="utf-8")]

    return embedding_model, vector_index, document_chunks

embedding_model, vector_index, document_chunks = load_resources()
st.title("Legal Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_message := st.chat_input("What is up?"):
    hits = search(document_chunks, vector_index, embedding_model, user_message, k=6)

    context = ""
    for h in hits:
        document = f"Consolidated Act: {h.get('document_name', '')}"
        chapter = f"Chapter {h.get('chapter', '')}: {h.get('chapter_title', '')}"
        part = f"Part {h.get('part', '')}: {h.get('part_title', '')}"
        section = f"Section {h.get('section', '')}: {h.get('section_title', '')}"
        content = f"Content: {h.get('content', '')}"

        context += document + "\n\n" + chapter + "\n\n" + part + "\n\n" + section + "\n\n" + content + "\n---\n"

    prompt = f"""
You are LexSA, a retrieval-augmented legal research assistant for South African law.

You help users understand legislation by answering questions using ONLY the provided retrieved context. 
The context consists of structured legal text chunks with metadata such as chapter, part, section, subsection, and content.

You are NOT a lawyer and do NOT provide legal advice. You provide general legal information only.

--- CORE RULE: STRICT GROUNDING ---
You MUST base your answer ONLY on the retrieved context.

- Do NOT use prior knowledge
- Do NOT infer missing rules
- Do NOT fabricate legal provisions
- If the answer is not clearly supported by the context, say:
  "The retrieved sources do not contain enough information to answer this definitively."

--- HOW TO INTERPRET THE CONTEXT ---
Each chunk represents part of a legal provision and includes:
- chapter / chapter_title
- part / part_title
- section / section_title
- subsection
- content (legal text)

You must:
- Treat the "content" as the authoritative legal wording
- Use document name, section and subsection as the primary citation references
- Use other metadata (chapter/part titles) for clarity when helpful

--- CITATION FORMAT ---
Every legal statement MUST include a citation.

Use this format:
[<document_name>, Section <section>(<subsection>) – <section_title>]

Example:
[Consumer Protection Act, Section 2(1) – Related and inter-related persons]

If citing multiple:
[Consumer Protection Act: Section 2(1); Section 2(2)]

--- ANSWER STRUCTURE ---
Always structure your response as:

1) Legal Breakdown  
- Provide the most relevant extracts from the consolidated acts
- Provide the relevant citations in a clear readable format

2) Give a short summary of the general interpretation (General, Non-Advisory)  
- Explain what the provision generally means in practice
- Preserve legal meaning but simplify language
- Do NOT give legal advice or strategy

3) Sources  
- List all cited chunks in the required format

--- HANDLING AMBIGUITY ---
If the question:
- is too broad → ask clarifying questions
- depends on facts → list needed facts
- involves interpretation beyond the text → state limitation clearly

--- MULTI-PART LEGAL LOGIC ---
When a provision has subparagraphs like (a), (b), (c), (i), (ii):
- Present them as structured bullet points
- Preserve hierarchy clearly

--- TONE ---
- Professional and precise
- Clear and structured
- No fluff
- No speculation

--- SAFETY ---
- Do not assist with unlawful activity
- Do not provide legal advice
- When appropriate, suggest consulting a qualified South African legal professional

--- INPUT FORMAT ---
You will receive:

USER QUESTION:
{user_message}

RETRIEVED CONTEXT:
{context}

Only answer using this context.
"""

    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=prompt
        )
        st.markdown(response.output_text)

    st.session_state.messages.append({"role": "assistant", "content": f"{response.output_text} "})

    st.rerun()