import json
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
from legal_bot.intent.intent_recognition import intent_logic
from legal_bot.prompts.prompt_manager import PromptManager
from legal_bot.llm.call_model import call_model
from legal_bot.rag.search import search, generate_context

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv()

@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    vector_index_path = BASE_DIR / "data" / "document_vectors" / "individual" / "companies_act_v1.index"
    document_chunks_path = BASE_DIR / "data" / "document_chunks" / "companies_act_section_chunks.jsonl"

    # Helpful error message instead of crashing deep inside FAISS
    if not vector_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {vector_index_path}")

    if not document_chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at: {document_chunks_path}")

    vector_index = faiss.read_index(str(vector_index_path))
    document_chunks = [json.loads(line) for line in document_chunks_path.open("r", encoding="utf-8")]

    return embedding_model, vector_index, document_chunks

embedding_model, vector_index, document_chunks = load_resources()

st.title("Legal Assistant")

welcome_message = """Hello, I am your personal legal assistant, ready to answer questions on any South African legislation documents. 
What can I help you with?"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": welcome_message}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_message := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    with st.chat_message("user"):
        st.markdown(user_message)

    user_intent = intent_logic(user_message, "intent_recognition_v1")
    pm = PromptManager()
    
    section_refs = []
    if user_intent.section_refs:
        section_refs = [chunk for chunk in document_chunks if chunk.get("section") in set(user_intent.section_refs)]

    if user_intent.intent == "fallback_response":
        if user_intent.clarify:
            with st.chat_message("assistant"):
                st.markdown(user_intent.clarifying_question)
                st.session_state.messages.append({"role": "assistant", "content": f"{user_intent.clarifying_question} "})
        else:
            prompt = pm.render(prompt_file="intent/fallback_response_v1.prompt", user_message=user_message)
            response = call_model(model="gpt-4o-mini", prompt=prompt, structure=None)

            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": f"{response} "})

    elif user_intent.clarify:
        with st.chat_message("assistant"):
            st.markdown(user_intent.clarifying_question)
            st.session_state.messages.append({"role": "assistant", "content": f"{user_intent.clarifying_question} "})

    elif user_intent.intent == "explain_section":
        # Add here search for direct section titles to add to section refs
        hits = search(document_chunks, vector_index, embedding_model, user_message, k=2)
        context = generate_context(hits, section_refs)

        prompt = pm.render(prompt_file="intent/explain_section_v2.prompt", user_message=user_message, context=context)
        response = call_model(model="gpt-4o-mini", prompt=prompt, structure=None)
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": f"{response}"})

    # elif user_intent.intent == "general_legal":
    #     with st.chat_message("assistant"):
    #         st.markdown("This intent is under maintenance.")
    #         st.session_state.messages.append({"role": "assistant", "content": "This intent is under maintenance."})

    # else:
    #     fallback = "I am sorry, I didn't quite get that. Could you please rephrase your question?"
    #     st.markdown(fallback)
    #     st.session_state.messages.append({"role": "assistant", "content": f"{fallback}"})

    # # hits = search(document_chunks, vector_index, embedding_model, user_message, k=2)
    
    # # prompt = build_prompt(user_message, hits, section_refs, "generation/legal_response_v1.prompt")



    # # with st.chat_message("assistant"):
    # #     response = call_model(model="gpt-4o-mini", prompt=prompt, structure=Structure)
    # #     st.markdown(response)

    # # st.session_state.messages.append({"role": "assistant", "content": f"{response} "})
    # # st.rerun()