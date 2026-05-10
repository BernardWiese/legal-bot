import streamlit as st
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ValidationError
from typing import Any, Type
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def call_model(model: str, prompt: str) -> Any:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.responses.parse(
            model=model,
            input=prompt
        )
        return response.output_text

    except ValidationError as ve:
        raise ValueError(f"Model output failed schema validation: {ve}") from ve

    except Exception as e:
        raise RuntimeError(f"Model call failed: {e}") from e

def embed(model, text):
    v = model.encode([text], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def search(meta, index, model, query, k=6):
    qv = embed(model, query)
    scores, idxs = index.search(qv, k)
    idxs, scores = idxs[0], scores[0]
    results = []
    for i, s in zip(idxs, scores):
        if i == -1:
            continue
        m = meta[i].copy()
        m["score"] = float(s)
        results.append(m)
    return results

