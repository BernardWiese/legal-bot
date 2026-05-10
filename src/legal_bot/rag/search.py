import numpy as np

def embed(model, text):
    v = model.encode([text], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def search(meta, index, model, query, k):
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

def generate_context(hits, section_refs):
    context = ""
    for h in hits:
        document = f"Consolidated Act: {h.get('document_name')}"
        chapter = f"Chapter {h.get('chapter')}: {h.get('chapter_title')}"
        part = f"Part {h.get('part')}: {h.get('part_title', '')}" if h.get('part') else ""
        section = f"Section {h.get('section')}: {h.get('section_title')}"
        content = f"Content: {h.get('content')}"

        context += document + "\n" + chapter + "\n" + part + "\n" + section + "\n" + content + "\n---\n"

    for section_str in section_refs:
        document = f"Consolidated Act: {section_str.get('document_name', '')}"
        chapter = f"Chapter {section_str.get('chapter', '')}: {section_str.get('chapter_title', '')}"
        part = f"Part {section_str.get('part')}: {section_str.get('part_title', '')}" if section_str.get('part') else ""
        section= f"Section {section_str.get('section', '')}: {section_str.get('section_title', '')}"
        content = f"Content: {section_str.get('content', '')}"

        context += document + "\n" + chapter + "\n" + part + "\n" + section + "\n" + content + "\n---\n"

    return context