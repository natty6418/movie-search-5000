"""Utilities for talking to a locally hosted LLM (e.g., Ollama)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from colorama import Fore, Style

load_dotenv()

LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_LLM_ENDPOINT = os.environ.get(
    "LOCAL_LLM_ENDPOINT", f"{LOCAL_LLM_BASE_URL.rstrip('/')}/api/generate"
)


class LocalLLMError(RuntimeError):
    """Raised when the local LLM cannot be reached or returns invalid data."""


def _call_local_llm(prompt: str) -> str:
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        LOCAL_LLM_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:  # noqa: PERF203 - clarity over micro-ops
        raise LocalLLMError(
            "Could not reach the local LLM endpoint. "
            "Ensure Ollama (or your chosen server) is running and the model is available."
        ) from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:  # noqa: B904 - preserve context for callers
        raise LocalLLMError("Local LLM returned invalid JSON.") from exc

    response_text = parsed.get("response")
    if not response_text:
        raise LocalLLMError("Local LLM response payload was empty.")
    return response_text.strip()


def enhance_query(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.
Query: "{query}"
If no errors, return the original query.
Corrected:"""
    try:
        response = _call_local_llm(prompt)
    except LocalLLMError as exc:
        print(f"Local LLM enhancement failed ({exc}). Using the original query.")
        return query

    cleaned = response.strip()
    return cleaned or query


def rewrite_query(query: str) -> str:
    prompt = f"""
You are an expert movie search query rewriter.

Your task is to convert a natural-language movie search query into a precise, database-friendly search query that captures the user’s intent.

Rules:
- Identify the core elements of the request: genre, tone, plot type, themes, style, era, or notable influences.
- If a movie is referenced (e.g. “movies like Knives Out”), infer its defining traits (e.g. genre, structure, tone) rather than repeating the title.
- Prefer descriptive attributes over vague wording.
- Use concise keyword-style phrasing (Google-style search).
- Keep the output under 10 words.
- Do NOT use boolean operators (AND, OR, NOT).
- Do NOT explain your reasoning.
- Do NOT include quotes.
- Output only the rewritten query.

Examples:
- "movies like Knives Out" → "modern whodunit mystery ensemble cast"
- "that bear movie where leo gets attacked" → "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" → "Paddington London marmalade"
- "scary movie with bear from few years ago" → "bear horror movie 2015-2020"
- "funny murder mystery like glass onion" → "satirical murder mystery ensemble comedy"

Original query:
{query}

Rewritten query:"""
    try:
        response = _call_local_llm(prompt)
    except LocalLLMError as exc:
        print(f"Local LLM enhancement failed ({exc}). Using the original query.")
        return query
    cleaned = response.strip()
    return cleaned or query


def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

**Only provide the rewritten query!**

Query: "{query}"
"""
    try:
        response = _call_local_llm(prompt)
    except LocalLLMError as exc:
        print(f"Local LLM enhancement failed ({exc}). Using the original query.")
        return query
    cleaned = response.strip()
    return cleaned or query


def rank_results(query: str, doc: dict) -> str:

    prompt = f"""Rate how well this movie matches the search query.
    
    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("description", "")}
    
    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness
    
    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.
    
    Score:"""

    try:
        response = _call_local_llm(prompt)
    except LocalLLMError as exc:
        print(f"Local LLM enhancement failed ({exc}). Using the original query.")
        return query
    cleaned = response.strip()
    return cleaned or query


def batch_rank_results(query: str, docs: list[dict]) -> list[str]:
    allowed_ids = [d["id"] for d in docs]

    movies_compact = [
        {
            "id": d["id"],
            "title": d.get("title", ""),
            "description": d.get("description", "")[:400],
        }
        for d in docs
    ]

    prompt = f"""
You are ranking movies by relevance to a search query.

Query: {query}

Movies (JSON):
{json.dumps(movies_compact, ensure_ascii=False)}

Rules:
- Output MUST be a valid JSON array of integers.
- Each integer MUST be one of these allowed IDs: {allowed_ids}
- Do NOT invent IDs.
- Do NOT include any text, explanation, or markdown—ONLY the JSON array.
- Return ALL provided IDs exactly once, sorted best to worst.

Output format example:
[1771, 12, 34]
""".strip()

    response = _call_local_llm(prompt).strip()
    return json.loads(response)


def judge_relevance(query: str, formatted_results: list[str]) -> list[str]:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = _call_local_llm(prompt).strip()
    return json.loads(response)


def rag_answer(query: str, docs: list[dict[str, str]]) -> str:

    print(Fore.YELLOW + "Query: " + query + Style.RESET_ALL)
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    
    Query: {query}
    
    Documents:
    {json.dumps(docs, ensure_ascii=False)}
    
    Provide a comprehensive answer that addresses the query:"""

    response = _call_local_llm(prompt).strip()
    return response


def summarize(query: str, docs: list[dict[str, str]]) -> str:
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{json.dumps(docs, ensure_ascii=False)}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    response = _call_local_llm(prompt).strip()
    return response
def _format_docs_for_prompt(docs: list[dict[str, str]]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        doc_id = d.get("id", "")
        title = d.get("title", "")
        text = d.get('description', "")  # adapt to your schema
        lines.append(
            f"[{i}] id={doc_id!r} title={title!r}\n"
            f"    text={text}\n"
        )
    return "\n".join(lines)
import re

def _strip_invalid_citations(answer: str, n_docs: int) -> str:
    def repl(m: re.Match) -> str:
        k = int(m.group(1))
        return m.group(0) if 1 <= k <= n_docs else ""
    return re.sub(r"\[(\d+)\]", repl, answer).strip()

def generate_citations(query: str, docs: list[dict[str, str]]) -> str:
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    if not docs:
        return "I don't have enough information."

    docs_block = _format_docs_for_prompt(docs)
    n = len(docs)

    prompt = f"""You are Hoopla's movie helper. Answer using ONLY the documents below.

Query: {query}

Documents (cite using these numbers ONLY):
{docs_block}

Rules:
- Use ONLY information explicitly supported by the documents.
- Every factual sentence must end with citations like [1] or [1][2].
- You may ONLY cite from [1]..[{n}]. Never cite anything else.
- If the documents don't contain enough information to answer, say: "I don't have enough information."
  Then give the best partial answer you can with citations.
- Do NOT invent plot details.

Answer:"""

    response = _call_local_llm(prompt).strip()
    response = _strip_invalid_citations(response, n)
    return response


def question_answering(question: str, docs: list[dict[str, str]]) -> str:
    print(Fore.MAGENTA + "Question: " + question + Style.RESET_ALL)
    context = _format_docs_for_prompt(docs)
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service. 

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation
- DO NOT make up answers that are not found in the documents.
- Do not ask for clarification.

Question: {question}

Answer:"""
    response = _call_local_llm(prompt).strip()
    return response