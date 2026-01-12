"""Utilities for talking to a local LLM via Ollama."""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from colorama import Fore, Style
from openai import OpenAI

load_dotenv()

# Ollama configuration
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")

class LocalLLMError(RuntimeError):
    """Raised when the LLM cannot be reached or returns invalid data."""

def _call_local_llm(prompt: str, system_prompt: str = None, json_mode: bool = False) -> str:
    """
    Calls the local Ollama instance using OpenAI-compatible API.
    """
    try:
        client = OpenAI(
            base_url=f"{LOCAL_LLM_URL}/v1",
            api_key="ollama"  # Ollama doesn't require a real key
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Set specific parameters for the generation
        response_format = {"type": "json_object"} if json_mode else None

        response = client.chat.completions.create(
            model=LOCAL_LLM_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            response_format=response_format
        )

        content = response.choices[0].message.content
        if not content:
             raise LocalLLMError("LLM response payload was empty.")

        return content.strip()

    except Exception as exc:
        raise LocalLLMError(
            f"Error calling Ollama for model {LOCAL_LLM_MODEL}: {exc}"
        ) from exc



def enhance_query(query: str) -> str:
    system = "You are a helpful assistant that fixes spelling errors for MovieSearch 5000™."
    prompt = f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.
Query: "{query}"
If no errors, return the original query.
Corrected:"""
    try:
        response = _call_local_llm(prompt, system_prompt=system)
    except LocalLLMError as exc:
        print(f"Local LLM enhancement failed ({exc}). Using the original query.")
        return query

    cleaned = response.strip()
    return cleaned or query


def rewrite_query(query: str) -> str:
    system = "You are an expert movie search query rewriter for MovieSearch 5000™."
    prompt = f"""
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
        response = _call_local_llm(prompt, system_prompt=system)
    except LocalLLMError as exc:
        print(f"Local LLM enhancement failed ({exc}). Using the original query.")
        return query
    cleaned = response.strip()
    return cleaned or query


def expand_query(query: str) -> str:
    system = "You are a movie search assistant for MovieSearch 5000™."
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
        response = _call_local_llm(prompt, system_prompt=system)
    except LocalLLMError as exc:
        print(f"Local LLM enhancement failed ({exc}). Using the original query.")
        return query
    cleaned = response.strip()
    return cleaned or query


def rank_results(query: str, doc: dict) -> str:
    system = "You are a movie relevance rater for MovieSearch 5000™."
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
        response = _call_local_llm(prompt, system_prompt=system)
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
    
    system = "You are a movie ranking system for MovieSearch 5000™."
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

    response = _call_local_llm(prompt, system_prompt=system, json_mode=True).strip()
    return json.loads(response)


def judge_relevance(query: str, formatted_results: list[str]) -> list[str]:
    system = "You are a relevance judge for MovieSearch 5000™."
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

    response = _call_local_llm(prompt, system_prompt=system, json_mode=True).strip()
    return json.loads(response)


def rag_answer(query: str, docs: list[dict[str, str]]) -> str:

    print(Fore.YELLOW + "Query: " + query + Style.RESET_ALL)
    system = "You are a helpful assistant for MovieSearch 5000™, a movie streaming service."
    
    context = _format_docs_for_prompt(docs)
    
    prompt = f"""You are a movie recommendation assistant.
    
    User Query: "{query}"
    
    Provided Movies (Context):
    {context}
    
    Task:
    - Analyze the provided movies.
    - Identify which ones are most relevant to the User Query.
    - Recommend them to the user with a brief explanation of why they fit.
    - If a movie in the list is NOT relevant (e.g. wrong actor, wrong genre), ignore it.
    - If NO movies are relevant, politely say so.
    
    Response:"""

    response = _call_local_llm(prompt, system_prompt=system).strip()
    return response


def summarize(query: str, docs: list[dict[str, str]]) -> str:
    system = "You are a helpful assistant for MovieSearch 5000™, a movie streaming service."
    
    context = _format_docs_for_prompt(docs)
    
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to MovieSearch 5000™ users.

Query: {query}
Search Results:
{context}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    response = _call_local_llm(prompt, system_prompt=system).strip()
    return response

def _format_docs_for_prompt(docs: list[dict[str, str]]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        title = d.get("title", "Unknown Title")
        text = d.get('description', "")
        lines.append(f"Movie {i}: {title}\nDescription: {text}\n")
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
    
    system = "You are MovieSearch 5000™'s movie helper."
    prompt = f"""Answer using ONLY the documents below.

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

    response = _call_local_llm(prompt, system_prompt=system).strip()
    response = _strip_invalid_citations(response, n)
    return response


def question_answering(question: str, docs: list[dict[str, str]]) -> str:
    print(Fore.MAGENTA + "Question: " + question + Style.RESET_ALL)
    context = _format_docs_for_prompt(docs)
    
    system = "You are a helpful assistant for MovieSearch 5000™. You are casual and conversational."
    prompt = f"""Answer the user's question based on the provided movies that are available on MovieSearch 5000™.

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
    response = _call_local_llm(prompt, system_prompt=system).strip()
    return response