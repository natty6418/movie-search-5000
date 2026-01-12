#!/usr/bin/env python3
"""Test script to run the Movie Search 5000 agent."""

import json
from typing import Any, Dict, List
from cli.lib.agent_copy import build_graph, AgentState
from cli.lib.hybrid_search import HybridSearch

def main():
    print("=" * 60)
    print("Movie Search 5000 Agent Test")
    print("=" * 60)
    
    # Load movies and build hybrid search engine
    print("\nLoading movie database...")
    movies_path = "data/movies.json"
    with open(movies_path, "r", encoding="utf-8") as f:
        movies_list = json.load(f)["movies"]
    
    print(f"Loaded {len(movies_list)} movies")
    
    # Build hybrid search engine
    print("Building hybrid search engine...")
    engine = HybridSearch(movies_list)
    
    # Build the agent
    print("Building agent graph...")
    agent = build_graph(engine)
    print("\n✓ Agent built successfully!\n")
    
    # Create initial state
    initial_state: AgentState = {
        "query": "I want a sci-fi movie with time travel",
        "messages": [],
        "retrieved_docs": [],
        "limit": 5,
        "action": ""
    }
    
    print(f"User Query: {initial_state['query']}")
    print("\nRunning agent...\n")
    print("-" * 60)
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 60)
    print("AGENT EXECUTION COMPLETE")
    print("=" * 60)
    
    print(f"\nFinal Query: {final_state.get('query', 'N/A')}")
    print(f"Number of Retrieved Docs: {len(final_state.get('retrieved_docs', []))}")
    print(f"Final Action: {final_state.get('action', 'N/A')}")
    
    print("\n" + "-" * 60)
    print("Retrieved Movies:")
    print("-" * 60)
    for i, doc in enumerate(final_state.get("retrieved_docs", []), 1):
        print(f"\n{i}. {doc['title']}")
        desc = doc['description']
        print(f"   {desc[:150]}{'...' if len(desc) > 150 else ''}")
    
    print("\n" + "-" * 60)
    print("Conversation History:")
    print("-" * 60)
    for i, msg in enumerate(final_state.get("messages", []), 1):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        print(f"\n[{i}] {role}:")
        print(f"{content[:300]}{'...' if len(content) > 300 else ''}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
