#!/usr/bin/env python3
"""Test script to visualize the agent graph."""

from cli.lib.agent_copy import build_movie_search_agent

def main():
    print("Building and visualizing Movie Search 5000 agent graph...\n")
    agent = build_movie_search_agent()
    
    try:
        # Try to display as ASCII (works in terminal)
        print(agent.get_graph().draw_ascii())
        print("\n✓ ASCII visualization successful!\n")
    except Exception as e:
        print(f"ASCII visualization failed: {e}\n")
    
    try:
        # Try to get mermaid syntax
        mermaid = agent.get_graph().draw_mermaid()
        print("Mermaid diagram syntax:")
        print(mermaid)
        print("\n✓ You can paste this into https://mermaid.live to visualize\n")
    except Exception as e:
        print(f"Mermaid visualization failed: {e}\n")

if __name__ == "__main__":
    main()
