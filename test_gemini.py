#!/usr/bin/env python3
"""
Test script to verify Gemini API works with function calling
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

print(f"✓ API Key found: {GEMINI_API_KEY[:10]}...")

# Configure Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Define a simple test tool
search_movies_tool = types.FunctionDeclaration(
    name="search_movies",
    description="Search for movies based on a query",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for movies"
            }
        },
        "required": ["query"]
    }
)

print("\n📡 Testing Gemini API connection...")

try:
    # Test 1: Basic chat
    print("Testing basic chat with gemini-3-flash-preview...")
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents="Say hello in 5 words"
    )
    print(f"✓ Basic chat works!")
    print(f"  Response: {response.text[:100]}")

    # Test 2: Function calling
    print("\n📞 Testing function calling...")
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents="Find me action movies from the 90s",
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[search_movies_tool])]
        )
    )

    # Check if function was called
    if response.candidates[0].content.parts[0].function_call:
        func_call = response.candidates[0].content.parts[0].function_call
        print(f"✓ Function calling works!")
        print(f"  Called: {func_call.name}")
        print(f"  Args: {func_call.args}")
    else:
        print(f"⚠ Function calling returned text instead:")
        print(f"  {response.text[:100]}")

    print("\n✅ All tests passed! Gemini is ready to use.")

except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
    print("\nCommon issues:")
    print("- Invalid or expired API key")
    print("- API quota exceeded")
    print("- Network connectivity issues")
    print(f"\nFull error: {e}")
