# models/deep_dive_model.py

import streamlit as st
from openai import OpenAI
import os

# Initialize the OpenAI client with your API key from Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def run_deep_dive(context: dict, queries: list[str]) -> dict:
    """
    For each query, construct a prompt that includes:
      - Form data
      - Demographics
      - POS data
    Then invoke the web-enabled GPT model (gpt-4o-mini) with the built-in browsing tool,
    showing a progress bar as it goes.
    """
    total = len(queries)
    prog = st.progress(0.0)
    responses = []

    for idx, q in enumerate(queries, start=1):
        # Build the combined prompt
        prompt_parts = [
    # 1) Role + high-level instructions
            "You are a data-driven café optimization assistant.  "
            "Using only the data provided (form fields, demographics, POS data and any relevant web context), "
            "produce a bullet-pointed analysis in the exact structure below. "
            "Be extremely specific, include concrete examples (citing where each insight came from), "
            "and suggest only things that haven’t already been done.",

    # 2) Desired response structure
            "— Response format —",
            "1. **Summary of current situation** (current business context + pros/cons + what the customer is asking)",
            "2. **Data summary** (key numbers & observations from Form/Demo/POS/Web)",
            "3. **So what?** (implications of the data)",
            "4. **Why it matters** (importance for this café)",
            "5. **Potential impact** (what implementing these could achieve)",
            "6. **Implementation intro** (brief how-to overview)",
            "7. **Recommendations** (short-term, actionable steps; each with clear sub-steps and examples)",

    # 3) Insert all your data sections
            "=== Form Data ===",
            *[f"{k}: {v}" for k, v in context["form"].items()],

            "=== Demographics ===",
            *[f"{k}: {v}" for k, v in context["demographics"].items()],

            "=== POS Data ===",
            *(str(r) for r in context.get("pos", []) or ["No POS data provided."]),

           # 4) The actual user question
            "=== Question ===",
            q
        ]

        prompt = "\n\n".join(prompt_parts)

        # Call the web-enabled GPT model
        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=prompt
        )

        # Capture the answer
        answer = resp.output_text
        responses.append({"query": q, "answer": answer})

        # Advance the progress bar
        prog.progress(idx / total)

    return {"responses": responses}
