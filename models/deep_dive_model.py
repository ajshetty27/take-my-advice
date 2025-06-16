# models/deep_dive_model.py

import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI client with your API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def run_deep_dive(context: dict, queries: list[str]) -> dict:
    """
    For each query, construct a prompt that includes:
      - Form data
      - Demographics
      - POS data
      - Extra additional insights
    Then invoke the web-enabled GPT model (gpt-4o-mini) with the built-in browsing tool,
    showing a progress bar as it goes.
    """
    total = len(queries)
    prog = st.progress(0.0)
    responses = []

    for idx, q in enumerate(queries, start=1):
        # Build the combined prompt
        prompt_parts = [
            # 1) Role & mission
            "You are a data-driven café optimization assistant. "
            "Using ONLY (a) the provided internal data (Form inputs, Demographics, POS, extra additional insights) and (b) reputable public sources on Gen Z/Millennial beverage & food trends, "
            "produce an ultra-specific, bullet-pointed analysis with citations.",

            # 2) Desired response structure
            "— Response format —",
            "1. **Summary of current situation**: business context, pros/cons, customer ask",
            "2. **Data summary**: key metrics from Form/Demo/POS seamlessly merged with 1–2 external stats (cite source)",
            "3. **So what?**: joint implications of your internal data & external trends",
            "4. **Why it matters**: link every insight back to café’s goals (e.g. lift AOV, drive foot traffic)",
            "5. **Potential impact**: quantify expected gain (%, $) with brief rationale",
            "6. **Implementation intro**: succinct pilot plan overview",
            "7. **Recommendations**: for each, use **STAR** (Situation → Task → Action → Result) and include:",
            "   • the exact menu/item tweak or promo (e.g. “Pumpkin Spice Latte + Bagel combo at $X”)",
            "   • clear sub-steps (e.g. POS trigger setup, signage, social post copy)",
            "   • linked back to data and goal",

            # 3) Insert all your data sections
            "=== Form Data ===",
            *[f"{k}: {v}" for k, v in context["form"].items()],

            "=== Demographics ===",
            *[f"{k}: {v}" for k, v in context["demographics"].items()],

            "=== POS Data ===",
            *(str(r) for r in context.get("pos", []) or ["No POS data provided."]),

           "=== Additional Insight ===",
            context["extra"] or "No additional insight provided.",

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
