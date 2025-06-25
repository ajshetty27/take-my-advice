import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def run_deep_dive(context: dict, queries: list[str]) -> dict:
    total = len(queries)
    prog = st.progress(0.0)
    responses = []

    for idx, q in enumerate(queries, start=1):
        # Consultant-style instruction
        system_prompt = (
            "You are a top-tier café consultant trusted by McKinsey and Bain. "
            "You specialize in uncovering deep, non-obvious insights from business data and offering "
            "laser-sharp, actionable strategies that drive measurable outcomes.\n\n"
            "Your answers must:\n"
            "- Think critically and holistically across all datasets\n"
            "- Highlight **why** the insight matters (link to goals)\n"
            "- Use STAR format for each recommendation:\n"
            "  • Situation: Current scenario from data\n"
            "  • Task: Business need or problem\n"
            "  • Action: Clear, evidence-backed intervention\n"
            "  • Result: Expected gain, quantified if possible\n\n"
            "Respond with only STAR insights—no summary or preamble. Be incisive, impactful, and practical."
        )

        # Build prompt
        prompt_parts = [
            "=== Form Data ===",
            *[f"{k}: {v}" for k, v in context["form"].items()],
            "=== Demographics ===",
            *[f"{k}: {v}" for k, v in context["demographics"].items()],
            "=== POS Data ===",
            *(str(r) for r in context.get("pos", []) or ["No POS data provided."]),
            "=== Persona Summary ===",
            *(
                [f"Cluster {c}: {d}" for c, d in context.get("persona_summary", {}).items()]
                if context.get("persona_summary") else ["No persona descriptions available."]
            ),
            "=== Top Items per Persona ===",
            *(
                [f"Cluster {c}: {i}" for c, i in context.get("top_items_summary", {}).items()]
                if context.get("top_items_summary") else ["No top item data available."]
            ),
            "=== Estimated People per Persona ===",
            *(
                [f"Cluster {c}: {count} people" for c, count in context.get("persona_sizes", {}).items()]
                if context.get("persona_sizes") else ["No persona size data available."]
            ),
            "=== Top Tags per Persona ===",
            *(
                [f"Cluster {c}: {', '.join(tags)}" for c, tags in context.get("persona_tags", {}).items()]
                if context.get("persona_tags") else ["No tag data available."]
            ),
            "=== Question ===",
            q
        ]

        # Combine into final prompt
        full_prompt = f"{system_prompt}\n\n" + "\n\n".join(prompt_parts)

        # Send to OpenAI Responses API (with web search tool)
        resp = client.responses.create(
            model="gpt-4o",
            input=full_prompt,
            tools=[{"type": "web_search_preview"}]
        )

        answer = resp.output_text.strip()
        responses.append({"query": q, "answer": answer})
        prog.progress(idx / total)

    return {"responses": responses}
