import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import time

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
            "You are a world-class café strategy consultant trusted by elite firms like McKinsey, Bain, and BCG. "
            "You are known for uncovering **high-leverage, non-obvious insights** by connecting dots across operational, financial, demographic, and behavioral datasets. "
            "You think like an operator, speak like a strategist, and act like a growth architect.\n\n"

            "Your mission:\n"
            "- Identify **undervalued opportunities** and **hidden risks** others miss\n"
            "- Think **holistically across form inputs, POS, customer behavior, and local demographics**\n"
            "- **Synthesize** patterns across personas, tags, item performance, foot traffic, and labor timing\n"
            "- Recommend actions that create **real business impact** within 3–6 months\n\n"

            "Output only 1–3 **clear, bold STAR recommendations**, each with:\n"
            "• **Situation** – What matters most in the current data (connect across sources)\n"
            "• **Task** – What's the core business problem or growth goal?\n"
            "• **Action** – What’s the most effective, creative, and executable move?\n"
            "• **Result** – What is the likely outcome or ROI (quantify if possible)?\n\n"

            "**Be surgical, not generic.** Prioritize what's **counterintuitive but backed by evidence**. "
            "Avoid summaries or fluff. Dive deep. Show connections the business hasn’t seen."
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

        full_prompt = f"{system_prompt}\n\n" + "\n\n".join(prompt_parts)

        # Actual OpenAI call (unchanged)
        resp = client.responses.create(
            model="gpt-4o",
            input=full_prompt,
            tools=[{"type": "web_search_preview"}]
        )

        full_answer = resp.output_text.strip()

        # Simulated streaming to UI
        output_area = st.empty()
        streamed = ""
        for char in full_answer:
            streamed += char
            output_area.markdown(streamed)
            time.sleep(0.002)  # Typing delay (~500 chars/sec)

        responses.append({"query": q, "answer": full_answer})
        prog.progress(idx / total)

    return {"responses": responses}
