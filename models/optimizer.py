# models/optimizer.py

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

OPTIMIZATION_PROMPTS = [
    "Using the provided POS sales volumes, menu recipes, and ingredient costs, identify 3 opportunities to reduce per-unit food costs—e.g. by bulk ordering, ingredient substitutions, or cross-utilization of stock—while maintaining product quality. For each, estimate annualized savings.",
    "Based on weekly sales seasonality and lead times for key ingredients, recommend an optimized ordering schedule (frequency & quantities) that minimizes spoilage and stockouts. Cite specific items and timing.",
    "Given sales mix, contribution margin per item, and local demographic willingness-to-pay, propose a revised menu pricing and item placement (e.g. highlight high-margin combos) designed to shift 10% of low-margin sales to higher-margin offerings.",
    "Analyze POS transaction timestamps and staff counts to suggest an optimized shift schedule (number of employees by hour) that meets demand peaks while cutting idle labor hours by at least 15%. Include a sample weekly rota.",
    "Using historical waste records (refunds, spoilage) alongside sales data, pinpoint the top 3 waste drivers and recommend process or recipe tweaks to reduce waste volume by 20%.",
    "Review your list of top-spend ingredients and suggest negotiation strategies or alternative suppliers (including approximate volume discounts) to lower unit costs by 5–10%.",
    "Based on café hours, equipment usage patterns, and utility rates, propose at least 3 energy-saving measures (e.g. equipment scheduling, low-flow fixtures) that could shave 10% off monthly utility bills.",
    "Using POS trends and local events schedule, develop a demand forecast model and outline how to adjust both prep volumes and staffing 48 hrs in advance to avoid stockouts and over-staffing.",
    "Identify the 3 highest-loyalty customer segments and design targeted upsell offers or add-on bundles (e.g. “Latte + pastry”) that have a lift in average order value of at least 15%, including messaging frameworks.",
    "Create a “profitability heatmap” by day-part and menu category (drinks vs food). Highlight the worst performing slots and recommend operational changes (e.g. limited specials, staffing tweaks) to boost under-leveraged time windows."
]

def auto_parse_pos(pos_df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "Qty sold": "qty_sold",
        "Avg. item price": "avg_item_price",
        "Net sales": "net_sales",
        "Waste count": "waste_count",
        "Item": "item",
    }
    parsed_df = pos_df.rename(columns={k: v for k, v in column_map.items() if k in pos_df.columns})
    return parsed_df[[col for col in column_map.values() if col in parsed_df.columns]].dropna()

def generate_extra_insights(pos_df: pd.DataFrame, labor_df: pd.DataFrame = None) -> str:
    insights = []

    try:
        if 'net_sales' in pos_df.columns:
            df = pd.DataFrame({
                'ds': pd.date_range(end=pd.Timestamp.today(), periods=len(pos_df)),
                'y': pos_df['net_sales'].values
            })
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)
            insights.append("Prophet forecast completed for weekly sales trends.")
    except Exception as e:
        insights.append(f"[Prophet] Error: {str(e)}")

    try:
        df = pos_df[['avg_item_price', 'qty_sold', 'net_sales']].dropna()
        kmeans = KMeans(n_clusters=3).fit(df)
        pos_df['cluster'] = kmeans.labels_
        insights.append("KMeans clustering applied to menu items by price, volume, and revenue.")
    except Exception as e:
        insights.append(f"[KMeans] Error: {str(e)}")

    try:
        if labor_df is not None:
            X = labor_df[['num_employees']]
            y = labor_df['transactions']
            reg = RandomForestRegressor().fit(X, y)
            score = reg.score(X, y)
            insights.append(f"Labor regression R² score: {score:.2f}")
    except Exception as e:
        insights.append(f"[Regression] Error: {str(e)}")

    return "\n".join(insights)

def run_optimization(context: dict, prompts: list[str] = OPTIMIZATION_PROMPTS) -> dict:
    total = len(prompts)
    prog = st.progress(0.0)
    responses = []

    pos_df_raw = pd.DataFrame(context.get("pos_raw", []))
    pos_df = auto_parse_pos(pos_df_raw)
    labor_df = pd.DataFrame(context.get("labor_raw", [])) if context.get("labor_raw") else None
    extra_insights = generate_extra_insights(pos_df, labor_df)

    def has_required_data(prompt_text: str) -> bool:
        if "waste" in prompt_text.lower() and 'waste_count' not in pos_df.columns:
            return False
        return True

    for idx, prompt_text in enumerate(prompts, start=1):
        if not has_required_data(prompt_text):
            continue

        prompt_parts = [
            "You are a senior café operations consultant optimizing restaurant operations across finance, staffing, waste, and demand.",
            "Using the internal data (Form, POS, Demographics, ML insights) and external assumptions where helpful,",
            "generate an optimized, actionable, quantified, and cross-functional recommendation. Then also rate its priority.",
            "",
            "— Response Format —",
            "1. Summary of issue",
            "2. Supporting data (internal + public)",
            "3. Why it matters (margin, efficiency, satisfaction)",
            "4. Estimated impact (quantified)",
            "5. Pilot plan",
            "6. STAR-formatted Recommendations (Situation, Task, Action, Result)",
            "7. Priority Score (1–5) based on ROI, Urgency, and Feasibility",
            "8. Where this fits in the next-best action sequence",
            "",
            "=== Form Data ===",
            *[f"{k}: {v}" for k, v in context.get("form", {}).items()],
            "",
            "=== Demographics ===",
            *[f"{k}: {v}" for k, v in context.get("demographics", {}).items()],
            "",
            "=== POS Data ===",
            *(str(r) for r in context.get("pos", []) or ["No POS data provided."]),
            "",
            "=== Additional Insight ===",
            context.get("extra", "No additional insight provided."),
            "",
            "=== Machine Learning Insights ===",
            extra_insights,
            "",
            "=== Optimization Prompt ===",
            prompt_text
        ]

        full_prompt = "\n\n".join(prompt_parts)

        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=full_prompt
        )

        answer = resp.output_text
        responses.append({"prompt": prompt_text, "answer": answer})
        prog.progress(idx / total)

    return {"responses": responses}
