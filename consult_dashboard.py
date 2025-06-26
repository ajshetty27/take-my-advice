# consult_dashboard.py
import json
import os
import base64
import streamlit as st
import re
import textwrap
import gspread
from google.oauth2.service_account import Credentials


from datetime import datetime
import pandas as pd
from gspread_dataframe import set_with_dataframe

from openai import OpenAI

import io
import matplotlib.pyplot as plt

from streamlit.components.v1 import html


from dotenv import load_dotenv
load_dotenv()


from models.arcgis_explorer import *
from models import buckets
from models.deep_dive_model import run_deep_dive

# --- GOOGLE SHEETS SETUP ---
SHEET_ID    = "1bviJIh9XYcg0I8V2B_DQ5S8Ar-qxo8_qFY-R-pS1ir8"
WS_NAME     = "Sheet 1"
AUTH_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# 1) Decode the base64 JSON
b64 = os.environ["GCLOUD_SA_KEY_B64"]
sa_json = base64.b64decode(b64).decode("utf-8")
info = json.loads(sa_json)

# 2) Build your creds & client
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_gspread_client():
    creds  = Credentials.from_service_account_info(info, scopes=AUTH_SCOPES)
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def get_worksheet():
    client = get_gspread_client()
    return client.open_by_key(SHEET_ID).worksheet(WS_NAME)

def load_remote_csv(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"❌ Failed to load file from {url}: {e}")
        return pd.DataFrame()

def load_uploaded_csvs(record):
    file_mappings = {
        "Sales Summary": "sales_summary_df",
        "Product Mix": "product_mix_df",
        "Order Details": "order_details_df",
        "Expenses": "expenses_df",
        "Employee Schedule": "employee_schedule_df",
        "Rewards Program": "rewards_program_df",
        "Menu PDF": "menu_pdf_df"  # if this is a CSV, otherwise skip loading
    }

    for col_name, session_key in file_mappings.items():
        if session_key not in st.session_state:
            url = str(record.get(col_name, "")).strip()
            if url.startswith("http"):
                df = load_remote_csv(url)
                if not df.empty:
                    st.session_state[session_key] = df

ws = get_worksheet()
# Cache the header row once
@st.cache_data(show_spinner=False)
def get_headers():
    return [h for h in ws.row_values(1) if h.strip()]

HEADERS = [

    # demographic keys...
    "Total Population (2024)", "Total Households (2024)", "Avg Household Size (2024)", "Median Household Income (2024)",
    "Per-Capita Income (2024)", "Diversity Index (2024)", "Pop. Growth Rate ’24–’29 (%)", "Median Home Value (2024)",
    "Avg Home Value (2024)", "Daytime Worker Pop. (2024)", "Median Age (2024)", "Owner-occupied Households (2024)",
    "Renter-occupied Households (2024)", "Vacant Housing Units (2024)", "Daytime Population (2024)", "Daytime Resident Pop. (2024)",
    "Population Age 18 (2024)", "Population Age 25 (2024)", "Population Age 35 (2024)", "Population Age 65+ (2024)",
    "Age Dependency Ratio (2024)", "White Population (2024)", "Black Population (2024)", "Asian Population (2024)",
    "Unemployment Rate (2024)", "Household Growth Rate ’24–’29 (%)", "Eating & Drinking Businesses (2024)", "Group Quarters Population (2024)"
]


# Check if demographic headers are already in the sheet
sheet_headers = ws.row_values(1)
if not any(header in sheet_headers for header in HEADERS):
    # Append demographic headers to the end of row 1
    updated_headers = sheet_headers + HEADERS
    ws.delete_rows(1)  # remove original header row
    ws.insert_row(updated_headers, index=1)

def append_row_to_sheet(row: list):
    ws.append_row(row, value_input_option="USER_ENTERED")


def load_sheet_df() -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    raw_headers = values[0]
    idxs = [i for i, h in enumerate(raw_headers) if h.strip()]
    headers = [raw_headers[i] for i in idxs]
    data = []
    for row in values[1:]:
        row += [""] * (len(raw_headers) - len(row))
        data.append([row[i] for i in idxs])

    df = pd.DataFrame(data, columns=headers)

    # ✅ Move RENAME_COLUMNS inside function before returning
    RENAME_COLUMNS = {
        "What is your business name?": "Business Name",
        "What is your Name and Position?": "Contact Name",
        "Position": "Contact Position",
        "What is your professional Email Address and Phone Number?": "Contact Info",
        "(XXX) XXX-XXXX": "Phone Number",
        "What is the Business Address?": "Location Address",
        "1. Which stage is most accurate to your business?": "Stage",
        "2. Describe in detail your target market.": "Target Market",
        "3. What key factors sets your café apart from competitors?": "Differentiators",
        "4. What are your top 3 goals in working with Take My [Advice]": "Goals",
        "5. If your café could be known for one thing, what would it be - and why does that matter to your customer?": "Signature Item",
        "6. How do you define success 6 months from now?": "Definition of Success",
        "7. What do you believe is currently holding your business back from achieving that success?": "Current Challenges",
        "8. If your café had a \"personality\" how would you describe it?": "Brand Personality",
        "9. How do you think your business is currently perceived by your customers?": "Customer Perception",
        "10. Are there specific competitors you admire or feel competitive pressure from? Why?": "Competitors",
        "11. What does your ideal customer journey look like from walking in -> to ordering -> to exiting?": "Customer Journey",
        "12. What are the biggest unknowns you hope we can help you answer?": "Unknowns",
        "13. Is there a demographic in the area that you feel you are missing out on servicing? Is there a demographic you are excelling in servicing?": "Demographic Gaps",
        "14. What does \"Growth\" mean to you right now?": "Growth Definition",
        "15. Are there any recent changes (new items, staff, renovations, marketing efforts) that we should be aware of when analyzing performance?": "Recent Changes",
        "16. Do you see the business staying in this current location long-term, or are you considering relocation or expansion?": "Location Plans",
        "17. If you could change one thing about your café immediately, what would it be?": "Immediate Change",
        "Sales Summary for the past 0-24 months (minimum 12 months)": "Sales Summary",
        "Sales by Item": "Product Mix",
        "Order Details": "Order Details",
        "Expenses": "Expenses",
        "Employee Schedule": "Employee Schedule",
        "Rewards Program Details (If Applicable)": "Rewards Program",
        "Downloadable Menu": "Menu PDF"
    }

    df = df.rename(columns=RENAME_COLUMNS)
    return df

def save_demographics_to_sheet(business_name, demo_dict):
    gs_client = get_gspread_client()
    sh = gs_client.open_by_key(SHEET_ID)
    ws = sh.worksheet(WS_NAME)

    # Get current headers
    headers = ws.row_values(1)
    header_set = set(headers)

    # Add missing demographic headers
    new_headers = [h for h in HEADERS if h not in header_set]
    if new_headers:
        updated_headers = headers + new_headers
        ws.delete_rows(1)
        ws.insert_row(updated_headers, index=1)
        headers = updated_headers  # update reference

    # Reload sheet data to find correct row index
    df = pd.DataFrame(ws.get_all_records())

    # Try both pre- and post-renamed business column keys
    business_col = "Business Name" if "Business Name" in df.columns else "What is your business name?"
    row_idx = df[df[business_col] == business_name].index

    if not row_idx.empty:
        row_number = row_idx[0] + 2  # +1 for header row, +1 because DataFrame is 0-indexed

        for key, value in demo_dict.items():
            if key in headers:
                col_number = headers.index(key) + 1  # 1-indexed for Sheets
                ws.update_cell(row_number, col_number, value)
    else:
        st.warning(f"⚠️ Could not find business name: '{business_name}' in the sheet.")

# --- STREAMLIT APP ---
def main():
    st.set_page_config(layout="wide")
    st.title("Take My Advice [...]")
    tab_summary, tab_deep = st.tabs(
        ["Summary", "Advice"]
    )

    if "cached_df" not in st.session_state:
        st.session_state.cached_df = load_sheet_df()
    df_global = st.session_state.cached_df

    # --- Form Tab ---
    with tab_summary:

        df = df_global
        st.header("📋 Café Submission Summary")

        if df.empty:
            st.info("No submissions available.")
            st.stop()

        cafes = df["Business Name"].dropna().unique().tolist()
        selected_cafe = st.selectbox("Select a café to view ", cafes, key="form_tab_select")

        # ✅ Store selection for use in Explore and Deep Dive
        st.session_state["target_cafe_name"] = selected_cafe

        matches = df[df["Business Name"] == selected_cafe]
        if matches.empty:
            st.warning("No data found for the selected café.")
            st.stop()

        record = matches.iloc[0]

        # ✅ Define grouped carousel structure
        grouped_fields = {
            "🏪 Business Overview": ["Business Name", "Location Address", "Stage"],
            "🎯 Target Market": ["Target Market"],
            "✨ What Sets Us Apart": ["Differentiators"],
            "⭐ Signature Item": ["Signature Item"],
            "📈 Goals": ["Goals"],
            "🧩 Current Challenges": ["Current Challenges"],
            "💬 Brand Personality": ["Brand Personality"],
            "👥 Customer Perception": ["Customer Perception"],
            "🏁 Success Means": ["Definition of Success"],
            "🥊 Key Competitors": ["Competitors"],
            "🛒 Customer Journey": ["Customer Journey"],
            "🧠 Unknowns / Strategic Questions": ["Unknowns"],
            "📊 Demographic Gaps": ["Demographic Gaps"],
            "🚀 Growth Definition": ["Growth Definition"],
            "📢 Recent Changes": ["Recent Changes"],
            "🗺️ Location Plans": ["Location Plans"],
            "⚡ Immediate Focus": ["Immediate Change"],
        }

        section_titles = list(grouped_fields.keys())
        total_sections = len(section_titles)

        if "summary_section_index" not in st.session_state:
            st.session_state.summary_section_index = 0

        # ✅ Arrow navigation buttons
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button("⬅️", use_container_width=True) and st.session_state.summary_section_index > 0:
                st.session_state.summary_section_index -= 1
        with col3:
            if st.button("➡️", use_container_width=True) and st.session_state.summary_section_index < total_sections - 1:
                st.session_state.summary_section_index += 1

        # ✅ CSS Styling for section box
        st.components.v1.html("""
        <style>
        .section-box {
          padding: 1.2em;
          background-color: #2b2b2b;
          border-radius: 15px;
          margin-bottom: 1.2em;
          color: #B0A698;
        }
        .section-box h3 {
          color: #bda967;
          margin-bottom: 0.3em;
        }
        .section-box ul {
          padding-left: 1.2em;
        }
        </style>
        """, height=0)


        # ✅ Render current section content
        section_key = section_titles[st.session_state.summary_section_index]
        field_labels = grouped_fields[section_key]

        html_content = f"""
<div style='padding: 1.2em; background-color: #2b2b2b; border-radius: 15px; margin-bottom: 1.2em; color: #B0A698;'>
    <h3 style='color: #bda967; margin-bottom: 0.3em;'>{section_key}</h3>
    <ul>
"""
        for label in field_labels:
            val = record.get(label, "")
            if val:
                lines = [line.strip() for line in str(val).split("\n") if line.strip()]
                for line in lines:
                    html_content += f"<li>{line}</li>"
        html_content += "</ul></div>"

        st.markdown(html_content, unsafe_allow_html=True)
        st.caption(f"{st.session_state.summary_section_index + 1} of {total_sections}")

        # ✅ Upload missing files section
        st.subheader("📎 Upload Missing Files")
        uploaded_file_fields = {
            "Sales Summary (0–24 months)": "Sales Summary",
            "Product Mix (by Item)": "Product Mix",
            "Order Details": "Order Details",
            "Expenses": "Expenses",
            "Employee Schedule": "Employee Schedule",
            "Rewards Program": "Rewards Program",
            "Menu PDF": "Menu PDF"
        }

        for label, key in uploaded_file_fields.items():
            col1, col2 = st.columns([2, 3])
            with col1:
                file_url = record.get(key, "")
                if file_url and str(file_url).startswith("http"):
                    st.markdown(f"[🔗 Download {label}]({file_url})")
                else:
                    st.markdown(f"❌ No link for {label}")
            with col2:
                upload_key = f"upload_{key.lower().replace(' ', '_')}"
                uploaded = st.file_uploader(f"Re-upload {label}", key=upload_key)

                if uploaded:
                    try:
                        df_uploaded = None
                        if uploaded.name.endswith(".csv"):
                            df_uploaded = pd.read_csv(uploaded)
                            st.write(f"CSV uploaded, shape: {df_uploaded.shape}")
                        elif uploaded.name.endswith((".xlsx", ".xls")):
                            df_uploaded = pd.read_excel(uploaded)
                            st.write(f"Excel uploaded, shape: {df_uploaded.shape}")
                        else:
                            st.warning(f"Unsupported file type: {uploaded.name.split('.')[-1]} — only CSV or Excel supported.")

                        if df_uploaded is not None:
                            st.session_state[f"{key.lower().replace(' ', '_')}_df"] = df_uploaded
                            st.success(f"{label} successfully re-uploaded!")
                    except Exception as e:
                        st.error(f"Failed to process {label}: {e}")




        with st.expander("Explore", expanded=False):
            st.header("📍 View whats around")

            selected = st.session_state.get("target_cafe_name", None)
            if not selected:
                st.info("Please select a café in the Summary tab first.")
                st.stop()

            df = df_global
            record = df[df["Business Name"] == selected].iloc[0]
            address = record["Location Address"]
            st.markdown(f"**Address:** {address}")

            # Only run exploration once or reuse results
            if "last_demo" not in st.session_state or "last_cafes" not in st.session_state:
                map_html, demo, cafes = run_explorer(address)
                st.session_state["last_demo"] = demo
                st.session_state["last_cafes"] = cafes
                st.session_state["target_cafe_address"] = address
                st.session_state["map_html"] = map_html  # Store initial map

            # Load previous results
            demo = st.session_state["last_demo"]
            cafes = st.session_state["last_cafes"]

            df_cafes = pd.DataFrame(cafes).fillna("").astype(str)

            # -- Selectbox must come BEFORE the updated map call
            selected_flash = st.selectbox("Select Café Nearby to View", df_cafes["Name"].tolist())

            # Regenerate map with green marker on selected_flash
            map_html, _, _ = run_explorer(address, selected_nearby_name=selected_flash)
            st.session_state["map_html"] = map_html  # Update map in session state

            col_map, col_info = st.columns([3, 2])

            def wrap_in_html_shell(html_snippet):
                return f"""
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        html, body {{
                            margin: 0;
                            padding: 0;
                            height: 100%;
                            background-color: #121212;
                        }}
                    </style>
                </head>
                <body>{html_snippet}</body>
                </html>
                """

            wrapped_map = wrap_in_html_shell(map_html)

            with col_map:
                st.subheader("📍 Map View (your cafe in red)")
                st.components.v1.html(wrapped_map, height=500)

                if st.button("💾 Save This Demographic Insight"):
                    save_demographics_to_sheet(selected, demo)
                    st.success("Demographic insight saved!")


            with col_info:
                st.subheader("📊 Key Demographics")
                selected_demo = st.selectbox("Select Demographic Metric", list(demo.keys()))
                demo_val = demo[selected_demo]

                demo_card_html = f"""
                <div style='padding: 1em; border-radius: 10px; background-color: #2b2b2b; color: #bda967; text-align: center;'>
                    <h2 style='font-size: 2em; margin: 0;'>{demo_val}</h2>
                    <div style='font-size: 1.1em; color: #B0A698;'>{selected_demo}</div>
                </div>
                """
                html(demo_card_html, height=120)

                st.subheader("☕ Nearby Cafés (marked in green)")
                selected_row = df_cafes[df_cafes["Name"] == selected_flash].iloc[0]
                cafe_card_html = f"""
                <div style='background-color:#2b2b2b; padding: 1em; border-radius: 10px; text-align: center;'>
                    <h3 style='color:#bda967;'>{selected_row["Name"]}</h3>
                    <p style='color:#B0A698;'>📍 {selected_row["Address"]}</p>
                </div>
                """
                html(cafe_card_html, height=120)

            # === New Comparison Block (More Efficient, Split) ===
            if "last_demo" in st.session_state:
                st.subheader("Compare with Similar Demographic Regions")

                city = st.text_input("Enter city to search for similar regions", "Los Angeles")
                k = 3

                if st.button("🔍 Find Similar Regions"):
                    token = get_token()
                    lat, lon = geocode(city, token)

                    if lat is None:
                        st.warning("Could not geocode city.")
                    else:
                        with st.spinner("Fetching regions..."):
                            # Cache batch to avoid refetching if city hasn't changed
                            if "cached_city" not in st.session_state or st.session_state["cached_city"] != city:
                                batch = fetch_demographics_batch(lat, lon, token, n=10)
                                st.session_state["cached_batch"] = batch
                                st.session_state["cached_city"] = city
                            else:
                                batch = st.session_state["cached_batch"]

                            top_k = get_similar_regions(st.session_state["last_demo"], batch, k)
                            st.session_state["top_k_regions"] = top_k

            # === Show Region Selector and Map/Cafés independently ===
            if "top_k_regions" in st.session_state:
                top_k = st.session_state["top_k_regions"]
                region_names = top_k["_region_name"].tolist()

                # Store selected region persistently
                selected_region = st.selectbox("Select region to view", region_names, key="region_select")

                if "last_selected_region" not in st.session_state or st.session_state["last_selected_region"] != selected_region:
                    st.session_state["last_selected_region"] = selected_region
                    selected_row = top_k[top_k["_region_name"] == selected_region].iloc[0]
                    region_lat, region_lon = selected_row["_lat"], selected_row["_lon"]
                    cafes_nearby = fetch_cafes_in_region(region_lat, region_lon)
                    st.session_state["selected_region_coords"] = (region_lat, region_lon)
                    st.session_state["selected_region_cafes"] = cafes_nearby

                # ✅ Always use latest values from session state (even if not newly selected)
                region_lat, region_lon = st.session_state.get("selected_region_coords", (None, None))
                cafes_nearby = st.session_state.get("selected_region_cafes", [])

                if region_lat is not None and region_lon is not None:
                    col_map, col_list = st.columns([2.5, 1.5])

                    with col_map:
                        m2 = folium.Map(location=[region_lat, region_lon], zoom_start=13)
                        folium.Marker(
                            [region_lat, region_lon],
                            tooltip=f"📍 {selected_region}",
                            icon=folium.Icon(color="red")
                        ).add_to(m2)

                        for cafe in cafes_nearby:
                            clat, clon = cafe["Lat"], cafe["Lon"]
                            if clat and clon:
                                folium.Marker(
                                    [clat, clon],
                                    popup=f"{cafe['Name']} ({cafe['Address']})",
                                    icon=folium.Icon(color="green", icon="coffee", prefix="fa")
                                ).add_to(m2)

                        st.components.v1.html(m2._repr_html_(), height=500)

                    with col_list:
                        st.markdown("### Nearby Cafés")
                        df_similar = pd.DataFrame(cafes_nearby)
                        for c in df_similar.columns:
                            df_similar[c] = df_similar[c].fillna("").astype(str)
                        st.dataframe(df_similar, use_container_width=True)


                # --- Exploration Tab ---

        with st.expander("Consumer Buckets", expanded=False):
            st.header("🧠 Consumer Buckets")

            # --- Load data ---
            df = df_global
            selected = st.session_state.get("target_cafe_name", None)
            if not selected:
                st.info("Please select a café in the Summary tab first.")
                st.stop()

            df = df_global
            record = df[df["Business Name"] == selected].iloc[0]

            columns = df.columns.tolist()
            menu_index = columns.index("Menu PDF") + 1
            form_data = record.loc[columns[:menu_index]].to_dict()
            demo_data = record.loc[columns[menu_index:]].dropna().to_dict()

            if "product_mix_df" not in st.session_state or "order_details_df" not in st.session_state:
                st.warning("Please upload Product mix and Order Details in the form tab first.")
                st.stop()

            pos_data = st.session_state["product_mix_df"].to_dict("records")
            order_data = st.session_state["order_details_df"].to_dict("records")

            context = {
                "form": form_data,
                "demographics": demo_data,
                "pos": pos_data,
                "order_details": order_data
            }

            # --- Run analysis if not cached ---
            if "persona_result" not in st.session_state:
                with st.spinner("Running persona clustering and assignment..."):
                    st.session_state.persona_result = buckets.run(context)

            result = st.session_state.persona_result

            cluster_df = result["persona_assignments"]
            cluster_personas = result["desciptions"]
            top_items_all = result["top_items"]
            bucket_sizes = result["footfall_estimates"]

            # === Format: Persona Descriptions ===
            persona_summary = {
                f"Cluster {i}": cluster_personas.get(i, "No description available.")
                for i in range(7)
                if i in cluster_personas
            }

            # === Format: Top Items Per Persona ===
            top_items_summary = {
                f"Cluster {i}": ", ".join([f"{k} ({v})" for k, v in top_items_all.get(i, {}).items()])
                for i in range(7)
                if top_items_all.get(i)
            }

            # === Format: Persona Sizes ===
            persona_sizes = {
                f"Cluster {i}": int(bucket_sizes.get(i, 0))
                for i in range(7)
                if bucket_sizes.get(i, 0) > 0
            }

            # === Format: Top Tags Per Persona ===
            persona_tags = {
                f"Cluster {i}": ", ".join(
                    cluster_df[cluster_df["Matched Order Cluster"] == i]["tag"]
                    .value_counts()
                    .head(3)
                    .index
                ) or "N/A"
                for i in range(7)
                if not cluster_df[cluster_df["Matched Order Cluster"] == i].empty
            }

            # ✅ Save into session state
            st.session_state["persona_gpt_summary"] = {
                "persona_summary": persona_summary,
                "top_items_summary": top_items_summary,
                "persona_sizes": persona_sizes,
                "persona_tags": persona_tags,
            }
            if result is None:
                st.error("Analysis failed. Please check logs.")
                return

            st.success("✅ Persona estimation complete!")
            cluster_df = result["persona_assignments"]
            cluster_personas = result["desciptions"]
            top_items_all = result["top_items"]
            summary_df = result["cluster_summary"]
            bucket_sizes = result["footfall_estimates"]

            # --- Top Row (3 Sections) ---
            st.markdown("### 🔍 High-Level Consumer Breakdown")
            col1, col2, col3 = st.columns([1, 1.2, 1])

            # Section 1: Pie chart of consumer personas
            with col1:
                fig, ax = plt.subplots()
                pd.Series(bucket_sizes).plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
                ax.set_ylabel('')
                ax.set_title("Consumer Persona Distribution")
                st.pyplot(fig)

            # Section 2: DataFrame of persona + tags + items
            with col2:
                def collect_tags(cluster):
                    tags = cluster_df[cluster_df["Matched Order Cluster"] == cluster]["tag"].value_counts().head(3).index.tolist()
                    return ", ".join(tags) if tags else "N/A"

                def collect_items(cluster):
                    items = top_items_all.get(cluster, {})
                    return ", ".join([f"{k} ({v})" for k, v in items.items()]) if items else "N/A"

                display_df = pd.DataFrame([
                    {
                        "Cluster": i,
                        "People": bucket_sizes.get(i, 0),
                        "Top Tags": collect_tags(i),
                        "Top Items": collect_items(i)
                    }
                    for i in range(7)
                ])
                st.dataframe(display_df)

            # Section 3: Persona Description
            with col3:
                selected_cluster = st.selectbox("Select Cluster for Description", list(range(7)), key="desc_cluster")
                desc = cluster_personas.get(selected_cluster, "No description available.")

                # ✅ Extract clean title without "Cluster 0 –"
                first_line = desc.splitlines()[0]
                clean_title = first_line.split("–", 1)[-1].strip()

                st.markdown(f"""### {clean_title}\n\n{desc}""")


            # --- Bottom Row (2 Pie Charts) ---
            st.markdown("### 👥 Deep Dive into Selected Persona")

            selected_cluster_bottom = st.selectbox("Persona to Visualize", list(range(7)), key="bottom_cluster")

            tags = (
                cluster_df[cluster_df["Matched Order Cluster"] == selected_cluster_bottom]["tag"]
                .value_counts()
            )
            items = top_items_all.get(selected_cluster_bottom, {})

            col4, col5 = st.columns(2)

            with col4:
                if not tags.empty:
                    fig1, ax1 = plt.subplots()
                    tags.plot.pie(autopct='%1.1f%%', ax=ax1, startangle=90)
                    ax1.set_ylabel('')
                    ax1.set_title("Tag Distribution in Persona")
                    st.pyplot(fig1)
                else:
                    st.info("No tag data available for this persona.")

            with col5:
                if items:
                    fig2, ax2 = plt.subplots()
                    pd.Series(items).plot.pie(autopct='%1.1f%%', ax=ax2, startangle=90)
                    ax2.set_ylabel('')
                    ax2.set_title("Top Order Items in Persona")
                    st.pyplot(fig2)
                else:
                    st.info("No item data available for this persona.")


    # --- Deep Dive Tab ---
    with tab_deep:
        st.header("Advice AI [...]")

        # 1) Load global submission
        df = df_global
        if df.empty:
            st.info("No data submitted yet.")
            st.stop()

        # 2) Load preselected café from Summary tab
        selected = st.session_state.get("target_cafe_name")
        if not selected:
            st.warning("Please select a café in the Summary tab first.")
            st.stop()

        record = df[df["Business Name"] == selected].iloc[0]

        # 3) Parse form + demo data
        columns = df.columns.tolist()
        form_data = record.loc[columns[:menu_index]].to_dict()
        demo_data = record.loc[columns[menu_index:]].dropna().to_dict()

        if "product_mix_df" not in st.session_state:
            st.warning("Please upload Product mix ")
            st.stop()

        pos_data = st.session_state["product_mix_df"].to_dict("records")

        # 4) Define the 7 final prompts
        AI_PROMPTS = [
            "What patterns in product performance, margin, and customer behavior reveal the biggest opportunities for smarter pricing, bundling, or removal?",
            "Which menu or inventory items represent the highest operational risk in terms of waste, workflow disruption, or service delay?",
            "Which customer segments (by purchase behavior or frequency) are being under-served or pose a risk to long-term retention and revenue growth?",
            "What changes in foot traffic or transaction volume would overwhelm current systems, and where are the choke points in labor, prep, and throughput?",
            "Which timeframes (by hour or day) show misalignment between customer demand and sales efficiency, and how can the business better match supply to demand?",
            "How does this business compare to similar businesses in the local area in terms of product mix, pricing, operational health, and revenue potential?",
            "Based on current trajectory and financial performance, what growth, reinvestment, or optimization strategy will yield the highest ROI in the next 3–6 months?"
        ]

        # 5) Run once
        context = {
            "form": form_data,
            "demographics": demo_data,
            "pos": pos_data
        }

        # ➕ Add persona clustering results if available
        if "persona_gpt_summary" in st.session_state:
            context["persona_summary"] = st.session_state["persona_gpt_summary"]["persona_summary"]
            context["top_items_summary"] = st.session_state["persona_gpt_summary"]["top_items_summary"]
            context["persona_sizes"] = st.session_state["persona_gpt_summary"]["persona_sizes"]
            context["persona_tags"] = st.session_state["persona_gpt_summary"]["persona_tags"]
        else:
            st.warning("Persona GPT summary not found in session. Deep dive may be limited.")


        if "deep_dive_responses" not in st.session_state:
            with st.spinner("Generating Advice [...]"):
                st.session_state.deep_dive_responses = run_deep_dive(context, AI_PROMPTS)["responses"]
                st.session_state.deep_followups = ["" for _ in AI_PROMPTS]
                st.session_state.deep_followup_answers = ["" for _ in AI_PROMPTS]

        # 6) Display each slide
        num_prompts = len(AI_PROMPTS)
        idx = st.slider("Slide", 1, num_prompts, 1, key="ai_slide_index") - 1

        prompt = st.session_state.deep_dive_responses[idx]["query"]
        answer = st.session_state.deep_dive_responses[idx]["answer"]

        st.subheader(f"Q{idx + 1}: {prompt}")
        st.write(answer)

        st.markdown("#### Ask a follow-up:")
        followup = st.text_area("Follow-up question", st.session_state.deep_followups[idx], key=f"followup_{idx}")

        if st.button("Ask", key=f"ask_btn_{idx}"):
            with st.spinner("Thinking..."):
                followup_resp = run_deep_dive(context, [followup])["responses"][0]["answer"]
                st.session_state.deep_followups[idx] = followup
                st.session_state.deep_followup_answers[idx] = followup_resp

        if st.session_state.deep_followup_answers[idx]:
            st.markdown("#### Response:")
            st.write(st.session_state.deep_followup_answers[idx])


if __name__ == "__main__":
    main()
