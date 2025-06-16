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
from streamlit.components.v1 import html
import pandas as pd
from gspread_dataframe import set_with_dataframe


from dotenv import load_dotenv
load_dotenv()


from models.foot_traffic_model import run as run_foot_traffic
from models.margin_model import run as run_margin
from models.arcgis_explorer import run_explorer
from models.deep_dive_model import run_deep_dive

# --- GOOGLE SHEETS SETUP ---
SHEET_ID    = "12Qvpi5jOdtWRaa1aL6yglCAJ5tFphW1fHsF8apTlEV4"
WS_NAME     = "Data"
AUTH_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# 1) Decode the base64 JSON
b64 = os.environ["GCLOUD_SA_KEY_B64"]
sa_json = base64.b64decode(b64).decode("utf-8")
info = json.loads(sa_json)

# 2) Build your creds & client
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(info, scopes= AUTH_SCOPES)
gc    = gspread.authorize(creds)
ws    = gc.open_by_key(SHEET_ID).worksheet(WS_NAME)


# --- ENSURE HEADER ROW EXISTS ---
HEADERS = [
    # — your original form fields —
    "Business Name","Location Address","Location Type","Years in Operation","Days/Hours of Operation",
    "Seating Indoor","Seating Outdoor","Full Kitchen","Average Weekly Sales","Average Transaction Value",
    "Top Item 1","Top Item 2","Top Item 3","% Sales Drinks","% Sales Food","Peak Hours",
    "Sales Monday","Sales Tuesday","Sales Wednesday","Sales Thursday","Sales Friday","Sales Saturday","Sales Sunday",
    "Avg Transactions/Day","Mobile/Online Ordering","Avg Wait Time","Target Age Range","Main Customer Segments",
    "% Regulars","Collect Contact Info","Run Surveys","# Competitors within 1mi","Competitor 1","Competitor 2","Competitor 3",
    "Near POI","Marketing Strategy","Partner Local","# Employees","Full-Time Staff","Part-Time Staff",
    "Track Labor % of Revenue","Staffing Challenges","Training Procedures","Goal 1","Goal 2","Expansion/Relocation",
    "Optimize Pricing/Product Mix","Want Location Report","POS Export Filename","Labor Report Filename","Benchmark Similar Businesses",
    "Timestamp",
    # — appended demographic keys (raw field codes) —
    "Total Population (2024)", "Total Households (2024)", "Avg Household Size (2024)", "Median Household Income (2024)",
    "Per-Capita Income (2024)", "Diversity Index (2024)", "Pop. Growth Rate ’24–’29 (%)", "Median Home Value (2024)",
    "Avg Home Value (2024)", "Daytime Worker Pop. (2024)", "Median Age (2024)", "Owner-occupied Households (2024)",
    "Renter-occupied Households (2024)", "Vacant Housing Units (2024)", "Daytime Population (2024)", "Daytime Resident Pop. (2024)",
    "Population Age 18 (2024)", "Population Age 25 (2024)", "Population Age 35 (2024)", "Population Age 65+ (2024)",
    "Age Dependency Ratio (2024)", "White Population (2024)", "Black Population (2024)", "Asian Population (2024)",
    "Unemployment Rate (2024)", "Household Growth Rate ’24–’29 (%)", "Eating & Drinking Businesses (2024)", "Group Quarters Population (2024)"

]

existing = [h for h in ws.row_values(1) if h.strip() != ""]

# If it doesn't exactly match, wipe & rewrite
if existing != HEADERS:
    ws.delete_rows(1)
    ws.insert_row(HEADERS, 1)

def append_row_to_sheet(row: list):
    ws.append_row(row, value_input_option="USER_ENTERED")

def save_demographics_to_sheet(address, demo):
    # 1) Find the correct row
    df = load_sheet_df()
    try:
        row_idx = df.index[df["Location Address"] == address][0] + 2
    except IndexError:
        st.error("Address not found!")
        return

    # 2) Get and clean current headers
    headers = [h for h in ws.row_values(1) if h.strip() != ""]

    # 3) Identify brand-new demo columns
    new_cols = [k for k in demo if k not in headers]
    if new_cols:
        headers += new_cols
        ws.update("A1:1", [headers])

    # 4) Write each cell individually (fast for ~10 cols)
    with st.spinner("Saving demographics…"):
        prog = st.progress(0)
        total = len(demo)
        for i, (k, v) in enumerate(demo.items(), start=1):
            col_idx = headers.index(k) + 1
            ws.update_cell(row_idx, col_idx, str(v))
            prog.progress(i / total)

    st.success("Demographics saved to sheet!")


def load_sheet_df():
    values = ws.get_all_values()  # raw matrix of strings
    if not values:
        return pd.DataFrame()

    # Clean headers (drop any blanks)
    raw_headers = values[0]
    idxs = [i for i, h in enumerate(raw_headers) if h.strip() != ""]
    headers = [raw_headers[i] for i in idxs]

    # Build rows
    data = []
    for row in values[1:]:
        row = row + [""] * (len(raw_headers) - len(row))
        data.append([row[i] for i in idxs])

    return pd.DataFrame(data, columns=headers)
# --- STREAMLIT APP ---
def main():
    st.title("Local Café Consulting Dashboard")
    tab_form, tab_results, tab_explore, tab_deep = st.tabs(
        ["Form","Results","Exploration","Deep Dive"]
    )

    # --- Form Tab ---
    with tab_form:
        st.header("CAFÉ PROFILE")
        business_name = st.text_input("Business Name")
        location_address = st.text_input("Location Address")
        location_type = st.radio("Standalone or inside another business?", ["Standalone","Inside another business"])
        years_in_operation = st.number_input("Years in operation", 0, step=1)
        days_hours = st.text_area("Days/hours of operation", help="e.g. Mon–Fri 7:00–19:00; Sat 8:00–16:00")
        seating_indoor = st.number_input("Seating capacity (indoor)", 0, step=1)
        seating_outdoor = st.number_input("Seating capacity (outdoor)", 0, step=1)
        full_kitchen = st.radio("Do you have a full kitchen?", ["Yes","No"])

        st.header("SALES & POS DATA")
        avg_weekly_sales = st.number_input("Average weekly sales (last 4 weeks)", 0.0, format="%.2f")
        avg_transaction_value = st.number_input("Average transaction value", 0.0, format="%.2f")
        st.subheader("Top 3 best-selling items")
        top1 = st.text_input("1st best-selling item")
        top2 = st.text_input("2nd best-selling item")
        top3 = st.text_input("3rd best-selling item")
        pct_drinks = st.slider("% of sales from drinks", 0,100,50)
        pct_food = 100 - pct_drinks
        peak_hours = st.multiselect("Peak hours (POS)", [f"{h}:00" for h in range(24)])
        st.subheader("Sales by day of week")
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        sales_by_day = {d: st.number_input(d, 0.0, format="%.2f") for d in days}
        avg_transactions = st.number_input("# transactions/day (avg)", 0, step=1)
        online_sales = st.radio("Mobile/online ordering?", ["Yes","No"])
        avg_wait_time = st.number_input("Avg wait time (min)", 0.0, format="%.1f")

        st.header("CUSTOMER DEMOGRAPHICS & INSIGHT")
        age_range = st.text_input("Target customer age range")
        segments = st.multiselect("Main customer segments", ["Students","Professionals","Tourists","Others"])
        pct_regulars = st.slider("% regular customers (3+ visits/mo)", 0,100,20)
        collect_emails = st.radio("Collect customer contacts?", ["Yes","No"])
        run_surveys = st.radio("Run surveys/feedback?", ["Yes","No"])

        st.header("COMPETITION & LOCATION")
        num_competitors = st.number_input("# competitors within 1 mile", 0, step=1)
        comp1 = st.text_input("Competitor 1 (name+loc)")
        comp2 = st.text_input("Competitor 2 (name+loc)")
        comp3 = st.text_input("Competitor 3 (name+loc)")
        near_poi = st.radio("Near major POI?", ["Yes","No"])
        marketing = st.selectbox("Marketing strategy", ["Social media","Events","None","Other"])
        partner_local = st.radio("Partner with local businesses/events?", ["Yes","No"])

        st.header("OPERATIONS & STAFFING")
        num_employees = st.number_input("Number of employees", 0, step=1)
        full_time = st.number_input("# full-time staff", 0, step=1)
        part_time = st.number_input("# part-time staff", 0, step=1)
        track_labor_pct = st.radio("Track labor % of revenue?", ["Yes","No"])
        staffing_challenges = st.text_area("Staffing challenges")
        training_procedures = st.radio("Standard training procedures?", ["Yes","No"])

        st.header("OWNER GOALS")
        goal1 = st.radio("Top goal #1", ["Increase foot traffic","Improve margins","Expand menu","Other"])
        if goal1=="Other": goal1 = st.text_input("Specify goal #1")
        goal2 = st.radio("Top goal #2", ["Increase foot traffic","Improve margins","Expand menu","Other"])
        if goal2=="Other": goal2 = st.text_input("Specify goal #2")
        expand_relocate = st.radio("Considering expansion/relocation?", ["Yes","No"])
        optimize_mix     = st.radio("Help optimizing pricing/product mix?", ["Yes","No"])
        location_report  = st.radio("Want location-based report?", ["Yes","No"])

        st.header("OPTIONAL TECHNICAL INPUTS")
        pos_export   = st.file_uploader("POS export (CSV)", type=["csv"])

        if pos_export is not None:
           st.markdown("✅ POS file loaded.")
           if st.button("Upload POS to sheet"):
               # Read the CSV
               df_pos = pd.read_csv(pos_export)
               # Try to open or else create the "POS Raw" sheet
               try:
                   ws_pos = gc.open_by_key(SHEET_ID).worksheet("POS Raw")
               except gspread.WorksheetNotFound:
                   ws_pos = gc.open_by_key(SHEET_ID).add_worksheet(
                       title="POS Raw",
                       rows=str(len(df_pos) + 10),
                       cols=str(len(df_pos.columns) + 5),
                   )
            # Clear old data and write new
               ws_pos.clear()
               set_with_dataframe(ws_pos, df_pos)
               st.success("POS CSV uploaded to tab ‘POS Raw’")
        labor_report = st.file_uploader("Employee schedule/report", type=["csv","xlsx","pdf"])
        benchmark    = st.radio("Benchmark against similar businesses?", ["Yes","No"])

        if st.button("Submit"):
            row = [
                business_name, location_address, location_type, years_in_operation, days_hours,
                seating_indoor, seating_outdoor, full_kitchen,
                avg_weekly_sales, avg_transaction_value,
                top1, top2, top3, pct_drinks, pct_food, ",".join(peak_hours),
                sales_by_day["Monday"], sales_by_day["Tuesday"], sales_by_day["Wednesday"],
                sales_by_day["Thursday"], sales_by_day["Friday"], sales_by_day["Saturday"],
                sales_by_day["Sunday"], avg_transactions, online_sales, avg_wait_time,
                age_range, ",".join(segments), pct_regulars, collect_emails, run_surveys,
                num_competitors, comp1, comp2, comp3, near_poi, marketing, partner_local,
                num_employees, full_time, part_time, track_labor_pct, staffing_challenges, training_procedures,
                goal1, goal2, expand_relocate, optimize_mix, location_report,
                pos_export.name if pos_export else "", labor_report.name if labor_report else "", benchmark,
                datetime.utcnow().isoformat()
            ]
            append_row_to_sheet(row)
            st.success("Form submitted!")

    # --- Results Tab ---
    with tab_results:
        st.header("Foot-Traffic Model Results")
        res = run_foot_traffic()
        if res["mae"] is not None:
            st.metric("MAE", f"{res['mae']:.2f}")
            st.pyplot(res["figure"])
        else:
            st.info("Not enough data yet.")

        st.header("Margin Optimization Results")
        mres = run_margin()
        if mres["metrics"]:
            st.metric("MAE", f"{mres['metrics']['MAE']:.2f}")
            st.metric("R²", f"{mres['metrics']['R2']:.2f}")
            st.pyplot(mres["figure"])
        else:
            st.info("Not enough data yet.")

    # --- Exploration Tab ---
    with tab_explore:
        st.header("Exploration")

        # Input address
        address = st.text_input("Enter café address")

        # When Explore is clicked, fetch and display map, demo, cafés
        if st.button("Explore"):
            map_html, demo, cafes = run_explorer(address)
            if map_html:
                # Store in session for later saving
                st.session_state["last_address"] = address
                st.session_state["last_demo"] = demo

                st.subheader("Map View")
                html(map_html, height=450)

                st.subheader("Key Demographics")
                df_demo = pd.DataFrame(list(demo.items()), columns=["Description", "Value"])
                df_demo["Value"] = df_demo["Value"].astype(str)
                st.table(df_demo)

                st.subheader("Nearby Cafés")
                df_cafes = pd.DataFrame(cafes)
                for c in df_cafes.columns:
                    df_cafes[c] = df_cafes[c].fillna("").astype(str)
                st.table(df_cafes)
            else:
                st.error("Map generation failed.")

        # After exploring, allow saving demographics
        if "last_demo" in st.session_state:
            if st.button("Save demographics"):
                with st.spinner("Saving demographics to sheet…"):
                    save_demographics_to_sheet(
                        st.session_state["last_address"],
                        st.session_state["last_demo"]
                    )

    # --- DEEP DIVE TAB ---
    with tab_deep:
       st.header("Deep Dive Insights")

       df = load_sheet_df()
       if df.empty:
           st.info("No data submitted yet.")
           return

    # Café selector
       cafelist = df["Business Name"].unique().tolist()
       selected = st.selectbox("Select café", cafelist)

    # Grab the selected record
       record = df[df["Business Name"] == selected].iloc[0]

    # Split form vs. demographics
       cols = ws.row_values(1)
       idx_ts = cols.index("Timestamp") + 1
       form_data = {c: record[c] for c in cols[:idx_ts]}
       demo_data = {c: record[c] for c in cols[idx_ts:] if pd.notna(record[c])}

       st.subheader("Form Data")
       st.dataframe(form_data)
       st.subheader("Demographics Data")
       st.dataframe(demo_data)

    # Load POS if present
       pos_data = []
       try:
           ws_pos = gc.open_by_key(SHEET_ID).worksheet("POS Raw")
           df_pos = pd.DataFrame(ws_pos.get_all_records())
           st.subheader("Raw POS Data")
           st.dataframe(df_pos)
           pos_data = df_pos.to_dict(orient="records")
       except gspread.WorksheetNotFound:
           st.info("No POS Raw sheet found. Upload POS first.")

    # Let user pick which deep-dive goals
       st.markdown("### Pick which Deep-Dive areas to run:")
       do_marketing = st.checkbox("🟦 Marketing", value=True)
       do_finance   = st.checkbox("🟥 Finance",   value=True)
       do_compete   = st.checkbox("🟩 Competitor Insight", value=True)

    # Define your question sets
       marketing_qs = [
           "Based on local ArcGIS demographics (age, income, ethnicity), do your current top-selling items align with dominant consumer preferences?",
           "What percentage of your highest AOV customers belong to the reported target group (e.g., 18–24, students)?",
           "Are there underserved demographic groups nearby (e.g., young professionals, working parents, older adults) who might respond to different offerings?",
       ]
       finance_qs = [
           "Which top 10 selling items yield the highest vs. lowest net profit margins, after accounting for modifiers, discounts, and waste?",
           "Are there high-volume, low-margin items (e.g., bagels, basic drinks) that could be bundled or upsold to increase AOV?",
           "What % of weekly revenue comes from your top 3 items (e.g., Bagel Combo, Latte, Matcha)?",
           "How do average order values differ between weekdays and weekends, and are certain hours consistently underperforming?",
           "Based on item-level waste and refund rates, which products contribute most to unnecessary cost?",
           "How many orders are fulfilled per employee during each shift, and where are the biggest bottlenecks?",
       ]
       competitor_qs = [
           "How does your demographic profile (students 18–24) compare to those of your top 3 competitors within 1 mile?",
           "Which competitors are leveraging seasonal specials, mobile ordering, or loyalty programs to drive repeat traffic — and what could you adapt?",
           "Which product types (e.g., specialty drinks, protein-rich meals) are you not offering that competitors succeed with?",
           "Is your revenue/sqft higher or lower than cafés with similar sales volume, seating, and staff count?",
           "Which of your current differentiators (e.g., indoor space, menu creativity, healthiness) are under-leveraged based on online comparisons?"
       ]

    # Build the final queries list
       queries: list[str] = []
       if do_marketing:
           queries += marketing_qs
       if do_finance:
           queries += finance_qs
       if do_compete:
           queries += competitor_qs

       if not queries:
           st.warning("Select at least one area to deep-dive on.")
       else:
        # Build context
           context = {
               "form": form_data,
               "demographics": demo_data,
               "pos": pos_data
           }

        # Run the Deep Dive
           deep = run_deep_dive(context, queries)
           for res in deep["responses"]:
               st.subheader(res["query"])
               st.write(res["answer"])

    # Freeform chat
       st.markdown("---")
       st.subheader("Chat with the Café AI")
       user_q = st.text_area("Your question")
       if st.button("Send"):
           follow = run_deep_dive(context, [user_q])
           st.write(follow["responses"][0]["answer"])

if __name__ == "__main__":
    main()
