import streamlit as st
import re
import json
import base64
import os
import gspread
import textwrap
from google.oauth2.service_account import Credentials
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# ...the rest of your code...


def load_data():
    # 1) Grab the entire sheet as a list of lists
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # 2) Clean headers: drop any blank entries
    raw_headers = values[0]
    idxs = [i for i, h in enumerate(raw_headers) if h.strip() != ""]
    headers = [raw_headers[i] for i in idxs]

    # 3) Build data rows, selecting only non-blank columns
    data_rows = []
    for row in values[1:]:
        # pad row to full width
        row += [""] * (len(raw_headers) - len(row))
        data_rows.append([row[i] for i in idxs])

    # 4) Create DataFrame
    df = pd.DataFrame(data_rows, columns=headers)
    return df

# models/margin_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def compute_margin(df: pd.DataFrame) -> pd.Series:
    # Convert the relevant columns from strings to numeric
    drinks = pd.to_numeric(df["% Sales Drinks"], errors="coerce")
    food   = pd.to_numeric(df["% Sales Food"],   errors="coerce")
    atv    = pd.to_numeric(df["Average Transaction Value"], errors="coerce")

    # Fill any NaNs with zero so math doesn’t break
    drinks = drinks.fillna(0)
    food   = food.fillna(0)
    atv    = atv.fillna(0)

    # Now compute estimated margin
    return (drinks/100 * 0.70 + food/100 * 0.50) * atv

def train_margin_model(df: pd.DataFrame):
    # 1) Compute target
    df = df.copy()
    df["Estimated_Margin"] = compute_margin(df)

    # 2) Select features and drop any rows with missing target
    feature_cols = [
        "% Sales Drinks","% Sales Food","Average Transaction Value",
        "Track Labor % of Revenue","Full Kitchen"
    ]
    # Convert all feature columns to numeric/boolean as needed
    df["Track Labor % of Revenue"] = (df["Track Labor % of Revenue"] == "Yes").astype(int)
    df["Full Kitchen"]             = (df["Full Kitchen"] == "Yes").astype(int)
    df["% Sales Drinks"]           = pd.to_numeric(df["% Sales Drinks"], errors="coerce").fillna(0)
    df["% Sales Food"]             = pd.to_numeric(df["% Sales Food"],   errors="coerce").fillna(0)
    df["Average Transaction Value"]= pd.to_numeric(df["Average Transaction Value"], errors="coerce").fillna(0)

    df = df.dropna(subset=["Estimated_Margin"])

    X = df[feature_cols]
    y = df["Estimated_Margin"]

    # 3) Train/test split
    if len(df) < 5:
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4) Fit model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 5) Predict & evaluate
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    # 6) Build feature-importance figure (same as before)…
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(feature_cols, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Margin Model Feature Importances")

    # 7) Build a test-set DataFrame for inspection
    test_df = X_test.copy()
    test_df["Actual"]    = y_test
    test_df["Predicted"] = preds

    return model, {"MAE": mae, "R2": r2}, feature_cols, test_df



def plot_importance(model, features):
    imp = model.feature_importances_
    idx = imp.argsort()[::-1]
    names = [features[i] for i in idx]
    fig, ax = plt.subplots()
    ax.barh(names, imp[idx])
    ax.set_title("Margin Model Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def run():
    df = load_data()
    if df.empty:
        st.warning("No data available for margin model.")
        return {"metrics": {}, "figure": None, "test_df": None}

    model, metrics, features, test_df = train_margin_model(df)
    if model is None:
        return {"metrics": metrics, "figure": None, "test_df": test_df}

    fig = plot_importance(model, features)
    return {"metrics": metrics, "figure": fig, "test_df": test_df}
