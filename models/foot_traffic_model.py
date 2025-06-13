import streamlit as st
import re
import json
import textwrap
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# --- GOOGLE SHEETS SETUP ---
SHEET_ID    = "12Qvpi5jOdtWRaa1aL6yglCAJ5tFphW1fHsF8apTlEV4"
WS_NAME     = "Data"
AUTH_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

GCP_SA_JSON = os.getenv("GCP_SA_JSON")
service_account_info = json.loads(GCP_SA_JSON)
creds = Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"],
)

gc    = gspread.authorize(creds)
ws    = gc.open_by_key(SHEET_ID).worksheet(WS_NAME)


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



def train_foot_traffic_model(df: pd.DataFrame):
    """
    Trains a RandomForestRegressor to predict average transactions per day.
    Returns the trained model, MAE on held-out test set, and feature names.
    If not enough samples, returns (None, None, features) and skips training.
    """
    # Encode boolean columns
    boolean_cols = ["Full Kitchen", "Mobile/Online Ordering", "Collect Contact Info"]
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    features = ["Years in Operation", "Seating Indoor", "Seating Outdoor"] + boolean_cols + ["% Regulars"]
    X = df[features].fillna(0)
    y = df["Avg Transactions/Day"].fillna(0)

    # If only one sample, cannot split; skip training
    if len(df) < 2:
        st.info("Not enough data to train model. Need at least 2 entries.")
        return None, None, features

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return model, mae, features


def plot_feature_importance(model, features):
    """
    Creates a horizontal bar chart of feature importances.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    sorted_feats = [features[i] for i in indices]

    fig, ax = plt.subplots()
    ax.barh(sorted_feats, importances[indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances for Foot Traffic Model")
    fig.tight_layout()
    return fig


def run():
    """
    Orchestrates loading data, training the model, and plotting results.
    Returns a dict with MAE and the matplotlib figure.
    Skips model if not enough data.
    """
    df = load_data()
    if df.empty:
        st.warning("No data found in sheet.")
        return {"mae": None, "figure": None}

    model, mae, features = train_foot_traffic_model(df)
    if model is None:
        return {"mae": None, "figure": None}

    fig = plot_feature_importance(model, features)
    return {"mae": mae, "figure": fig}
