# models/arcgis_explorer.py

import streamlit as st
import os
import requests
import folium
import json
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GEOCODE_URL      = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
DEMOGRAPHICS_URL = "https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/Geoenrichment/enrich"
TOKEN_URL        = "https://www.arcgis.com/sharing/rest/generateToken"
OVERPASS_URL     = "https://overpass-api.de/api/interpreter"

USERNAME = os.getenv("ARCGIS_USERNAME")
PASSWORD = os.getenv("ARCGIS_PASSWORD")
if not USERNAME or not PASSWORD:
    st.error("ArcGIS credentials missing.")
    st.stop()

KEY_LABELS = {
    "TOTPOP_CY":    "Total Population (2024)",
    "TOTHH_CY":     "Total Households (2024)",
    "AVGHHSZ_CY":   "Avg Household Size (2024)",
    "MEDHINC_CY":   "Median Household Income (2024)",
    "PCI_CY":       "Per-Capita Income (2024)",
    "DIVINDX_CY":   "Diversity Index (2024)",
    "POPGRWCYFY":   "Pop. Growth Rate ’24–’29 (%)",
    "MEDVAL_CY":    "Median Home Value (2024)",
    "AVGVAL_CY":    "Avg Home Value (2024)",
    "DPOPWRK_CY":   "Daytime Worker Pop. (2024)",
    "MEDAGE_CY":    "Median Age (2024)",
    "OWNER_CY":     "Owner-occupied Households (2024)",
    "RENTER_CY":    "Renter-occupied Households (2024)",
    "VACANT_CY":    "Vacant Housing Units (2024)",
    "DPOP_CY":      "Daytime Population (2024)",
    "DPOPRES_CY":   "Daytime Resident Pop. (2024)",
    "AGE18_CY":     "Population Age 18 (2024)",
    "AGE25_CY":     "Population Age 25 (2024)",
    "AGE35_CY":     "Population Age 35 (2024)",
    "AGE65_CY":     "Population Age 65+ (2024)",
    "AGEDEP_CY":    "Age Dependency Ratio (2024)",
    "WHITE_CY":     "White Population (2024)",
    "BLACK_CY":     "Black Population (2024)",
    "ASIAN_CY":     "Asian Population (2024)",
    "UNEMPRT_CY":   "Unemployment Rate (2024)",
    "HHGRWCYFY":    "Household Growth Rate ’24–’29 (%)",
    "S16_BUS":      "Eating & Drinking Businesses (2024)",
    "GQPOP_CY":     "Group Quarters Population (2024)"
}

def get_token():
    r = requests.post(TOKEN_URL, data={
        "f": "json", "username": USERNAME, "password": PASSWORD,
        "referer": "https://arcgis.com", "expiration": 60
    })
    return r.json().get("token")

def geocode(address, token):
    r = requests.get(GEOCODE_URL, params={
        "SingleLine": address, "f": "json", "maxLocations": 1, "token": token
    })
    c = r.json().get("candidates", [])
    if not c:
        return None, None
    loc = c[0]["location"]
    return loc["y"], loc["x"]

def fetch_demographics(lat, lon, token, prog=None):
    if prog: prog.progress(0.2)
    study_areas = [{
        "geometry": {"x": lon, "y": lat, "spatialReference": {"wkid": 4326}},
        "areaType": "esriEnrichmentStudyArea",
        "buffer": {"distance": 1, "units": "esriMiles"}
    }]
    analysis_vars = ",".join(KEY_LABELS.keys())
    r = requests.get(DEMOGRAPHICS_URL, params={
        "f": "json",
        "studyAreas": json.dumps(study_areas),
        "dataCollections": json.dumps(["KeyUSFacts"]),
        "analysisVariables": analysis_vars,
        "token": token
    })
    j = r.json()
    if prog: prog.progress(0.4)
    merged = {}
    for res in j.get("results", []):
        fs = res.get("value", {}).get("FeatureSet", [{}])[0]
        feats = fs.get("features", [])
        if feats:
            merged.update(feats[0].get("attributes", {}))
    return {KEY_LABELS[k]: merged.get(k, "N/A") for k in KEY_LABELS}

def fetch_nearby_cafes(lat, lon, radius_km=1):
    radius = int(radius_km * 1000)
    query = f'''
        [out:json][timeout:25];
        node["amenity"="cafe"](around:{radius},{lat},{lon});
        out center tags;
    '''
    r = requests.post(OVERPASS_URL, data={"data": query})
    elems = r.json().get("elements", [])
    cafes = []
    for el in elems:
        tags = el.get("tags", {})
        cafes.append({
            "Name": tags.get("name", "–"),
            "Lat": el.get("lat") or el.get("center", {}).get("lat"),
            "Lon": el.get("lon") or el.get("center", {}).get("lon"),
            "Address": tags.get("addr:full") or tags.get("addr:street", "–")
        })
    return cafes

def create_map(lat, lon, cafes, prog=None):
    if prog: prog.progress(0.8)
    m = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], tooltip="Target Café", icon=folium.Icon(color="red")).add_to(m)
    for cafe in cafes:
        clat, clon = cafe["Lat"], cafe["Lon"]
        if clat and clon:
            folium.Marker(
                [clat, clon],
                popup=f"{cafe['Name']} ({cafe['Address']})",
                icon=folium.Icon(color="blue", icon="coffee", prefix="fa")
            ).add_to(m)
    if prog: prog.progress(1.0)
    return m

def run_explorer(address):
    prog = st.progress(0.0)
    token = get_token()
    if not token:
        return None, {}, []
    prog.progress(0.1)
    lat, lon = geocode(address, token)
    if lat is None:
        return None, {}, []
    prog.progress(0.15)
    demo = fetch_demographics(lat, lon, token, prog)
    cafes = fetch_nearby_cafes(lat, lon)
    map_obj = create_map(lat, lon, cafes, prog)
    return map_obj._repr_html_(), demo, cafes

def reverse_geocode(lat, lon, token):
    r = requests.get(GEOCODE_URL, params={
        "location": f"{lon},{lat}",
        "f": "json",
        "token": token
    })
    j = r.json()
    cands = j.get("candidates", [])
    if not cands:
        return "Unnamed Region"

    address = cands[0].get("address")
    if not address or "Unnamed" in address:
        return "Unnamed Region"

    return address


# NEW FUNCTION: Get multiple demo regions near a location
def fetch_demographics_batch(lat, lon, token, n=5):
    offsets = [(i / 1000.0, i / 1000.0) for i in range(1, n + 1)]
    regions = []
    for dy, dx in offsets:
        new_lat = lat + dy
        new_lon = lon + dx
        demo = fetch_demographics(new_lat, new_lon, token)
        region_name = reverse_geocode(new_lat, new_lon, token)
        regions.append({
            **demo,
            "_lat": new_lat,
            "_lon": new_lon,
            "_region_name": region_name
        })
    return regions


# NEW FUNCTION: Compare demo similarity and return top k
def get_similar_regions(main_demo, all_demos, k):
    feature_keys = list(main_demo.keys())
    df = pd.DataFrame(all_demos)

    # Save region metadata
    region_names = df[["_region_name", "_lat", "_lon"]]

    # Compute similarity
    df_features = df[feature_keys].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_features)
    main_vec = scaler.transform([pd.Series(main_demo).apply(pd.to_numeric, errors="coerce").fillna(0).values])
    similarities = cosine_similarity(main_vec, df_scaled)[0]

    df_result = region_names.copy()
    df_result["similarity"] = similarities
    df_result = pd.concat([df_result, df_features], axis=1)

    return df_result.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)



# NEW FUNCTION: Fetch cafés around one of the similar regions
def fetch_cafes_in_region(lat, lon):
    return fetch_nearby_cafes(lat, lon)

# NEW FUNCTION: Return structured comparison of café demographics
def compare_cafes(cafe1, cafe2):
    return pd.DataFrame({
        "Feature": list(cafe1.keys()),
        "Target Café": list(cafe1.values()),
        "Competitor Café": list(cafe2.values())
    })
