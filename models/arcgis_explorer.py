# models/arcgis_explorer.py

import streamlit as st
import requests
import folium
import json

GEOCODE_URL      = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
DEMOGRAPHICS_URL = "https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/Geoenrichment/enrich"
TOKEN_URL        = "https://www.arcgis.com/sharing/rest/generateToken"
OVERPASS_URL     = "https://overpass-api.de/api/interpreter"

USERNAME = st.secrets["arcgis_username"]
PASSWORD = st.secrets["arcgis_password"]
if not USERNAME or not PASSWORD:
    st.error("ArcGIS credentials missing.")
    st.stop()

KEY_LABELS = {
    # ── Core Demographics ───────────────────────────────────────────────
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

    # ── Added Household Tenure & Vacancies ─────────────────────────────
    "OWNER_CY":     "Owner-occupied Households (2024)",
    "RENTER_CY":    "Renter-occupied Households (2024)",
    "VACANT_CY":    "Vacant Housing Units (2024)",


    # ── Added Daytime Totals ────────────────────────────────────────────
    "DPOP_CY":      "Daytime Population (2024)",
    "DPOPRES_CY":   "Daytime Resident Pop. (2024)",

    # ── Added Age Bands & Dependency ───────────────────────────────────
    "AGE18_CY":     "Population Age 18 (2024)",
    "AGE25_CY":     "Population Age 25 (2024)",
    "AGE35_CY":     "Population Age 35 (2024)",
    "AGE65_CY":     "Population Age 65+ (2024)",
    "AGEDEP_CY":    "Age Dependency Ratio (2024)",

    # ── Race/Ethnicity Counts ────────────────────────────────────────────
    "WHITE_CY":    "White Population (2024)",
    "BLACK_CY":    "Black Population (2024)",
    "ASIAN_CY":    "Asian Population (2024)",

    # ── Diversity ────────────────────────────────────────────────────────
    "DIVINDX_CY":   "Diversity Index (2024)",

    # ── Added Labor & Growth Rates ─────────────────────────────────────
    "UNEMPRT_CY":   "Unemployment Rate (2024)",
    "HHGRWCYFY":    "Household Growth Rate ’24–’29 (%)",

    # ── Added Local Business Density ───────────────────────────────────
    "S16_BUS":      "Eating & Drinking Businesses (2024)",

    # ── Added Group‐Quarters Pop. ──────────────────────────────────────
    "GQPOP_CY":     "Group Quarters Population (2024)"
}



def get_token():
    r = requests.post(TOKEN_URL, data={
        "f":"json","username":USERNAME,"password":PASSWORD,
        "referer":"https://arcgis.com","expiration":60
    })
    return r.json().get("token")

def geocode(addr, token):
    r = requests.get(GEOCODE_URL, params={
        "SingleLine":addr,"f":"json","maxLocations":1,"token":token
    })
    c = r.json().get("candidates",[])
    if not c: return None,None
    loc = c[0]["location"]
    return loc["y"], loc["x"]

def fetch_demographics(lat, lon, token, prog):
    prog.progress(0.2)
    study_areas = [{
        "geometry":{"x":lon,"y":lat,"spatialReference":{"wkid":4326}},
        "areaType":"esriEnrichmentStudyArea",
        "buffer":{"distance":1,"units":"esriMilesUnits"}
    }]

    analysis_vars = ",".join(KEY_LABELS.keys())
    params = {
        "f":"json",
        "studyAreas": json.dumps(study_areas),
        "dataCollections": json.dumps([
            "KeyGlobalFacts","KeyUSFacts","DetailedDemographics","Spending","1yearincrements","5yearincrements","AgeDependency","agebyracebysex","businesses","basicFactsForMobileApps","RaceAndHispanicOrigin"
        ]),
        "analysisVaraibles": analysis_vars,
        "token": token
    }
    r = requests.get(DEMOGRAPHICS_URL, params=params)
    r.raise_for_status()
    j = r.json()

    results = j.get("results",[]) if isinstance(j,dict) else []
    merged = {}
    for res in results:
        fs = res.get("value",{}).get("FeatureSet") or {}
        if isinstance(fs,list): fs = fs[0]
        feats = fs.get("features",[]) or []
        if feats:
            merged.update(feats[0].get("attributes",{}))
    prog.progress(0.4)

    out = {label: merged.get(raw,"N/A") for raw,label in KEY_LABELS.items()}
    return out

def fetch_nearby_cafes(lat, lon, prog):
    prog.progress(0.6)
    radius = int(1 * 1609.34)
    query = f'''
      [out:json][timeout:25];
      node["amenity"="cafe"](around:{radius},{lat},{lon});
      out center tags;
    '''
    r = requests.post(OVERPASS_URL, data={"data":query})
    elems = r.json().get("elements",[])
    cafes=[]
    for el in elems:
        t = el.get("tags",{})
        cafes.append({
            "Name":    t.get("name","–"),
            "Lat":     el.get("lat") or el.get("center",{}).get("lat"),
            "Lon":     el.get("lon") or el.get("center",{}).get("lon"),
            "Address": t.get("addr:full") or t.get("addr:street","–")
        })
    return cafes

def create_map(lat, lon, cafes, prog):
    prog.progress(0.8)
    m = folium.Map(location=[lat,lon], zoom_start=13)
    folium.Marker([lat,lon], tooltip="This Café", icon=folium.Icon(color="red")).add_to(m)
    for cafe in cafes:
        clat,clon = cafe["Lat"], cafe["Lon"]
        if clat and clon:
            folium.Marker([clat,clon],
                popup=f"{cafe['Name']} ({cafe['Address']})",
                icon=folium.Icon(color="blue", icon="coffee", prefix="fa")
            ).add_to(m)
    prog.progress(1.0)
    return m

def run_explorer(address):
    prog = st.progress(0.0)
    token = get_token(); prog.progress(0.1)
    if not token: return None,{},[]
    lat,lon = geocode(address, token); prog.progress(0.15)
    if lat is None: return None,{},[]
    demo  = fetch_demographics(lat, lon, token, prog)
    cafes = fetch_nearby_cafes(lat, lon, prog)
    m     = create_map(lat, lon, cafes, prog)
    return m._repr_html_(), demo, cafes
