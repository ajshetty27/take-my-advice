# buckets.py

import os
import json
import pandas as pd
import streamlit as st
import numpy as np
import requests
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from models.arcgis_explorer import get_token, geocode
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import re
import hashlib

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Persona Descriptions
cluster_personas = {
    0: """☕ – The Classic Combo Loyalist
Demographics:
- Age: 28–35
- Income: $60k–75k
- Occupation: Graduate students, university staff

Ordering Behavior:
- Favorite Items: Bagel Combo, Cream Cheese, Hot Coffee
- Time: Late mornings (10 AM–12 PM)
- Frequency: 3–5 times/week
- Avg Spend: ~$9.35
- Behavior Tags: Routine-driven, customizable orders, weekday loyalist
""",

    1: """🧊 – The Iced Latte Devotee
Demographics:
- Age: 19–26
- Income: $20k–40k
- Occupation: Students, freelancers

Ordering Behavior:
- Favorite Items: Iced Latte, Alt Milk, Vanilla Syrup
- Time: 1–4 PM
- Frequency: Daily
- Avg Spend: ~$6.14
- Behavior Tags: Workspace users, aesthetic focus, loyalty reward seekers
""",

    2: """🥯 – The Power Snacker
Demographics:
- Age: 30–40
- Income: $70k–95k
- Occupation: Healthcare workers, small business staff

Ordering Behavior:
- Favorite Items: Large Sandwiches, Bagel + Add-ons, Bottled Drinks
- Time: Lunch hours (12–2 PM)
- Frequency: 2–3x/week
- Avg Spend: $12+
- Behavior Tags: Group orders, high-value cart, consistent lunch routine
""",

    3: """🌅 – The Morning Latte Regular
Demographics:
- Age: 27–35
- Income: $50k–65k
- Occupation: Teachers, office professionals

Ordering Behavior:
- Favorite Items: Latte, Muffin, Grab-n-go Breakfast
- Time: Early AM (7–9 AM)
- Frequency: 4+ times/week
- Avg Spend: ~$6.09
- Behavior Tags: Morning ritualist, in-and-out, speed-driven
""",

    4: """🕓 – The Afternoon Refresher
Demographics:
- Age: 22–35
- Income: $40k–75k
- Occupation: Students, junior professionals

Ordering Behavior:
- Favorite Items: Matcha Latte, Refreshers, Alt Milk Drinks
- Time: 2–5 PM
- Frequency: 2–4x/week
- Avg Spend: ~$6.19
- Behavior Tags: Light drinkers, social relaxers, solo visitors
""",

    5: """🍵  – The Matcha Minimalist
Demographics:
- Age: 21–28
- Income: $45k–60k
- Occupation: Influencers, student-athletes

Ordering Behavior:
- Favorite Items: Matcha, Almond Milk, Toast or Smoothie
- Time: 11 AM–2 PM
- Frequency: 2–3x/week
- Avg Spend: ~$6.44
- Behavior Tags: Health-focused, minimalist cart, midday routine
""",

    6: """🥯  – The Early Bagel Purist
Demographics:
- Age: 35–50
- Income: $40k–55k
- Occupation: Blue-collar, logistics, delivery

Ordering Behavior:
- Favorite Items: Plain Bagel, Black Coffee, No Modifiers
- Time: Very Early (6–8 AM)
- Frequency: 3–4x/week
- Avg Spend: ~$3.94
- Behavior Tags: Quick turnaround, low-modification orders, consistency seekers
"""
}


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class DemoToOrderProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


@st.cache_data(show_spinner="Generating personas...", max_entries=50)
def cached_gpt_personas(prompt: str) -> list:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Simulation expert"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=4096
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```json"): raw = raw.replace("```json", "").replace("```", "").strip()
    elif raw.startswith("```"): raw = raw.replace("```", "").strip()
    return json.loads(raw)

def build_deterministic_prompt(demo_data: dict, poi_types: dict) -> str:
# --- Group POIs into high-level zones ---
    poi_summary = {
        "college campus": sum(poi_types.get(k, 0) for k in ["college", "university", "library"]),
        "office-heavy": sum(poi_types.get(k, 0) for k in ["office", "bank", "company", "post_office"]),
        "residential": sum(poi_types.get(k, 0) for k in ["apartment", "residential", "dormitory"]),
        "hospital zone": sum(poi_types.get(k, 0) for k in ["hospital", "clinic", "pharmacy"]),
        "retail district": sum(poi_types.get(k, 0) for k in ["mall", "supermarket", "convenience", "department_store"]),
        "tourist zone": sum(poi_types.get(k, 0) for k in ["museum", "theatre", "attraction"]),
        "transport hub": sum(poi_types.get(k, 0) for k in ["bus_station", "train_station", "airport"]),
    }

    poi_context = "\n".join([f"- {k}: {v} count" for k, v in poi_summary.items() if v > 0])
    demo_text   = "\n".join([f"- {k}: {v}" for k, v in demo_data.items()])


    return f"""
You are a simulation expert generating 7 synthetic customer personas for a café using demographic and neighborhood context.

The area surrounding the café is characterized by the following amenities:
{poi_context}

Key demographic indicators:
{demo_text}

Based on these inputs, generate exactly 7 customer personas as a structured JSON array. Each persona should represent a distinct slice of the local customer base, influenced by the amenities nearby.

Each persona must include:
- "cluster": integer from 0 to 6
- "age", "income", "household_size", "race", "tag"
- "percentage": proportion of total population they represent (must sum to 1.0)

Return ONLY the JSON — no prose, no markdown.

Example format:
[
  {{
    "cluster": 0,
    "age": 29,
    "income": 62000,
    "household_size": 2,
    "race": "White",
    "tag": "young professional",
    "percentage": 0.15
  }},
  ...
]
"""

def train_demo_to_order_mapping(demo_embeddings, target_order_centroids, assignments, device, epochs=200, lr=1e-3):
    model = DemoToOrderProjector(demo_embeddings.shape[1], output_dim=target_order_centroids.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    demo_embeddings = demo_embeddings.to(device)
    target_centroids = target_order_centroids.to(device)
    assignments = torch.tensor(assignments).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(demo_embeddings)  # [N, output_dim]
        targets = target_centroids[assignments]  # True order embedding per person

        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model

def embed_and_assign_population_from_centroids(synthetic_centroids,persona_centroids,device,autoencoder,trained_projector=None):
    demo_features = ["age", "income", "household_size"]
    race_col = "race"

    # Step 1: Expand into full population (1 row = 1 person)
    expanded = []
    for _, row in synthetic_centroids.iterrows():
        for _ in range(int(row["estimated_people"])):
            expanded.append({
                "age": row["age"],
                "income": row["income"],
                "household_size": row["household_size"],
                "race": row["race"],
                "tag": row["tag"],
                "estimated_people": 1
            })
    expanded_df = pd.DataFrame(expanded)

    # Step 2: Normalize numeric features
    X_numeric = expanded_df[demo_features].astype(float)
    X_scaled = StandardScaler().fit_transform(X_numeric)

    # Step 3: One-hot encode race
    race_onehot = pd.get_dummies(expanded_df[race_col])
    X_all = np.hstack([X_scaled, race_onehot.values])

    # Step 4: Convert to tensor
    X_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)

    # Step 5: Encode + Project (optional) + Normalize
    with torch.no_grad():
        demo_embedded = autoencoder.encoder(X_tensor)
        demo_embedded = torch.nn.functional.normalize(demo_embedded, dim=1)

        if trained_projector is not None:
            projected_embedded = trained_projector(demo_embedded)
            embedded = torch.nn.functional.normalize(projected_embedded, dim=1)
        else:
            embedded = demo_embedded

        normalized_centroids = torch.nn.functional.normalize(persona_centroids, dim=1)

        # Step 6: Similarity & Probabilities
        similarities = torch.matmul(embedded, normalized_centroids.T)  # [N, 7]


        temperature = 0.05
        scaled_similarities = similarities / temperature
        probs = torch.nn.functional.softmax(scaled_similarities, dim=1).cpu().numpy()

    # Step 7: Store probabilities, entropy, etc.
    for i in range(7):
        expanded_df[f"prob_cluster_{i}"] = probs[:, i]

    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    expanded_df["assignment_entropy"] = entropy

    top_probs = probs.max(axis=1)
    top_indices = probs.argmax(axis=1)
    expanded_df["max_prob"] = top_probs

    # Soft assignment fallback logic
    conf_threshold = 0.5
    soft_assignments = [
        idx if prob > conf_threshold else np.random.choice(7, p=row)
        for row, prob, idx in zip(probs, top_probs, top_indices)
    ]
    expanded_df["Matched Order Cluster"] = soft_assignments

    # Weighted similarity metric
    expanded_df["weighted_similarity"] = (probs * similarities.cpu().numpy()).sum(axis=1)

    # === Step 8: Recenter Persona Centroids using Soft Assignments ===
    new_centroids = []
    probs_tensor = torch.tensor(probs, dtype=torch.float32).to(device)

    for i in range(7):
        weights = probs_tensor[:, i].unsqueeze(1)  # [N, 1]
        weighted_sum = (weights * embedded).sum(dim=0)
        total_weight = weights.sum()
        new_centroid = weighted_sum / (total_weight + 1e-9)
        new_centroids.append(new_centroid)

    updated_centroids = torch.stack(new_centroids)
    updated_centroids = torch.nn.functional.normalize(updated_centroids, dim=1)

    print("✅ Updated Centroids via Soft Assignment")

    return expanded_df

def run(context):

    global cluster_personas

    # --- Extract context ---
    demo_data  = context.get("demographics", {})
    pos_data   = context.get("pos", [])
    order_data = context.get("order_details", [])
    form_data  = context.get("form", {})
    address    = form_data.get("Location Address", "")
    if not address: raise ValueError("No address in form.")

    # --- Geocode for POIs ---
    token = get_token()
    lat, lon = geocode(address, token)
    if lat is None: raise ValueError("Geocode failed.")

    half_lat, half_lon = 0.00725, 0.0091
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];(
      node["amenity"]({lat - half_lat},{lon - half_lon},{lat + half_lat},{lon + half_lon});
      way["amenity"]({lat - half_lat},{lon - half_lon},{lat + half_lat},{lon + half_lon});
      relation["amenity"]({lat - half_lat},{lon - half_lon},{lat + half_lat},{lon + half_lon});
    );out center;
    """
    pois = requests.get(overpass_url, params={"data": query}).json()
    poi_types = {}
    for el in pois.get("elements", []):
        tags = el.get("tags", {})
        if "amenity" in tags:
            poi_types[tags["amenity"]] = poi_types.get(tags["amenity"], 0) + 1
    print("🔍 POIs:", json.dumps(poi_types, indent=2))

    prompt = build_deterministic_prompt(demo_data, poi_types)
    persona_data = cached_gpt_personas(prompt)
    persona_df = pd.DataFrame(persona_data)

    # Total population from demographic context
    total_pop = int(float(demo_data.get("Total Population (2024)", 1000)))

    # Add estimated people count per persona
    persona_df["estimated_people"] = (persona_df["percentage"] * total_pop).round().astype(int)

    print("✅ Synthetic personas with population counts:")
    print(persona_df[["cluster", "tag", "estimated_people"]])

    # Rename to maintain compatibility with rest of pipeline
    population = persona_df

    demo_features = ["age", "income", "household_size"]
    race_col = "race"

    X_numeric = population[demo_features].astype(float)
    X_scaled = StandardScaler().fit_transform(X_numeric)
    race_onehot = pd.get_dummies(population[race_col])
    X_demo = np.hstack([X_scaled, race_onehot.values])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_autoencoder = Autoencoder(input_dim=X_demo.shape[1]).to(device)
    demo_loader = DataLoader(TensorDataset(torch.tensor(X_demo, dtype=torch.float32)), batch_size=4, shuffle=True)
    opt_demo = torch.optim.Adam(demo_autoencoder.parameters(), lr=1e-3)

    for _ in range(1000):
        for batch, in demo_loader:
            batch = batch.to(device)
            recon, _ = demo_autoencoder(batch)
            loss = nn.functional.mse_loss(recon, batch)
            opt_demo.zero_grad(); loss.backward(); opt_demo.step()

    print("✅ Demo autoencoder trained.")


    # --- Clean and embed order data ---
    df = pd.DataFrame(order_data)
    has_orders = not df.empty

    if has_orders:
        df["Item"] = df["Item"].fillna("")
        df["Modifiers"] = df["Modifiers"].fillna("")
        df["Qty"] = df["Qty"].fillna(1)
        df["Price"] = df["Price"].fillna(0)
        df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour.fillna(-1)

        def is_valid_item(item):
            if not isinstance(item, str): return False
            item = item.strip()
            return (
                len(item) > 1 and
                not bool(re.match(r"^\$?\d+(\.\d{1,2})?$", item)) and
                item.lower() not in ["", "free", "none", "n/a"]
            )

        # Initial fill and cleaning
        df["Item"] = df["Item"].fillna("").astype(str)
        df["Modifiers"] = df["Modifiers"].fillna("").astype(str)
        df["Qty"] = df["Qty"].fillna(1)
        df["Price"] = df["Price"].fillna(0)
        df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour.fillna(-1)

        # Filter invalid items
        df["Item"] = df["Item"].apply(lambda x: x if is_valid_item(x) else None)
        df["Item"] = df["Item"].fillna("").astype(str)  # 🚨 FIX LINE

        # Vectorize
        tfidf_item = TfidfVectorizer(max_features=50)
        tfidf_mod  = TfidfVectorizer(max_features=50)
        item_vecs = tfidf_item.fit_transform(df["Item"])
        mod_vecs  = tfidf_mod.fit_transform(df["Modifiers"])
        numeric = StandardScaler().fit_transform(df[["Qty", "Price", "Hour"]].values)
        X_order = hstack([item_vecs, mod_vecs, numeric]).toarray()


        # --- Autoencoder directly on order features ---
        autoencoder = Autoencoder(input_dim=X_order.shape[1]).to(device)
        loader = DataLoader(TensorDataset(torch.tensor(X_order, dtype=torch.float32)), batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # --- Train ---
        for epoch in range(10):  # keep short during debugging
            for batch, in loader:
                batch = batch.to(device)
                recon, _ = autoencoder(batch)
                loss = loss_fn(recon, batch)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        with torch.no_grad():
            latent_vecs = autoencoder.encoder(torch.tensor(X_order, dtype=torch.float32).to(device))
        latent_vecs_np = latent_vecs.cpu().numpy()

        kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(latent_vecs_np)

        df["Persona Cluster"] = clusters

        centroids = []
        for i in range(7):
            cluster_pts = latent_vecs[clusters == i]
            centroids.append(cluster_pts.mean(dim=0))
        persona_centroids = torch.stack(centroids)


        persona_centroids_norm = F.normalize(persona_centroids, dim=1)


        # -- Compute persona embedding vectors from synthetic demographic profiles --
        X_demo_tensor = torch.tensor(X_demo, dtype=torch.float32).to(device)
        with torch.no_grad():
            population_emb = demo_autoencoder.encoder(X_demo_tensor)  # shape: [7, latent_dim]

        # -- Normalize both sets of embeddings --
        population_emb_norm = F.normalize(population_emb, dim=1)
        persona_centroids_norm = F.normalize(persona_centroids, dim=1)

        # -- Compute similarity matrix --
        similarity_matrix = torch.matmul(population_emb_norm, persona_centroids_norm.T)

        # === NEW: Train projection model from demographic embeddings → order cluster centroids ===
        hard_assignments = similarity_matrix.argmax(dim=1)  # Best matching cluster per synthetic person

        projector_model = train_demo_to_order_mapping(
            demo_embeddings=population_emb,
            target_order_centroids=persona_centroids,
            assignments=hard_assignments,
            device = device,
            epochs=300,
            lr=1e-3
        )

# === NEW: Use projection model to predict soft persona centroids from demographic data ===
        with torch.no_grad():
            projected = projector_model(population_emb)  # shape: [7, latent_dim]
            projected_norm = F.normalize(projected, dim=1)

        # Soft similarity between projected vectors and order cluster centroids
        soft_similarity = torch.matmul(projected_norm, F.normalize(persona_centroids, dim=1).T)
        soft_assignment = soft_similarity.softmax(dim=1)  # Shape [7, 7]

        # Assign the most likely cluster and store max prob
        best_clusters = soft_assignment.argmax(dim=1).cpu().numpy()
        max_probs = soft_assignment.max(dim=1).values.cpu().numpy()

        population["Matched Order Cluster"] = best_clusters
        population["max_prob"] = max_probs

        # Replace hard alignment with soft best cluster
        df["Persona Cluster (Original)"] = df["Persona Cluster"]
        df["Persona Cluster"] = df["Persona Cluster"].map(lambda x: best_clusters[x])


        print("🔁 Re-mapped KMeans clusters to match persona descriptions:")
        if 'alignment' in locals():
                    for i, persona_idx in enumerate(alignment):
                        print(f"KMeans Cluster {i} → Persona Cluster {persona_idx}")

        # Top items per cluster
        top_items = df.groupby("Persona Cluster")["Item"].agg(lambda x: x.value_counts().head(3).to_dict())
    else:
        print("⚠️ No order data uploaded, skipping order clustering.")
        df = pd.DataFrame()
        top_items = {}
        persona_centroids = torch.zeros((7, 16)).to(device)  # dummy for assignment


    # --- Autoencoder Training with Early Stopping ---
    EPOCHS = 50
    PATIENCE = 5
    MIN_DELTA = 1e-4

    best_loss = float("inf")
    epochs_no_improve = 0
    loss_history = []

    print("🧠 Training autoencoder...")
    for epoch in range(EPOCHS):
        epoch_losses = []
        for batch, in loader:
            batch = batch.to(device)
            recon, _ = autoencoder(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.5f}")

        # Early stopping check
        if best_loss - avg_loss > MIN_DELTA:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    print("✅ Autoencoder training complete.")


    # --- Cluster and embed personas ---
    kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(latent_vecs_np)
    df["Persona Cluster"] = clusters

    centroids = []
    with torch.no_grad():
        for i in range(7):
            cluster_pts = torch.tensor(latent_vecs_np[clusters == i], dtype=torch.float32).to(device)
            centroids.append(cluster_pts.mean(dim=0))
    persona_centroids = torch.stack(centroids)

    # -- Compute persona embedding vectors from synthetic demographic profiles --
    X_demo_tensor = torch.tensor(X_demo, dtype=torch.float32).to(device)
    with torch.no_grad():
        population_emb = demo_autoencoder.encoder(X_demo_tensor)  # [7, latent_dim]

    # -- Normalize both sets of embeddings for cosine similarity --
    persona_centroids_norm = torch.nn.functional.normalize(persona_centroids, dim=1)
    population_emb_norm = torch.nn.functional.normalize(population_emb, dim=1)


    # -- Compute cosine distance matrix for Hungarian assignment (we want max similarity → min negative similarity) --
    cost_matrix = -torch.matmul(population_emb_norm, persona_centroids_norm.T).cpu().numpy()  # shape [7, 7]

    # -- Solve optimal assignment (minimize total negative similarity) --
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    alignment = dict(zip(row_ind, col_ind))  # order_cluster → persona_cluster


# Convert row/column assignment from Hungarian method into a list
    best_match = [col_ind[row] for row in row_ind]

    # Assign back
    population["Closest Order Cluster"] = best_match
    population["Similarity Score"] = (torch.matmul(population_emb_norm, persona_centroids_norm.T).max(dim=1).values.cpu().numpy())

    print("\n🧬 Persona to Order Cluster Assignment:")
    for i, c in enumerate(best_match):
        print(f"Persona {i} → Order Cluster {c} | Similarity: {population['Similarity Score'].iloc[i]:.3f}")

    # 🔁 Use projected centroids to improve the population matching
    projected_centroids = projector_model(population_emb)
    projected_centroids = torch.nn.functional.normalize(projected_centroids, dim=1)


    # Step 1: Reset index to get clean cluster IDs
    # === Prepare population with a 'cluster' column ===
    population = population.reset_index(drop=True)
    population["cluster"] = population.index

    # === Get persona assignments ===
    pop_with_persona = embed_and_assign_population_from_centroids(
        synthetic_centroids=population,
        persona_centroids=projected_centroids,
        device=device,
        autoencoder=demo_autoencoder
    )

    # === Ensure 'cluster' exists for merging ===
    pop_with_persona = pop_with_persona.reset_index(drop=True)
    pop_with_persona["cluster"] = pop_with_persona.index

    # === Merge with population info ===
# Merge and keep only what's needed
    pop_with_persona = pop_with_persona.merge(
        population[["cluster", "Similarity Score", "max_prob"]],
        on="cluster",
        how="left"
    )

    # Rename to ensure consistency
    if "Matched Order Cluster_x" in pop_with_persona.columns:
        pop_with_persona.rename(columns={"Matched Order Cluster_x": "Matched Order Cluster"}, inplace=True)
    elif "Matched Order Cluster_y" in pop_with_persona.columns:
        pop_with_persona.rename(columns={"Matched Order Cluster_y": "Matched Order Cluster"}, inplace=True)

    # Confirm fix
    print("✅ Final Columns:", pop_with_persona.columns.tolist())
    assert "Matched Order Cluster" in pop_with_persona.columns, "Matched Order Cluster is still missing!"




    top_tags_per_cluster = (
    pop_with_persona.groupby("Matched Order Cluster")["tag"]
    .value_counts()
    .groupby(level=0)
    .head(3)
    .reset_index(name="count")
    )


    bucket_sizes = (pop_with_persona.groupby("Matched Order Cluster")["estimated_people"].sum().reindex(range(7), fill_value = 0).sort_index())
    print("\n📊 Estimated Footfall per Bucket:")
    for i, count in bucket_sizes.items():
        print(f"Cluster {i}: {count} people")

    # Build final persona summary table
    cluster_summary = []
    for i in range(7):
        if i in pop_with_persona["Matched Order Cluster"].values:
            # Get top 2-3 tags by frequency
            tag_counts = (
                pop_with_persona[pop_with_persona["Matched Order Cluster"] == i]["tag"]
                .value_counts()
                .head(3)
                .index
                .tolist()
            )
            tag = ", ".join(tag_counts)
        else:
            tag = "N/A"

        top_items_str = ", ".join(top_items.get(i, {}).keys())
        estimated = bucket_sizes.get(i, 0)
        cluster_summary.append({
            "Cluster": i,
            "Estimated People": estimated,
            "tag": tag,
            "Top Items": top_items_str
        })


    cluster_df = pd.DataFrame(cluster_summary)
    # Map order cluster (from KMeans) to persona cluster based on reverse alignment
# Example: if Persona 2 is closest to Order Cluster 0, then Order Cluster 0 → Persona 2
    cluster_mapping = {order_cluster: persona_cluster for persona_cluster, order_cluster in enumerate(best_match)}

    print("🧭 Order Cluster → Persona Cluster Mapping:")
    for order_cluster, persona_cluster in alignment.items():
        print(f"Order Cluster {order_cluster} → Persona Cluster {persona_cluster}")



    df["Persona Cluster (Original)"] = df["Persona Cluster"]
    df["Persona Cluster"] = df["Persona Cluster"].map(lambda x: alignment[x])


    pop_with_persona["Mapped Persona Cluster"] = pop_with_persona["Matched Order Cluster"].map(cluster_mapping)
    mapped_cluster_sizes = (
        pop_with_persona.groupby("Mapped Persona Cluster")["tag"]
        .count().reindex(range(7), fill_value=0).sort_index()
    )


    # --- Top items by persona ---
    top_items = df.groupby("Persona Cluster")["Item"].agg(lambda x: x.value_counts().head(3).to_dict())

    # Step 1: Convert top items to text
    cluster_texts = {
        cluster: " ".join(items.keys())
        for cluster, items in top_items.items()
    }

    # Step 2: Prepare corpus: persona descriptions + top item clusters
    persona_texts = [desc for _, desc in cluster_personas.items()]
    order_texts = [cluster_texts.get(i, "") for i in range(7)]
    corpus = persona_texts + order_texts

    # Step 3: TF-IDF and cosine similarity
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)
    persona_vecs = tfidf_matrix[:7]
    order_vecs = tfidf_matrix[7:]

    similarity_matrix = cosine_similarity(persona_vecs, order_vecs)

    # Step 4: Optimal matching (Hungarian method)
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Max similarity

    persona_to_order = dict(zip(row_ind, col_ind))  # Persona cluster → Order cluster

    # --- Remap cluster_personas so that each order cluster shows correct description ---
    revised_personas = {}
    for persona_cluster, order_cluster in persona_to_order.items():
        revised_personas[order_cluster] = cluster_personas[persona_cluster]

    # Fill gaps if any clusters didn't match (fallback)
    for i in range(7):
        if i not in revised_personas:
            revised_personas[i] = f"📦 Cluster {i} – No specific description available."

    # Replace old mapping
    cluster_personas = revised_personas


    print("\n🧠 Matched Persona Descriptions to Order Clusters:")
    for persona, order in persona_to_order.items():
        sim = similarity_matrix[persona, order]
        print(f"Persona {persona} → Order Cluster {order} | Cosine Sim: {sim:.3f}")

    print("🧩 Top Items by Cluster:")
    for i, items in top_items.items():
        print(f"Cluster {i}: {items}")

    print("\n📐 Avg Similarity by Cluster:")
    for i in range(7):
        sims = pop_with_persona[pop_with_persona["Matched Order Cluster"] == i]["Similarity Score"]

        print(f"Cluster {i}: {sims.mean():.3f} avg similarity")

    if "max_prob_x" in pop_with_persona.columns:
        pop_with_persona.rename(columns={"max_prob_x": "max_prob"}, inplace=True)
    elif "max_prob_y" in pop_with_persona.columns:
        pop_with_persona.rename(columns={"max_prob_y": "max_prob"}, inplace=True)

    convertible_people = pop_with_persona[pop_with_persona["max_prob"] < 0.55]  # adjust threshold as needed

    type_summary = pop_with_persona.groupby("tag").agg({
    "age": "mean",
    "income": "mean",
    "household_size": "mean",
    "race": lambda x: x.value_counts().index[0],  # most common race
    "Mapped Persona Cluster": lambda x: x.value_counts().index[0],  # most common cluster
    "estimated_people": "sum"
    }).reset_index()


    return {
        "persona_assignments": pop_with_persona,
        "order_clusters": df,
        "top_items": top_items,
        "desciptions": cluster_personas,
        "footfall_estimates": bucket_sizes.to_dict(),
        "cluster_summary": cluster_df,
        "top_tags_per_persona": top_tags_per_cluster,
        "convertible_individuals": convertible_people,
        "type_summaries": type_summary
    }
