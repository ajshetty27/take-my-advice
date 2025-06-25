import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from tqdm import tqdm
import json
import requests


from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Cluster ID → Full Persona Descriptions
cluster_personas = {
    0: """☕ Cluster 0 – The Classic Combo Loyalist
Profile: Midday regular ordering Bagel Combos with flavorful modifiers.
Age: 28–35 | Income: $60k–75k | Occupation: Grad student/staff | Dining: 3–5x/week, late mornings
Spending: ~$9.35 | Needs: Quality, routine, customizable
Media: Instagram, YouTube, Yelp""",

    1: """🧊 Cluster 1 – The Iced Latte Devotee
Profile: Daily drink-seekers, often students/freelancers using café space.
Age: 19–26 | Income: $20k–40k | Dining: Daily drinks | Spending: ~$6.14
Needs: Quick, aesthetic drinks, loyalty rewards
Media: TikTok, Instagram Stories, Spotify""",

    2: """🥯 Cluster 2 – The Power Snacker
Profile: Heavy lunch orderers, possibly ordering for multiple people.
Age: 30–40 | Income: $70k–95k | Occupation: Small business / hospital worker
Dining: 2–3x/week | Spending: $12+ | Needs: Filling, consistent, fast
Media: Facebook, Yelp, AM radio""",

    3: """🌅 Cluster 3 – The Morning Latte Regular
Profile: In-and-out loyalists who start the day with coffee.
Age: 27–35 | Income: $50k–65k | Occupation: Teacher, office worker
Dining: 4x+/week | Spending: ~$6.09 | Needs: Quick grab, friendly familiarity
Media: Podcasts, Reddit, Apple News""",

    4: """🕓 Cluster 4 – The Afternoon Refresher
Profile: Later-day customers looking for energy and calm.
Age: 22–35 | Income: $40k–75k | Dining: 2–4x/week | Time: 1–3 PM
Spending: ~$6.19 | Needs: Light refreshers, seating space
Media: Medium, productivity TikTok, Spotify""",

    5: """🍵 Cluster 5 – The Matcha Minimalist
Profile: Health- and aesthetic-driven visitors
Age: 21–28 | Income: $45k–60k | Occupation: Student-athlete, influencer
Dining: 2–3x/week | Time: early afternoon | Spending: ~$6.44
Needs: Clean ingredients, alt milk, aesthetic
Media: YouTube wellness, Substack, Instagram""",

    6: """🥯 Cluster 6 – The Early Bagel Purist
Profile: Early-morning, fast-service loyalists with no frills.
Age: 35–50 | Income: $40k–55k | Occupation: Labor, delivery, maintenance
Dining: 3–4x/week | Spending: ~$3.94 | Needs: Consistency, simplicity
Media: Talk radio, ESPN, Facebook"""
}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load data
df = pd.read_csv("data/eruta_1000_orders.csv")

# Parse hour
df["Parsed Time"] = pd.to_datetime(df["Time"], errors="coerce")
df["Hour"] = df["Parsed Time"].dt.hour.fillna(-1)

# Clean and fill
df["Item"] = df["Item"].fillna("")
df["Modifiers"] = df["Modifiers"].fillna("")
df["Qty"] = df["Qty"].fillna(1)
df["Price"] = df["Price"].fillna(0)

# Text vectorization
tfidf_item = TfidfVectorizer(max_features=50)
tfidf_mod = TfidfVectorizer(max_features=50)

item_vecs = tfidf_item.fit_transform(df["Item"])
mod_vecs = tfidf_mod.fit_transform(df["Modifiers"])

# Numeric features
numeric = df[["Hour", "Qty", "Price"]].values
numeric_scaled = StandardScaler().fit_transform(numeric)

# Combine
X_combined = hstack([item_vecs, mod_vecs, numeric_scaled])

# Dimensionality reduction
pca = PCA(n_components=30, random_state=42)
X_reduced = pca.fit_transform(X_combined.toarray())

# Clustering
kmeans = KMeans(n_clusters=7, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_reduced)

# Save results
df.to_csv("clustered_orders.csv", index=False)
print(df[["Time", "Item", "Modifiers", "Qty", "Price", "Cluster"]].head())

# === 🎨 Visualization ===
pca_viz = PCA(n_components=2, random_state=42)
X_2d = pca_viz.fit_transform(X_combined.toarray())
df["PCA1"] = X_2d[:, 0]
df["PCA2"] = X_2d[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", alpha=0.7)
plt.title("KMeans Clustering of Café Orders")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("cluster_visualization.png")
plt.show()

# === 📊 Cluster Summary ===
def top_n(series, n=3):
    return series.value_counts().nlargest(n).to_dict()

summary_df = df.groupby("Cluster").agg({
    "Item": top_n,
    "Modifiers": top_n,
    "Qty": "mean",
    "Price": "mean",
    "Order #": "count" if "Order #" in df.columns else ("Time", "count")
}).rename(columns={"Item": "Top Items", "Modifiers": "Top Modifiers", "Order #": "Count", "Time": "Count"})

print("\n📌 Cluster Summary:\n")
print(summary_df)

import numpy as np
import pandas as pd

# --- Demographic inputs ---
total_pop = 51512
unemp_rate_overall = 11.2 / 100

# Raw age bucket counts from your data
age_counts = {
    "18-24": 767,
    "25-34": 2162,
    "35-64": 680,
    "65+": 319
}
total_age_count = sum(age_counts.values())

# Step 1: Normalize and randomize age group probabilities
np.random.seed(42)  # for reproducibility
age_probs_noisy = {
    k: (v / total_age_count) * np.random.uniform(0.95, 1.05)
    for k, v in age_counts.items()
}
age_total_noisy = sum(age_probs_noisy.values())
age_probs_normalized = {
    k: v / age_total_noisy for k, v in age_probs_noisy.items()
}

# Base race proportions from your actual data
white_pct = 10744 / total_pop
black_pct = 5852 / total_pop
asian_pct = 9878 / total_pop
other_pct = max(1 - (white_pct + black_pct + asian_pct), 0)

race_cats = ["White", "Black", "Asian", "Other"]
base_probs = np.array([white_pct, black_pct, asian_pct, other_pct])

# Add SMALL noise (±1%) *without distorting original ratios*
noise = np.random.normal(loc=1.0, scale=0.01, size=4)  # tightly centered noise
noisy_probs = base_probs * noise
noisy_probs = np.clip(noisy_probs, 0.0001, None)  # prevent zeros

# Normalize to ensure valid probability distribution
race_probs = (noisy_probs / noisy_probs.sum()).tolist()


# Step 3: Employment rate by age group
emp_probs = {
    "18-24": 0.60,
    "25-34": 0.85,
    "35-64": 0.80,
    "65+": 0.20
}

# --- Function to create one synthetic individual ---
def generate_person():
    age_group = np.random.choice(list(age_probs_normalized.keys()), p=list(age_probs_normalized.values()))

    age_map = {
        "18-24": np.random.randint(18, 25),
        "25-34": np.random.randint(25, 35),
        "35-64": np.random.randint(35, 65),
        "65+": np.random.randint(65, 90)
    }
    age = age_map[age_group]

    employed = np.random.rand() < emp_probs[age_group] * (1 - unemp_rate_overall)

    if employed:
        income = {
            "18-24": np.random.normal(20000, 5000),
            "25-34": np.random.normal(30000, 7000),
            "35-64": np.random.normal(35000, 7000),
            "65+": np.random.normal(20000, 5000)
        }[age_group]
    else:
        income = np.random.normal(8000, 3000)

    race = np.random.choice(race_cats, p=race_probs)
    household_size = np.random.choice([1, 2, 3, 4, 5], p=[0.29, 0.35, 0.25, 0.08, 0.03])

    return {
        "age": age,
        "age_group": age_group,
        "employed": employed,
        "income": max(0, int(income)),
        "race": race,
        "household_size": household_size
    }

# --- Generate synthetic population ---
n = 10000  # or scale up/down as needed
synthetic_population = pd.DataFrame([generate_person() for _ in range(n)])

# View sample
synthetic_population.head()

distinct_values = synthetic_population.apply(lambda col: col.unique())

# Format as a DataFrame for easier display
distinct_df = pd.DataFrame({col: [distinct_values[col]] for col in distinct_values.index}).T
distinct_df.columns = ["Unique Values"]

print(synthetic_population.head())
# Optional: display the distinct values nicely
print(distinct_df)

# --- Step 1: Define bounding box for region (Los Angeles downtown example) ---
# Center of USC Village
center_lat = 34.0266
center_lon = -118.2854

# Approximate conversion: 1 mile ≈ 0.0145 degrees latitude and 0.0182 degrees longitude
half_mile_lat = 0.00725
half_mile_lon = 0.0091

lat_min = center_lat - half_mile_lat
lat_max = center_lat + half_mile_lat
lon_min = center_lon - half_mile_lon
lon_max = center_lon + half_mile_lon


# --- Step 2: Query Overpass API for POIs ---
overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = f"""
[out:json];
(
  node["amenity"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["amenity"]({lat_min},{lon_min},{lat_max},{lon_max});
  relation["amenity"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out center;
"""
response = requests.get(overpass_url, params={'data': overpass_query})
data = response.json()

poi_types = {}
for el in data['elements']:
    tags = el.get('tags', {})
    amenity = tags.get('amenity')
    if amenity:
        poi_types[amenity] = poi_types.get(amenity, 0) + 1

print("\nNearby POIs:")
for k, v in poi_types.items():
    print(f"{k}: {v}")

# --- Step 3: Estimate distribution (mocked current cluster counts) ---
mock_cluster_counts = np.random.dirichlet(np.ones(7), size=1)[0]
current_distribution = {i: float(f"{p:.3f}") for i, p in enumerate(mock_cluster_counts)}

poi_text = "\n".join([f"{k}: {v}" for k, v in poi_types.items()])
distribution_prompt = f"""
You are an expert in urban analytics and behavioral segmentation.

Based on the following distribution of café customer personas:
{json.dumps(current_distribution, indent=2)}

And based on the POIs found in the area:
{poi_text}

Adjust the cluster distribution to better reflect the context of this neighborhood.
Respond with a new JSON distribution (cluster IDs 0–6) summing to 1.0.
"""

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for adjusting demographic distributions based on neighborhood context."},
        {"role": "user", "content": distribution_prompt}
    ],
    temperature=0.2,
    max_tokens=512
)

response_text = response.choices[0].message.content.strip()

try:
    new_distribution = json.loads(response_text)
    print("\n📊 Adjusted Distribution Based on POIs:")
    print(json.dumps(new_distribution, indent=2))
except json.JSONDecodeError:
    print("❌ Failed to parse JSON response. Here is the raw output:")
    print(response_text)


'''# Parameters
batch_size = 50
num_people = 1000  # Use len(synthetic_population) for full-scale
persona_assignments = {}

# Ensure reproducibility
np.random.seed(42)

# --- Batching GPT Classification ---
for start in tqdm(range(0, num_people, batch_size), desc="Classifying Personas"):
    end = min(start + batch_size, num_people)
    batch = synthetic_population.iloc[start:end].copy().reset_index(drop=True)

    person_blocks = []
    for i, row in batch.iterrows():
        person_blocks.append(
            f"{i}. Age: {row['age']}, Income: ${row['income']}, Race: {row['race']}, "
            f"Employed: {'Yes' if row['employed'] else 'No'}, Household Size: {row['household_size']}"
        )
    people_text = "\n".join(person_blocks)

    prompt = f"""
You are an expert in customer segmentation.

Below are 7 pre-defined café personas:
{cluster_personas}

Now you are given {batch_size} synthetic individuals. Each includes demographics and lifestyle features.

Assign each individual to one of the above clusters (0–6) based on the best match.

Respond in JSON format like:
{{
  "0": {{
    "person": {{ "age": ..., "income": ..., "race": "...", "employed": ..., "household_size": ... }},
    "cluster": <int>,
    "reason": "..."
  }},
  ...
}}

Here are the people:
{people_text}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for clustering behavioral personas."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=4096
    )

    try:
        reply_text = response.choices[0].message.content
        parsed = json.loads(reply_text)
        for pid, data in parsed.items():
            global_index = start + int(pid)
            data["original_index"] = global_index
            persona_assignments[global_index] = data
    except Exception as e:
        print(f"⚠️ Error parsing batch {start}-{end}: {e}\nResponse:\n{reply_text}")

# === Merge Results ===
result_df = pd.DataFrame.from_dict(persona_assignments, orient="index")
merged = synthetic_population.iloc[:num_people].copy()
merged["Persona Cluster"] = result_df["cluster"].astype(int).values
merged["Reason"] = result_df["reason"].values

# Save merged file
merged.to_csv("persona_classified_population.csv", index=False)

# === Print Sample Matches ===
print("\n🧪 Example Persona Matches:")
print(merged[["age", "income", "race", "employed", "household_size", "Persona Cluster", "Reason"]].head(5))

# === Plot Distribution ===
plt.figure(figsize=(10, 5))
sns.countplot(data=merged, x="Persona Cluster", palette="Set3")
plt.title("Distribution of Café Personas")
plt.xlabel("Persona Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("persona_bar.png")
plt.show()

# Pie chart
plt.figure(figsize=(6, 6))
merged["Persona Cluster"].value_counts().sort_index().plot.pie(
    autopct="%1.1f%%", startangle=90, counterclock=False
)
plt.title("Persona Proportions (Pie Chart)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("persona_pie.png")
plt.show()

# === Print Probabilities ===
cluster_probs = merged["Persona Cluster"].value_counts(normalize=True).sort_index()
print("\n📊 Persona Probability Distribution:")
print(cluster_probs.apply(lambda x: f"{x:.2%}"))'''
