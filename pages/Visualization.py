import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import re
from pathlib import Path
import plotly.express as px
import numpy as np


st.set_page_config(page_title="Dialect Change Over Time", layout="wide")
st.title("Visualization Page")
st.write("Here are the visualizations.")


st.title("U.S. Dialect Word Usage Over Time")

# -------------------------------
# Load data from Google Drive (fixed caching)
# -------------------------------
@st.cache_data(show_spinner="Fetching data from Google Drive…")
def load_from_drive(_file_map):  # leading underscore fixes UnhashableParamError
    dfs = {}
    data_folder = Path("data")
    for name, link in _file_map.items():
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
        file_id = match.group(1) if match else link
        url = f"https://drive.google.com/uc?id={file_id}"
        output = data_folder / f"{name}.csv"
        
        if not output.exists():
            gdown.download(url, str(output), quiet=False)
        dfs[name] = pd.read_csv(output, low_memory=False)
    return dfs

# -------------------------------
# Load all CSVs from secrets.toml
# -------------------------------
try:
    data = load_from_drive(st.secrets["drive_files"])
except KeyError:
    st.error("❌ Missing `drive_files` in secrets.toml! Add it under `[drive_files]`.")
    st.stop()

# Unpack data
questions = data["questions"]
choices = data["choices"]
users = data["users"]
responses = data["responses"]

st.success("✅ All four datasets loaded successfully!")

# --- Focus on the "soda/pop" question (id=2) ---
soda_qid = 2
responses_soda = responses[responses["question_id"] == soda_qid].copy()

# Merge user info (birth year)
responses_soda = responses_soda.merge(
    users[["id", "year"]], left_on="user_id", right_on="id", how="left"
)

# Merge choice values if available
responses_soda = responses_soda.merge(
    choices[["id", "value"]], left_on="choice_id", right_on="id", how="left"
)

# Use 'value' from choices, or fallback to 'other' column
responses_soda["term"] = responses_soda["value"].combine_first(responses_soda["other"])

# --- Clean & Transform ---
# Drop missing years or terms
responses_soda = responses_soda.dropna(subset=["year", "term"])

# Compute birth decade
responses_soda["decade"] = (responses_soda["year"] // 10 * 10).astype(int)

# Normalize term text
responses_soda["term"] = responses_soda["term"].str.strip().str.lower()

# --- Aggregate ---
counts = (
    responses_soda.groupby(["decade", "term"])
    .size()
    .reset_index(name="count")
)

# Compute % respondents per decade
totals = counts.groupby("decade")["count"].transform("sum")
counts["percent"] = (counts["count"] / totals * 100).round(1)

# --- Get Top 5 Terms Overall ---
top_terms = (
    counts.groupby("term")["count"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)

# Sidebar filters
st.sidebar.header("Filter Controls")
selected_terms = st.sidebar.multiselect(
    "Select terms to display:",
    options=counts["term"].unique(),
    default=top_terms
)

filtered = counts[counts["term"].isin(selected_terms)]

# --- Plot ---
fig = px.line(
    filtered,
    x="decade",
    y="percent",
    color="term",
    markers=True,
    labels={"percent": "% Respondents Using Term", "decade": "Birth Decade"},
    hover_data={
        "term": True,
        "percent": True,
        "decade": True,
    },
    title="Change in Word Usage Over Birth Decades (Top 5 Terms, Normalized by Birth Year)"
)

# --- Make hover work per bullet instead of per year ---
fig.update_traces(
    mode="lines+markers",
    hovertemplate="<b>%{customdata[0]}</b><br>Decade: %{x}<br>% Respondents: %{y:.1f}%<extra></extra>"
)
fig.update_layout(
    hovermode="closest",  # <— this is the key change
    legend_title_text="Term",
    plot_bgcolor="white"
)

# --- Display ---
st.plotly_chart(fig, use_container_width=True)


# --- Focus on Soda/Pop Question ---
QUESTION_ID = 2  # Only the "sweetened carbonated beverage" question
responses = responses[responses["question_id"] == QUESTION_ID].copy()

# Merge user info
responses = responses.merge(
    users[["id", "year", "gender", "state"]],
    left_on="user_id",
    right_on="id",
    how="left"
)

# Merge choice values
responses = responses.merge(
    choices[["id", "value"]],
    left_on="choice_id",
    right_on="id",
    how="left"
)

# Combine choice value with typed 'other' responses
responses["term"] = responses["value"].combine_first(responses["other"]).str.lower().str.strip()

# Keep only soda vs pop
responses = responses[responses["term"].isin(["soda", "pop"])]

# Drop missing states or terms
responses = responses.dropna(subset=["state", "term"])

# --- Shannon Entropy Function ---
def shannon_entropy(series):
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

# Compute lexical diversity per state
entropy_by_state = (
    responses.groupby("state")["term"]
    .apply(shannon_entropy)
    .reset_index(name="entropy")
)

# --- Sidebar Filters (optional placeholders) ---
st.sidebar.header("Filter Controls")

# Set year slider range based on the dataset
min_year = int(responses["year"].min())
max_year = int(responses["year"].max())
year_range = st.sidebar.slider(
    "Filter by birth year:",
    min_year,
    max_year,
    value=(min_year, max_year)
)

# Gender filter
existing_genders = responses["gender"].dropna().unique().tolist()
all_genders = sorted(set(existing_genders + ["f","m","o","x"]))
gender_filter = st.sidebar.multiselect(
    "Filter by gender:",
    options=all_genders,
    default=all_genders
)

# Apply filters
filtered = responses[
    (responses["year"].between(year_range[0], year_range[1])) &
    (responses["gender"].isin(gender_filter))
]
entropy_by_state = (
    filtered.groupby("state")["term"]
    .apply(shannon_entropy)
    .reset_index(name="entropy")
)

# --- Create Map ---
fig = px.choropleth(
    entropy_by_state,
    locations="state",
    locationmode="USA-states",
    color="entropy",
    color_continuous_scale="plasma",
    scope="usa",
    labels={"entropy": "Lexical Diversity (Shannon Entropy)"},
    hover_data={"state": True, "entropy": True}
)

fig.update_layout(
    geo=dict(bgcolor="rgba(0,0,0,0)"),
    coloraxis_colorbar=dict(title="Lexical Diversity"),
    margin=dict(l=10, r=10, t=60, b=10),
)

# --- Display ---
st.title("Lexical Diversity (Shannon Entropy) by U.S. State — Soda vs. Pop")
st.plotly_chart(fig, use_container_width=True)



st.markdown("""
**Shannon entropy** measures how diverse word choices are within each state  
(high = high diversity, no single dominant response; low = low diversity, one response dominates).  

👉 *This example is **ONLY SODA VS. POP!***
""")