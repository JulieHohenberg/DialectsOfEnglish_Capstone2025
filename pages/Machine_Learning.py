import streamlit as st

st.set_page_config(page_title="Machine Learning", layout="wide")
st.title("Machine Learning Page")

st.write("Here are the ML results.")

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import gdown
import re

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

# --- Make sure the 'responses' DataFrame exists ---
# Normalize columns: strip spaces and lowercase
responses.columns = responses.columns.str.strip().str.lower()

# 1. Count unique users per question
users_answered = (
    responses
    .dropna(subset=['choice_id'])
    .groupby('question_id')['user_id']
    .nunique()
    .reset_index(name='users_answered')
)

# 2. Compute summary statistics
mean_val = users_answered['users_answered'].mean()
median_val = users_answered['users_answered'].median()
min_val = users_answered['users_answered'].min()
max_val = users_answered['users_answered'].max()
total_users = responses['user_id'].nunique()

# 3. Display results in Streamlit
st.subheader("User Coverage per Question")
st.write(f"**Mean:** {mean_val:.2f}")
st.write(f"**Median:** {median_val}")
st.write(f"**Min:** {min_val}")
st.write(f"**Max:** {max_val}")
st.write(f"**Total unique users:** {total_users}")

# 4. Plot histogram
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

sns.histplot(
    data=users_answered,
    x='users_answered',
    bins=50,           # adjust as needed
    color='steelblue',
    edgecolor='black',
    alpha=0.7
)

ax.tick_params(axis='both', labelsize=14)
ax.set_title("Distribution of Number of Users Who Answered Each Question", fontsize=16, pad=12)
ax.set_xlabel("Number of Users Who Answered (per question)", fontsize=14)
ax.set_ylabel("Number of Questions", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# 5. Show in Streamlit
st.pyplot(fig)

