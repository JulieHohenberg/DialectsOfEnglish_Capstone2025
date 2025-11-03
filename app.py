import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import re
from pathlib import Path

# Safe debug output (works on Cloud)
st.write("🔍 Keys found in secrets:", list(st.secrets.keys()))

st.set_page_config(page_title="Dialect Change Over Time", layout="wide")
st.title("📈 U.S. Dialect Word Usage Over Time")

# -------------------------------
# Load data from Google Drive (fixed caching)
# -------------------------------
@st.cache_data(show_spinner="Fetching data from Google Drive…")
def load_from_drive(_file_map):  # leading underscore fixes UnhashableParamError
    dfs = {}
    for name, link in _file_map.items():
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
        file_id = match.group(1) if match else link
        url = f"https://drive.google.com/uc?id={file_id}"
        output = Path(f"{name}.csv")
        if not output.exists():
            gdown.download(url, str(output), quiet=False)
        dfs[name] = pd.read_csv(output)
    return dfs

# -------------------------------
# Load all CSVs from secrets.toml
# -------------------------------
try:
    st.write("✅ Secrets loaded successfully:", list(st.secrets["drive_files"].keys()))
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
