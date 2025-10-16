import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import re
from pathlib import Path

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(page_title="Dialect Change Over Time", layout="wide")
st.title("📈 U.S. Dialect Word Usage Over Time")

# -------------------------------
# Load data from Google Drive (works for full links or IDs)
# -------------------------------
@st.cache_data(show_spinner="Fetching data from Google Drive…")
def load_from_drive(file_map):
    dfs = {}
    for name, link in file_map.items():
        # Extract file ID from either full link or raw ID
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
        file_id = match.group(1) if match else link
        url = f"https://drive.google.com/uc?id={file_id}"
        output = Path(f"{name}.csv")
        if not output.exists():
            gdown.download(url, str(output), quiet=False)
        dfs[name] = pd.read_csv(output)
    return dfs

# Load all CSVs from Streamlit secrets
st.write("✅ Secrets loaded successfully:", list(st.secrets["drive_files"].keys()))
data = load_from_drive(st.secrets["drive_files"])

# Unpack the data
questions = data["questions"]
choices   = data["choices"]
users     = data["users"]
responses = data["responses"]

st.success("✅ All four datasets loaded successfully!")

# -------------------------------
# Select question from dropdown
# -------------------------------
question_texts = questions["text"].tolist()
selected_question = st.selectbox("Select a question:", question_texts)

qid = questions.loc[questions["text"] == selected_question, "id"].iloc[0]
st.write(f"Using question ID: **{qid}**")

# -------------------------------
# Merge responses with user data
# -------------------------------
choice_map = choices[choices["question_id"] == qid][["id", "value"]]
merged = (
    responses[responses["question_id"] == qid]
    .merge(choice_map, left_on="choice_id", right_on="id", how="left")
    .merge(users[["id", "year"]], left_on="user_id", right_on="id", how="left")
)

# Clean data
merged = merged.dropna(subset=["value", "year"])
merged = merged[merged["year"].between(1900, 2025)]
merged["birth_decade"] = (merged["year"] // 10) * 10

# -------------------------------
# Frequency table
# -------------------------------
freq = (
    merged.groupby(["birth_decade", "value"])
    .size()
    .reset_index(name="count")
)
freq["total"] = freq.groupby("birth_decade")["count"].transform("sum")
freq["pct"] = freq["count"] / freq["total"] * 100

# -------------------------------
# Keep top 5 most common terms
# -------------------------------
top_terms = (
    freq.groupby("value")["count"].sum()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)
freq_filtered = freq[freq["value"].isin(top_terms)]
st.write(f"Top {len(top_terms)} terms: {top_terms}")

# -------------------------------
# Plot with seaborn
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=freq_filtered,
    x="birth_decade",
    y="pct",
    hue="value",
    marker="o",
    ax=ax,
    linewidth=2.5,
)
ax.set_title("Change in Word Usage Over Birth Decades", fontsize=16, pad=12)
ax.set_xlabel("Birth Decade", fontsize=12)
ax.set_ylabel("Percentage of Respondents Using Term", fontsize=12)
ax.grid(alpha=0.3)
ax.legend(title="Term", bbox_to_anchor=(1.02, 1), loc="upper left")
st.pyplot(fig)

# -------------------------------
# Summary table
# -------------------------------
st.write("### Approximate Term Usage (%) by Decade")
summary = freq_filtered.pivot_table(
    index="birth_decade", columns="value", values="pct", fill_value=0
)
st.dataframe(summary.round(1))
