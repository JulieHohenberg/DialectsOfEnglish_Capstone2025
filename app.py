import streamlit as st
import seaborn as sns
import re

# Load data
@st.cache_data(show_spinner="Fetching data from Google Drive…")
def load_from_drive(file_map):
    dfs = {}
    for name, link in file_map.items():
        # Extract file ID from either full link or raw ID
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
        if match:
            file_id = match.group(1)
        else:
            file_id = link  # if already just an ID
        
        url = f"https://drive.google.com/uc?id={file_id}"
        output = Path(f"{name}.csv")
        if not output.exists():
            gdown.download(url, str(output), quiet=False)
        dfs[name] = pd.read_csv(output)
    return dfs

## define
questions = data["questions"]
choices   = data["choices"]
users     = data["users"]
responses = data["responses"]

st.success("✅ All four datasets loaded successfully!")

st.header("📈 Word Usage Over Time")

# -------------------------------
# Identify the selected question again
# -------------------------------
qid = questions.loc[questions["text"] == selected_question, "id"].iloc[0]

# Merge responses, choices, and user birth year
choice_map = choices[choices["question_id"] == qid][["id", "value"]]
merged = (
    responses[responses["question_id"] == qid]
    .merge(choice_map, left_on="choice_id", right_on="id", how="left")
    .merge(users[["id", "year"]], left_on="user_id", right_on="id", how="left")
)

# Clean
merged = merged.dropna(subset=["value", "year"])
merged = merged[merged["year"].between(1900, 2025)]

# Birth decade bins
merged["birth_decade"] = (merged["year"] // 10) * 10

# Frequency table
freq = (
    merged.groupby(["birth_decade", "value"])
    .size()
    .reset_index(name="count")
)
freq["total"] = freq.groupby("birth_decade")["count"].transform("sum")
freq["pct"] = freq["count"] / freq["total"] * 100

# Identify top 5 most common terms overall
top_terms = (
    freq.groupby("value")["count"].sum().sort_values(ascending=False).head(5).index.tolist()
)
freq_filtered = freq[freq["value"].isin(top_terms)]

# -------------------------------
# Plot using seaborn
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

# Optional table display
st.write("### Approximate Term Usage (%) by Decade")
summary = freq_filtered.pivot_table(
    index="birth_decade", columns="value", values="pct", fill_value=0
)
st.dataframe(summary.round(1))
