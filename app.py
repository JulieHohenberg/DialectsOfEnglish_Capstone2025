import streamlit as st

st.write("✅ Secrets loaded successfully:",
         list(st.secrets["drive_files"].keys()))
