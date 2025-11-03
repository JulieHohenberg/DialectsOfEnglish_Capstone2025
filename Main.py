import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import re
from pathlib import Path


st.set_page_config(page_title="Dialect App", layout="wide")
st.title("Welcome to the Dialect App")
st.write("Use the sidebar to navigate between pages!")
