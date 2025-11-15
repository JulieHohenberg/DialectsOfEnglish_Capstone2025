import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import gdown
import re
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import numpy as np
import io
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import pydeck as pdk

# -------------------------------
# Basic page config
# -------------------------------
st.set_page_config(page_title="Machine Learning", layout="wide")
st.title("Machine Learning Page")
st.write("Here are the ML results.")

# -------------------------------
# Load data from Google Drive (cached)
# -------------------------------
@st.cache_data(show_spinner="Fetching data from Google Drive…")
def load_from_drive(_file_map):
    dfs = {}
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    for name, link in _file_map.items():
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
        file_id = match.group(1) if match else link
        url = f"https://drive.google.com/uc?id={file_id}"
        output = data_folder / f"{name}.csv"

        if not output.exists():
            gdown.download(url, str(output), quiet=False)

        dfs[name] = pd.read_csv(output, low_memory=False)

    return dfs

# -------------------------------
# Load all CSVs
# -------------------------------
try:
    data = load_from_drive(st.secrets["drive_files"])
except KeyError:
    st.error("❌ Missing `drive_files` in secrets.toml! Add it under `[drive_files]`.")
    st.stop()

questions = data["questions"]
choices = data["choices"]
users = data["users"]
responses = data["responses"]

st.success("✅ All four datasets loaded successfully!")

# -------------------------------
# Prepare user data
# -------------------------------
required_cols = {"id", "lat", "lng"}
if not required_cols.issubset(users.columns):
    st.error(
        f"`users` must contain columns {required_cols}, "
        f"but has {set(users.columns)}."
    )
    st.stop()

users_df = users[["id", "lat", "lng"]].copy()

# -------------------------------
# NORTH AMERICA FILTER
# -------------------------------
def filter_north_america(df):
    return df[
        (df["lng"] >= -140) & (df["lng"] <= -60) &
        (df["lat"] >= 20) & (df["lat"] <= 60)
    ].copy()

# -------------------------------
# KMeans Clustering on NA only
# -------------------------------
@st.cache_data(show_spinner="Running KMeans clustering on North America…")
def run_kmeans_na(users_df: pd.DataFrame, n_clusters: int):

    # Filter NA only
    na_df = filter_north_america(users_df).dropna(subset=["lat", "lng"])
    if len(na_df) < n_clusters:
        raise ValueError(
            f"Not enough NA points ({len(na_df)}) for {n_clusters} clusters."
        )

    # ---- NEW FIX: Project to Web Mercator before clustering ----
    gdf = gpd.GeoDataFrame(
        na_df,
        geometry=[Point(xy) for xy in zip(na_df["lng"], na_df["lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Use meters instead of lat/lng degrees
    coord_matrix = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(coord_matrix)

    encoded_df = na_df[["id"]].copy()
    encoded_df["cluster"] = labels.astype(int)

    # Compute cluster centers in geographic space (median lat/lng)
    cluster_geo_summary = (
        na_df.merge(encoded_df, on="id")
        .groupby("cluster")
        .agg(
            median_lat=("lat", "median"),
            median_lng=("lng", "median"),
        )
        .reset_index()
    )

    return encoded_df, cluster_geo_summary


# ---------------------------------------
# MAIN PLOT FUNCTION
# ---------------------------------------
def make_cluster_figure(encoded_df, users_df, cluster_geo_summary,
                        max_points=200_000, zoom=4):

    df = encoded_df.merge(users_df, on="id", how="left")
    df = df.dropna(subset=["lat", "lng"])

    # plotting NA only
    df = filter_north_america(df)

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    if df.empty:
        raise ValueError("No NA user coordinates available for plotting.")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lng"], df["lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    n_clusters = gdf["cluster"].nunique()
    cmap = plt.get_cmap("tab10", n_clusters)
    boundaries = np.arange(n_clusters + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(20, 12), dpi=200)

    # Hard-coded NA bounds in WebMercator
    ax.set_xlim([-1.4e7, -7.5e6])
    ax.set_ylim([2.8e6, 6.5e6])

    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        crs=gdf.crs,
        zoom=zoom,
    )

    sc = ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf["cluster"],
        cmap=cmap,
        norm=norm,
        s=3,
        alpha=0.8,
        linewidths=0,
        zorder=3,
    )

    # Plot cluster centers
    centers_gdf = gpd.GeoDataFrame(
        cluster_geo_summary,
        geometry=[
            Point(xy)
            for xy in zip(cluster_geo_summary["median_lng"],
                          cluster_geo_summary["median_lat"])
        ],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    ax.set_axis_off()
    plt.title("User Clusters by Geographic Region", fontsize=18, pad=15)

    cbar = plt.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        pad=0.02,
        fraction=0.03,
        ticks=np.arange(n_clusters),
    )
    cbar.set_label("Cluster ID", fontsize=14)
    cbar.ax.tick_params(labelsize=10)

    return fig

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
def main():
    st.subheader("North America User Cluster Map")

    st.sidebar.header("Clustering settings")
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 15, 5)

    try:
        encoded_df, cluster_geo_summary = run_kmeans_na(users_df, n_clusters)
    except Exception as e:
        st.error(f"Error while clustering NA users: {e}")
        return

    try:
        fig = make_cluster_figure(
            encoded_df=encoded_df,
            users_df=users_df,
            cluster_geo_summary=cluster_geo_summary,
        )
    except Exception as e:
        st.error(f"Error while generating map: {e}")
        return

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    st.image(buf, use_container_width=True)

    st.download_button(
        "Download map as PNG",
        data=buf,
        file_name="na_user_clusters_map.png",
        mime="image/png",
    )

if __name__ == "__main__":
    main()
