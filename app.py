import streamlit as st
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Pitching Metric Correlations", layout="wide")
st.title("Pitching Metric Correlations")

DATA_DIR = Path(__file__).parent

ALL_METRICS = [
    "proStuff+", "proStuff_xRV",
    "tjStuff+", "ERA", "xERA", "FIP", "xFIP", "K-BB", "Stuff+", "Pitching+",
    "botOvr", "botStf", "SIERA", "RA9", "DRA", "cFIP",
    "StuffPro", "PitchPro", "RV100", "wOBA",
]

DEFAULT_TARGETS = ["ERA", "K-BB", "FIP", "xERA", "SIERA", "wOBA"]


# ---------------------------------------------------------------------------
# Weighted correlation helpers
# ---------------------------------------------------------------------------
def _wm(x, w):
    return np.sum(x * w) / np.sum(w)


def _wcov(x, y, w):
    return np.sum(w * (x - _wm(x, w)) * (y - _wm(y, w))) / np.sum(w)


def _wcorr(x, y, w):
    denom = np.sqrt(_wcov(x, x, w) * _wcov(y, y, w))
    return np.nan if denom == 0 else _wcov(x, y, w) / denom


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    tj24 = (
        pl.read_csv(DATA_DIR / "tjstats_pitcher_tjstuff__2024.csv")
        .rename({"pitcher_id": "MLBAMID", "stuff_overall": "tjStuff+"})
        .select(["MLBAMID", "tjStuff+"])
    )
    tj25 = (
        pl.read_csv(DATA_DIR / "tjstats_pitcher_tjstuff__2025.csv")
        .rename({"pitcher_id": "MLBAMID", "stuff_overall": "tjStuff+"})
        .select(["MLBAMID", "tjStuff+"])
    )

    fg_cols = [
        "MLBAMID", "IP", "ERA", "xERA", "FIP", "xFIP",
        "Stuff+", "Pitching+", "botOvr", "botStf", "SIERA", "K-BB",
    ]
    fg24 = pl.read_csv(DATA_DIR / "fg_2024.csv").select(fg_cols)
    fg25 = pl.read_csv(DATA_DIR / "fg_2025.csv").select(fg_cols)

    dra24 = (
        pl.read_csv(DATA_DIR / "dra_2024.csv")
        .rename({"mlbid": "MLBAMID"})
        .select(["MLBAMID", "RA9", "DRA", "cFIP"])
    )
    dra25 = (
        pl.read_csv(DATA_DIR / "dra_2025.csv")
        .rename({"mlbid": "MLBAMID"})
        .select(["MLBAMID", "RA9", "DRA", "cFIP"])
    )

    sp_rename = {
        "mlbid": "MLBAMID",
        "Pitch Type Probability": "ptp",
        "Surprise Factor": "sf",
        "Movement Spread": "mvmtsp",
        "Velocity Spread": "velosp",
    }
    sp24 = (
        pl.read_csv(DATA_DIR / "stuffpro_2024.csv")
        .rename(sp_rename)
        .select(["MLBAMID", "StuffPro", "PitchPro"])
    )
    sp25 = (
        pl.read_csv(DATA_DIR / "stuffpro_2025.csv")
        .rename(sp_rename)
        .select(["MLBAMID", "StuffPro", "PitchPro"])
    )

    sc24 = pl.read_csv(DATA_DIR / "statcast_2024.csv")
    sc25 = pl.read_csv(DATA_DIR / "statcast_2025.csv")

    # proStuff+ (Oracle Pitch Profiler)
    ps24 = (
        pl.read_csv(DATA_DIR / "prostuff_2024.csv")
        .rename({"mlbam_id": "MLBAMID", "stuff_plus": "proStuff+", "stuff_xRV": "proStuff_xRV"})
        .select(["MLBAMID", "proStuff+", "proStuff_xRV"])
    )
    ps25 = (
        pl.read_csv(DATA_DIR / "prostuff_2025.csv")
        .rename({"mlbam_id": "MLBAMID", "stuff_plus": "proStuff+", "stuff_xRV": "proStuff_xRV"})
        .select(["MLBAMID", "proStuff+", "proStuff_xRV"])
    )

    return (tj24, tj25), (fg24, fg25), (dra24, dra25), (sp24, sp25), (sc24, sc25), (ps24, ps25)


# ---------------------------------------------------------------------------
# Build merged frames
# ---------------------------------------------------------------------------
def _pd_merge_across(src24, src25):
    return src24.to_pandas().merge(
        src25.to_pandas(), on="MLBAMID", suffixes=["_2024", "_2025"]
    )


def build_across(data, ip_min_24, ip_min_25, weight_method):
    (tj24, tj25), (fg24, fg25), (dra24, dra25), (sp24, sp25), (sc24, sc25), (ps24, ps25) = data

    merged = pl.from_pandas(
        _pd_merge_across(tj24, tj25)
        .merge(_pd_merge_across(fg24, fg25), on="MLBAMID", how="inner")
        .merge(_pd_merge_across(dra24, dra25), on="MLBAMID", how="inner")
        .merge(_pd_merge_across(sp24, sp25), on="MLBAMID", how="inner")
        .merge(_pd_merge_across(sc24, sc25), on="MLBAMID", how="inner")
        .merge(_pd_merge_across(ps24, ps25), on="MLBAMID", how="inner")
    )

    merged = merged.filter(
        (pl.col("IP_2024") >= ip_min_24) & (pl.col("IP_2025") >= ip_min_25)
    )

    if weight_method == "Mean IP":
        w = (pl.col("IP_2024") + pl.col("IP_2025")) / 2
    elif weight_method == "Harmonic mean of sqrt(IP)":
        w = (
            pl.col("IP_2024").sqrt() * pl.col("IP_2025").sqrt()
            / (pl.col("IP_2024").sqrt() + pl.col("IP_2025").sqrt())
        )
    elif weight_method == "Harmonic mean of IP":
        w = (
            pl.col("IP_2024") * pl.col("IP_2025")
            / (pl.col("IP_2024") + pl.col("IP_2025"))
        )
    else:
        w = pl.lit(1.0)

    return merged.with_columns(w.alias("weight")).drop_nulls()


def build_same(data, ip_min_24, ip_min_25, weight_method):
    (tj24, tj25), (fg24, fg25), (dra24, dra25), (sp24, sp25), (sc24, sc25), (ps24, ps25) = data

    fg24_f = fg24.filter(pl.col("IP") >= ip_min_24)
    fg25_f = fg25.filter(pl.col("IP") >= ip_min_25)

    merged = pl.from_pandas(
        pl.concat([tj24, tj25]).to_pandas()
        .merge(pl.concat([fg24_f, fg25_f]).to_pandas(), on="MLBAMID", how="inner")
        .merge(pl.concat([dra24, dra25]).to_pandas(), on="MLBAMID", how="inner")
        .merge(pl.concat([sp24, sp25]).to_pandas(), on="MLBAMID", how="inner")
        .merge(pl.concat([sc24, sc25]).to_pandas(), on="MLBAMID", how="inner")
        .merge(pl.concat([ps24, ps25]).to_pandas(), on="MLBAMID", how="inner")
    )

    if weight_method == "IP":
        w = pl.col("IP")
    elif weight_method == "sqrt(IP)":
        w = pl.col("IP").sqrt()
    else:
        w = pl.lit(1.0)

    return merged.with_columns(w.alias("weight")).drop_nulls()


# ---------------------------------------------------------------------------
# Compute correlation
# ---------------------------------------------------------------------------
def compute_corr(x, y, w, use_spearman):
    if use_spearman:
        x = pd.Series(x).rank().to_numpy()
        y = pd.Series(y).rank().to_numpy()
    return _wcorr(x, y, w)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

st.sidebar.subheader("IP Filters")
ip_min_24 = st.sidebar.number_input("Min IP (2024)", 0, 300, 30, step=5)
ip_min_25 = st.sidebar.number_input("Min IP (2025)", 0, 300, 30, step=5)

corr_type = st.sidebar.radio("Correlation Type", ["Spearman", "Pearson"])
use_spearman = corr_type == "Spearman"

st.sidebar.subheader("Weighting")
same_weight = st.sidebar.selectbox(
    "Same Season Weight", ["sqrt(IP)", "IP", "None"], index=0
)
next_weight = st.sidebar.selectbox(
    "Next Season Weight",
    ["Harmonic mean of sqrt(IP)", "Mean IP", "Harmonic mean of IP", "None"],
    index=0,
)

st.sidebar.subheader("Row Metrics")
row_metrics = st.sidebar.multiselect(
    "Metrics to evaluate (rows)", ALL_METRICS, default=ALL_METRICS
)

st.sidebar.subheader("Next Season Columns")
next_targets = st.sidebar.multiselect(
    "Next season target outcomes", ALL_METRICS, default=DEFAULT_TARGETS
)
include_self = st.sidebar.checkbox("Include Next Season Self", value=True)
mean_next_options = next_targets if next_targets else []
mean_next = st.sidebar.multiselect(
    "Include in Next Season Mean",
    mean_next_options,
    default=[c for c in DEFAULT_TARGETS if c in mean_next_options],
)

st.sidebar.subheader("Same Season Columns")
same_targets = st.sidebar.multiselect(
    "Same season target outcomes", ALL_METRICS, default=DEFAULT_TARGETS
)
mean_same_options = same_targets if same_targets else []
mean_same = st.sidebar.multiselect(
    "Include in Same Season Mean",
    mean_same_options,
    default=[c for c in DEFAULT_TARGETS if c in mean_same_options],
)

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
if not row_metrics:
    st.warning("Select at least one row metric.")
    st.stop()

if not next_targets and not same_targets:
    st.warning("Select at least one target column.")
    st.stop()

data = load_data()
comps_across = build_across(data, ip_min_24, ip_min_25, next_weight)
comps_same = build_same(data, ip_min_24, ip_min_25, same_weight)

st.caption(f"Next season: {len(comps_across)} pitcher-pairs | Same season: {len(comps_same)} pitcher-seasons")

# Build result columns
result_columns = []
for t in next_targets:
    result_columns.append(f"Next {t}")
if mean_next:
    result_columns.append("Next Mean")
if include_self:
    result_columns.append("Next Self")
for t in same_targets:
    result_columns.append(f"Same {t}")
if mean_same:
    result_columns.append("Same Mean")

corr_df = pd.DataFrame(index=row_metrics, columns=result_columns, dtype=float)

w_across = comps_across["weight"].to_numpy()
w_same = comps_same["weight"].to_numpy()

for metric in row_metrics:
    # Next season correlations: metric_2024 vs target_2025
    for t in next_targets:
        col_label = f"Next {t}"
        x = comps_across[f"{metric}_2024"].to_numpy()
        y = comps_across[f"{t}_2025"].to_numpy()
        corr_df.loc[metric, col_label] = compute_corr(x, y, w_across, use_spearman)

    # Next season self
    if include_self:
        x = comps_across[f"{metric}_2024"].to_numpy()
        y = comps_across[f"{metric}_2025"].to_numpy()
        corr_df.loc[metric, "Next Self"] = compute_corr(x, y, w_across, use_spearman)

    # Same season correlations
    for t in same_targets:
        col_label = f"Same {t}"
        x = comps_same[metric].to_numpy()
        y = comps_same[t].to_numpy()
        corr_df.loc[metric, col_label] = compute_corr(x, y, w_same, use_spearman)

# Compute means
if mean_next:
    mean_cols = [f"Next {t}" for t in mean_next]
    corr_df["Next Mean"] = corr_df[mean_cols].mean(axis=1)

if mean_same:
    mean_cols = [f"Same {t}" for t in mean_same]
    corr_df["Same Mean"] = corr_df[mean_cols].mean(axis=1)

# Take absolute values and round
display_df = corr_df.abs().astype(float).round(3)

# Sort controls
sort_col = st.selectbox(
    "Sort by",
    display_df.columns.tolist(),
    index=0,
)
sort_asc = st.checkbox("Ascending", value=False)
display_df = display_df.sort_values(by=sort_col, ascending=sort_asc)

# Style: separate background gradients for next-season, self, and same-season groups
next_group = [c for c in display_df.columns if c.startswith("Next ") and c != "Next Self" and c != "Next Mean"]
next_extra = [c for c in ["Next Mean"] if c in display_df.columns]
self_group = [c for c in ["Next Self"] if c in display_df.columns]
same_group = [c for c in display_df.columns if c.startswith("Same ") and c != "Same Mean"]
same_extra = [c for c in ["Same Mean"] if c in display_df.columns]

styler = display_df.style.format("{:.3f}")

if next_group + next_extra:
    styler = styler.background_gradient(axis=0, cmap="Reds", subset=next_group + next_extra)
if self_group:
    styler = styler.background_gradient(axis=0, cmap="Reds", subset=self_group)
if same_group + same_extra:
    styler = styler.background_gradient(axis=0, cmap="Reds", subset=same_group + same_extra)

# Caption
weight_desc_same = same_weight if same_weight != "None" else "unweighted"
weight_desc_next = next_weight if next_weight != "None" else "unweighted"
styler = styler.set_caption(
    f"2024/2025 {corr_type} Correlations | "
    f"Same season: {weight_desc_same} | Next season: {weight_desc_next} | "
    f"IP filter: {ip_min_24}/{ip_min_25}"
)

st.dataframe(styler, use_container_width=True, height=35 * len(row_metrics) + 60)
