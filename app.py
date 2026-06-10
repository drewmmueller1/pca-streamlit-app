import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (confusion_matrix, accuracy_score, silhouette_score,
                             mean_squared_error, classification_report,
                             roc_curve, auc, roc_auc_score)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_valid_hex(s):
    s = s.strip().lstrip('#')
    return len(s) == 6 and all(c in '0123456789abcdefABCDEF' for c in s)

DEFAULT_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#66c2a5','#fc8d62',
    '#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494',
    '#b3b3b3','#1b9e77','#d95f02','#7570b3','#e7298a',
]

def apply_white_theme(fig):
    """
    Force white background + black text for publication-ready downloads.
    Only active when the Publication / White Background Mode toggle is ON.
    When OFF, returns the figure unchanged (browser theme used instead).
    """
    if not use_white_theme:
        return fig
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(
            gridcolor='#e0e0e0',
            zerolinecolor='#cccccc',
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
        ),
        yaxis=dict(
            gridcolor='#e0e0e0',
            zerolinecolor='#cccccc',
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
        ),
        legend=dict(
            font=dict(color='black'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
        ),
        title_font=dict(color='black'),
        coloraxis_colorbar=dict(
            tickfont=dict(color='black'),
            title_font=dict(color='black'),
        ),
    )
    for key in list(fig.layout):
        if key.startswith('xaxis') or key.startswith('yaxis'):
            try:
                fig.layout[key].update(
                    gridcolor='#e0e0e0',
                    zerolinecolor='#cccccc',
                    tickfont=dict(color='black'),
                    title_font=dict(color='black'),
                )
            except Exception:
                pass
    return fig

def diagnose_pca(X_pca, var_ratios):
    issues = []
    check_pcs = min(X_pca.shape[1], 5)
    for i in range(check_pcs):
        col, col_std = X_pca[:, i], np.std(X_pca[:, i])
        if col_std < 1e-8:
            issues.append({'pc': i+1, 'type': 'zero_variance', 'detail': (
                f"**PC{i+1} has essentially zero variance** (std ≈ {col_std:.2e}). "
                "All points collapse onto this axis.\n\n**Likely causes:**\n"
                "- A constant or near-constant feature column.\n- Duplicate columns.\n"
                "- Normalization made all samples identical for one variable.\n\n**Fix:**\n"
                "1. Check raw data for constant columns and remove them.\n"
                "2. Inspect the loadings for this PC.")})
        elif var_ratios[i] < 0.001 and i < 3:
            issues.append({'pc': i+1, 'type': 'near_zero_variance', 'detail': (
                f"**PC{i+1} explains < 0.1% of variance** ({var_ratios[i]:.4%}). "
                "Points will appear compressed.\n\n**Likely causes:**\n"
                "- One feature with a much larger numeric scale dominating the covariance matrix.\n"
                "- Near-duplicate or highly correlated features.\n\n**Fix:**\n"
                "1. Ensure Z-score standardization is applied before decomposition.\n"
                "2. Check the Scree Plot for a single dominant PC.")})
    for i in range(check_pcs):
        col = X_pca[:, i]
        if np.std(col) < 1e-8: continue
        span = col.max() - col.min()
        if span < 1e-8: continue
        near_min = np.sum(np.abs(col - col.min()) < 0.05 * span)
        near_max = np.sum(np.abs(col - col.max()) < 0.05 * span)
        if (near_min + near_max) / len(col) >= 0.80 and len(col) > 5:
            issues.append({'pc': i+1, 'type': 'axis_clustering', 'detail': (
                f"**PC{i+1} shows strong axis-end clustering** — most points pile up at the extremes.\n\n"
                "**Likely causes:**\n- A binary or near-binary variable dominating this component.\n"
                "- Severe class imbalance.\n\n**Fix:**\n"
                f"1. Inspect the PC{i+1} loadings — the top variable is almost certainly the cause.\n"
                "2. Consider removing or re-encoding that variable.")})
    return issues

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.title("Multivariate Analysis App for Lab Data")
st.markdown("""
Upload a CSV file with:
- **Row-wise:** Each row is a sample; first column must be named `label`.
- **Column-wise:** Each column is a sample; first row has sample names, column 1 has variable names. Enable *Transpose* in Data Prep.

Replicate measurements should share a name prefix separated by `_` (e.g. `Sample1_1`, `Sample1_2`).
""")

# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOADER
# ══════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader("Upload feature CSV", type="csv")
if uploaded_file is None:
    st.info("Please upload a feature CSV to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
st.subheader("Raw Data Preview")
st.dataframe(df.head())
if 'label' not in df.columns:
    st.warning("No 'label' column found — will be generated from column names if Transpose is enabled.")

# ── Publication theme toggle (must be defined before any chart is drawn) ─────
use_white_theme = st.toggle(
    "📄 Publication / White Background Mode",
    value=False,
    help=(
        "**OFF (default):** plots use your browser's theme — dark mode friendly, "
        "great for exploratory analysis on screen.\n\n"
        "**ON:** all plots switch to a white background with black text and light gray gridlines. "
        "Turn this on when preparing figures for reports, publications, or presentations "
        "where a clean white background is required. "
        "Downloaded plots will match exactly what you see on screen."
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – ANALYSIS MODE
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Analysis Mode", expanded=True):
    analysis_mode = st.radio(
        "Decomposition method:",
        ["PCA (Principal Component Analysis)",
         "PCR (Principal Component Regression)",
         "PLS (Partial Least Squares)"],
        index=0,
        help=(
            "**PCA:** Unsupervised — finds directions of maximum variance in X. "
            "Use when you want to explore structure with no target variable.\n\n"
            "**PCR:** Supervised regression — runs PCA on X first, then regresses "
            "selected PCs against a continuous Y target. Good when X is highly collinear.\n\n"
            "**PLS:** Supervised — finds latent components that maximally co-vary with Y. "
            "Generally outperforms PCR when the target is strongly related to specific features.\n\n"
            "**None:** Skips decomposition entirely. Applies normalization and standardization, "
            "then runs classification directly on the standardized features. "
            "Use this as a baseline to check whether PCA/PCR/PLS actually improves accuracy."
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – DATA PREP
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Data Prep Options", expanded=False):
    transpose_data = st.checkbox(
        "Transpose Dataset (samples are columns)", value=False,
        help="Swaps rows/cols. Use when samples are columns and features are rows."
    )
    preprocess_option = st.radio(
        "Standardization (applied last, after normalization):",
        ['None', 'SNV', 'Z-score'], index=2,
        help=(
            "**SNV (Standard Normal Variate):** Centers and scales each *sample* (row) by its own "
            "mean and standard deviation. Best when all variables share the same unit "
            "(e.g., absorbance across wavelengths) and you want to remove sample-to-sample "
            "baseline/intensity differences.\n\n"
            "**Z-score:** Centers and scales each *variable* (column) across all samples using "
            "the population mean and std. Best when variables are on widely different scales "
            "(e.g., mixing concentrations in mM with temperatures in °C), so every variable "
            "contributes equally to the decomposition."
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Normalization Options", expanded=True):
    st.markdown("Applied **before** standardization.")
    norm_option = st.radio(
        "Normalization method:",
        ['None', 'Unit Area (Total Sum)', 'Unit Vector (L2 Norm)',
         'Min-Max (per sample)', 'Internal Standard'],
        index=0,
        help=(
            "**Unit Area:** divides each sample by its total sum — removes concentration differences.\n\n"
            "**Unit Vector (L2):** divides each sample by its Euclidean length — scale-invariant.\n\n"
            "**Min-Max:** rescales each sample to [0, 1] using its own min and max.\n\n"
            "**Internal Standard:** each variable in a sample is divided by the IS row value "
            "for that same variable. Select the IS row in the selector below."
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – SPARSE / BINARY ANALYSIS (Hierarchical clustering + presence heatmap)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Sparse / Binary Analysis", expanded=False):
    run_hierarchical = st.checkbox(
        "Run Hierarchical Clustering + Presence Heatmap", value=False,
        help=(
            "Designed for sparse presence/absence data (e.g. drug profiling where most "
            "compounds are zero in most samples). Runs on raw detected/not-detected patterns "
            "BEFORE any standardization, so results are valid regardless of standardization choice."
        )
    )
    if run_hierarchical:
        hc_distance = st.radio(
            "Binary distance metric:",
            ["Jaccard", "Dice"],
            index=0,
            help=(
                "**Jaccard:** treats shared presence and shared absence symmetrically — "
                "use when a compound being absent in both samples is not meaningful similarity.\n\n"
                "**Dice:** gives double weight to shared presences (co-occurring compounds) — "
                "use when two samples sharing the same detected compounds matters more than "
                "them both lacking a compound."
            )
        )
        hc_linkage = st.selectbox(
            "Linkage method:",
            ["average", "complete", "ward", "single"],
            index=0,
            help="How clusters are merged. 'average' and 'complete' work well for binary distances. "
                 "'ward' minimizes within-cluster variance (works on the distance matrix here)."
        )
        hc_n_clusters = st.slider(
            "Number of clusters to highlight", 2, 30, 4,
            help="Colors the dendrogram and heatmap cluster groups by this many clusters."
        )
        hc_presence_threshold = st.number_input(
            "Presence threshold (value > this = 'detected')",
            value=0.0, step=1.0,
            help="Any value strictly greater than this is treated as 'present/detected' (1), "
                 "everything else as 'absent' (0). Usually 0 for peak-area data."
        )
    else:
        hc_distance = "Jaccard"; hc_linkage = "average"
        hc_n_clusters = 4; hc_presence_threshold = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# TRANSPOSE
# ══════════════════════════════════════════════════════════════════════════════
if transpose_data:
    if df.shape[1] < 2:
        st.error("Dataset too narrow for transpose.")
        st.stop()
    features      = df.iloc[:, 0].values
    data          = df.iloc[:, 1:].T.copy()
    data.columns  = features
    sample_names  = df.columns[1:]
    data['label'] = [name.split('_')[0] for name in sample_names]
    df = data.reset_index(drop=True)
    st.success(f"Transposed: {df.shape[0]} samples × {df.shape[1]-1} features.")
else:
    if 'label' not in df.columns:
        st.error("CSV must have a 'label' column (or enable Transpose).")
        st.stop()

df['label'] = df['label'].astype(str).str.split('_').str[0]
st.info(f"Simplified labels: {df['label'].nunique()} unique classes")

X_raw = df.drop('label', axis=1).select_dtypes(include=[np.number])
y_raw = df['label']
if X_raw.empty:
    st.error("No numerical columns found.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL STANDARD ROW SELECTOR
# (IS = a specific ROW in the data, identified by its label value)
# After transpose, column-wise data becomes row-wise, so this works uniformly.
# ══════════════════════════════════════════════════════════════════════════════
is_label_val = None
if norm_option == 'Internal Standard':
    is_label_val = st.sidebar.selectbox(
        "Internal Standard row (label)",
        options=sorted(y_raw.unique()),
        help=(
            "Select the label that identifies your Internal Standard row(s). "
            "For each variable (column), all samples are divided by the IS value "
            "for that variable — i.e., sample[col] / IS[col] for every column independently. "
            "If multiple IS replicates exist, their values are averaged per column first. "
            "This works identically whether your original data was row-wise or column-wise "
            "(after transpose they are both row-wise at this point)."
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION (per-sample, before standardization)
# ══════════════════════════════════════════════════════════════════════════════
# Always cast to float64 first — integer columns (e.g. peak areas) can overflow
# when squared (L2) or produce integer-division truncation in other methods.
X = X_raw.astype(float).copy()

if norm_option != 'None':
    if norm_option == 'Unit Area (Total Sum)':
        # Row sums — vectorized: divide each row by its own total
        row_sums = X.sum(axis=1)                      # Series, one value per sample
        zero_rows = row_sums[row_sums == 0].index
        if len(zero_rows) > 0:
            st.warning(f"{len(zero_rows)} sample(s) have row sum = 0 — Unit Area skipped for those rows.")
        # Replace zeros with 1 so division is a no-op for zero-sum rows
        row_sums_safe = row_sums.replace(0, 1)
        X = X.div(row_sums_safe, axis=0)
        st.success("Normalization applied: Unit Area / Total Sum (per sample)")

    elif norm_option == 'Unit Vector (L2 Norm)':
        # L2 norm per row — vectorized
        row_l2 = np.sqrt((X ** 2).sum(axis=1))        # Series, one value per sample
        zero_rows = row_l2[row_l2 == 0].index
        if len(zero_rows) > 0:
            st.warning(f"{len(zero_rows)} sample(s) have L2 = 0 — Unit Vector skipped for those rows.")
        row_l2_safe = row_l2.replace(0, 1)
        X = X.div(row_l2_safe, axis=0)
        st.success("Normalization applied: Unit Vector / L2 Norm (per sample)")

    elif norm_option == 'Min-Max (per sample)':
        # Min and max per row — vectorized
        row_min  = X.min(axis=1)
        row_max  = X.max(axis=1)
        row_span = row_max - row_min
        const_rows = row_span[row_span == 0].index
        if len(const_rows) > 0:
            st.warning(f"{len(const_rows)} sample(s) are constant — Min-Max skipped for those rows.")
        row_span_safe = row_span.replace(0, 1)
        X = X.sub(row_min, axis=0).div(row_span_safe, axis=0)
        st.success("Normalization applied: Min-Max (per sample)")

    elif norm_option == 'Internal Standard':
        if is_label_val is None:
            st.error("Select an Internal Standard label in the sidebar.")
            st.stop()
        is_mask = (y_raw == is_label_val)
        if is_mask.sum() == 0:
            st.error(f"No rows found with label '{is_label_val}'.")
            st.stop()
        # Average IS rows → one IS value per variable (column)
        is_values = X.loc[is_mask].mean(axis=0)       # Series, one value per column
        zero_cols = is_values[is_values == 0].index.tolist()
        if zero_cols:
            st.warning(
                f"{len(zero_cols)} variable(s) have IS value = 0 and will NOT be normalized: "
                f"{zero_cols[:5]}{'...' if len(zero_cols) > 5 else ''}"
            )
        # Divide each column by its IS value (leave zero-IS columns untouched)
        is_values_safe = is_values.copy()
        is_values_safe[is_values_safe == 0] = 1       # no-op divisor for zero-IS columns
        X = X.div(is_values_safe, axis=1)
        st.success(
            f"Normalization applied: Internal Standard "
            f"(IS label = '{is_label_val}', {is_mask.sum()} IS row(s) averaged per variable)"
        )

y = y_raw.copy()

# ══════════════════════════════════════════════════════════════════════════════
# STANDARDIZATION (applied after normalization)
# ══════════════════════════════════════════════════════════════════════════════
if preprocess_option == 'SNV':
    # Per-sample (row-wise): center and scale each sample by its own mean and std — vectorized
    row_mean = X.mean(axis=1)
    row_std  = X.std(axis=1)
    zero_std = row_std[row_std == 0].index
    if len(zero_std) > 0:
        st.warning(f"{len(zero_std)} sample(s) have zero variance — SNV skipped for those rows.")
    row_std_safe = row_std.replace(0, 1)
    X_snv    = X.sub(row_mean, axis=0).div(row_std_safe, axis=0)
    X_scaled = X_snv.values
    st.success("Standardization applied: SNV (per-sample, row-wise)")

elif preprocess_option == 'Z-score':
    # Per-variable (column-wise): center and scale each variable by population mean and std
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.success("Standardization applied: Z-score (per-variable, population-wise)")

else:
    # No standardization — still Z-score before decomposition to keep PCA meaningful
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.info("No standardization selected — Z-score applied automatically before decomposition to ensure equal feature weighting.")

# ── Processed Data Overview ───────────────────────────────────────────────────
# Build a human-readable label for what was applied
_steps = []
if norm_option != 'None':
    _steps.append(norm_option)
if preprocess_option != 'None':
    _steps.append(f"{preprocess_option} standardization")
else:
    _steps.append("Z-score standardization (auto-applied)")
_pipeline_label = " → ".join(_steps)

st.subheader("Processed Data Overview")
st.caption(
    f"Pipeline applied: **{_pipeline_label}**. "
    "This table shows the exact values that will be passed to PCA / PCR / PLS. "
    "All values are shown before the Data Filtering exclusion step below."
)
# Reconstruct as a readable DataFrame: label column + processed feature columns
_X_processed_df = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)
_X_processed_df.insert(0, 'label', y_raw)   # use y_raw (before filtering) to match full index
st.dataframe(_X_processed_df.head(10).style.format("{:.4f}", subset=X.columns.tolist()))
st.caption(
    f"Showing first 10 rows of {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features. "
    "Download the full processed dataset from the Download Results section below."
)

# ══════════════════════════════════════════════════════════════════════════════
# SPARSE / BINARY ANALYSIS — Hierarchical clustering + presence/absence heatmap
# Placed AFTER the Processed Data Overview.
# Presence/absence is derived from the RAW detection pattern (a peak was either
# detected or not), which is the scientifically correct basis for binary distance —
# standardization/normalization changes magnitudes but not whether a compound was
# detected. The note below makes this explicit to the user.
# ══════════════════════════════════════════════════════════════════════════════
if run_hierarchical:
    st.header("Hierarchical Clustering & Presence/Absence Analysis")
    st.caption(
        "Presence/absence is determined from the **raw detection pattern** (value above the "
        "threshold = detected), independent of normalization/standardization — because a compound "
        "is either detected or not regardless of scaling. This keeps the binary distance valid "
        "even when standardization is set to 'None'."
    )

    # Build presence/absence (binary) matrix from RAW data
    X_binary = (X_raw.astype(float) > hc_presence_threshold).astype(int)

    # Drop variables that are all-zero or all-one (no information, break binary distances)
    col_sums = X_binary.sum(axis=0)
    informative_cols = col_sums[(col_sums > 0) & (col_sums < X_binary.shape[0])].index
    n_dropped = X_binary.shape[1] - len(informative_cols)
    X_binary_info = X_binary[informative_cols]

    # Drop samples with zero detections (can't compute binary distance)
    row_sums = X_binary_info.sum(axis=1)
    valid_rows = row_sums[row_sums > 0].index
    n_empty_samples = X_binary_info.shape[0] - len(valid_rows)
    X_binary_valid = X_binary_info.loc[valid_rows]
    y_valid        = y_raw.loc[valid_rows]

    info_bits = [f"{X_binary_valid.shape[0]} samples × {X_binary_valid.shape[1]} informative variables"]
    if n_dropped > 0:
        info_bits.append(f"{n_dropped} non-informative variable(s) dropped (all-present or all-absent)")
    if n_empty_samples > 0:
        info_bits.append(f"{n_empty_samples} sample(s) with no detections dropped")
    st.info(" | ".join(info_bits))

    if X_binary_valid.shape[0] < 3:
        st.warning("Fewer than 3 valid samples after filtering — cannot cluster.")
    else:
        metric_name = hc_distance.lower()  # 'jaccard' or 'dice'
        try:
            dist_condensed = pdist(X_binary_valid.values, metric=metric_name)
            if np.isnan(dist_condensed).any():
                dist_condensed = np.nan_to_num(dist_condensed, nan=1.0)

            Z = linkage(dist_condensed, method=hc_linkage)

            # Guard cluster count against number of samples
            max_clusters_possible = X_binary_valid.shape[0]
            eff_n_clusters = min(hc_n_clusters, max_clusters_possible)
            if eff_n_clusters < hc_n_clusters:
                st.warning(f"Reduced clusters from {hc_n_clusters} to {eff_n_clusters} "
                           f"(only {max_clusters_possible} samples).")

            cluster_ids = fcluster(Z, t=eff_n_clusters, criterion='maxclust')

            # Leaf order from the dendrogram (shared by all three figures)
            dendro_data  = dendrogram(Z, no_plot=True)
            dendro_order = dendro_data['leaves']
            X_ordered    = X_binary_valid.iloc[dendro_order]
            y_ordered    = y_valid.iloc[dendro_order]
            clusters_ord = cluster_ids[dendro_order]
            col_order    = X_ordered.sum(axis=0).sort_values(ascending=False).index
            X_ordered    = X_ordered[col_order]
            n_samples_hc = X_ordered.shape[0]
            n_vars_hc    = X_ordered.shape[1]

            from matplotlib.colors import ListedColormap, to_hex
            from matplotlib.patches import Patch

            # ── Cluster colors: fixed, maximally-distinct palette ─────────────
            # Hand-picked high-contrast colors; cycles only if clusters exceed the list.
            DISTINCT_CLUSTER_COLORS = [
                '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080',
                '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000',
                '#000075', '#ffd8b1', '#000000', '#a9a9a9', '#ffe119',
                '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
                '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
            ]
            cluster_hex = [DISTINCT_CLUSTER_COLORS[i % len(DISTINCT_CLUSTER_COLORS)]
                           for i in range(eff_n_clusters)]

            txt_color = 'black' if use_white_theme else None

            # ══════════════════════════════════════════════════════════════════
            # 1. STANDALONE DENDROGRAM (square-ish, clear)
            # ══════════════════════════════════════════════════════════════════
            st.subheader(f"Dendrogram ({hc_distance} distance, {hc_linkage} linkage)")

            # A link_color_func that colors branches by cluster membership
            color_thresh = None
            if eff_n_clusters > 1 and len(Z) >= (eff_n_clusters - 1):
                color_thresh = float(Z[-(eff_n_clusters - 1), 2])

            # Map scipy's default coloring to our cluster colors via set_link_color_palette
            from scipy.cluster.hierarchy import set_link_color_palette
            set_link_color_palette([c for c in cluster_hex])

            # Equal, generous dimensions; width grows slightly with sample count but stays balanced
            d_w = float(np.clip(n_samples_hc * 0.10, 8, 20))
            d_h = float(np.clip(d_w * 0.55, 5, 11))
            fig_d, ax_d = plt.subplots(figsize=(d_w, d_h))
            if use_white_theme:
                fig_d.patch.set_facecolor('white'); ax_d.set_facecolor('white')

            show_leaf_labels = n_samples_hc <= 80
            dkw = dict(
                ax=ax_d,
                leaf_rotation=90,
                leaf_font_size=max(5, min(9, 700 // max(n_samples_hc, 1))),
                color_threshold=color_thresh,
                above_threshold_color='#999999',
            )
            if color_thresh is None:
                dkw.pop('color_threshold')
            if show_leaf_labels:
                dkw['labels'] = list(y_ordered.values) if False else list(y_valid.values)
            else:
                dkw['no_labels'] = True
            dendrogram(Z, **dkw)
            set_link_color_palette(None)  # reset global palette

            if txt_color:
                ax_d.set_title(f"Hierarchical Clustering — {hc_distance}, {hc_linkage}", color=txt_color)
                ax_d.set_ylabel("Distance", color=txt_color)
                ax_d.tick_params(colors=txt_color)
                for sp in ax_d.spines.values(): sp.set_edgecolor('#cccccc')
            else:
                ax_d.set_title(f"Hierarchical Clustering — {hc_distance}, {hc_linkage}")
                ax_d.set_ylabel("Distance")
            st.pyplot(fig_d, bbox_inches='tight'); plt.close(fig_d)
            if not show_leaf_labels:
                st.caption(f"Leaf labels hidden ({n_samples_hc} samples too many to display legibly). "
                           "Use the Cluster Assignments table below to identify samples.")

            # ══════════════════════════════════════════════════════════════════
            # 2. STANDALONE INTERACTIVE HEATMAP (Plotly)
            # ══════════════════════════════════════════════════════════════════
            st.subheader("Presence/Absence Heatmap (interactive)")
            fig_heat = go.Figure(go.Heatmap(
                z=X_ordered.values,
                x=list(X_ordered.columns),
                y=[str(lbl) for lbl in y_ordered.values],
                colorscale=[[0, '#f3f3f3'], [1, '#2166ac']],
                showscale=True,
                colorbar=dict(title="Detected", tickvals=[0, 1], ticktext=["Absent", "Present"]),
                hovertemplate="Sample: %{y}<br>Variable: %{x}<br>%{z}<extra></extra>",
                xgap=0.5, ygap=0.3,
            ))
            fig_heat.update_layout(
                title="Compound Presence/Absence Across Samples (clustered order)",
                xaxis_title="Variables (compounds)",
                yaxis_title="Samples (ordered by dendrogram)",
                height=max(500, n_samples_hc * 13),
                xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=7), autorange='reversed'),
            )
            apply_white_theme(fig_heat)
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                "Interactive: hover any cell for sample/variable detail, zoom and pan freely. "
                "Rows are in dendrogram order; columns are ordered by detection frequency. "
                + ("Dice gives 2× weight to shared detections relative to Jaccard."
                   if hc_distance == "Dice"
                   else "Jaccard weights shared presence and absence symmetrically.")
            )

            # ══════════════════════════════════════════════════════════════════
            # 3. COMBINED DENDROGRAM + HEATMAP (aligned)
            # ══════════════════════════════════════════════════════════════════
            st.subheader("Combined Dendrogram + Heatmap")
            from matplotlib.gridspec import GridSpec

            fig_h2 = float(np.clip(n_samples_hc * 0.16, 6, 24))
            fig_w2 = float(np.clip(n_vars_hc * 0.22 + 4, 9, 26))
            show_sample_labels = n_samples_hc <= 60
            show_var_labels    = n_vars_hc <= 80

            fig_cm = plt.figure(figsize=(fig_w2, fig_h2))
            gs = GridSpec(1, 3, width_ratios=[0.20, 0.03, 0.77], wspace=0.02, figure=fig_cm)
            if use_white_theme:
                fig_cm.patch.set_facecolor('white')

            # Left: horizontal dendrogram
            ax_dend = fig_cm.add_subplot(gs[0, 0])
            if use_white_theme: ax_dend.set_facecolor('white')
            set_link_color_palette([c for c in cluster_hex])
            dkw2 = dict(orientation='left', ax=ax_dend, no_labels=True,
                        color_threshold=color_thresh, above_threshold_color='#999999')
            if color_thresh is None: dkw2.pop('color_threshold')
            dendrogram(Z, **dkw2)
            set_link_color_palette(None)
            ax_dend.invert_yaxis()
            ax_dend.set_xticks([]); ax_dend.set_yticks([])
            for sp in ax_dend.spines.values(): sp.set_visible(False)
            if txt_color:
                ax_dend.set_title("Dendrogram", fontsize=10, color=txt_color)
            else:
                ax_dend.set_title("Dendrogram", fontsize=10)

            # Middle: cluster color strip using resolved cluster_hex
            ax_strip = fig_cm.add_subplot(gs[0, 1])
            strip = (clusters_ord - 1).reshape(-1, 1)
            ax_strip.imshow(strip, aspect='auto',
                            cmap=ListedColormap(cluster_hex),
                            interpolation='nearest', vmin=0, vmax=eff_n_clusters - 1)
            ax_strip.set_xticks([]); ax_strip.set_yticks([])
            for sp in ax_strip.spines.values(): sp.set_visible(False)
            if txt_color:
                ax_strip.set_title("Cluster", fontsize=9, color=txt_color)
            else:
                ax_strip.set_title("Cluster", fontsize=9)

            # Right: heatmap
            ax_heat = fig_cm.add_subplot(gs[0, 2])
            ax_heat.imshow(X_ordered.values, aspect='auto',
                           cmap=ListedColormap(['#f3f3f3', '#2166ac']),
                           interpolation='nearest', vmin=0, vmax=1)
            if show_var_labels:
                ax_heat.set_xticks(range(n_vars_hc))
                ax_heat.set_xticklabels(list(X_ordered.columns), rotation=90,
                                        fontsize=max(5, min(9, 700 // max(n_vars_hc, 1))),
                                        color=txt_color if txt_color else 'black')
            else:
                ax_heat.set_xticks([])
                ax_heat.set_xlabel(f"{n_vars_hc} variables", fontsize=9,
                                   color=txt_color if txt_color else 'black')
            if show_sample_labels:
                ax_heat.set_yticks(range(n_samples_hc))
                ax_heat.set_yticklabels(list(y_ordered.values),
                                        fontsize=max(5, min(9, 600 // max(n_samples_hc, 1))),
                                        color=txt_color if txt_color else 'black')
            else:
                ax_heat.set_yticks([])
                ax_heat.set_ylabel(f"{n_samples_hc} samples", fontsize=9,
                                   color=txt_color if txt_color else 'black')
            ax_heat.yaxis.tick_right(); ax_heat.yaxis.set_label_position('right')
            if txt_color: ax_heat.tick_params(colors=txt_color)

            legend_elems = [
                Patch(facecolor='#2166ac', label='Detected'),
                Patch(facecolor='#f3f3f3', edgecolor='#cccccc', label='Not detected'),
            ]
            leg = ax_heat.legend(handles=legend_elems, loc='upper left',
                                 bbox_to_anchor=(1.06, 1.0), fontsize=8,
                                 frameon=True, title="Presence")
            if txt_color:
                leg.get_title().set_color(txt_color)
                for t in leg.get_texts(): t.set_color(txt_color)

            sup_color = txt_color if txt_color else 'black'
            fig_cm.suptitle(
                f"Clustered Presence/Absence Map "
                f"({hc_distance}, {hc_linkage}, {eff_n_clusters} clusters)",
                fontsize=12, color=sup_color, y=0.995
            )
            st.pyplot(fig_cm, bbox_inches='tight'); plt.close(fig_cm)
            st.caption(
                "Left: dendrogram. Middle strip: cluster membership (colors match your selection). "
                "Right: presence/absence heatmap aligned to the dendrogram rows. "
                + ("" if show_sample_labels else
                   f"Sample labels hidden ({n_samples_hc} too many) — see Cluster Assignments below.")
            )

            # ══════════════════════════════════════════════════════════════════
            # Cluster membership table + download
            # ══════════════════════════════════════════════════════════════════
            st.subheader("Cluster Assignments")
            df_clusters = pd.DataFrame({
                'Sample': y_valid.values,
                'Cluster': cluster_ids,
                'N_compounds_detected': X_binary_valid.sum(axis=1).values,
            }).sort_values(['Cluster', 'Sample'])
            st.dataframe(df_clusters, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Cluster Assignments CSV",
                df_clusters.to_csv(index=False),
                "hierarchical_clusters.csv", "text/csv",
                help="Cluster ID and compound count for each sample."
            )

        except Exception as e:
            st.error(f"Hierarchical clustering failed: {e}")

    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DATA FILTERING
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Data Filtering")
unique_labels   = sorted(y.unique())
excluded_labels = st.multiselect("Labels to exclude", unique_labels, default=[])
mask_include    = ~y.isin(excluded_labels)
X_scaled        = X_scaled[mask_include]
X_df_filt       = X.loc[mask_include]   # keep pandas version for IS label in pipeline card
y               = y[mask_include].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# PCR / PLS TARGET VARIABLE (for supervised modes)
# ══════════════════════════════════════════════════════════════════════════════
y_target = None
if analysis_mode != "PCA (Principal Component Analysis)":
    st.subheader("Regression Target Variable")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for regression target.")
        st.stop()
    target_col = st.selectbox(
        "Select target variable (Y) for regression:",
        numeric_cols,
        help="This column will be used as the Y vector for PCR or PLS regression."
    )
    y_target_full = df[target_col].values
    y_target = y_target_full[mask_include]
    st.info(f"Target: **{target_col}** | Range: {y_target.min():.3g} – {y_target.max():.3g} | Mean: {y_target.mean():.3g}")

# ══════════════════════════════════════════════════════════════════════════════
# DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
pca_full   = None
pls_model  = None
X_scores   = None
var_ratios = None
component_label = "PC"

n_max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])

if analysis_mode == "PCA (Principal Component Analysis)":
    pca_full   = PCA()
    X_scores   = pca_full.fit_transform(X_scaled)
    var_ratios = pca_full.explained_variance_ratio_
    n_total_pcs = X_scores.shape[1]
    component_label = "PC"

elif analysis_mode == "PCR (Principal Component Regression)":
    # Step 1: full PCA on X
    pca_full   = PCA()
    X_scores   = pca_full.fit_transform(X_scaled)
    var_ratios = pca_full.explained_variance_ratio_
    n_total_pcs = X_scores.shape[1]
    component_label = "PC"

elif analysis_mode == "PLS (Partial Least Squares)":
    if y_target is None:
        st.error("PLS requires a target variable — select one above.")
        st.stop()
    with st.sidebar.expander("PLS Options", expanded=True):
        n_pls_components = st.slider(
            "Number of PLS components", 2, min(20, n_max_components), 5,
            help="Number of latent variables for PLS. Use the CV plot to choose."
        )
    pls_model  = PLSRegression(n_components=n_pls_components, scale=False)
    pls_model.fit(X_scaled, y_target)
    X_scores   = pls_model.x_scores_
    # Approximate variance explained per LV from x_weights
    ss_total  = np.sum(X_scaled**2)
    var_ratios = np.array([
        np.sum((X_scaled @ pls_model.x_weights_[:, i:i+1] @ pls_model.x_weights_[:, i:i+1].T)**2) / ss_total
        for i in range(n_pls_components)
    ])
    var_ratios = np.clip(var_ratios, 0, 1)
    n_total_pcs = n_pls_components
    component_label = "LV"

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════

cum_var = np.cumsum(var_ratios)
n_95  = int(np.argmax(cum_var >= 0.95)) + 1 if np.any(cum_var >= 0.95) else n_total_pcs
n_99  = int(np.argmax(cum_var >= 0.99)) + 1 if np.any(cum_var >= 0.99) else n_total_pcs

norm_detail_map = {
    'None':                  "None — raw data passed to standardization.",
    'Unit Area (Total Sum)': "Unit Area: each sample ÷ its own total sum (per-sample).",
    'Unit Vector (L2 Norm)': "Unit Vector (L2): each sample ÷ its own Euclidean length (per-sample).",
    'Min-Max (per sample)':  "Min-Max: each sample rescaled to [0,1] using its own min/max (per-sample).",
    'Internal Standard':     f"Internal Standard: each variable ÷ IS row avg for that variable (IS = '{is_label_val}').",
}
std_detail_map = {
    'None':     "Z-score auto-applied (population-wise) — no user standardization selected.",
    'SNV':      "SNV: each sample (row) centered and scaled by its own mean and std (per-sample).",
    'Z-score':  "Z-score: each variable (column) centered and scaled by population mean and std (population-wise).",
}

with st.expander(f"📋 {analysis_mode.split('(')[0].strip()} Pipeline Details", expanded=True):
    st.markdown(f"""
**Input:** {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features  
**Step 1 — Normalization:** {norm_detail_map.get(norm_option, norm_option)}  
**Step 2 — Standardization:** {std_detail_map.get(preprocess_option, preprocess_option)}  
**Step 3 — Decomposition:** `{analysis_mode}`  

| Component | Variance Explained | Cumulative |
|---|---|---|
""" + "\n".join([
        f"| {component_label}{i+1} | {var_ratios[i]:.2%} | {cum_var[i]:.2%} |"
        for i in range(min(5, n_total_pcs))
    ]) + f"""

**95% cumulative variance at:** {component_label}{n_95}  
**99% cumulative variance at:** {component_label}{n_99}
""")
    top_n = min(10, n_total_pcs)
    fig_sum = go.Figure(go.Bar(
        x=[f"{component_label}{i+1}" for i in range(top_n)],
        y=[v*100 for v in var_ratios[:top_n]],
        marker_color='steelblue', opacity=0.85,
        text=[f"{v*100:.1f}%" for v in var_ratios[:top_n]], textposition='outside'
    ))
    fig_sum.update_layout(
        title=f"Variance Explained — Top {top_n} {component_label}s",
        xaxis_title="Component", yaxis_title="% Variance Explained",
        height=280, margin=dict(t=40,b=30,l=40,r=10),
        yaxis=dict(range=[0, max(var_ratios[0]*115, 1)])
    )
    apply_white_theme(fig_sum)
    st.plotly_chart(fig_sum, use_container_width=True)

# PCA Diagnostics
if pca_full is not None:
    pca_issues = diagnose_pca(X_scores, var_ratios)
    if pca_issues:
        st.subheader("⚠️ PCA Diagnostics")
        st.warning("Issues detected that may cause points to appear collapsed onto an axis.")
        label_map = {'zero_variance':'Zero Variance','near_zero_variance':'Near-Zero Variance','axis_clustering':'Axis-End Clustering'}
        for issue in pca_issues:
            with st.expander(f"⚠️ {component_label}{issue['pc']} — {label_map.get(issue['type'], issue['type'])}", expanded=True):
                st.markdown(issue['detail'])
                col_data = X_scores[:, issue['pc']-1]
                fig_diag = go.Figure(go.Histogram(x=col_data, nbinsx=30, marker_color='salmon', opacity=0.8))
                fig_diag.update_layout(title=f"{component_label}{issue['pc']} Score Distribution",
                                       xaxis_title="Score", yaxis_title="Count",
                                       height=250, margin=dict(t=35,b=30,l=30,r=10))
                apply_white_theme(fig_diag)
                st.plotly_chart(fig_diag, use_container_width=True)

# Always set these — used by classification/clustering
if n_total_pcs >= 2:
    X_scores_2d_global = X_scores[:, :2]
    y_global           = y
else:
    X_scores_2d_global = None
    y_global           = None

# ══════════════════════════════════════════════════════════════════════════════
# COLOR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Label Color Configuration")
unique_labels_all = sorted(y.unique())
n_labels          = len(unique_labels_all)
use_custom_colors = st.toggle("Enable custom label colors (enter hex codes)", value=False)
color_map_hex = {}
if use_custom_colors:
    st.markdown("Type a 6-digit hex code for each label. A color swatch confirms your choice.")
    n_cols     = min(4, n_labels)
    col_groups = st.columns(n_cols)
    for idx, lbl in enumerate(unique_labels_all):
        default_hex = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        with col_groups[idx % n_cols]:
            st.markdown(f"**`{lbl}`**")
            user_input = st.text_input(label=f"Hex color for {lbl}", value=default_hex,
                                       key=f"color_{lbl}", label_visibility="collapsed")
            user_input = user_input.strip()
            chosen = ('#' + user_input.lstrip('#')) if (user_input and is_valid_hex(user_input)) else default_hex
            color_map_hex[lbl] = chosen
            st.markdown(f'<div style="width:100%;height:18px;border-radius:4px;background:{chosen};'
                        f'border:1px solid #ccc;margin-bottom:6px;"></div>', unsafe_allow_html=True)
else:
    for idx, lbl in enumerate(unique_labels_all):
        color_map_hex[lbl] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – PLOT OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Plot Options", expanded=False):
    show_2d         = st.checkbox("Show 2D Scores Plot (Static)", value=True)
    legend_separate = st.checkbox("Separate legend figure (PCA plots)", value=False)
    show_3d         = st.checkbox("Show 3D Scores Plot (Interactive)", value=True)
    show_scree      = st.checkbox("Show Scree Plot", value=True)
    if show_scree:
        n_99_s  = np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else n_total_pcs
        n_999_s = np.argmax(cum_var >= 0.999)+ 1 if np.any(cum_var >= 0.999) else n_total_pcs
        n_scree = st.slider(f"Number of {component_label}s in Scree", 1, max(n_999_s,2), max(n_99_s,2))
    else:
        n_scree = n_total_pcs
        n_99_s  = n_total_pcs

    db_legend_mode = st.radio(
        "Decision Boundary legend",
        ["Inside plot", "Separate figure", "No legend"],
        index=0,
        help=(
            "**Inside plot:** legend is shown inside the interactive Plotly figure. "
            "Click labels to toggle visibility.\n\n"
            "**Separate figure:** legend rendered as a standalone figure below the plot — "
            "useful when there are many labels.\n\n"
            "**No legend:** legend is hidden entirely — best when you have large numbers of labels "
            "that would otherwise truncate the plot."
        )
    )

    show_loadings = st.checkbox("Show Loadings / Weights Plot", value=True)
    if show_loadings:
        loadings_type = st.selectbox(
            "Loadings Plot Type",
            ["Bar Graph (Discrete, e.g., GCMS)",
             "Connected Scatterplot (Continuous, e.g., Spectroscopy)"],
            index=0
        )
    else:
        loadings_type = "Bar Graph (Discrete, e.g., GCMS)"

    if show_2d and n_total_pcs >= 2:
        st.markdown(f"**2D Plot {component_label} Axes**")
        cx_2d  = st.selectbox("X-axis", [f"{component_label}{i+1}" for i in range(n_total_pcs)], index=0, key="cx_2d")
        cy_2d  = st.selectbox("Y-axis", [f"{component_label}{i+1}" for i in range(n_total_pcs)], index=1, key="cy_2d")
        cx_idx = int(cx_2d[len(component_label):]) - 1
        cy_idx = int(cy_2d[len(component_label):]) - 1
    else:
        cx_idx, cy_idx = 0, 1
        cx_2d = f"{component_label}1"; cy_2d = f"{component_label}2"

with st.sidebar.expander("Download Options", expanded=False):
    num_save_pcs = st.slider("Number of components to save", 1, min(10, n_total_pcs), 3)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Classification Options", expanded=False):
    run_da = st.checkbox("Run Discriminant Analysis")
    if run_da:
        da_type = st.selectbox("DA Type", ["LDA","QDA","GaussianNB"], index=0)
    else:
        da_type = "LDA"
    run_knn = st.checkbox("Run KNN")
    k       = st.slider("K value", 1, 20, 5) if run_knn else 5
    run_dt  = st.checkbox("Run Decision Tree")
    if run_dt:
        dt_max_depth    = st.slider("Max tree depth", 1, 10, 3,
                                     help="Limits how deep the tree grows. Shallower = simpler, less overfit. "
                                          "For ≤75 samples, depth 2–4 is usually best.")
        dt_criterion    = st.selectbox("Split criterion", ["gini","entropy"], index=0,
                                        help="**Gini:** measures impurity — fast and works well in practice.\n\n"
                                             "**Entropy:** information gain — can produce slightly different splits.")
        dt_min_samples  = st.slider("Min samples per leaf", 1, 10, 2,
                                     help="Minimum number of samples required at a leaf node. "
                                          "Higher values prevent over-fitting on small datasets.")
    else:
        dt_max_depth = 3; dt_criterion = "gini"; dt_min_samples = 2
    run_kmeans = st.checkbox("Run K-Means Clustering")
    if run_kmeans:
        auto_optimize_k      = st.checkbox("Auto-optimize K", value=False)
        n_clusters           = st.slider("Clusters", 2, 10, 3) if not auto_optimize_k else 3
        show_elbow           = st.checkbox("Elbow Plot", value=True)
        show_silhouette      = st.checkbox("Silhouette Plot", value=True)
        show_cluster_profile = st.checkbox("Cluster Profile Plot", value=True)
    else:
        auto_optimize_k = False; show_elbow = False
        show_silhouette = False; show_cluster_profile = False; n_clusters = 3

with st.sidebar.expander("Classification Input Options", expanded=True):
    max_pcs_cls = min(10, n_total_pcs)
    n_pcs_for_classification = st.slider(
        "PCs / LVs for Classification", 1, max_pcs_cls, 2,
        help="How many components to feed into LDA/QDA/KNN. Decision boundary plots need exactly 2."
    )

# ══════════════════════════════════════════════════════════════════════════════
# LABEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Label Configuration")
label_mode = st.radio("Label Mode", ["Default Labels","Combined Groups"], index=0)
y_plot     = y.copy()
if label_mode == "Default Labels":
    st.info("Using default simplified labels.")
    selected_for_a = []; selected_for_b = []; apply_to_plots = False
    rename_a = "Group A"; rename_b = "Group B"
else:
    if n_total_pcs >= 2:
        uc = sorted(y_global.unique())
        selected_for_a = st.multiselect("Labels for Group A", uc, default=uc[:1])
        selected_for_b = st.multiselect("Labels for Group B", uc, default=uc[1:2])
        rename_a       = st.text_input("Rename Group A", value=f"Group A ({', '.join(selected_for_a)})")
        rename_b       = st.text_input("Rename Group B", value=f"Group B ({', '.join(selected_for_b)})")
        apply_to_plots = st.checkbox("Use combined labels for plots", value=True)
        if apply_to_plots and selected_for_a and selected_for_b:
            y_plot = y_plot.replace({l: rename_a for l in selected_for_a})
            y_plot = y_plot.replace({l: rename_b for l in selected_for_b})
    else:
        selected_for_a=[]; selected_for_b=[]; rename_a="Group A"; rename_b="Group B"; apply_to_plots=False

unique_plot_labels = sorted(y_plot.unique())
plot_color_map = {lbl: color_map_hex.get(lbl, DEFAULT_COLORS[i%len(DEFAULT_COLORS)])
                  for i,lbl in enumerate(unique_plot_labels)}

# ══════════════════════════════════════════════════════════════════════════════
# PCR / PLS REGRESSION DIAGNOSTICS  (shown first so prediction quality is
# immediately visible before exploring the component space)
# ══════════════════════════════════════════════════════════════════════════════
if analysis_mode != "PCA (Principal Component Analysis)" and y_target is not None:
    st.subheader(f"{analysis_mode.split('(')[0].strip()} Regression Diagnostics")

    if analysis_mode == "PCR (Principal Component Regression)":
        st.markdown("#### Cross-Validation MSE vs. Number of PCs")
        with st.sidebar.expander("PCR Options", expanded=True):
            max_pcr_comps = st.slider("Max PCs to evaluate (CV)", 2, min(20, n_total_pcs), min(10, n_total_pcs))
            n_pcr_final   = st.slider("PCs to use for final model", 2, max_pcr_comps, 3,
                                       help="Number of PCs selected for the final PCR prediction.")
        from sklearn.linear_model import LinearRegression
        cv_mse = []
        for n_pc in range(1, max_pcr_comps+1):
            X_sw  = X_scores[:, :n_pc]
            kf    = KFold(n_splits=min(5, len(y_target)), shuffle=True, random_state=42)
            preds = np.zeros(len(y_target))
            for tr, te in kf.split(X_sw):
                lr = LinearRegression().fit(X_sw[tr], y_target[tr])
                preds[te] = lr.predict(X_sw[te])
            cv_mse.append(mean_squared_error(y_target, preds))
        best_pcr = int(np.argmin(cv_mse)) + 1
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(x=list(range(1,max_pcr_comps+1)), y=cv_mse,
                                     mode='lines+markers', line=dict(color='royalblue',width=2),
                                     marker=dict(size=7), name='CV MSE'))
        fig_cv.add_vline(x=best_pcr, line_dash='dot', line_color='green',
                          annotation_text=f"Best ({best_pcr} PCs, MSE={cv_mse[best_pcr-1]:.4f})",
                          annotation_position="top right")
        fig_cv.add_vline(x=n_pcr_final, line_dash='dash', line_color='orange',
                          annotation_text=f"Selected ({n_pcr_final} PCs)",
                          annotation_position="top left")
        fig_cv.update_layout(title="PCR: CV MSE vs. Number of PCs",
                              xaxis_title="Number of PCs", yaxis_title="CV MSE",
                              xaxis=dict(tickmode='linear',dtick=1), height=400)
        apply_white_theme(fig_cv)
        st.plotly_chart(fig_cv, use_container_width=True)

        from sklearn.linear_model import LinearRegression
        lr_final = LinearRegression().fit(X_scores[:, :n_pcr_final], y_target)
        y_pred   = lr_final.predict(X_scores[:, :n_pcr_final])
        r2       = 1 - np.sum((y_target - y_pred)**2) / np.sum((y_target - y_target.mean())**2)
        mse_full = mean_squared_error(y_target, y_pred)
        st.info(f"PCR final model — {n_pcr_final} PCs | R² = {r2:.4f} | MSE = {mse_full:.4f}")

    else:  # PLS
        st.markdown("#### Cross-Validation MSE vs. Number of Latent Variables")
        max_pls_cv = min(20, n_max_components)
        cv_mse_pls = []
        for n_lv in range(1, max_pls_cv+1):
            kf    = KFold(n_splits=min(5, len(y_target)), shuffle=True, random_state=42)
            preds = np.zeros(len(y_target))
            for tr, te in kf.split(X_scaled):
                m = PLSRegression(n_components=n_lv, scale=False)
                m.fit(X_scaled[tr], y_target[tr])
                preds[te] = m.predict(X_scaled[te]).ravel()
            cv_mse_pls.append(mean_squared_error(y_target, preds))
        best_pls = int(np.argmin(cv_mse_pls)) + 1
        fig_cv_pls = go.Figure()
        fig_cv_pls.add_trace(go.Scatter(x=list(range(1,max_pls_cv+1)), y=cv_mse_pls,
                                         mode='lines+markers', line=dict(color='royalblue',width=2),
                                         marker=dict(size=7), name='CV MSE'))
        fig_cv_pls.add_vline(x=best_pls, line_dash='dot', line_color='green',
                              annotation_text=f"Best ({best_pls} LVs, MSE={cv_mse_pls[best_pls-1]:.4f})",
                              annotation_position="top right")
        fig_cv_pls.add_vline(x=n_pls_components, line_dash='dash', line_color='orange',
                              annotation_text=f"Selected ({n_pls_components} LVs)",
                              annotation_position="top left")
        fig_cv_pls.update_layout(title="PLS: CV MSE vs. Number of Latent Variables",
                                  xaxis_title="Number of LVs", yaxis_title="CV MSE",
                                  xaxis=dict(tickmode='linear',dtick=1), height=400)
        apply_white_theme(fig_cv_pls)
        st.plotly_chart(fig_cv_pls, use_container_width=True)

        y_pred   = pls_model.predict(X_scaled).ravel()
        r2       = 1 - np.sum((y_target-y_pred)**2)/np.sum((y_target-y_target.mean())**2)
        mse_full = mean_squared_error(y_target, y_pred)
        st.info(f"PLS model — {n_pls_components} LVs | R² = {r2:.4f} | MSE = {mse_full:.4f}")

    # Predicted vs Actual
    st.markdown("#### Predicted vs. Actual")
    df_pred = pd.DataFrame({'Actual': y_target, 'Predicted': y_pred, 'label': y.values})
    fig_pva = px.scatter(df_pred, x='Actual', y='Predicted', color='label',
                          color_discrete_map=plot_color_map,
                          title="Predicted vs. Actual",
                          labels={'Actual':'Actual Y','Predicted':'Predicted Y'})
    mn_val, mx_val = float(y_target.min()), float(y_target.max())
    fig_pva.add_trace(go.Scatter(x=[mn_val,mx_val], y=[mn_val,mx_val],
                                  mode='lines', line=dict(dash='dash',color='gray'),
                                  name='1:1 line', showlegend=True))
    fig_pva.update_layout(height=450)
    apply_white_theme(fig_pva)
    st.plotly_chart(fig_pva, use_container_width=True)

    # Residuals
    st.markdown("#### Residuals Plot")
    residuals = y_target - y_pred
    df_res    = pd.DataFrame({'Predicted': y_pred, 'Residual': residuals, 'label': y.values})
    fig_res   = px.scatter(df_res, x='Predicted', y='Residual', color='label',
                            color_discrete_map=plot_color_map,
                            title="Residuals vs. Predicted",
                            labels={'Predicted':'Predicted Y','Residual':'Residual (Actual − Predicted)'})
    fig_res.add_hline(y=0, line_dash='dash', line_color='gray')
    fig_res.update_layout(height=420)
    apply_white_theme(fig_res)
    st.plotly_chart(fig_res, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCORES PLOTS, SCREE, LOADINGS, DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
if show_2d and n_total_pcs >= 2:
    st.subheader(f"2D {component_label} Scores Plot ({cx_2d} vs {cy_2d})")
    xv, yv = X_scores[:, cx_idx], X_scores[:, cy_idx]
    xvar, yvar = var_ratios[cx_idx], var_ratios[cy_idx]
    df_2d = pd.DataFrame({cx_2d: xv, cy_2d: yv, 'label': y_plot.values})
    ul2d       = sorted(df_2d['label'].unique())
    mpl_colors = [plot_color_map.get(l, DEFAULT_COLORS[i%len(DEFAULT_COLORS)]) for i,l in enumerate(ul2d)]
    if legend_separate:
        fig_m, ax_m = plt.subplots(figsize=(8,6))
        if use_white_theme:
            fig_m.patch.set_facecolor('white'); ax_m.set_facecolor('white')
        for lbl, col in zip(ul2d, mpl_colors):
            m = df_2d['label']==lbl
            ax_m.scatter(df_2d[m][cx_2d], df_2d[m][cy_2d], c=col, label=lbl, s=50)
        lbl_kw = dict(color='black') if use_white_theme else {}
        ax_m.set_xlabel(f"{cx_2d} ({xvar:.1%})", **lbl_kw)
        ax_m.set_ylabel(f"{cy_2d} ({yvar:.1%})", **lbl_kw)
        ax_m.set_title(f"2D {component_label} Scores", **lbl_kw)
        if use_white_theme:
            ax_m.tick_params(colors='black')
            ax_m.grid(True, alpha=0.3, color='#cccccc')
            for sp in ax_m.spines.values(): sp.set_edgecolor('#cccccc')
        else:
            ax_m.grid(True, alpha=0.3)
        st.pyplot(fig_m, bbox_inches='tight'); plt.close(fig_m)
        fig_lg, ax_lg = plt.subplots(figsize=(2, len(ul2d)*0.5))
        if use_white_theme:
            fig_lg.patch.set_facecolor('white'); ax_lg.set_facecolor('white')
        ax_lg.axis('off')
        handles = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=8,label=l)
                   for l,c in zip(ul2d,mpl_colors)]
        leg = ax_lg.legend(handles=handles, loc='center')
        if use_white_theme:
            for txt in leg.get_texts(): txt.set_color('black')
        st.pyplot(fig_lg, bbox_inches='tight'); plt.close(fig_lg)
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        if use_white_theme:
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        for lbl, col in zip(ul2d, mpl_colors):
            m = df_2d['label']==lbl
            ax.scatter(df_2d[m][cx_2d], df_2d[m][cy_2d], c=col, label=lbl, s=50)
        lbl_kw = dict(color='black') if use_white_theme else {}
        ax.set_xlabel(f"{cx_2d} ({xvar:.1%})", **lbl_kw)
        ax.set_ylabel(f"{cy_2d} ({yvar:.1%})", **lbl_kw)
        ax.set_title(f"2D {component_label} Scores Plot", **lbl_kw)
        if use_white_theme:
            ax.tick_params(colors='black')
            ax.grid(True, alpha=0.3, color='#cccccc')
            for sp in ax.spines.values(): sp.set_edgecolor('#cccccc')
            leg = ax.legend()
            for txt in leg.get_texts(): txt.set_color('black')
        else:
            ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig, bbox_inches='tight'); plt.close(fig)
elif show_2d:
    st.warning("Need at least 2 components for 2D plot.")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 3D SCORES PLOT
# ══════════════════════════════════════════════════════════════════════════════
if show_3d and n_total_pcs >= 3:
    st.subheader(f"3D {component_label} Scores Plot (Interactive)")
    df_3d = pd.DataFrame(X_scores[:,:3], columns=[f"{component_label}{i+1}" for i in range(3)])
    df_3d['label'] = y_plot.values
    fig_3d = px.scatter_3d(df_3d, x=f"{component_label}1", y=f"{component_label}2",
                            z=f"{component_label}3", color='label',
                            color_discrete_map=plot_color_map)
    fig_3d.update_traces(marker=dict(size=5))
    if legend_separate:
        fig_3d.update_layout(showlegend=False)
    else:
        fig_3d.update_layout(
            title=f"Interactive 3D {component_label} Scores",
            scene=dict(xaxis_title=f"{component_label}1 ({var_ratios[0]:.1%})",
                       yaxis_title=f"{component_label}2 ({var_ratios[1]:.1%})",
                       zaxis_title=f"{component_label}3 ({var_ratios[2]:.1%})"))
    apply_white_theme(fig_3d)
    st.plotly_chart(fig_3d, use_container_width=True)
elif show_3d:
    st.warning("Need at least 3 components for 3D plot.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SCREE PLOT
# ══════════════════════════════════════════════════════════════════════════════
if show_scree:
    st.subheader(f"Scree Plot: Variance Explained ({component_label}s)")
    var_pct = var_ratios[:n_scree] * 100
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=[f"{component_label}{i+1}" for i in range(n_scree)],
        y=var_pct, name='% Variance', marker_color='lightblue'
    ))
    for i, v in enumerate(var_pct):
        fig_scree.add_annotation(x=f"{component_label}{i+1}", y=v, text=f"{v:.1f}%",
                                  showarrow=False, yshift=10, font=dict(size=10))
    fig_scree.update_layout(
        title=f"Scree Plot — {n_scree} {component_label}s",
        xaxis_title="Component", yaxis_title="% Variance Explained",
        yaxis=dict(range=[0, var_pct.max()*1.15])
    )
    apply_white_theme(fig_scree)
    st.plotly_chart(fig_scree, use_container_width=True)
    st.info(f"Shown: {np.sum(var_ratios[:n_scree]):.1%} | ≥99% at {component_label}{n_99_s}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LOADINGS / WEIGHTS PLOT
# ══════════════════════════════════════════════════════════════════════════════
if show_loadings:
    # Determine loadings matrix depending on mode
    if pca_full is not None:
        loadings_matrix = pd.DataFrame(
            pca_full.components_,
            columns=X.columns,
            index=[f"{component_label}{i+1}" for i in range(n_total_pcs)]
        )
        loadings_title_suffix = "Factor Loadings"
    elif pls_model is not None:
        loadings_matrix = pd.DataFrame(
            pls_model.x_weights_.T,
            columns=X.columns,
            index=[f"{component_label}{i+1}" for i in range(n_pls_components)]
        )
        loadings_title_suffix = "X Weights"
    else:
        loadings_matrix = None

    if loadings_matrix is None:
        st.warning("Loadings not available.")
    else:
        # Part A: Top-3 grouped
        st.subheader(f"{loadings_title_suffix} Plot (Top 3 {component_label}s)")
        max_c       = min(3, n_total_pcs)
        valid_idx   = [i for i in range(max_c) if var_ratios[i] > 0]
        if not valid_idx:
            st.warning(f"No {component_label}s with >0% variance.")
        else:
            lt3     = loadings_matrix.iloc[valid_idx].abs()
            bar_clr = px.colors.qualitative.Set3[:len(valid_idx)]
            if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                sv = lt3.max(axis=0).sort_values(ascending=False).index
                fg = go.Figure()
                for i, pc in enumerate(lt3.index):
                    fg.add_trace(go.Bar(x=sv, y=lt3.loc[pc,sv].values, name=pc,
                                        marker_color=bar_clr[i], width=0.25, base=0, offsetgroup=i))
                fg.update_layout(barmode='group', height=400, showlegend=True,
                                  title=f"{loadings_title_suffix}: Grouped Bar (Abs, Top 3 {component_label}s)",
                                  xaxis_title="Variables", yaxis_title="Loading Magnitude")
                fg.update_xaxes(tickangle=45, tickfont=dict(size=9))
            else:
                orig = X.columns.tolist()
                melt = lt3.reset_index().melt(id_vars='index', var_name='Variable', value_name='Loading')
                melt['Component'] = melt['index']
                melt['Variable']  = pd.Categorical(melt['Variable'], categories=orig, ordered=True)
                melt = melt.sort_values(['Component','Variable'])
                fg = px.line(melt, x='Variable', y='Loading', color='Component', markers=False,
                             title=f"{loadings_title_suffix}: Connected Line (Abs, Top 3 {component_label}s)")
                fg.update_traces(line=dict(width=2)); fg.update_xaxes(tickangle=45, tickfont=dict(size=9))
            apply_white_theme(fg)
            st.plotly_chart(fg, use_container_width=True)
            st.subheader(f"{loadings_title_suffix} Table (Top 3 {component_label}s)")
            st.dataframe(loadings_matrix.iloc[valid_idx])

        # Part B: Single component explorer
        st.subheader(f"{loadings_title_suffix} — Single {component_label} Explorer")
        sel_comp_num = st.slider(f"Select {component_label} to explore", 1, min(10,n_total_pcs), 1)
        sel_comp_idx = sel_comp_num - 1
        if sel_comp_idx >= n_total_pcs or var_ratios[sel_comp_idx] == 0:
            st.warning(f"{component_label}{sel_comp_num} has zero variance.")
        else:
            lrow     = loadings_matrix.iloc[sel_comp_idx]
            lrow_abs = lrow.abs()
            st.info(f"Variance by {component_label}{sel_comp_num}: {var_ratios[sel_comp_idx]:.1%}")
            if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                sv2  = lrow_abs.sort_values(ascending=False).index
                fg2  = go.Figure()
                fg2.add_trace(go.Bar(x=sv2, y=lrow_abs.loc[sv2].values,
                                      marker_color='steelblue', name=f"{component_label}{sel_comp_num}"))
                fg2.update_layout(title=f"{loadings_title_suffix}: {component_label}{sel_comp_num} (Abs)",
                                   xaxis_title="Variables", yaxis_title="Magnitude", height=400)
                fg2.update_xaxes(tickangle=45, tickfont=dict(size=9))
            else:
                fg2 = go.Figure()
                fg2.add_trace(go.Scatter(x=X.columns.tolist(), y=lrow_abs.values,
                                          mode='lines', line=dict(width=2),
                                          name=f"{component_label}{sel_comp_num}"))
                fg2.update_layout(title=f"{loadings_title_suffix}: {component_label}{sel_comp_num} (Abs, Line)",
                                   xaxis_title="Variables", yaxis_title="Magnitude", height=400)
                fg2.update_xaxes(tickangle=45, tickfont=dict(size=9))
            apply_white_theme(fg2)
            st.plotly_chart(fg2, use_container_width=True)
            st.subheader(f"{loadings_title_suffix} Table — {component_label}{sel_comp_num}")
            st.dataframe(lrow.to_frame(name=f"{component_label}{sel_comp_num}"))

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Download Results")

# Processed (normalized + standardized) data — always available
_X_dl = pd.DataFrame(X_scaled, columns=X.columns)
_X_dl.insert(0, 'label', y.values)
st.download_button(
    "⬇ Processed Data CSV (normalized + standardized)",
    _X_dl.to_csv(index=False),
    "processed_data.csv", "text/csv",
    help="The exact values used as input to PCA / PCR / PLS — after all normalization and standardization steps."
)

col1, col2 = st.columns(2)

with col1:
    df_scores_dl = pd.DataFrame(
        X_scores[:, :num_save_pcs],
        columns=[f"{component_label}{i+1}" for i in range(num_save_pcs)]
    )
    df_scores_dl['label'] = y.values
    st.download_button("⬇ Scores CSV", df_scores_dl.to_csv(index=False),
                        f"{component_label.lower()}_scores.csv", "text/csv",
                        help=f"Scores for the top {num_save_pcs} {component_label}s, one row per sample.")

with col2:
    df_scree_dl = pd.DataFrame({
        "Component":             [f"{component_label}{i+1}" for i in range(n_total_pcs)],
        "Variance_Explained_%":  [round(v*100, 4) for v in var_ratios],
        "Cumulative_Variance_%": [round(c*100, 4) for c in np.cumsum(var_ratios)],
    })
    st.download_button("⬇ Scree Data CSV", df_scree_dl.to_csv(index=False),
                        "scree_data.csv", "text/csv",
                        help="Individual and cumulative variance for all components.")

col3, col4 = st.columns(2)
with col3:
    if pca_full is not None or pls_model is not None:
        lm = loadings_matrix if 'loadings_matrix' in dir() and loadings_matrix is not None else None
        if lm is not None:
            lraw_T = lm.iloc[:num_save_pcs].T.reset_index().rename(columns={'index':'Variable'})
            st.download_button("⬇ Loadings CSV (raw/signed)", lraw_T.to_csv(index=False),
                                "loadings_raw.csv", "text/csv",
                                help=f"Signed loadings/weights for {component_label}1–{component_label}{num_save_pcs}. Rows = variables.")
    else:
        st.info("Loadings not available.")

with col4:
    if pca_full is not None or pls_model is not None:
        lm2 = loadings_matrix if 'loadings_matrix' in dir() and loadings_matrix is not None else None
        if lm2 is not None:
            labs_T = lm2.iloc[:num_save_pcs].abs().T.reset_index().rename(columns={'index':'Variable'})
            st.download_button("⬇ Loadings CSV (absolute)", labs_T.to_csv(index=False),
                                "loadings_abs.csv", "text/csv",
                                help=f"Absolute loadings/weights for {component_label}1–{component_label}{num_save_pcs}.")
    else:
        st.info("Loadings not available.")

# PCR/PLS regression results download
if analysis_mode != "PCA (Principal Component Analysis)" and y_target is not None:
    df_reg_dl = pd.DataFrame({
        'label': y.values, 'Actual_Y': y_target,
        'Predicted_Y': y_pred, 'Residual': residuals
    })
    st.download_button("⬇ Regression Results CSV", df_reg_dl.to_csv(index=False),
                        "regression_results.csv", "text/csv",
                        help="Actual vs predicted Y values and residuals for each sample.")

st.caption(
    f"Scores and loadings include top {num_save_pcs} components. "
    "Scree data includes all components. Loadings rows = variables, columns = components."
)

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
if run_kmeans and X_scores_2d_global is not None:
    st.header("Clustering Results")
    if auto_optimize_k:
        inertias   = [KMeans(n_clusters=ki,random_state=42,n_init=10).fit(X_scores_2d_global).inertia_
                      for ki in range(1,11)]
        n_clusters = np.argmin(np.diff(np.diff(inertias))) + 2
        st.info(f"Auto-optimized K: {n_clusters}")
    kmeans      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clabs       = kmeans.fit_predict(X_scores_2d_global)
    df_cl       = pd.DataFrame(X_scores_2d_global, columns=[f"{component_label}1",f"{component_label}2"])
    df_cl['cluster'] = clabs.astype(str)
    fig_cl = px.scatter(df_cl, x=f"{component_label}1", y=f"{component_label}2", color='cluster',
                         title=f"K-Means (k={n_clusters}) on {component_label}1 vs {component_label}2",
                         color_discrete_sequence=px.colors.qualitative.Set1)
    apply_white_theme(fig_cl)
    st.plotly_chart(fig_cl, use_container_width=True)
    if show_elbow:
        inertias = [KMeans(n_clusters=ki,random_state=42,n_init=10).fit(X_scores_2d_global).inertia_ for ki in range(1,11)]
        fg_el = px.line(x=range(1,11), y=inertias, markers=True, title="Elbow Plot")
        fg_el.update_layout(xaxis_title="K", yaxis_title="Inertia"); st.plotly_chart(fg_el)
    if show_silhouette:
        silhs = [silhouette_score(X_scores_2d_global,
                  KMeans(n_clusters=ki,random_state=42,n_init=10).fit_predict(X_scores_2d_global))
                 for ki in range(2,11)]
        fg_sil = px.line(x=range(2,11), y=silhs, markers=True, title="Silhouette Score")
        fg_sil.update_layout(xaxis_title="K", yaxis_title="Score"); st.plotly_chart(fg_sil)
    if show_cluster_profile:
        df_cen = pd.DataFrame(kmeans.cluster_centers_, columns=[f"{component_label}1",f"{component_label}2"])
        df_cen['cluster'] = range(n_clusters)
        fg_pr = px.bar(df_cen.melt(id_vars='cluster'), x='cluster', y='value',
                        color='variable', barmode='group', title="Cluster Centroids")
        apply_white_theme(fg_pr)
        st.plotly_chart(fg_pr)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
st.header("Classification Results")

if n_total_pcs < n_pcs_for_classification:
    st.error(f"Not enough components. Selected {n_pcs_for_classification}, only {n_total_pcs} available.")
else:
    X_class = X_scores[:, :n_pcs_for_classification]
    if label_mode == "Default Labels":
        y_selected           = y_global
        title_suffix         = " (Multi-class)"
        X_pca_full_for_sweep = X_scores
    else:
        if not selected_for_a or not selected_for_b:
            st.warning("Select groups to run combined classification.")
            st.stop()
        mga = y_global.isin(selected_for_a); mgb = y_global.isin(selected_for_b)
        mgs = mga | mgb
        X_class              = X_class[mgs]
        X_pca_full_for_sweep = X_scores[mgs]
        y_selected           = np.where(mga[mgs], 0, 1)
        title_suffix         = ""

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_selected)
    unique_y  = le.classes_

    split_data = st.checkbox("Split into train/test sets")
    if split_data:
        # ── Split advisor ─────────────────────────────────────────────────────
        # For stratified splitting, every class needs at least 1 sample in test
        # and 1 in train, so the valid test fractions are constrained by class sizes.
        n_total   = len(y_encoded)
        class_counts = np.bincount(y_encoded)
        min_class_n  = int(class_counts.min())
        n_classes_n  = len(class_counts)

        # A test fraction f is valid when floor(min_class_n * f) >= 1
        # i.e. f >= 1/min_class_n, AND floor(min_class_n * (1-f)) >= 1
        # i.e. f <= 1 - 1/min_class_n
        # Generate the discrete valid fractions: test_n = 1, 2, … min_class_n-1
        valid_test_sizes = []
        for test_n in range(1, min_class_n):
            frac = test_n / min_class_n
            if 0.05 <= frac <= 0.60:
                valid_test_sizes.append(round(frac, 4))

        if not valid_test_sizes:
            st.error(
                f"Cannot split: the smallest class has only {min_class_n} sample(s). "
                "You need at least 2 samples per class to split into train and test. "
                "Either collect more data or uncheck 'Split into train/test sets'."
            )
            st.stop()

        # Find the fraction closest to 20%
        target     = 0.20
        best_frac  = min(valid_test_sizes, key=lambda f: abs(f - target))
        best_pct   = int(round(best_frac * 100))
        train_pct  = 100 - best_pct

        # Describe the replicate structure
        replicate_note = ""
        if min_class_n <= 10:
            replicate_note = (
                f" Your smallest class has **{min_class_n} samples**, so valid test sizes "
                f"are multiples of 1/{min_class_n} of that class."
            )

        if len(valid_test_sizes) == 1:
            test_size = valid_test_sizes[0]
            st.info(
                f"**Split advisor:** with {n_total} total samples across {n_classes_n} classes "
                f"(smallest class = {min_class_n} samples).{replicate_note}  \n"
                f"**Only one valid split is possible:** {int(round(test_size*100))}% test / "
                f"{int(round((1-test_size)*100))}% train. This will be used automatically."
            )
        else:
            st.info(
                f"**Split advisor:** with {n_total} total samples across {n_classes_n} classes "
                f"(smallest class = {min_class_n} samples).{replicate_note}  \n"
                f"**Recommended split:** {best_pct}% test / {train_pct}% train "
                f"(closest valid split to 20%).  \n"
                f"**All valid test sizes:** {', '.join([f'{int(round(f*100))}%' for f in valid_test_sizes])}"
            )
            test_size = st.select_slider(
                "Test size",
                options=valid_test_sizes,
                value=best_frac,
                format_func=lambda f: f"{int(round(f*100))}% test / {int(round((1-f)*100))}% train",
                help="Only fractions that guarantee at least 1 sample per class in each split are shown."
            )

        try:
            X_train, X_test, y_train_enc, y_test_enc = train_test_split(
                X_class, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
        except ValueError as e:
            st.error(f"Split failed: {e}. Try a different test size or uncheck the split option.")
            st.stop()
    else:
        test_size = None
        X_train, X_test, y_train_enc, y_test_enc = X_class, X_class, y_encoded, y_encoded

    def accuracy_vs_pcs_plot(clf_factory, label, X_full, y_enc, n_selected, n_scree_max):
        n_samples, n_classes = len(y_enc), len(np.unique(y_enc))
        if n_samples < 2 * n_classes:
            return None, f"⚠️ Not enough samples ({n_samples}) for {n_classes} classes (need ≥{2*n_classes})."
        cv_folds = min(5, n_samples // n_classes)
        if cv_folds < 2:
            return None, f"⚠️ Too few samples per class for CV (cv={cv_folds}, need ≥2)."
        max_sweep = min(n_scree_max, X_full.shape[1], 30)
        if max_sweep < 2:
            return None, f"⚠️ Only {X_full.shape[1]} component(s) — need ≥2 to sweep."
        sweep_pcs, sweep_mean, sweep_std, errs = [], [], [], []
        for n_pc in range(2, max_sweep+1):
            try:
                scores = cross_val_score(clf_factory(), X_full[:,:n_pc], y_enc,
                                          cv=cv_folds, scoring='accuracy', error_score='raise')
                sweep_pcs.append(n_pc); sweep_mean.append(float(scores.mean())); sweep_std.append(float(scores.std()))
            except Exception as e:
                errs.append(f"{component_label}{n_pc}: {type(e).__name__}: {str(e)[:100]}")
        if errs:
            with st.expander(f"⚠️ {len(errs)} step(s) skipped — click to see why"):
                [st.text(m) for m in errs]
        if not sweep_pcs:
            return None, f"⚠️ Sweep could not be computed. First error: {errs[0] if errs else 'unknown'}"
        arr, std = np.array(sweep_mean), np.array(sweep_std)
        best = int(np.argmax(arr))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep_pcs, y=arr, mode='lines+markers',
                                  line=dict(color='royalblue',width=2), marker=dict(size=7),
                                  name='CV Accuracy',
                                  error_y=dict(type='data',array=list(std),visible=True,color='lightblue')))
        if n_selected in sweep_pcs:
            fig.add_vline(x=n_selected, line_dash='dash', line_color='orange',
                           annotation_text=f"Selected ({n_selected})", annotation_position="top left")
        if sweep_pcs[best] != n_selected:
            fig.add_vline(x=sweep_pcs[best], line_dash='dot', line_color='green',
                           annotation_text=f"Best ({sweep_pcs[best]}, {arr[best]:.3f})",
                           annotation_position="top right")
        fig.update_layout(
            title=f"{label} — CV Accuracy vs. Number of {component_label}s (CV folds={cv_folds})",
            xaxis_title=f"Number of {component_label}s (cumulative)",
            yaxis_title="CV Accuracy",
            yaxis=dict(range=[max(0., float(arr.min())-0.1), 1.02]),
            xaxis=dict(tickmode='linear',dtick=1), height=420)
        sel_txt = ""
        if n_selected in sweep_pcs:
            si = sweep_pcs.index(n_selected)
            sel_txt = f"  |  Selected {n_selected} → {arr[si]:.3f} ± {std[si]:.3f}"
        return fig, f"Best: **{arr[best]:.3f} ± {std[best]:.3f}** at **{sweep_pcs[best]} {component_label}s**{sel_txt}"

    # ── shared visualisation helpers (called for every classifier) ────────────

    def show_train_test_accuracy(clf_name, clf, X_tr, X_te, y_tr, y_te, split_active):
        """Shows train accuracy, test accuracy, and overfitting delta."""
        acc_train = accuracy_score(y_tr, clf.predict(X_tr))
        acc_test  = accuracy_score(y_te, clf.predict(X_te))
        delta     = acc_train - acc_test

        if split_active:
            col_tr, col_te, col_d = st.columns(3)
            col_tr.metric("🔵 Training Accuracy", f"{acc_train:.3f}",
                           help="Accuracy on the training set — data the model has already seen.")
            col_te.metric("🟢 Test Accuracy", f"{acc_test:.3f}",
                           help="Accuracy on the held-out test set — never seen during training.")
            col_d.metric("⚠️ Train − Test Gap", f"{delta:+.3f}",
                          delta=f"{delta:+.3f}", delta_color="inverse",
                          help="Positive = model fits training better than test (potential overfitting).")
            if delta > 0.15:
                st.warning(
                    f"**Possible overfitting** — {clf_name} scores {delta:.1%} higher on training than test. "
                    "Consider fewer PCs, stronger regularisation, or more data."
                )
            elif delta < -0.05:
                st.warning(
                    f"**Unusual result** — test accuracy ({acc_test:.3f}) exceeds training ({acc_train:.3f}). "
                    "Check for data leakage or a very small test set."
                )
            else:
                st.success(f"Train/test gap is small ({delta:+.3f}) — no strong sign of overfitting.")
        else:
            st.metric("📊 Accuracy (in-sample — no split)", f"{acc_train:.3f}",
                       help="No split applied. Train = test = full dataset. May be optimistic.")
            st.caption(
                "⚠️ **In-sample result:** the model was evaluated on the same data it was trained on. "
                "This accuracy measures memorisation, not generalisation. "
                "Enable 'Split into train/test sets' above for an unbiased estimate."
            )

    def render_decision_boundary(clf_name, clf, X_2d, y_enc, unique_classes,
                                  cx_label, cy_label, title_suffix, legend_mode):
        """
        Plotly decision boundary plot.
        legend_mode: "Inside plot" | "Separate figure" | "No legend"
        - Real class label names in legend (not numeric codes)
        - Custom color map for regions and points
        - Text visibility fixed (dark font on transparent bg, not white-on-white)
        """
        x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
        y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
        pad_x = (x_max - x_min) * 0.10
        pad_y = (y_max - y_min) * 0.10
        res   = 200
        xx, yy = np.meshgrid(
            np.linspace(x_min - pad_x, x_max + pad_x, res),
            np.linspace(y_min - pad_y, y_max + pad_y, res)
        )
        try:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        except Exception as e:
            st.warning(f"Decision boundary mesh prediction failed: {e}")
            return

        n_classes  = len(unique_classes)
        # Build discrete colorscale: each encoded int → its hex color
        colorscale = []
        for i, cls in enumerate(unique_classes):
            hex_c = color_map_hex.get(str(cls), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            lo = i / n_classes
            hi = (i + 1) / n_classes
            colorscale.append([lo, hex_c])
            colorscale.append([hi, hex_c])

        show_inline = (legend_mode == "Inside plot")
        fig_db = go.Figure()

        # Filled decision regions
        fig_db.add_trace(go.Heatmap(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale=colorscale, zmin=0, zmax=n_classes - 1,
            showscale=False, opacity=0.30,
            hoverinfo='skip', name='Decision region'
        ))

        # Data points with actual label names
        for i, cls in enumerate(unique_classes):
            mask  = (y_enc == i)
            hex_c = color_map_hex.get(str(cls), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            fig_db.add_trace(go.Scatter(
                x=X_2d[mask, 0], y=X_2d[mask, 1],
                mode='markers',
                marker=dict(color=hex_c, size=9, line=dict(color='white', width=0.8)),
                name=str(cls),
                showlegend=show_inline
            ))

        legend_cfg = dict(
            orientation='v',
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            # Transparent background so Plotly uses the theme background — avoids white-on-white
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(150,150,150,0.5)',
            borderwidth=1,
            # Explicit dark font so text is always readable regardless of theme
            font=dict(color='#222222', size=12),
        ) if show_inline else dict(visible=False)

        fig_db.update_layout(
            title=f"{clf_name} Decision Boundary{title_suffix}",
            xaxis_title=cx_label, yaxis_title=cy_label,
            height=520,
            legend=legend_cfg,
        )
        apply_white_theme(fig_db)
        st.plotly_chart(fig_db, use_container_width=True)

        if legend_mode == "Separate figure":
            fig_leg = go.Figure()
            for i, cls in enumerate(unique_classes):
                hex_c = color_map_hex.get(str(cls), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                fig_leg.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(color=hex_c, size=10),
                    name=str(cls), showlegend=True
                ))
            fig_leg.update_layout(
                height=max(80, n_classes * 30 + 40),
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(
                    orientation='v', x=0, y=1,
                    font=dict(color='#222222', size=12),
                    bgcolor='rgba(0,0,0,0)',
                )
            )
            apply_white_theme(fig_leg)
            st.plotly_chart(fig_leg, use_container_width=True)

    def render_classification_report(clf_name, y_true, y_pred, classes):
        """Formatted classification report table."""
        st.subheader(f"{clf_name} — Classification Report")
        report = classification_report(y_true, y_pred,
                                        target_names=[str(c) for c in classes],
                                        output_dict=True, zero_division=0)
        rows = []
        for cls in [str(c) for c in classes]:
            if cls in report:
                rows.append({
                    "Class":     cls,
                    "Precision": round(report[cls]["precision"], 3),
                    "Recall":    round(report[cls]["recall"],    3),
                    "F1-Score":  round(report[cls]["f1-score"],  3),
                    "Support":   int(report[cls]["support"]),
                })
        for avg in ["macro avg", "weighted avg"]:
            if avg in report:
                rows.append({
                    "Class":     avg,
                    "Precision": round(report[avg]["precision"], 3),
                    "Recall":    round(report[avg]["recall"],    3),
                    "F1-Score":  round(report[avg]["f1-score"],  3),
                    "Support":   int(report[avg]["support"]),
                })
        df_report = pd.DataFrame(rows)
        st.dataframe(df_report, use_container_width=True, hide_index=True)
        st.caption(
            "**Precision** = of all predicted positives, how many were correct.  "
            "**Recall** = of all actual positives, how many were found.  "
            "**F1** = harmonic mean of precision and recall.  "
            "**Support** = number of true samples in that class."
        )
        return df_report

    def render_roc_curve(clf_name, clf, X_tr, X_te, y_tr, y_te, classes, has_proba):
        """
        ROC curve with multiselect to choose which classes to display.
        Binary: single curve. Multi-class: one-vs-rest, filtered by user selection.
        """
        st.subheader(f"{clf_name} — ROC Curve")
        n_classes = len(classes)
        try:
            if has_proba:
                y_score = clf.predict_proba(X_te)
            else:
                y_score = clf.decision_function(X_te)
                if n_classes == 2 and y_score.ndim == 1:
                    y_score = y_score.reshape(-1, 1)

            roc_colors = px.colors.qualitative.Set1

            if n_classes == 2:
                # Binary — no selection needed, single curve
                score_col = y_score[:, 1] if y_score.ndim == 2 else y_score.ravel()
                fpr, tpr, _ = roc_curve(y_te, score_col)
                roc_auc_val = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                              line=dict(dash='dash', color='gray', width=1),
                                              name='Random (AUC=0.50)'))
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                              line=dict(width=2, color='royalblue'),
                                              name=f"{classes[1]} (AUC={roc_auc_val:.3f})"))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.02]),
                    height=460, legend=dict(x=0.55, y=0.05)
                )
                apply_white_theme(fig_roc)
                st.plotly_chart(fig_roc, use_container_width=True)
                st.caption(f"**AUC = {roc_auc_val:.3f}** | 1.0 = perfect classifier, 0.5 = random.")

            else:
                # Multi-class — compute all curves first, then let user filter
                from sklearn.preprocessing import label_binarize
                y_te_bin = label_binarize(y_te, classes=list(range(n_classes)))

                # Pre-compute all valid curves
                computed = {}   # class_name -> (fpr, tpr, roc_auc)
                for i, cls_name in enumerate(classes):
                    if y_score.ndim != 2 or i >= y_score.shape[1]:
                        continue
                    if len(np.unique(y_te_bin[:, i])) < 2:
                        continue
                    fpr_i, tpr_i, _ = roc_curve(y_te_bin[:, i], y_score[:, i])
                    computed[str(cls_name)] = (fpr_i, tpr_i, auc(fpr_i, tpr_i))

                if not computed:
                    st.warning("Could not compute ROC curves — test set may not contain all classes.")
                    return

                # Class selector — all selected by default
                selected_classes = st.multiselect(
                    f"Select classes to display on ROC plot ({clf_name})",
                    options=list(computed.keys()),
                    default=list(computed.keys()),
                    key=f"roc_select_{clf_name}"
                )

                if not selected_classes:
                    st.info("Select at least one class above to display the ROC curve.")
                    return

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                              line=dict(dash='dash', color='gray', width=1),
                                              name='Random (AUC=0.50)'))
                displayed_aucs = []
                for i, cls_name in enumerate(classes):
                    sn = str(cls_name)
                    if sn not in selected_classes or sn not in computed:
                        continue
                    fpr_i, tpr_i, roc_auc_i = computed[sn]
                    displayed_aucs.append(roc_auc_i)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr_i, y=tpr_i, mode='lines',
                        line=dict(width=2, color=roc_colors[i % len(roc_colors)]),
                        name=f"{sn} vs Rest (AUC={roc_auc_i:.3f})"
                    ))

                fig_roc.update_layout(
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.02]),
                    height=460, legend=dict(x=0.55, y=0.05)
                )
                apply_white_theme(fig_roc)
                st.plotly_chart(fig_roc, use_container_width=True)
                all_aucs = [v[2] for v in computed.values()]
                st.caption(
                    f"Showing {len(selected_classes)} of {len(computed)} classes. "
                    f"Macro-avg AUC (all classes) = **{np.mean(all_aucs):.3f}** | "
                    "Each curve is one-vs-rest: that class treated as positive, all others as negative."
                )

        except Exception as e:
            st.warning(f"ROC curve could not be computed: {e}")

    # ── DA ────────────────────────────────────────────────────────────────────
    if run_da:
        if da_type == "LDA":
            best_da = LDA()
            da_params = {"solver":"svd","shrinkage":"None","tol":1e-4}
            da_desc = ("**Linear Discriminant Analysis (LDA):** finds linear combinations of features "
                       "that best separate classes. Assumes equal covariance matrices across classes "
                       "(homoscedastic). Decision boundaries are linear hyperplanes.")
        elif da_type == "QDA":
            best_da = QDA(reg_param=0.001)
            da_params = {"reg_param":0.001,"tol":1e-4}
            da_desc = ("**Quadratic Discriminant Analysis (QDA):** fits a separate covariance matrix "
                       "per class — allows curved (quadratic) decision boundaries. More flexible than "
                       "LDA but needs more data. `reg_param=0.001` prevents singular covariance issues.")
        else:
            best_da = GaussianNB()
            da_params = {"var_smoothing":1e-9}
            da_desc = ("**Gaussian Naive Bayes:** models each class as a multivariate Gaussian and "
                       "assumes all features are independent given the class label. Very fast; works "
                       "well when the independence assumption roughly holds.")
        best_da.fit(X_train, y_train_enc)
        with st.expander(f"📋 {da_type} Model Details", expanded=True):
            st.markdown(da_desc)
            st.markdown("**Parameters:** " + ", ".join([f"`{k}` = `{v}`" for k,v in da_params.items()]))
            st.markdown(
                f"**Input:** {X_train.shape[0]} training samples × {n_pcs_for_classification} {component_label}s  \n"
                f"**Classes:** {list(unique_y)}  \n"
                f"**Split:** " + (f"Yes — {int((1-test_size)*100)}% train / {int(test_size*100)}% test"
                                   if (split_data and test_size is not None) else "No split — full dataset for train and eval")
            )
        y_pred_da = best_da.predict(X_test)
        st.subheader(f"{da_type} Confusion Matrix{title_suffix}")
        cm_da = confusion_matrix(y_test_enc, y_pred_da)
        fig_cm = px.imshow(cm_da, text_auto=True, x=list(unique_y), y=list(unique_y),
                            color_continuous_scale='Blues',
                            title=f"{da_type} Confusion Matrix — {'Test Set' if split_data else 'In-Sample (No Split)'}{title_suffix}")
        apply_white_theme(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        show_train_test_accuracy(da_type, best_da, X_train, X_test, y_train_enc, y_test_enc, split_data)

        st.subheader(f"{da_type} Accuracy vs. Number of {component_label}s ({component_label}2 → {component_label}{n_pcs_for_classification})")
        if da_type=="LDA":   cf = lambda: LDA()
        elif da_type=="QDA": cf = lambda rp=0.001: QDA(reg_param=rp)
        else:                cf = lambda: GaussianNB()
        fig_sw, sw_sum = accuracy_vs_pcs_plot(cf, da_type, X_pca_full_for_sweep, y_encoded, n_pcs_for_classification, n_pcs_for_classification)
        if fig_sw: apply_white_theme(fig_sw); st.plotly_chart(fig_sw, use_container_width=True); st.info(sw_sum)
        else: st.warning(sw_sum)

        if n_pcs_for_classification == 2:
            st.subheader(f"{da_type} Decision Boundary{title_suffix}")
            render_decision_boundary(
                da_type, best_da, X_class[:, :2], y_encoded, unique_y,
                f"{component_label}1", f"{component_label}2",
                title_suffix, db_legend_mode
            )
        else:
            st.info(f"Decision boundary only for exactly 2 {component_label}s (currently {n_pcs_for_classification}).")

        # ── Formal visuals ─────────────────────────────────────────────────────
        render_classification_report(da_type, y_test_enc, y_pred_da, unique_y)

        da_has_proba = hasattr(best_da, "predict_proba")
        render_roc_curve(da_type, best_da, X_train, X_test, y_train_enc, y_test_enc, unique_y, da_has_proba)

    # ── KNN ───────────────────────────────────────────────────────────────────
    if run_knn:
        # Ensure k never exceeds number of training samples (minimum 1)
        n_train_knn = len(y_train_enc)
        k_safe = max(1, min(k, n_train_knn))
        if k_safe < k:
            st.warning(
                f"K was reduced from {k} to {k_safe} — training set has only {n_train_knn} samples."
            )

        try:
            best_knn = KNeighborsClassifier(n_neighbors=k_safe)
            best_knn.fit(X_train, y_train_enc)
        except Exception as e:
            st.error(f"KNN failed to fit: {e}")
            best_knn = None

        if best_knn is not None:
            with st.expander("📋 KNN Model Details", expanded=True):
                st.markdown(
                    f"**K-Nearest Neighbors:** classifies each sample by majority vote among its "
                    f"**{k_safe} nearest neighbors** in {component_label} space (Euclidean distance). "
                    "No explicit training — the model memorizes the training set and searches at prediction time."
                )
                st.markdown(f"**Parameters:** `n_neighbors`=`{k_safe}`, `metric`=`minkowski(p=2)`, `weights`=`uniform`, `algorithm`=`auto`")
                st.markdown(
                    f"**Input:** {n_train_knn} training samples × {n_pcs_for_classification} {component_label}s  \n"
                    f"**Classes:** {list(unique_y)}  \n"
                    f"**Split:** " + (f"Yes — {int((1-test_size)*100)}% / {int(test_size*100)}%"
                                       if (split_data and test_size is not None) else "No split")
                )
            y_pred_knn = best_knn.predict(X_test)
            st.subheader(f"KNN Confusion Matrix{title_suffix}")
            cm_knn = confusion_matrix(y_test_enc, y_pred_knn)
            fig_ck = px.imshow(cm_knn, text_auto=True, x=list(unique_y), y=list(unique_y),
                                color_continuous_scale='Blues',
                                title=f"KNN Confusion Matrix — {'Test Set' if split_data else 'In-Sample (No Split)'}{title_suffix}")
            apply_white_theme(fig_ck)
            st.plotly_chart(fig_ck, use_container_width=True)
            show_train_test_accuracy(f"KNN (k={k_safe})", best_knn, X_train, X_test, y_train_enc, y_test_enc, split_data)

            st.subheader(f"KNN Accuracy vs. Number of {component_label}s ({component_label}2 → {component_label}{n_pcs_for_classification}, k={k_safe})")
            fig_ksw, ksw_sum = accuracy_vs_pcs_plot(
                lambda k=k_safe: KNeighborsClassifier(n_neighbors=k), f"KNN (k={k_safe})",
                X_pca_full_for_sweep, y_encoded, n_pcs_for_classification, n_pcs_for_classification)
            if fig_ksw:
                apply_white_theme(fig_ksw)
                st.plotly_chart(fig_ksw, use_container_width=True)
                st.info(ksw_sum)
            else:
                st.warning(ksw_sum)

            if n_pcs_for_classification == 2:
                st.subheader(f"KNN Decision Boundary{title_suffix}")
                render_decision_boundary(
                    f"KNN (k={k_safe})", best_knn, X_class[:, :2], y_encoded, unique_y,
                    f"{component_label}1", f"{component_label}2",
                    title_suffix, db_legend_mode
                )
            else:
                st.info(f"Decision boundary only for exactly 2 {component_label}s (currently {n_pcs_for_classification}).")

            render_classification_report(f"KNN (k={k_safe})", y_test_enc, y_pred_knn, unique_y)
            render_roc_curve(f"KNN (k={k_safe})", best_knn, X_train, X_test,
                              y_train_enc, y_test_enc, unique_y, has_proba=True)
    # ── DECISION TREE ─────────────────────────────────────────────────────────
    if run_dt:
        best_dt = DecisionTreeClassifier(
            max_depth=dt_max_depth,
            criterion=dt_criterion,
            min_samples_leaf=dt_min_samples,
            random_state=42
        )
        best_dt.fit(X_train, y_train_enc)

        with st.expander("📋 Decision Tree Model Details", expanded=True):
            st.markdown(
                "**Decision Tree:** recursively partitions the component space by finding the single "
                "split threshold on one component at a time that best separates classes. Each internal "
                "node is a yes/no question (`PCn ≤ threshold`); each leaf assigns a class. "
                "Fast, interpretable, and handles multi-class natively with no distributional assumptions. "
                "Prone to over-fitting on small datasets — use `max_depth` and `min_samples_leaf` to control."
            )
            st.markdown(
                f"**Parameters:** `max_depth`=`{dt_max_depth}`, `criterion`=`{dt_criterion}`, "
                f"`min_samples_leaf`=`{dt_min_samples}`, `random_state`=`42`"
            )
            st.markdown(
                f"**Input:** {X_train.shape[0]} training samples × {n_pcs_for_classification} {component_label}s  \n"
                f"**Classes:** {list(unique_y)}  \n"
                f"**Split:** " + (f"Yes — {int((1-test_size)*100)}% train / {int(test_size*100)}% test"
                                   if (split_data and test_size is not None) else "No split — full dataset for train and eval")
            )

        y_pred_dt = best_dt.predict(X_test)

        st.subheader(f"Decision Tree Confusion Matrix{title_suffix}")
        cm_dt = confusion_matrix(y_test_enc, y_pred_dt)
        fig_cm_dt = px.imshow(cm_dt, text_auto=True, x=list(unique_y), y=list(unique_y),
                               color_continuous_scale='Blues',
                               title=f"Decision Tree Confusion Matrix — {'Test Set' if split_data else 'In-Sample (No Split)'}{title_suffix}")
        apply_white_theme(fig_cm_dt)
        st.plotly_chart(fig_cm_dt, use_container_width=True)
        show_train_test_accuracy(f"Decision Tree (depth={dt_max_depth})", best_dt, X_train, X_test, y_train_enc, y_test_enc, split_data)

        # Accuracy vs PCs sweep
        st.subheader(f"Decision Tree Accuracy vs. Number of {component_label}s ({component_label}2 → {component_label}{n_pcs_for_classification})")
        fig_dtsw, dtsw_sum = accuracy_vs_pcs_plot(
            lambda md=dt_max_depth, cr=dt_criterion, ms=dt_min_samples: DecisionTreeClassifier(
                max_depth=md, criterion=cr, min_samples_leaf=ms, random_state=42),
            f"Decision Tree (depth={dt_max_depth})",
            X_pca_full_for_sweep, y_encoded, n_pcs_for_classification, n_pcs_for_classification
        )
        if fig_dtsw:
            apply_white_theme(fig_dtsw)
            st.plotly_chart(fig_dtsw, use_container_width=True)
            st.info(dtsw_sum)
        else:
            st.warning(dtsw_sum)

        # Decision boundary (2 components only)
        if n_pcs_for_classification == 2:
            st.subheader(f"Decision Tree Decision Boundary{title_suffix}")
            render_decision_boundary(
                f"Decision Tree (depth={dt_max_depth})", best_dt, X_class[:, :2], y_encoded, unique_y,
                f"{component_label}1", f"{component_label}2",
                title_suffix, db_legend_mode
            )
        else:
            st.info(f"Decision boundary only available for exactly 2 {component_label}s (currently {n_pcs_for_classification}).")

        # Tree visualization
        st.subheader("Decision Tree Structure")
        actual_depth = best_dt.get_depth()
        n_leaves     = best_dt.get_n_leaves()
        st.caption(f"Actual depth: {actual_depth} | Leaves: {n_leaves} | "
                   f"Features used: {best_dt.n_features_in_} {component_label}s")

        # Visual tree diagram
        fig_tree, ax_tree = plt.subplots(figsize=(max(10, n_leaves * 1.5), max(5, actual_depth * 2)))
        if use_white_theme:
            fig_tree.patch.set_facecolor('white')
            ax_tree.set_facecolor('white')
        plot_tree(
            best_dt,
            feature_names=[f"{component_label}{i+1}" for i in range(n_pcs_for_classification)],
            class_names=[str(c) for c in unique_y],
            filled=True, rounded=True, fontsize=9, ax=ax_tree,
            impurity=True,   # show Gini/entropy split criterion
            value=False,     # hide "value = [n, n, ...]" class distribution counts
        )
        title_kw = dict(color='black') if use_white_theme else {}
        ax_tree.set_title(f"Decision Tree (depth={actual_depth}, criterion={dt_criterion})", fontsize=11, **title_kw)
        st.pyplot(fig_tree, bbox_inches='tight'); plt.close(fig_tree)

        # Feature importance bar chart
        st.subheader(f"Feature Importance ({component_label}s ranked by split contribution)")
        importances = best_dt.feature_importances_
        feat_names  = [f"{component_label}{i+1}" for i in range(n_pcs_for_classification)]
        df_imp = pd.DataFrame({'Component': feat_names, 'Importance': importances})
        df_imp = df_imp.sort_values('Importance', ascending=False)
        fig_imp = px.bar(df_imp, x='Component', y='Importance',
                          title=f"Decision Tree Feature Importances (Gini/Entropy impurity reduction)",
                          color='Importance', color_continuous_scale='Blues',
                          labels={'Importance': 'Relative Importance'})
        fig_imp.update_layout(height=350, showlegend=False)
        fig_imp.update_coloraxes(showscale=False)
        apply_white_theme(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Importance = total reduction in impurity weighted by the number of samples reaching each split. "
                   "Values sum to 1.0 across all components used.")

        # ── Formal visuals ─────────────────────────────────────────────────────
        render_classification_report(f"Decision Tree (depth={dt_max_depth})",
                                      y_test_enc, y_pred_dt, unique_y)

        render_roc_curve(f"Decision Tree (depth={dt_max_depth})", best_dt,
                          X_train, X_test, y_train_enc, y_test_enc, unique_y, has_proba=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "v29 — Hierarchical analysis: separate clear dendrogram + standalone interactive heatmap + "
    "combined view; fixed combined-figure color=None crash; fixed distinct cluster palette (no picker)."
)
