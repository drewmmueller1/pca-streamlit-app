import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def confidence_ellipse_params(x, y, n_std=2.0):
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle  = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0))
    return np.mean(x), np.mean(y), width, height, angle

def is_valid_hex(s):
    s = s.strip().lstrip('#')
    return len(s) == 6 and all(c in '0123456789abcdefABCDEF' for c in s)

DEFAULT_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#66c2a5','#fc8d62',
    '#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494',
    '#b3b3b3','#1b9e77','#d95f02','#7570b3','#e7298a',
]

def diagnose_pca(X_pca, var_ratios):
    """Return list of issue dicts describing degenerate PCA patterns."""
    issues = []
    n_samples, n_pcs = X_pca.shape
    check_pcs = min(n_pcs, 5)

    for i in range(check_pcs):
        col     = X_pca[:, i]
        col_std = np.std(col)

        if col_std < 1e-8:
            issues.append({
                'pc': i + 1, 'type': 'zero_variance',
                'detail': (
                    f"**PC{i+1} has essentially zero variance** (std ≈ {col_std:.2e}). "
                    "All points collapse onto this axis, appearing as a single line of dots. "
                    "\n\n**Likely causes:**\n"
                    "- A constant or near-constant feature column in your raw data (every sample has the same value).\n"
                    "- Duplicate columns that cancel each other out during centering.\n"
                    "- Normalization (e.g. Unit Area) turned all samples identical for one variable.\n"
                    "\n**What to do:**\n"
                    "1. Check your raw data for constant columns and remove them.\n"
                    "2. Look at the loadings for this PC — the dominant variable is the culprit.\n"
                    "3. Try a different preprocessing or normalization method."
                )
            })
        elif var_ratios[i] < 0.001 and i < 3:
            issues.append({
                'pc': i + 1, 'type': 'near_zero_variance',
                'detail': (
                    f"**PC{i+1} explains < 0.1% of variance** ({var_ratios[i]:.4%}). "
                    "Points along this axis will appear compressed into a thin line. "
                    "\n\n**Likely causes:**\n"
                    "- One feature with a much larger numeric scale than others is dominating "
                    "the covariance matrix (standardization may not have been applied).\n"
                    "- Near-duplicate or highly correlated features.\n"
                    "\n**What to do:**\n"
                    "1. Ensure Z-score or SNV standardization is applied before PCA.\n"
                    "2. Check the Scree Plot — if most variance is in PC1, there may be a scaling issue.\n"
                    "3. Inspect raw data for features with very different magnitudes."
                )
            })

    for i in range(check_pcs):
        col = X_pca[:, i]
        if np.std(col) < 1e-8:
            continue
        col_min, col_max = col.min(), col.max()
        span = col_max - col_min
        if span < 1e-8:
            continue
        near_min = np.sum(np.abs(col - col_min) < 0.05 * span)
        near_max = np.sum(np.abs(col - col_max) < 0.05 * span)
        if (near_min + near_max) / len(col) >= 0.80 and len(col) > 5:
            issues.append({
                'pc': i + 1, 'type': 'axis_clustering',
                'detail': (
                    f"**PC{i+1} shows strong axis-end clustering** — most points pile up "
                    "near the minimum or maximum, forming two distinct bands parallel to the other axis. "
                    f"\n\n**Likely causes:**\n"
                    "- A binary or near-binary variable (e.g., a 0/1-coded feature) is dominating this component.\n"
                    "- Severe class imbalance is pulling the PCA axis toward one group.\n"
                    "\n**What to do:**\n"
                    f"1. Inspect the PC{i+1} loadings plot — the top loading variable is almost certainly the cause.\n"
                    "2. Consider removing or re-encoding that variable.\n"
                    "3. If this is expected (e.g. a deliberate binary flag), this pattern may be informative."
                )
            })

    return issues

# ══════════════════════════════════════════════════════════════════════════════
# TITLE & INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════
st.title("PCA Visualization App for Lab Data")
st.markdown("""
Upload a CSV file with:
- **Row-wise:** Each row is a sample; first column is `label`. Column 1 row 1 must say `label`.
- **Column-wise:** Each column is a sample; first row is the label row. Column 1 = variables (e.g. wavelengths). Enable *Transpose* in Data Prep.
- **Pre-computed PCs:** Columns named PC1, PC2, … plus a `label` column. Enable the checkbox in Mode Selection.

Replicate measurements should share a name prefix separated by `_` (e.g. `Sample1_1`, `Sample1_2`).
""")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – MODE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Mode Selection", expanded=False):
    use_precomputed = st.checkbox("Use pre-computed PC scores", value=False)

# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOADERS
# ══════════════════════════════════════════════════════════════════════════════
uploaded_file    = st.file_uploader("Upload feature CSV", type="csv")
pc_uploaded_file = st.file_uploader("Upload PC scores CSV", type="csv")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
if use_precomputed:
    if pc_uploaded_file is None:
        st.info("Please upload the PC scores CSV to proceed.")
        st.stop()
    df = pd.read_csv(pc_uploaded_file)
    st.success(f"Loaded pre-computed PC dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    st.subheader("Data Overview")
    st.dataframe(df.head())
    if 'label' not in df.columns:
        st.error("PC CSV must have a 'label' column.")
        st.stop()
    df['label'] = df['label'].astype(str).str.split('_').str[0]
    st.info(f"Simplified labels: {df['label'].nunique()} unique classes")
    X = df.drop('label', axis=1).select_dtypes(include=[np.number])
    y = df['label']
    if X.empty:
        st.error("No numerical PC columns found.")
        st.stop()
    X_scaled       = X.values
    X_pca          = X_scaled
    n_total_pcs    = X_pca.shape[1]
    var_ratios     = np.full(n_total_pcs, 1.0 / n_total_pcs)
    pca_full       = None
    is_precomputed = True

else:
    if uploaded_file is None:
        st.info("Please upload the feature CSV to proceed.")
        st.stop()
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    st.subheader("Data Overview")
    st.dataframe(df.head())
    if 'label' not in df.columns:
        st.warning("No 'label' column found. Will be generated from column names if transposing.")

    # ── Sidebar: Data Prep ────────────────────────────────────────────────────
    with st.sidebar.expander("Data Prep Options", expanded=False):
        transpose_data    = st.checkbox(
            "Transpose Dataset (samples are columns)", value=False,
            help="Swaps rows/cols. Use when samples are columns and features are rows."
        )
        preprocess_option = st.radio(
            "Spectral preprocessing:", ['None', 'SNV', 'Z-score'], index=2
        )

    # ── Sidebar: Normalization ────────────────────────────────────────────────
    with st.sidebar.expander("Normalization Options", expanded=True):
        st.markdown("Applied **after** spectral preprocessing, **before** PCA.")
        norm_option = st.radio(
            "Normalization method:",
            [
                'None',
                'Unit Area (Total Sum)',
                'Unit Vector (L2 Norm)',
                'Min-Max (per sample)',
                'Internal Standard',
            ],
            index=0,
            help=(
                "**Unit Area:** divides each sample by its total sum. "
                "Removes concentration differences; good for chromatography area normalization.\n\n"
                "**Unit Vector (L2):** divides each sample by its Euclidean length. "
                "Scale-invariant; commonly used for spectral data.\n\n"
                "**Min-Max:** rescales each sample so its minimum = 0 and maximum = 1.\n\n"
                "**Internal Standard:** each variable in a sample is divided by that same "
                "variable's value in the designated IS rows. "
                "No cross-variable mixing — each variable is normalized to its own IS value."
            )
        )

    # ── Transpose ─────────────────────────────────────────────────────────────
    if transpose_data:
        if df.shape[1] < 2:
            st.error("Dataset too narrow for transpose.")
            st.stop()
        features     = df.iloc[:, 0].values
        data         = df.iloc[:, 1:].T
        data.columns = features
        sample_names = df.columns[1:]
        data['label'] = [name.split('_')[0] for name in sample_names]
        df = data.reset_index(drop=True)
        st.success(f"Transposed: {df.shape[0]} samples, {df.shape[1]-1} features.")
    else:
        if 'label' not in df.columns:
            st.error("CSV must have a 'label' column (or enable Transpose).")
            st.stop()

    df['label'] = df['label'].astype(str).str.split('_').str[0]
    st.info(f"Simplified labels: {df['label'].nunique()} unique classes")

    X = df.drop('label', axis=1).select_dtypes(include=[np.number])
    y = df['label']
    if X.empty:
        st.error("No numerical columns found.")
        st.stop()

    # Internal Standard feature selector (uses column/feature names, not sample labels)
    is_feature_col = None
    if norm_option == 'Internal Standard':
        is_feature_col = st.sidebar.selectbox(
            "Internal Standard feature (column)",
            options=list(X.columns),
            help=(
                "Select the feature column (row 1 of your CSV) that is your Internal Standard. "
                "Each sample's value for every feature is divided by that sample's own IS value. "
                "This is per-sample normalization — each chromatogram is divided by its own IS peak, "
                "independently of all other samples."
            )
        )

    # ── Spectral Preprocessing (per-sample) ───────────────────────────────────
    # SNV: mean-centers and scales each sample (row) by its own std — strictly per-sample.
    # Z-score: mean-centers and scales each feature (column) across all samples — per-feature/population.
    # Both are applied row-wise or column-wise as labeled; no cross-sample mixing for SNV.
    X_processed = X.copy()
    if preprocess_option == 'SNV':
        # Per-sample: each row is normalized by its own mean and std
        for i in range(X_processed.shape[0]):
            row_mean = np.mean(X_processed.iloc[i])
            row_std  = np.std(X_processed.iloc[i])
            if row_std > 0:
                X_processed.iloc[i] = (X_processed.iloc[i] - row_mean) / row_std
            else:
                st.warning(f"Row {i+1} zero variance — SNV skipped.")
    elif preprocess_option == 'Z-score':
        # Per-feature across all samples (population-level scaling)
        scaler      = StandardScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed),
                                    columns=X.columns, index=X.index)
    X = X_processed
    if preprocess_option != 'None':
        st.success(f"Spectral preprocessing applied: {preprocess_option}")

    # ── Normalization (all methods below are strictly per-sample) ─────────────
    # Each row (sample/chromatogram) is normalized using only that row's own values.
    # No values from other samples are used — this is not population normalization.
    if norm_option != 'None':
        X_norm = X.copy()

        if norm_option == 'Unit Area (Total Sum)':
            # Per-sample: divide each sample by its own total area sum
            for i in range(X_norm.shape[0]):
                row_sum = X_norm.iloc[i].sum()
                if row_sum != 0:
                    X_norm.iloc[i] = X_norm.iloc[i] / row_sum
                else:
                    st.warning(f"Row {i+1} sum = 0 — Unit Area normalization skipped.")
            st.success("Normalization applied: Unit Area / Total Sum (per sample)")

        elif norm_option == 'Unit Vector (L2 Norm)':
            # Per-sample: divide each sample by its own Euclidean length
            for i in range(X_norm.shape[0]):
                r_l2 = np.sqrt(np.sum(X_norm.iloc[i] ** 2))
                if r_l2 > 0:
                    X_norm.iloc[i] = X_norm.iloc[i] / r_l2
                else:
                    st.warning(f"Row {i+1} L2 = 0 — Unit Vector normalization skipped.")
            st.success("Normalization applied: Unit Vector / L2 Norm (per sample)")

        elif norm_option == 'Min-Max (per sample)':
            # Per-sample: rescale each sample to [0, 1] using its own min and max
            for i in range(X_norm.shape[0]):
                r_min, r_max = X_norm.iloc[i].min(), X_norm.iloc[i].max()
                if r_max > r_min:
                    X_norm.iloc[i] = (X_norm.iloc[i] - r_min) / (r_max - r_min)
                else:
                    st.warning(f"Row {i+1} constant — Min-Max normalization skipped.")
            st.success("Normalization applied: Min-Max (per sample)")

        elif norm_option == 'Internal Standard':
            if is_feature_col is None:
                st.error("Select an Internal Standard feature column in the sidebar.")
                st.stop()
            if is_feature_col not in X_norm.columns:
                st.error(f"Feature '{is_feature_col}' not found in the dataset.")
                st.stop()
            # Per-sample: each sample is divided by its own IS value for each feature.
            # sample[feature_i] /= sample[IS_feature]  — no cross-sample mixing at all.
            is_col_values = X_norm[is_feature_col].copy()  # IS value for each sample (row)
            zero_is_rows  = is_col_values[is_col_values == 0].index.tolist()
            if zero_is_rows:
                st.warning(
                    f"{len(zero_is_rows)} sample(s) have IS value = 0 and will NOT be normalized. "
                    f"Row indices: {zero_is_rows[:5]}{'...' if len(zero_is_rows) > 5 else ''}"
                )
            for i in X_norm.index:
                is_val = is_col_values.loc[i]
                if is_val != 0:
                    # Divide every feature in this sample by this sample's IS value
                    X_norm.loc[i] = X_norm.loc[i] / is_val
                # If IS = 0 for this sample, leave it un-normalized
            st.success(
                f"Normalization applied: Internal Standard "
                f"(IS feature = '{is_feature_col}', normalized per-sample to each sample's own IS value)"
            )

        X = X_norm

    # ── Final scaling before PCA ──────────────────────────────────────────────
    if preprocess_option != 'Z-score':
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    is_precomputed = False

# ══════════════════════════════════════════════════════════════════════════════
# DATA FILTERING
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Data Filtering")
unique_labels   = sorted(y.unique())
excluded_labels = st.multiselect("Labels to exclude", unique_labels, default=[])
mask_include    = ~y.isin(excluded_labels)
X_scaled        = X_scaled[mask_include]
y               = y[mask_include]

# ══════════════════════════════════════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════════════════════════════════════
if is_precomputed:
    X_pca       = X_scaled
    n_total_pcs = X_pca.shape[1]
    var_ratios  = np.full(n_total_pcs, 1.0/n_total_pcs) if n_total_pcs > 0 else np.array([])
    pca_full    = None
else:
    pca_full    = PCA()
    X_pca       = pca_full.fit_transform(X_scaled)
    n_total_pcs = X_pca.shape[1]
    var_ratios  = pca_full.explained_variance_ratio_

    # ── PCA Pipeline Summary ──────────────────────────────────────────────────
    st.subheader("PCA Pipeline Summary")

    # Determine what final scaling was applied immediately before PCA
    if preprocess_option == 'Z-score':
        pca_scaling_method = "Z-score (applied as spectral preprocessing)"
        pca_scaling_detail = (
            "Each **feature (column)** was mean-centered and divided by its population "
            "standard deviation across all samples (scikit-learn `StandardScaler`). "
            "This is population-wise standardization: mean and std are computed from the "
            "full dataset, so every feature contributes equally to the PCA covariance matrix."
        )
    else:
        pca_scaling_method = "Z-score (final PCA scaling, population-wise)"
        pca_scaling_detail = (
            "Before PCA, each **feature (column)** was mean-centered and divided by its "
            "population standard deviation across all samples (scikit-learn `StandardScaler`). "
            "This is applied automatically when spectral preprocessing is not Z-score, "
            "ensuring all features are on the same scale before decomposition."
        )

    norm_detail_map = {
        'None':                   "None — raw preprocessed values passed to PCA scaling.",
        'Unit Area (Total Sum)':  "Unit Area: each sample divided by its own total peak sum (per-sample).",
        'Unit Vector (L2 Norm)':  "Unit Vector (L2): each sample divided by its own Euclidean length (per-sample).",
        'Min-Max (per sample)':   "Min-Max: each sample rescaled to [0, 1] using its own min and max (per-sample).",
        'Internal Standard':      f"Internal Standard: each sample divided by its own IS peak value (per-sample, IS feature = '{is_feature_col if is_feature_col else 'N/A'}').",
    }

    n_samples_pca, n_features_pca = X_scaled.shape
    cum_var = np.cumsum(var_ratios)
    n_95 = int(np.argmax(cum_var >= 0.95)) + 1 if np.any(cum_var >= 0.95) else n_total_pcs
    n_99 = int(np.argmax(cum_var >= 0.99)) + 1 if np.any(cum_var >= 0.99) else n_total_pcs

    with st.expander("📋 Full PCA Pipeline Details", expanded=True):
        st.markdown(f"""
**Input data:** {n_samples_pca} samples × {n_features_pca} features

**Step 1 — Spectral preprocessing:** `{preprocess_option}`
{"- *SNV (Standard Normal Variate):* each sample row is mean-centered and divided by its own standard deviation. This is **per-sample** normalization — no information from other samples is used." if preprocess_option == 'SNV' else ""}
{"- *Z-score:* each feature column is mean-centered and divided by its population std across all samples. **Population-wise** (uses all samples to compute mean/std)." if preprocess_option == 'Z-score' else ""}
{"- *None:* no spectral preprocessing applied." if preprocess_option == 'None' else ""}

**Step 2 — Sample normalization:** `{norm_option}`
- {norm_detail_map.get(norm_option, norm_option)}

**Step 3 — PCA scaling (always applied before decomposition):** `{pca_scaling_method}`
- {pca_scaling_detail}

**Step 4 — PCA decomposition:** `sklearn.decomposition.PCA(svd_solver='full')`
- Algorithm: Singular Value Decomposition (SVD) on the covariance matrix of the standardized data.
- Components computed: **{n_total_pcs}** (all available)
- **PC1 explains:** {var_ratios[0]:.2%} of total variance
- **PC2 explains:** {var_ratios[1]:.2%} of total variance  
- **PC3 explains:** {var_ratios[2]:.2%} of total variance
- **95% variance reached at:** PC{n_95} ({cum_var[n_95-1]:.2%} cumulative)
- **99% variance reached at:** PC{n_99} ({cum_var[n_99-1]:.2%} cumulative)
""")

        # Quick variance bar for top 10 PCs
        top_n = min(10, n_total_pcs)
        fig_summary = go.Figure(go.Bar(
            x=[f"PC{i+1}" for i in range(top_n)],
            y=[v*100 for v in var_ratios[:top_n]],
            marker_color='steelblue', opacity=0.85,
            text=[f"{v*100:.1f}%" for v in var_ratios[:top_n]],
            textposition='outside'
        ))
        fig_summary.update_layout(
            title=f"Variance Explained — Top {top_n} PCs",
            xaxis_title="Principal Component",
            yaxis_title="% Variance Explained",
            height=280, margin=dict(t=40, b=30, l=40, r=10),
            yaxis=dict(range=[0, var_ratios[0]*110])
        )
        st.plotly_chart(fig_summary, use_container_width=True)

    # ── PCA Diagnostics ───────────────────────────────────────────────────────
    pca_issues = diagnose_pca(X_pca, var_ratios)
    if pca_issues:
        st.subheader("⚠️ PCA Diagnostics")
        st.warning(
            "Issues detected that may cause points to appear collapsed onto an axis "
            "(straight lines of dots along x or y). Expand each issue below for details and fixes."
        )
        for issue in pca_issues:
            label_map = {'zero_variance': 'Zero Variance', 'near_zero_variance': 'Near-Zero Variance',
                         'axis_clustering': 'Axis-End Clustering'}
            with st.expander(
                f"⚠️ PC{issue['pc']} — {label_map.get(issue['type'], issue['type'])}",
                expanded=True
            ):
                st.markdown(issue['detail'])
                col_data = X_pca[:, issue['pc'] - 1]
                fig_diag = go.Figure(go.Histogram(x=col_data, nbinsx=30,
                                                   marker_color='salmon', opacity=0.8))
                fig_diag.update_layout(
                    title=f"PC{issue['pc']} Score Distribution",
                    xaxis_title=f"PC{issue['pc']} Score", yaxis_title="Count",
                    height=250, margin=dict(t=35, b=30, l=30, r=10)
                )
                st.plotly_chart(fig_diag, use_container_width=True)

if n_total_pcs >= 2:
    X_pca_2d_global = X_pca[:, :2]
    y_global        = y
else:
    X_pca_2d_global = None
    y_global        = None

# ══════════════════════════════════════════════════════════════════════════════
# COLOR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Label Color Configuration")
unique_labels_all = sorted(y.unique())
n_labels          = len(unique_labels_all)

use_custom_colors = st.toggle(
    "Enable custom label colors (enter hex codes)", value=False,
    help="OFF = default palette. ON = type a hex code per label."
)

color_map_hex = {}
if use_custom_colors:
    st.markdown("Type a 6-digit hex code for each label. A color swatch confirms your choice.")
    n_cols     = min(4, n_labels)
    col_groups = st.columns(n_cols)
    for idx, lbl in enumerate(unique_labels_all):
        default_hex = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        with col_groups[idx % n_cols]:
            st.markdown(f"**`{lbl}`**")
            user_input = st.text_input(
                label=f"Hex color for {lbl}", value=default_hex,
                key=f"color_{lbl}", label_visibility="collapsed",
            )
            user_input = user_input.strip()
            chosen_hex = ('#' + user_input.lstrip('#')) if (user_input and is_valid_hex(user_input)) else default_hex
            color_map_hex[lbl] = chosen_hex
            st.markdown(
                f'<div style="width:100%;height:18px;border-radius:4px;background:{chosen_hex};'
                f'border:1px solid #ccc;margin-bottom:6px;"></div>',
                unsafe_allow_html=True
            )
else:
    for idx, lbl in enumerate(unique_labels_all):
        color_map_hex[lbl] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – PLOT OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Plot Options", expanded=False):
    show_2d         = st.checkbox("Show 2D PCA Plot (Static)", value=True)
    legend_separate = st.checkbox("Show legend in separate figure for PCA plots", value=False)
    show_3d         = st.checkbox("Show 3D PCA Plot (Interactive)", value=True)
    show_scree      = st.checkbox("Show Scree Plot", value=True)
    if show_scree:
        cum_var = np.cumsum(var_ratios)
        n_99    = np.argmax(cum_var >= 0.99) + 1  if np.any(cum_var >= 0.99)  else n_total_pcs
        n_999   = np.argmax(cum_var >= 0.999) + 1 if np.any(cum_var >= 0.999) else n_total_pcs
        n_scree = st.slider("Number of PCs to Show in Scree Plot", 1, n_999, n_99)
    else:
        n_scree = n_total_pcs
        n_99    = n_total_pcs

    show_loadings = st.checkbox("Show Loadings Plot", value=True)
    if show_loadings and not is_precomputed:
        loadings_type = st.selectbox(
            "Loadings Plot Type",
            ["Bar Graph (Discrete, e.g., GCMS)",
             "Connected Scatterplot (Continuous, e.g., Spectroscopy)"],
            index=0
        )
    else:
        loadings_type = "Bar Graph (Discrete, e.g., GCMS)"

    if show_2d and n_total_pcs >= 2:
        st.markdown("**2D Plot PC Axes**")
        pc_x_2d  = st.selectbox("X-axis PC", [f"PC{i+1}" for i in range(n_total_pcs)], index=0, key="pc_x_2d")
        pc_y_2d  = st.selectbox("Y-axis PC", [f"PC{i+1}" for i in range(n_total_pcs)], index=1, key="pc_y_2d")
        pc_x_idx = int(pc_x_2d[2:]) - 1
        pc_y_idx = int(pc_y_2d[2:]) - 1
    else:
        pc_x_idx, pc_y_idx = 0, 1
        pc_x_2d, pc_y_2d   = "PC1", "PC2"

with st.sidebar.expander("Download Options", expanded=False):
    num_save_pcs = st.slider("Number of PCs to Save", 1, min(10, n_total_pcs), 3)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – CLASSIFICATION OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar.expander("Classification Options", expanded=False):
    run_da = st.checkbox("Run Discriminant Analysis")
    if run_da:
        da_type      = st.selectbox("Discriminant Analysis Type", ["LDA","QDA","GaussianNB"], index=0)
        show_ellipse = st.checkbox("Show Ellipse (JMP-style) Plot", value=True)
        ellipse_std  = st.slider("Ellipse confidence (σ)", 1.0, 3.0, 2.0, 0.5)
    else:
        da_type = "LDA"
        show_ellipse = False; ellipse_std = 2.0

    run_knn = st.checkbox("Run K-Nearest Neighbors (KNN)")
    k       = st.slider("K value", 1, 20, 5) if run_knn else 5

    run_kmeans = st.checkbox("Run K-Means Clustering")
    if run_kmeans:
        auto_optimize_k      = st.checkbox("Auto-optimize K", value=False)
        n_clusters           = st.slider("Number of clusters", 2, 10, 3) if not auto_optimize_k else 3
        show_elbow           = st.checkbox("Show Elbow Plot", value=True)
        show_silhouette      = st.checkbox("Show Silhouette Plot", value=True)
        show_cluster_profile = st.checkbox("Show Cluster Profile Plots", value=True)
    else:
        auto_optimize_k = False; show_elbow = False
        show_silhouette = False; show_cluster_profile = False; n_clusters = 3

with st.sidebar.expander("Classification Input Options", expanded=True):
    max_pcs_for_class        = min(10, n_total_pcs)
    n_pcs_for_classification = st.slider(
        "Number of PCs used for Classification", 1, max_pcs_for_class, 2,
        help="How many PCs to feed into LDA / QDA / KNN. Decision boundary plots need exactly 2."
    )

# ══════════════════════════════════════════════════════════════════════════════
# LABEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Label Configuration")
label_mode = st.radio("Label Mode", ["Default Labels","Combined Groups"], index=0)
y_plot     = y.copy()

if label_mode == "Default Labels":
    st.info("Using default simplified labels for plots and models.")
    selected_for_a = []; selected_for_b = []
    rename_a = "Group A"; rename_b = "Group B"
    apply_to_plots = False
else:
    if n_total_pcs >= 2:
        unique_classes = sorted(y_global.unique())
        selected_for_a = st.multiselect("Select labels for Group A", unique_classes, default=unique_classes[:1])
        selected_for_b = st.multiselect("Select labels for Group B", unique_classes, default=unique_classes[1:2])
        rename_a       = st.text_input("Rename Group A", value=f"Group A ({', '.join(selected_for_a)})")
        rename_b       = st.text_input("Rename Group B", value=f"Group B ({', '.join(selected_for_b)})")
        apply_to_plots = st.checkbox("Use combined labels for plots", value=True)
        if apply_to_plots and selected_for_a and selected_for_b:
            y_plot = y_plot.replace({lbl: rename_a for lbl in selected_for_a})
            y_plot = y_plot.replace({lbl: rename_b for lbl in selected_for_b})
    else:
        selected_for_a=[]; selected_for_b=[]; rename_a="Group A"; rename_b="Group B"; apply_to_plots=False

unique_plot_labels = sorted(y_plot.unique())
plot_color_map = {lbl: color_map_hex.get(lbl, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                  for i, lbl in enumerate(unique_plot_labels)}

# ══════════════════════════════════════════════════════════════════════════════
# 1. 2D PCA PLOT
# ══════════════════════════════════════════════════════════════════════════════
if show_2d and n_total_pcs >= 2:
    st.subheader(f"2D PCA Plot ({pc_x_2d} vs {pc_y_2d})")
    xv   = X_pca[:, pc_x_idx]
    yv   = X_pca[:, pc_y_idx]
    xvar = var_ratios[pc_x_idx]
    yvar = var_ratios[pc_y_idx]

    df_plot_2d          = pd.DataFrame({pc_x_2d: xv, pc_y_2d: yv})
    df_plot_2d['label'] = y_plot.values
    ul2d       = sorted(df_plot_2d['label'].unique())
    mpl_colors = [plot_color_map.get(l, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) for i, l in enumerate(ul2d)]

    if legend_separate:
        fig_main, ax_main = plt.subplots(figsize=(8, 6))
        for lbl, col in zip(ul2d, mpl_colors):
            mask = df_plot_2d['label'] == lbl
            ax_main.scatter(df_plot_2d[mask][pc_x_2d], df_plot_2d[mask][pc_y_2d], c=col, label=lbl, s=50)
        ax_main.set_xlabel(f"{pc_x_2d} ({xvar:.1%})"); ax_main.set_ylabel(f"{pc_y_2d} ({yvar:.1%})")
        ax_main.set_title("Static 2D PCA Plot"); ax_main.grid(True, alpha=0.3)
        st.pyplot(fig_main); plt.close(fig_main)
        fig_legend, ax_leg = plt.subplots(figsize=(2, len(ul2d)*0.5))
        ax_leg.axis('off')
        handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
                   for l, c in zip(ul2d, mpl_colors)]
        ax_leg.legend(handles=handles, loc='center')
        st.pyplot(fig_legend); plt.close(fig_legend)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for lbl, col in zip(ul2d, mpl_colors):
            mask = df_plot_2d['label'] == lbl
            ax.scatter(df_plot_2d[mask][pc_x_2d], df_plot_2d[mask][pc_y_2d], c=col, label=lbl, s=50)
        ax.set_xlabel(f"{pc_x_2d} ({xvar:.1%})"); ax.set_ylabel(f"{pc_y_2d} ({yvar:.1%})")
        ax.set_title("Static 2D PCA Plot"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)
elif show_2d:
    st.warning("Need at least 2 features for 2D plot.")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 3D PCA PLOT
# ══════════════════════════════════════════════════════════════════════════════
if show_3d and n_total_pcs >= 3:
    st.subheader("3D PCA Plot (Interactive)")
    if is_precomputed:
        X_pca_3d    = X_pca[:, :3]
        explained_3d = var_ratios[:3]
    else:
        pca_3d       = PCA(n_components=3)
        X_pca_3d     = pca_3d.fit_transform(X_scaled)
        explained_3d = pca_3d.explained_variance_ratio_
    df_plot          = pd.DataFrame(X_pca_3d, columns=['PC1','PC2','PC3'])
    df_plot['label'] = y_plot.values

    fig_3d = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                            color='label', color_discrete_map=plot_color_map)
    fig_3d.update_traces(marker=dict(size=5))
    if legend_separate:
        fig_3d.update_layout(showlegend=False)
    else:
        fig_3d.update_layout(title="Interactive 3D PCA Plot",
                              scene=dict(xaxis_title=f"PC1 ({explained_3d[0]:.1%})",
                                         yaxis_title=f"PC2 ({explained_3d[1]:.1%})",
                                         zaxis_title=f"PC3 ({explained_3d[2]:.1%})"))
    st.plotly_chart(fig_3d, use_container_width=True)
elif show_3d:
    st.warning("Need at least 3 features for 3D plot.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SCREE PLOT
# ══════════════════════════════════════════════════════════════════════════════
if show_scree:
    st.subheader("Scree Plot: Variance Explained")
    if is_precomputed:
        st.warning("Using equal-variance assumption for pre-computed PCs.")
    var_ratio = var_ratios[:n_scree] * 100
    fig_scree = make_subplots(specs=[[{"secondary_y": False}]])
    fig_scree.add_trace(
        go.Bar(x=[f'PC{i+1}' for i in range(n_scree)], y=var_ratio,
               name='% Variance', marker_color='lightblue'), secondary_y=False
    )
    for i, v in enumerate(var_ratio):
        fig_scree.add_annotation(x=f'PC{i+1}', y=v, text=f'{v:.1f}%',
                                  showarrow=False, yshift=10, font=dict(size=10))
    fig_scree.update_layout(title=f"Scree Plot (Showing {n_scree} PCs)",
                             xaxis_title="Principal Components",
                             yaxis_title="% Variance Explained")
    fig_scree.update_yaxes(range=[0, var_ratio.max()*1.1], secondary_y=False)
    st.plotly_chart(fig_scree, use_container_width=True)
    st.info(f"Total variance by shown PCs: {np.sum(var_ratios[:n_scree]):.1%} | ≥99% at PC{n_99}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FACTOR LOADINGS
# ══════════════════════════════════════════════════════════════════════════════
if show_loadings:
    if pca_full is None:
        st.warning("Loadings not available for pre-computed PC mode.")
    else:
        # Part A: Top-3 PCs grouped
        st.subheader("Factor Loadings Plot (Top 3 PCs)")
        max_pcs       = min(3, n_total_pcs)
        valid_indices = [i for i in range(max_pcs) if var_ratios[i] > 0]
        num_valid     = len(valid_indices)
        if num_valid == 0:
            st.warning("No PCs with >0% variance.")
        else:
            st.info(f"Showing loadings for {num_valid} valid PCs (out of top 3)")
            loadings_top3     = pd.DataFrame(
                pca_full.components_[valid_indices],
                columns=X.columns,
                index=[f'PC{i+1}' for i in valid_indices]
            )
            loadings_top3_abs = loadings_top3.abs()

            if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                fig_top3    = go.Figure()
                bar_colors  = px.colors.qualitative.Set3[:num_valid]
                sorted_vars = loadings_top3_abs.max(axis=0).sort_values(ascending=False).index
                for i, pc in enumerate(loadings_top3.index):
                    fig_top3.add_trace(go.Bar(
                        y=loadings_top3_abs.loc[pc].loc[sorted_vars].values,
                        x=sorted_vars, name=pc,
                        marker_color=bar_colors[i], width=0.25, base=0, offsetgroup=i
                    ))
                fig_top3.update_layout(
                    barmode='group', height=400, showlegend=True,
                    title="Loadings: Grouped Bar (Abs, Top 3 PCs)",
                    xaxis_title="Variables", yaxis_title="Loading Magnitude"
                )
                fig_top3.update_xaxes(tickangle=45, tickfont=dict(size=9))
            else:
                original_vars = X.columns.tolist()
                loadings_melt = loadings_top3_abs.reset_index().melt(
                    id_vars='index', var_name='Variable', value_name='Loading'
                )
                loadings_melt['PC']       = loadings_melt['index']
                loadings_melt['Variable'] = pd.Categorical(
                    loadings_melt['Variable'], categories=original_vars, ordered=True
                )
                loadings_melt = loadings_melt.sort_values(['PC','Variable'])
                fig_top3 = px.line(
                    loadings_melt, x='Variable', y='Loading', color='PC', markers=False,
                    title="Loadings: Connected Line Plot (Abs, Top 3 PCs)",
                    labels={'Variable':'Variables','Loading':'Loading Magnitude'}
                )
                fig_top3.update_traces(line=dict(width=2, dash='solid'))
                fig_top3.update_xaxes(tickangle=45, tickfont=dict(size=9))
                if len(original_vars) > 50:
                    st.warning("Many variables (>50) — zoom/pan for details.")

            st.plotly_chart(fig_top3, use_container_width=True)
            st.subheader("Loadings Table (Top 3 PCs)")
            st.dataframe(loadings_top3)

        # Part B: Single-PC explorer
        st.subheader("Factor Loadings — Single PC Explorer")
        sel_pc_num   = st.slider("Select PC to explore", 1, min(10, n_total_pcs), 1,
                                  help="Inspect any individual PC's factor loadings.")
        sel_pc_idx   = sel_pc_num - 1
        sel_pc_label = f"PC{sel_pc_num}"

        if sel_pc_idx >= n_total_pcs or var_ratios[sel_pc_idx] == 0:
            st.warning(f"{sel_pc_label} has zero variance — no loadings to display.")
        else:
            loadings_row     = pd.Series(pca_full.components_[sel_pc_idx], index=X.columns)
            loadings_abs_row = loadings_row.abs()
            st.info(f"Variance explained by {sel_pc_label}: {var_ratios[sel_pc_idx]:.1%}")

            if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                sorted_vars = loadings_abs_row.sort_values(ascending=False).index
                fig_single  = go.Figure()
                fig_single.add_trace(go.Bar(
                    x=sorted_vars, y=loadings_abs_row.loc[sorted_vars].values,
                    marker_color='steelblue', name=sel_pc_label
                ))
                fig_single.update_layout(
                    title=f"Loadings: {sel_pc_label} (Abs Values)",
                    xaxis_title="Variables", yaxis_title="Loading Magnitude", height=400
                )
                fig_single.update_xaxes(tickangle=45, tickfont=dict(size=9))
            else:
                original_vars = X.columns.tolist()
                fig_single    = go.Figure()
                fig_single.add_trace(go.Scatter(
                    x=original_vars, y=loadings_abs_row.loc[original_vars].values,
                    mode='lines', line=dict(width=2), name=sel_pc_label
                ))
                fig_single.update_layout(
                    title=f"Loadings: {sel_pc_label} (Connected Line, Abs Values)",
                    xaxis_title="Variables", yaxis_title="Loading Magnitude", height=400
                )
                fig_single.update_xaxes(tickangle=45, tickfont=dict(size=9))
                if len(X.columns) > 50:
                    st.warning("Many variables (>50) — zoom/pan for details.")

            st.plotly_chart(fig_single, use_container_width=True)
            st.subheader(f"Loadings Table — {sel_pc_label}")
            st.dataframe(loadings_row.to_frame(name=sel_pc_label))

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD PCA RESULTS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Download PCA Results")
col1, col2 = st.columns(2)
with col1:
    if is_precomputed:
        X_pca_save = X_pca[:, :num_save_pcs]
    else:
        pca_save   = PCA(n_components=num_save_pcs)
        X_pca_save = pca_save.fit_transform(X_scaled)
    df_scores          = pd.DataFrame(X_pca_save, columns=[f'PC{i+1}' for i in range(num_save_pcs)])
    df_scores['label'] = y.values
    st.download_button("Download PC Scores CSV", df_scores.to_csv(index=False), "pc_scores.csv", "text/csv")
with col2:
    if pca_full is not None:
        loadings_save = pd.DataFrame(pca_full.components_[:num_save_pcs],
                                      columns=X.columns,
                                      index=[f'PC{i+1}' for i in range(num_save_pcs)])
        st.download_button("Download Loadings CSV", loadings_save.to_csv(index=True), "pca_loadings.csv", "text/csv")
    else:
        st.info("Loadings not available for pre-computed mode.")
st.info(f"Downloads include top {num_save_pcs} PCs.")

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
if run_kmeans and X_pca_2d_global is not None:
    st.header("Clustering Results")
    if auto_optimize_k:
        inertias = [KMeans(n_clusters=k_i, random_state=42, n_init=10).fit(X_pca_2d_global).inertia_
                    for k_i in range(1, 11)]
        n_clusters = np.argmin(np.diff(np.diff(inertias))) + 2
        st.info(f"Auto-optimized K: {n_clusters}")

    kmeans       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labs = kmeans.fit_predict(X_pca_2d_global)
    df_cluster   = pd.DataFrame(X_pca_2d_global, columns=['PC1','PC2'])
    df_cluster['cluster'] = cluster_labs.astype(str)
    fig_cluster  = px.scatter(df_cluster, x='PC1', y='PC2', color='cluster',
                               title=f"K-Means Clustering (k={n_clusters}) on PC1 vs PC2",
                               color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_cluster, use_container_width=True)

    if show_elbow:
        st.subheader("Elbow Plot")
        inertias = [KMeans(n_clusters=k_i, random_state=42, n_init=10).fit(X_pca_2d_global).inertia_
                    for k_i in range(1, 11)]
        fig_elbow = px.line(x=range(1,11), y=inertias, markers=True, title="Elbow Plot")
        fig_elbow.update_layout(xaxis_title="K", yaxis_title="Inertia")
        st.plotly_chart(fig_elbow)
    if show_silhouette:
        st.subheader("Silhouette Plot")
        silhouettes = [silhouette_score(X_pca_2d_global,
                        KMeans(n_clusters=k_i, random_state=42, n_init=10).fit_predict(X_pca_2d_global))
                       for k_i in range(2, 11)]
        fig_sil = px.line(x=range(2,11), y=silhouettes, markers=True, title="Silhouette Score for Optimal K")
        fig_sil.update_layout(xaxis_title="K", yaxis_title="Silhouette Score")
        st.plotly_chart(fig_sil)
    if show_cluster_profile:
        st.subheader("Cluster Profile Plots")
        df_cents = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1','PC2'])
        df_cents['cluster'] = range(n_clusters)
        fig_prof = px.bar(df_cents.melt(id_vars='cluster'), x='cluster', y='value',
                           color='variable', barmode='group', title="Cluster Centroids on PC1 and PC2")
        st.plotly_chart(fig_prof)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
st.header("Classification Results")

if n_total_pcs < n_pcs_for_classification:
    st.error(f"Not enough PCs. Selected {n_pcs_for_classification}, only {n_total_pcs} exist.")
else:
    X_class = X_pca[:, :n_pcs_for_classification]

    if label_mode == "Default Labels":
        y_selected           = y_global
        title_suffix         = " (Multi-class)"
        X_pca_full_for_sweep = X_pca          # all PCs available for sweep
    else:
        if not selected_for_a or not selected_for_b:
            st.warning("Select groups to run combined classification.")
            st.stop()
        mask_group_a         = y_global.isin(selected_for_a)
        mask_group_b         = y_global.isin(selected_for_b)
        mask_selected        = mask_group_a | mask_group_b
        X_class              = X_class[mask_selected]
        X_pca_full_for_sweep = X_pca[mask_selected]
        y_selected           = np.where(mask_group_a[mask_selected], 0, 1)
        title_suffix         = ""

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_selected)
    unique_y  = le.classes_

    split_data = st.checkbox("Split into train/test sets")
    if split_data:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train_enc, y_test_enc = train_test_split(
            X_class, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train_enc, y_test_enc = X_class, X_class, y_encoded, y_encoded

    # ── helper: build accuracy-vs-PCs figure ─────────────────────────────────
    def accuracy_vs_pcs_plot(clf_factory, label, X_full, y_enc, n_selected, n_scree_max):
        """
        Cross-validates clf_factory() at 2, 3, 4 … max_sweep PCs.
        Starts at 2 PCs (not 1) so LDA/QDA always have at least a 2D input.
        Returns (plotly_figure, summary_string).
        """
        n_samples  = len(y_enc)
        n_classes  = len(np.unique(y_enc))

        # Need at least 2*n_classes samples for stratified CV to work
        if n_samples < 2 * n_classes:
            return None, (
                f"⚠️ Not enough samples to cross-validate "
                f"({n_samples} samples, {n_classes} classes). "
                f"Need at least {2 * n_classes}."
            )

        # Require cv >= 2 and every fold must have at least 1 sample per class
        cv_folds = min(5, n_samples // n_classes)
        if cv_folds < 2:
            return None, (
                f"⚠️ Too few samples per class for cross-validation "
                f"(cv would be {cv_folds}; need ≥ 2). "
                f"Add more replicates or switch to 'no split' mode."
            )

        # Cap the sweep at the number of PCs actually available and a reasonable ceiling
        max_sweep = min(n_scree_max, X_full.shape[1], 30)

        # Start at 2 PCs (1-PC input causes LDA/QDA issues with multi-class)
        start_pc = 2

        if max_sweep < start_pc:
            return None, (
                f"⚠️ Only {X_full.shape[1]} PC(s) available — need at least 2 to sweep."
            )

        sweep_pcs, sweep_mean, sweep_std, sweep_errors = [], [], [], []

        for n_pc in range(start_pc, max_sweep + 1):
            X_sw = X_full[:, :n_pc]
            try:
                scores = cross_val_score(
                    clf_factory(), X_sw, y_enc,
                    cv=cv_folds, scoring='accuracy',
                    error_score='raise'      # surface errors instead of hiding them
                )
                sweep_pcs.append(n_pc)
                sweep_mean.append(float(scores.mean()))
                sweep_std.append(float(scores.std()))
            except Exception as e:
                sweep_errors.append(f"PC{n_pc}: {type(e).__name__}: {str(e)[:120]}")

        # Surface any errors as an expander so the user can see what went wrong
        if sweep_errors:
            with st.expander(f"⚠️ {len(sweep_errors)} PC count(s) skipped during sweep — click to see why"):
                for msg in sweep_errors:
                    st.text(msg)

        if not sweep_pcs:
            err_preview = " | ".join(sweep_errors[:3]) if sweep_errors else "unknown error"
            return None, (
                f"⚠️ Accuracy sweep could not be computed for any PC count. "
                f"First error: {err_preview}"
            )

        arr      = np.array(sweep_mean)
        std_arr  = np.array(sweep_std)
        best_idx = int(np.argmax(arr))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_pcs, y=arr,
            mode='lines+markers',
            line=dict(color='royalblue', width=2),
            marker=dict(size=7),
            name='CV Accuracy',
            error_y=dict(type='data', array=list(std_arr), visible=True, color='lightblue')
        ))

        # Mark currently selected PC count (only if it falls within the swept range)
        if n_selected in sweep_pcs:
            fig.add_vline(
                x=n_selected, line_dash='dash', line_color='orange',
                annotation_text=f"Selected ({n_selected} PCs)",
                annotation_position="top left"
            )

        # Mark best PC count (avoid double-annotation if same as selected)
        if sweep_pcs[best_idx] != n_selected:
            fig.add_vline(
                x=sweep_pcs[best_idx], line_dash='dot', line_color='green',
                annotation_text=f"Best ({sweep_pcs[best_idx]} PCs, {arr[best_idx]:.3f})",
                annotation_position="top right"
            )

        fig.update_layout(
            title=f"{label} — CV Accuracy vs. Number of PCs (CV folds = {cv_folds})",
            xaxis_title="Number of PCs included (cumulative: PC1+PC2+…+PCn)",
            yaxis_title="Cross-Validated Accuracy",
            yaxis=dict(range=[max(0.0, float(arr.min()) - 0.1), 1.02]),
            xaxis=dict(tickmode='linear', dtick=1),
            height=430,
        )

        # Summary text — look up selected PC by value, not index
        sel_txt = ""
        if n_selected in sweep_pcs:
            si = sweep_pcs.index(n_selected)
            sel_txt = f"  |  Selected {n_selected} PCs → {arr[si]:.3f} ± {std_arr[si]:.3f}"
        summary = (
            f"Best CV accuracy: **{arr[best_idx]:.3f} ± {std_arr[best_idx]:.3f}** "
            f"at **{sweep_pcs[best_idx]} PCs**{sel_txt}"
        )
        return fig, summary

    # ══════════════════════════════════════════════════════════════════════════
    # DA MODEL
    # ══════════════════════════════════════════════════════════════════════════
    if run_da:
        if da_type == "LDA":
            best_da = LDA()
            da_params = {
                "solver": "svd",
                "shrinkage": "None",
                "n_components": "min(n_classes-1, n_features)",
                "store_covariance": False,
                "tol": 1e-4,
            }
            da_description = (
                "**Linear Discriminant Analysis (LDA)** finds linear combinations of features "
                "that best separate the classes. It assumes each class shares the **same covariance matrix** "
                "(homoscedastic). The decision boundary between any two classes is a straight line/hyperplane."
            )
        elif da_type == "QDA":
            best_da = QDA(reg_param=0.001)
            da_params = {
                "reg_param": 0.001,
                "store_covariance": False,
                "tol": 1e-4,
            }
            da_description = (
                "**Quadratic Discriminant Analysis (QDA)** fits a separate covariance matrix per class, "
                "allowing **curved (quadratic) decision boundaries**. More flexible than LDA but requires "
                "more data per class. `reg_param=0.001` adds a small regularization to prevent singular "
                "covariance matrices on small datasets."
            )
        else:
            best_da = GaussianNB()
            da_params = {
                "var_smoothing": 1e-9,
            }
            da_description = (
                "**Gaussian Naive Bayes** models each class as a multivariate Gaussian and assumes "
                "**all features are independent** given the class label (naive assumption). "
                "Very fast and works well when the independence assumption roughly holds."
            )

        best_da.fit(X_train, y_train_enc)

        # Model parameter card
        with st.expander(f"📋 {da_type} Model Details", expanded=True):
            st.markdown(da_description)
            st.markdown("**Parameters used:**")
            param_rows = "\n".join([f"- `{k}` = `{v}`" for k, v in da_params.items()])
            st.markdown(param_rows)
            st.markdown(
                f"**Input:** PC scores matrix — {X_train.shape[0]} training samples × "
                f"{n_pcs_for_classification} PCs  \n"
                f"**Classes:** {list(unique_y)}  \n"
                f"**Train/test split:** "
                + (f"Yes — {int((1-test_size)*100)}% train / {int(test_size*100)}% test"
                   if split_data else "No split — full dataset used for both training and evaluation")
            )
        y_pred_da = best_da.predict(X_test)
        acc_da    = accuracy_score(y_test_enc, y_pred_da)

        st.subheader(f"{da_type} Confusion Matrix{title_suffix}")
        cm_da = confusion_matrix(y_test_enc, y_pred_da)
        fig_cm_da = px.imshow(cm_da, text_auto=True, x=list(unique_y), y=list(unique_y),
                               color_continuous_scale='Blues',
                               title=f"{da_type} Confusion Matrix{title_suffix}")
        st.plotly_chart(fig_cm_da, use_container_width=True)
        st.write(f"**Accuracy at {n_pcs_for_classification} PCs:** {acc_da:.3f}")

        # ── Accuracy vs PCs sweep (PC2 → n_scree) ────────────────────────────
        st.subheader(f"{da_type} Accuracy vs. Number of PCs (PC2 → PC{n_scree})")
        if da_type == "LDA":
            clf_factory = lambda: LDA()
        elif da_type == "QDA":
            clf_factory = lambda: QDA(reg_param=0.001)
        else:
            clf_factory = lambda: GaussianNB()

        fig_sweep, sweep_summary = accuracy_vs_pcs_plot(
            clf_factory, da_type, X_pca_full_for_sweep,
            y_encoded, n_pcs_for_classification, n_scree
        )
        if fig_sweep:
            st.plotly_chart(fig_sweep, use_container_width=True)
            st.info(sweep_summary)
        else:
            st.warning(sweep_summary)

        # ── 2D decision boundary ──────────────────────────────────────────────
        if n_pcs_for_classification == 2:
            st.subheader(f"{da_type} Decision Boundary{title_suffix}")
            try:
                from mlxtend.plotting import plot_decision_regions
                fig_da, ax_da = plt.subplots(figsize=(8, 6))
                plot_decision_regions(X_class, y_encoded, clf=best_da, legend=2, ax=ax_da)
                ax_da.set_xlabel('PC1'); ax_da.set_ylabel('PC2')
                ax_da.set_title(f'{da_type} Decision Boundary{title_suffix}')
                st.pyplot(fig_da); plt.close(fig_da)
            except Exception as e:
                st.warning(f"Decision boundary plot failed: {e}")
        else:
            st.info(f"Decision boundary plot only available for exactly 2 PCs (currently {n_pcs_for_classification}).")

        # ── Ellipse (JMP-style) plot ──────────────────────────────────────────
        if show_ellipse and n_pcs_for_classification >= 2:
            st.subheader(f"{da_type} Ellipse Plot (JMP-style, {ellipse_std:.0f}σ)")
            fig_ell = go.Figure()
            for enc_val, cls_name in enumerate(unique_y):
                mask_c  = (y_encoded == enc_val)
                Xc      = X_class[mask_c, :2]
                if len(Xc) < 3:
                    continue
                col_hex = color_map_hex.get(str(cls_name), DEFAULT_COLORS[enc_val % len(DEFAULT_COLORS)])
                fig_ell.add_trace(go.Scatter(
                    x=Xc[:, 0], y=Xc[:, 1], mode='markers',
                    marker=dict(color=col_hex, size=7, opacity=0.8),
                    name=str(cls_name)
                ))
                ep = confidence_ellipse_params(Xc[:,0], Xc[:,1], n_std=ellipse_std)
                if ep:
                    mx, my, w, h, angle = ep
                    theta = np.linspace(0, 2*np.pi, 200)
                    cos_a = np.cos(np.radians(angle)); sin_a = np.sin(np.radians(angle))
                    ex = (w/2)*np.cos(theta)*cos_a - (h/2)*np.sin(theta)*sin_a + mx
                    ey = (w/2)*np.cos(theta)*sin_a + (h/2)*np.sin(theta)*cos_a + my
                    fig_ell.add_trace(go.Scatter(
                        x=ex, y=ey, mode='lines',
                        line=dict(color=col_hex, width=2, dash='solid'),
                        showlegend=False
                    ))
            fig_ell.update_layout(
                title=f"{da_type} Confidence Ellipses ({ellipse_std:.0f}σ) — PC1 vs PC2",
                xaxis_title="PC1", yaxis_title="PC2", height=550, legend_title="Class"
            )
            st.plotly_chart(fig_ell, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # KNN
    # ══════════════════════════════════════════════════════════════════════════
    if run_knn:
        best_knn = KNeighborsClassifier(n_neighbors=k)
        best_knn.fit(X_train, y_train_enc)

        # Model parameter card
        with st.expander(f"📋 KNN Model Details", expanded=True):
            st.markdown(
                f"**K-Nearest Neighbors (KNN)** classifies each sample by majority vote among its "
                f"**{k} nearest neighbors** in PC space, using Euclidean distance. "
                "There is no explicit training step — the algorithm memorizes the training set and "
                "classifies at prediction time by searching for the closest points."
            )
            st.markdown("**Parameters used:**")
            st.markdown(
                f"- `n_neighbors` = `{k}` — number of neighbors to vote\n"
                f"- `metric` = `minkowski` (p=2, equivalent to Euclidean distance)\n"
                f"- `weights` = `uniform` — all neighbors vote equally\n"
                f"- `algorithm` = `auto` — auto-selects ball_tree / kd_tree / brute\n"
            )
            st.markdown(
                f"**Input:** PC scores matrix — {X_train.shape[0]} training samples × "
                f"{n_pcs_for_classification} PCs  \n"
                f"**Classes:** {list(unique_y)}  \n"
                f"**Train/test split:** "
                + (f"Yes — {int((1-test_size)*100)}% train / {int(test_size*100)}% test"
                   if split_data else "No split — full dataset used for both training and evaluation")
            )

        y_pred_knn = best_knn.predict(X_test)
        acc_knn    = accuracy_score(y_test_enc, y_pred_knn)

        knn_title = f"KNN Confusion Matrix{title_suffix}"
        st.subheader(knn_title)
        cm_knn = confusion_matrix(y_test_enc, y_pred_knn)
        fig_cm_knn = px.imshow(cm_knn, text_auto=True, x=list(unique_y), y=list(unique_y),
                                color_continuous_scale='Blues', title=knn_title)
        st.plotly_chart(fig_cm_knn, use_container_width=True)
        st.write(f"**Accuracy at {n_pcs_for_classification} PCs:** {acc_knn:.3f}")

        # ── KNN Accuracy vs PCs sweep ─────────────────────────────────────────
        st.subheader(f"KNN Accuracy vs. Number of PCs (PC2 → PC{n_scree}, k={k})")
        fig_knn_sweep, knn_sweep_summary = accuracy_vs_pcs_plot(
            lambda: KNeighborsClassifier(n_neighbors=k),
            f"KNN (k={k})", X_pca_full_for_sweep,
            y_encoded, n_pcs_for_classification, n_scree
        )
        if fig_knn_sweep:
            st.plotly_chart(fig_knn_sweep, use_container_width=True)
            st.info(knn_sweep_summary)
        else:
            st.warning(knn_sweep_summary)

        if n_pcs_for_classification == 2:
            st.subheader(f"KNN Decision Boundary{title_suffix}")
            try:
                from mlxtend.plotting import plot_decision_regions
                fig_knn_db, ax_knn = plt.subplots(figsize=(8, 6))
                plot_decision_regions(X_class, y_encoded, clf=best_knn, legend=2, ax=ax_knn)
                ax_knn.set_xlabel('PC1'); ax_knn.set_ylabel('PC2')
                ax_knn.set_title(f'KNN Decision Boundary{title_suffix}')
                st.pyplot(fig_knn_db); plt.close(fig_knn_db)
            except Exception as e:
                st.warning(f"KNN decision boundary plot failed: {e}")
        else:
            st.info(f"Decision boundary plot only available for exactly 2 PCs (currently {n_pcs_for_classification}).")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "v13 — PCA pipeline summary card; model parameter detail cards for LDA/QDA/GaussianNB/KNN; "
    "sweep fixes (starts at PC2, surfaces errors, safe cv_folds)."
)
