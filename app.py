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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score, mean_squared_error
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
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
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
st.subheader("Data Overview")
st.dataframe(df.head())
if 'label' not in df.columns:
    st.warning("No 'label' column found — will be generated from column names if Transpose is enabled.")

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
            "Generally outperforms PCR when the target is strongly related to specific features."
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
X = X_raw.copy()

if norm_option != 'None':
    if norm_option == 'Unit Area (Total Sum)':
        for i in range(X.shape[0]):
            s = X.iloc[i].sum()
            if s != 0:
                X.iloc[i] = X.iloc[i] / s
            else:
                st.warning(f"Row {i+1} sum = 0 — Unit Area skipped for this sample.")
        st.success("Normalization applied: Unit Area / Total Sum (per sample)")

    elif norm_option == 'Unit Vector (L2 Norm)':
        for i in range(X.shape[0]):
            l2 = np.sqrt(np.sum(X.iloc[i]**2))
            if l2 > 0:
                X.iloc[i] = X.iloc[i] / l2
            else:
                st.warning(f"Row {i+1} L2 = 0 — Unit Vector skipped for this sample.")
        st.success("Normalization applied: Unit Vector / L2 Norm (per sample)")

    elif norm_option == 'Min-Max (per sample)':
        for i in range(X.shape[0]):
            rmin, rmax = X.iloc[i].min(), X.iloc[i].max()
            if rmax > rmin:
                X.iloc[i] = (X.iloc[i] - rmin) / (rmax - rmin)
            else:
                st.warning(f"Row {i+1} is constant — Min-Max skipped for this sample.")
        st.success("Normalization applied: Min-Max (per sample)")

    elif norm_option == 'Internal Standard':
        if is_label_val is None:
            st.error("Select an Internal Standard label in the sidebar.")
            st.stop()
        # Identify IS rows by label
        is_mask = (y_raw == is_label_val)
        if is_mask.sum() == 0:
            st.error(f"No rows found with label '{is_label_val}'.")
            st.stop()
        # Average IS values across replicates — one IS value per variable (column)
        is_values = X.loc[is_mask].mean(axis=0)   # shape: (n_features,)
        zero_cols = is_values[is_values == 0].index.tolist()
        if zero_cols:
            st.warning(
                f"{len(zero_cols)} variable(s) have IS value = 0 and will NOT be normalized: "
                f"{zero_cols[:5]}{'...' if len(zero_cols) > 5 else ''}"
            )
        # Divide every sample by the IS value for each column independently
        for col in X.columns:
            if is_values[col] != 0:
                X[col] = X[col] / is_values[col]
        st.success(
            f"Normalization applied: Internal Standard "
            f"(IS label = '{is_label_val}', {is_mask.sum()} IS row(s) averaged per variable)"
        )

y = y_raw.copy()

# ══════════════════════════════════════════════════════════════════════════════
# STANDARDIZATION (applied after normalization)
# ══════════════════════════════════════════════════════════════════════════════
if preprocess_option == 'SNV':
    # Per-sample (row-wise): center and scale each sample by its own mean and std
    X_proc = X.copy()
    for i in range(X_proc.shape[0]):
        row_mean = X_proc.iloc[i].mean()
        row_std  = X_proc.iloc[i].std()
        if row_std > 0:
            X_proc.iloc[i] = (X_proc.iloc[i] - row_mean) / row_std
        else:
            st.warning(f"Row {i+1} zero variance — SNV skipped for this sample.")
    X_scaled = X_proc.values
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
X_scores   = None   # scores matrix (samples × components), used everywhere below
var_ratios = None
component_label = "PC"   # "PC" for PCA/PCR, "LV" for PLS

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

else:  # PLS
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
        yaxis=dict(range=[0, var_ratios[0]*115])
    )
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
                st.plotly_chart(fig_diag, use_container_width=True)

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
    legend_separate = st.checkbox("Separate legend figure", value=False)
    show_3d         = st.checkbox("Show 3D Scores Plot (Interactive)", value=True)
    show_scree      = st.checkbox("Show Scree Plot", value=True)
    if show_scree:
        n_99_s  = np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else n_total_pcs
        n_999_s = np.argmax(cum_var >= 0.999)+ 1 if np.any(cum_var >= 0.999) else n_total_pcs
        n_scree = st.slider(f"Number of {component_label}s in Scree", 1, max(n_999_s,2), max(n_99_s,2))
    else:
        n_scree = n_total_pcs
        n_99_s  = n_total_pcs

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
        da_type      = st.selectbox("DA Type", ["LDA","QDA","GaussianNB"], index=0)
        show_ellipse = st.checkbox("Show Ellipse (JMP-style) Plot", value=True)
        ellipse_std  = st.slider("Ellipse σ", 1.0, 3.0, 2.0, 0.5)
    else:
        da_type = "LDA"; show_ellipse = False; ellipse_std = 2.0
    run_knn = st.checkbox("Run KNN")
    k       = st.slider("K value", 1, 20, 5) if run_knn else 5
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
# 1. 2D SCORES PLOT
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
        for lbl, col in zip(ul2d, mpl_colors):
            m = df_2d['label']==lbl
            ax_m.scatter(df_2d[m][cx_2d], df_2d[m][cy_2d], c=col, label=lbl, s=50)
        ax_m.set_xlabel(f"{cx_2d} ({xvar:.1%})"); ax_m.set_ylabel(f"{cy_2d} ({yvar:.1%})")
        ax_m.set_title(f"2D {component_label} Scores"); ax_m.grid(True,alpha=0.3)
        st.pyplot(fig_m); plt.close(fig_m)
        fig_lg, ax_lg = plt.subplots(figsize=(2, len(ul2d)*0.5))
        ax_lg.axis('off')
        handles = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=8,label=l)
                   for l,c in zip(ul2d,mpl_colors)]
        ax_lg.legend(handles=handles,loc='center'); st.pyplot(fig_lg); plt.close(fig_lg)
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        for lbl, col in zip(ul2d, mpl_colors):
            m = df_2d['label']==lbl
            ax.scatter(df_2d[m][cx_2d], df_2d[m][cy_2d], c=col, label=lbl, s=50)
        ax.set_xlabel(f"{cx_2d} ({xvar:.1%})"); ax.set_ylabel(f"{cy_2d} ({yvar:.1%})")
        ax.set_title(f"2D {component_label} Scores Plot"); ax.legend(); ax.grid(True,alpha=0.3)
        st.pyplot(fig); plt.close(fig)
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
    st.plotly_chart(fig_scree, use_container_width=True)
    st.info(f"Shown: {np.sum(var_ratios[:n_scree]):.1%} | ≥99% at {component_label}{n_99_s}")

# ══════════════════════════════════════════════════════════════════════════════
# PCR / PLS REGRESSION DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════
if analysis_mode != "PCA (Principal Component Analysis)" and y_target is not None:
    st.subheader(f"{analysis_mode.split('(')[0].strip()} Regression Diagnostics")

    if analysis_mode == "PCR (Principal Component Regression)":
        # --- CV / MSE plot across number of PCs ---
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
        st.plotly_chart(fig_cv, use_container_width=True)

        # Final PCR model
        from sklearn.linear_model import LinearRegression
        lr_final = LinearRegression().fit(X_scores[:, :n_pcr_final], y_target)
        y_pred   = lr_final.predict(X_scores[:, :n_pcr_final])
        r2       = 1 - np.sum((y_target - y_pred)**2) / np.sum((y_target - y_target.mean())**2)
        mse_full = mean_squared_error(y_target, y_pred)
        st.info(f"PCR final model — {n_pcr_final} PCs | R² = {r2:.4f} | MSE = {mse_full:.4f}")

    else:  # PLS
        # --- CV / MSE plot across number of LVs ---
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
        st.plotly_chart(fig_cv_pls, use_container_width=True)

        y_pred   = pls_model.predict(X_scaled).ravel()
        r2       = 1 - np.sum((y_target-y_pred)**2)/np.sum((y_target-y_target.mean())**2)
        mse_full = mean_squared_error(y_target, y_pred)
        st.info(f"PLS model — {n_pls_components} LVs | R² = {r2:.4f} | MSE = {mse_full:.4f}")

    # --- Predicted vs Actual ---
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
    st.plotly_chart(fig_pva, use_container_width=True)

    # --- Residuals ---
    st.markdown("#### Residuals Plot")
    residuals = y_target - y_pred
    df_res    = pd.DataFrame({'Predicted': y_pred, 'Residual': residuals, 'label': y.values})
    fig_res   = px.scatter(df_res, x='Predicted', y='Residual', color='label',
                            color_discrete_map=plot_color_map,
                            title="Residuals vs. Predicted",
                            labels={'Predicted':'Predicted Y','Residual':'Residual (Actual − Predicted)'})
    fig_res.add_hline(y=0, line_dash='dash', line_color='gray')
    fig_res.update_layout(height=420)
    st.plotly_chart(fig_res, use_container_width=True)

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
            st.plotly_chart(fg2, use_container_width=True)
            st.subheader(f"{loadings_title_suffix} Table — {component_label}{sel_comp_num}")
            st.dataframe(lrow.to_frame(name=f"{component_label}{sel_comp_num}"))

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Download Results")
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
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train_enc, y_test_enc = train_test_split(
            X_class, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
    else:
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
                                   if split_data else "No split — full dataset for train and eval")
            )
        y_pred_da = best_da.predict(X_test)
        acc_da    = accuracy_score(y_test_enc, y_pred_da)
        st.subheader(f"{da_type} Confusion Matrix{title_suffix}")
        cm_da = confusion_matrix(y_test_enc, y_pred_da)
        fig_cm = px.imshow(cm_da, text_auto=True, x=list(unique_y), y=list(unique_y),
                            color_continuous_scale='Blues',
                            title=f"{da_type} Confusion Matrix{title_suffix}")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.write(f"**Accuracy at {n_pcs_for_classification} {component_label}s:** {acc_da:.3f}")

        st.subheader(f"{da_type} Accuracy vs. Number of {component_label}s ({component_label}2 → {component_label}{n_scree})")
        if da_type=="LDA": cf = lambda: LDA()
        elif da_type=="QDA": cf = lambda: QDA(reg_param=0.001)
        else: cf = lambda: GaussianNB()
        fig_sw, sw_sum = accuracy_vs_pcs_plot(cf, da_type, X_pca_full_for_sweep, y_encoded, n_pcs_for_classification, n_scree)
        if fig_sw: st.plotly_chart(fig_sw, use_container_width=True); st.info(sw_sum)
        else: st.warning(sw_sum)

        if n_pcs_for_classification == 2:
            st.subheader(f"{da_type} Decision Boundary{title_suffix}")
            try:
                from mlxtend.plotting import plot_decision_regions
                fig_db, ax_db = plt.subplots(figsize=(8,6))
                plot_decision_regions(X_class, y_encoded, clf=best_da, legend=2, ax=ax_db)
                ax_db.set_xlabel(f"{component_label}1"); ax_db.set_ylabel(f"{component_label}2")
                ax_db.set_title(f"{da_type} Decision Boundary{title_suffix}")
                st.pyplot(fig_db); plt.close(fig_db)
            except Exception as e:
                st.warning(f"Decision boundary plot failed: {e}")
        else:
            st.info(f"Decision boundary only for exactly 2 {component_label}s (currently {n_pcs_for_classification}).")

        if show_ellipse and n_pcs_for_classification >= 2:
            st.subheader(f"{da_type} Ellipse Plot (JMP-style, {ellipse_std:.0f}σ)")
            fig_ell = go.Figure()
            for ev, cn in enumerate(unique_y):
                mc  = (y_encoded==ev)
                Xc  = X_class[mc,:2]
                if len(Xc)<3: continue
                ch  = color_map_hex.get(str(cn), DEFAULT_COLORS[ev%len(DEFAULT_COLORS)])
                fig_ell.add_trace(go.Scatter(x=Xc[:,0],y=Xc[:,1],mode='markers',
                                              marker=dict(color=ch,size=7,opacity=0.8),name=str(cn)))
                ep = confidence_ellipse_params(Xc[:,0], Xc[:,1], n_std=ellipse_std)
                if ep:
                    mx,my,w,h,ang = ep
                    t  = np.linspace(0,2*np.pi,200)
                    ca = np.cos(np.radians(ang)); sa = np.sin(np.radians(ang))
                    ex = (w/2)*np.cos(t)*ca-(h/2)*np.sin(t)*sa+mx
                    ey = (w/2)*np.cos(t)*sa+(h/2)*np.sin(t)*ca+my
                    fig_ell.add_trace(go.Scatter(x=ex,y=ey,mode='lines',
                                                  line=dict(color=ch,width=2,dash='solid'),showlegend=False))
            fig_ell.update_layout(
                title=f"{da_type} Confidence Ellipses ({ellipse_std:.0f}σ) — {component_label}1 vs {component_label}2",
                xaxis_title=f"{component_label}1", yaxis_title=f"{component_label}2", height=550)
            st.plotly_chart(fig_ell, use_container_width=True)

    # ── KNN ───────────────────────────────────────────────────────────────────
    if run_knn:
        best_knn = KNeighborsClassifier(n_neighbors=k)
        best_knn.fit(X_train, y_train_enc)
        with st.expander("📋 KNN Model Details", expanded=True):
            st.markdown(f"**K-Nearest Neighbors:** classifies each sample by majority vote among its **{k} nearest "
                        f"neighbors** in {component_label} space (Euclidean distance). No explicit training — the model "
                        "memorizes the training set and searches at prediction time.")
            st.markdown(f"**Parameters:** `n_neighbors`=`{k}`, `metric`=`minkowski(p=2)`, `weights`=`uniform`, `algorithm`=`auto`")
            st.markdown(
                f"**Input:** {X_train.shape[0]} training samples × {n_pcs_for_classification} {component_label}s  \n"
                f"**Classes:** {list(unique_y)}  \n"
                f"**Split:** " + (f"Yes — {int((1-test_size)*100)}% / {int(test_size*100)}%"
                                   if split_data else "No split")
            )
        y_pred_knn = best_knn.predict(X_test)
        acc_knn    = accuracy_score(y_test_enc, y_pred_knn)
        st.subheader(f"KNN Confusion Matrix{title_suffix}")
        cm_knn = confusion_matrix(y_test_enc, y_pred_knn)
        fig_ck = px.imshow(cm_knn, text_auto=True, x=list(unique_y), y=list(unique_y),
                            color_continuous_scale='Blues', title=f"KNN Confusion Matrix{title_suffix}")
        st.plotly_chart(fig_ck, use_container_width=True)
        st.write(f"**Accuracy at {n_pcs_for_classification} {component_label}s:** {acc_knn:.3f}")

        st.subheader(f"KNN Accuracy vs. Number of {component_label}s ({component_label}2 → {component_label}{n_scree}, k={k})")
        fig_ksw, ksw_sum = accuracy_vs_pcs_plot(
            lambda: KNeighborsClassifier(n_neighbors=k), f"KNN (k={k})",
            X_pca_full_for_sweep, y_encoded, n_pcs_for_classification, n_scree)
        if fig_ksw: st.plotly_chart(fig_ksw, use_container_width=True); st.info(ksw_sum)
        else: st.warning(ksw_sum)

        if n_pcs_for_classification == 2:
            st.subheader(f"KNN Decision Boundary{title_suffix}")
            try:
                from mlxtend.plotting import plot_decision_regions
                fig_kdb, ax_kdb = plt.subplots(figsize=(8,6))
                plot_decision_regions(X_class, y_encoded, clf=best_knn, legend=2, ax=ax_kdb)
                ax_kdb.set_xlabel(f"{component_label}1"); ax_kdb.set_ylabel(f"{component_label}2")
                ax_kdb.set_title(f"KNN Decision Boundary{title_suffix}")
                st.pyplot(fig_kdb); plt.close(fig_kdb)
            except Exception as e:
                st.warning(f"KNN decision boundary failed: {e}")
        else:
            st.info(f"Decision boundary only for exactly 2 {component_label}s (currently {n_pcs_for_classification}).")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "v15 — Removed pre-computed PC mode; normalization before standardization; "
    "IS uses row label for per-variable column normalization; SNV/Z-score guidance text; "
    "PCA / PCR / PLS toggle with CV/MSE, predicted vs actual, residuals, and full downloads."
)
