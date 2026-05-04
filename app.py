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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# ── helpers ──────────────────────────────────────────────────────────────────

def confidence_ellipse_params(x, y, n_std=2.0):
    """Return (mean_x, mean_y, width, height, angle_deg) for a confidence ellipse."""
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0))
    return np.mean(x), np.mean(y), width, height, angle

def hex_to_rgb_str(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    return f"rgb({r},{g},{b})"

def is_valid_hex(s):
    s = s.strip().lstrip('#')
    return len(s) == 6 and all(c in '0123456789abcdefABCDEF' for c in s)

DEFAULT_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#66c2a5','#fc8d62',
    '#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494',
    '#b3b3b3','#1b9e77','#d95f02','#7570b3','#e7298a',
]

# ── Title and instructions ────────────────────────────────────────────────────
st.title("PCA Visualization App for Lab Data")
st.markdown("""
Upload a CSV file with:
- For features: Numerical columns for features (e.g., measurements).
- Replicate measurements must have same name or separated by _. For example, "Sample1_1","Sample1_2","Sample2_1","Sample2_2".
- Data can be uploaded row-wise or column-wise with the following options:
  - If row-wise: Each row is a sample, and the first column is a 'label' column. Column 1 row 1 must be labeled "label"
  - If column-wise: Each column is a sample, and the first row is a 'label' row. Column 1 must be the variables (i.e. wavelengths).
- For pre-computed PCs: Columns named PC1, PC2, ..., and a 'label' column. Enable the checkbox in the sidebar to use this mode and upload the PC CSV.
""")

# ── Sidebar mode selection ───────────────────────────────────────────────────
with st.sidebar.expander("Mode Selection", expanded=False):
    use_precomputed = st.checkbox("Use pre-computed PC scores", value=False)

# ── File uploaders ───────────────────────────────────────────────────────────
uploaded_file    = st.file_uploader("Upload feature CSV", type="csv")
pc_uploaded_file = st.file_uploader("Upload PC scores CSV", type="csv")

if use_precomputed:
    if pc_uploaded_file is None:
        st.info("Please upload the PC scores CSV to proceed.")
        st.stop()
    df = pd.read_csv(pc_uploaded_file)
    st.success(f"Loaded pre-computed PC dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    st.subheader("Data Overview")
    st.dataframe(df.head())
    if 'label' not in df.columns:
        st.error("PC CSV must have a 'label' column for coloring points.")
        st.stop()
    df['label'] = df['label'].astype(str).str.split('_').str[0]
    st.info(f"Simplified labels: Unique classes now {df['label'].nunique()}")
    X = df.drop('label', axis=1).select_dtypes(include=[np.number])
    y = df['label']
    if X.empty:
        st.error("No numerical PC columns found.")
        st.stop()
    X_scaled    = X.values
    X_pca       = X_scaled
    n_total_pcs = X_pca.shape[1]
    var_ratios  = np.full(n_total_pcs, 1.0 / n_total_pcs)
    pca_full    = None
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
        st.warning("No 'label' column found. Will generate from column names if transposing.")

    with st.sidebar.expander("Data Prep Options", expanded=False):
        transpose_data     = st.checkbox("Transpose Dataset (if samples are in columns)", value=False)
        preprocess_option  = st.radio("Select preprocessing:", ['SNV', 'Z-score'], index=1)
        normalize_option   = st.radio(
            "Select normalization (applied after preprocessing, before PCA):",
            ['None','Min-Max (per sample)','Sum to 1 (per sample)','L2 Norm (sqrt(sum squares) per sample)'],
            index=0
        )

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
        st.success(f"Dataset transposed: {df.shape[0]} samples, {df.shape[1]-1} features.")
    else:
        if 'label' not in df.columns:
            st.error("CSV must have a 'label' column.")
            st.stop()

    df['label'] = df['label'].astype(str).str.split('_').str[0]
    st.info(f"Simplified labels: Unique classes now {df['label'].nunique()}")

    X = df.drop('label', axis=1).select_dtypes(include=[np.number])
    y = df['label']
    if X.empty:
        st.error("No numerical columns found.")
        st.stop()

    X_processed = X.copy()
    if preprocess_option == 'SNV':
        for i in range(X_processed.shape[0]):
            row_mean = np.mean(X_processed.iloc[i])
            row_std  = np.std(X_processed.iloc[i])
            if row_std > 0:
                X_processed.iloc[i] = (X_processed.iloc[i] - row_mean) / row_std
            else:
                st.warning(f"Row {i+1} has zero variance—SNV skipped.")
    elif preprocess_option == 'Z-score':
        scaler = StandardScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed), columns=X.columns, index=X.index)
    X = X_processed
    st.success(f"Preprocessing applied: {preprocess_option}")

    if normalize_option != 'None':
        X_normalized = X.copy()
        for i in range(X_normalized.shape[0]):
            if normalize_option == 'Min-Max (per sample)':
                r_min, r_max = np.min(X_normalized.iloc[i]), np.max(X_normalized.iloc[i])
                if r_max > r_min:
                    X_normalized.iloc[i] = (X_normalized.iloc[i] - r_min) / (r_max - r_min)
                else:
                    st.warning(f"Row {i+1} constant—Min-Max skipped.")
            elif normalize_option == 'Sum to 1 (per sample)':
                r_sum = np.sum(X_normalized.iloc[i])
                if r_sum > 0:
                    X_normalized.iloc[i] /= r_sum
                else:
                    st.warning(f"Row {i+1} zero sum—skipped.")
            elif normalize_option == 'L2 Norm (sqrt(sum squares) per sample)':
                r_l2 = np.sqrt(np.sum(X_normalized.iloc[i]**2))
                if r_l2 > 0:
                    X_normalized.iloc[i] /= r_l2
                else:
                    st.warning(f"Row {i+1} zero L2—skipped.")
        st.success(f"{normalize_option} normalization applied.")
        X = X_normalized

    if preprocess_option != 'Z-score':
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    is_precomputed = False

# ── Data Filtering ───────────────────────────────────────────────────────────
st.subheader("Data Filtering")
unique_labels    = sorted(y.unique())
excluded_labels  = st.multiselect("Labels to exclude", unique_labels, default=[])
mask_include     = ~y.isin(excluded_labels)
X_scaled         = X_scaled[mask_include]
y                = y[mask_include]

# ── PCA ──────────────────────────────────────────────────────────────────────
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

if n_total_pcs >= 2:
    X_pca_2d_global = X_pca[:, :2]
    y_global        = y
else:
    X_pca_2d_global = None
    y_global        = None

# ── Color Configuration ──────────────────────────────────────────────────────
# Must happen before plots so every section can reference color_map_hex
st.subheader("Label Color Configuration")
unique_labels_all = sorted(y.unique())
n_labels          = len(unique_labels_all)

use_custom_colors = st.toggle(
    "Enable custom label colors (enter hex codes)",
    value=False,
    help="When OFF the default color palette is used. When ON you can type a hex color for each label."
)

color_map_hex = {}
if use_custom_colors:
    st.markdown("Type a 6-digit hex code for each label below (e.g. `#FF5733`). A live color swatch confirms your choice. Leave blank to keep the default.")
    n_cols = min(4, n_labels)
    col_groups = st.columns(n_cols)
    for idx, lbl in enumerate(unique_labels_all):
        default_hex = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        with col_groups[idx % n_cols]:
            # Show label name prominently so it's unambiguous which label this controls
            st.markdown(f"**`{lbl}`**")
            user_input = st.text_input(
                label=f"Hex color for {lbl}",
                value=default_hex,
                key=f"color_{lbl}",
                label_visibility="collapsed",   # label already shown above as bold text
            )
            user_input = user_input.strip()
            chosen_hex = ('#' + user_input.lstrip('#')) if (user_input != '' and is_valid_hex(user_input)) else default_hex
            color_map_hex[lbl] = chosen_hex
            # Live swatch
            st.markdown(
                f'<div style="width:100%;height:18px;border-radius:4px;background:{chosen_hex};'
                f'border:1px solid #ccc;margin-bottom:6px;"></div>',
                unsafe_allow_html=True
            )
else:
    for idx, lbl in enumerate(unique_labels_all):
        color_map_hex[lbl] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

# ── Sidebar – Plot Options ───────────────────────────────────────────────────
with st.sidebar.expander("Plot Options", expanded=False):
    show_2d        = st.checkbox("Show 2D PCA Plot (Static)", value=True)
    legend_separate= st.checkbox("Show legend in separate figure for PCA plots", value=False)
    show_3d        = st.checkbox("Show 3D PCA Plot (Interactive)", value=True)
    show_scree     = st.checkbox("Show Scree Plot", value=True)
    if show_scree:
        cum_var = np.cumsum(var_ratios)
        n_99    = np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else n_total_pcs
        n_999   = np.argmax(cum_var >= 0.999)+ 1 if np.any(cum_var >= 0.999) else n_total_pcs
        n_scree = st.slider("Number of PCs to Show in Scree Plot", 1, n_999, n_99)
    else:
        n_scree = n_total_pcs
    show_loadings  = st.checkbox("Show Loadings Plot", value=True)
    if show_loadings and not is_precomputed:
        loadings_type = st.selectbox(
            "Loadings Plot Type",
            ["Bar Graph (Discrete, e.g., GCMS)", "Connected Scatterplot (Continuous, e.g., Spectroscopy)"],
            index=0
        )
        # ── NEW: which PC to display in loadings ──────────────────────────────
        loadings_pc_idx = st.slider(
            "Select PC to view loadings",
            min_value=1, max_value=min(10, n_total_pcs), value=1,
            help="Choose which principal component's factor loadings to display."
        ) - 1   # convert to 0-based index
    else:
        loadings_type   = "Bar Graph (Discrete, e.g., GCMS)"
        loadings_pc_idx = 0

    # ── NEW: PC axis selectors for 2D plot ───────────────────────────────────
    if show_2d and n_total_pcs >= 2:
        st.markdown("**2D Plot PC Axes**")
        pc_x_2d = st.selectbox("X-axis PC", [f"PC{i+1}" for i in range(n_total_pcs)], index=0, key="pc_x_2d")
        pc_y_2d = st.selectbox("Y-axis PC", [f"PC{i+1}" for i in range(n_total_pcs)], index=1, key="pc_y_2d")
        pc_x_idx = int(pc_x_2d[2:]) - 1
        pc_y_idx = int(pc_y_2d[2:]) - 1
    else:
        pc_x_idx, pc_y_idx = 0, 1
        pc_x_2d, pc_y_2d  = "PC1", "PC2"

with st.sidebar.expander("Download Options", expanded=False):
    num_save_pcs = st.slider("Number of PCs to Save", 1, min(10, n_total_pcs), 3)

# ── Classification Options ───────────────────────────────────────────────────
with st.sidebar.expander("Classification Options", expanded=False):
    run_da     = st.checkbox("Run Discriminant Analysis")
    if run_da:
        da_type     = st.selectbox("Discriminant Analysis Type", ["LDA","QDA","GaussianNB"], index=0)
        optimize_da = st.checkbox("Optimize Discriminant Analysis parameters")
        show_ellipse   = st.checkbox("Show Ellipse (JMP-style) Plot", value=True)
        ellipse_std    = st.slider("Ellipse confidence (σ)", 1.0, 3.0, 2.0, 0.5,
                                    help="Number of standard deviations for the confidence ellipses.")
    else:
        da_type = "LDA"; optimize_da = False
        show_ellipse = False; ellipse_std = 2.0

    run_knn   = st.checkbox("Run K-Nearest Neighbors (KNN)")
    optimize_knn = st.checkbox("Optimize KNN parameters") if run_knn else False
    k = st.slider("K value", 1, 20, 5) if (run_knn and not optimize_knn) else 5
    run_kmeans = st.checkbox("Run K-Means Clustering")
    if run_kmeans:
        auto_optimize_k    = st.checkbox("Auto-optimize K", value=False)
        n_clusters         = st.slider("Number of clusters", 2, 10, 3) if not auto_optimize_k else 3
        show_elbow         = st.checkbox("Show Elbow Plot", value=True)
        show_silhouette    = st.checkbox("Show Silhouette Plot", value=True)
        show_cluster_profile = st.checkbox("Show Cluster Profile Plots", value=True)
    else:
        auto_optimize_k = False; show_elbow = False
        show_silhouette = False; show_cluster_profile = False; n_clusters = 3

with st.sidebar.expander("Classification Input Options", expanded=True):
    max_pcs_for_class   = min(10, n_total_pcs)
    n_pcs_for_classification = st.slider(
        "Number of PCs used for Classification", 1, max_pcs_for_class, 2,
        help="How many principal components to feed into LDA / QDA / KNN. "
             "Decision boundary plots only shown for exactly 2 PCs."
    )

# ── Label Configuration ──────────────────────────────────────────────────────
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
        rename_a  = st.text_input("Rename Group A (optional)", value=f"Group A ({', '.join(selected_for_a)})")
        rename_b  = st.text_input("Rename Group B (optional)", value=f"Group B ({', '.join(selected_for_b)})")
        apply_to_plots = st.checkbox("Use combined labels for plots", value=True)
        if apply_to_plots and selected_for_a and selected_for_b:
            y_plot = y_plot.replace({lbl: rename_a for lbl in selected_for_a})
            y_plot = y_plot.replace({lbl: rename_b for lbl in selected_for_b})
    else:
        selected_for_a=[]; selected_for_b=[]; rename_a="Group A"; rename_b="Group B"; apply_to_plots=False

# ── build a y_plot color map (may have renamed labels) ───────────────────────
unique_plot_labels = sorted(y_plot.unique())
# map renamed labels back to a hex color by finding overlapping originals
plot_color_map = {}
for idx, lbl in enumerate(unique_plot_labels):
    if lbl in color_map_hex:
        plot_color_map[lbl] = color_map_hex[lbl]
    else:
        plot_color_map[lbl] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

# ── 1. 2D PCA Plot ───────────────────────────────────────────────────────────
if show_2d and n_total_pcs >= 2:
    st.subheader(f"2D PCA Plot ({pc_x_2d} vs {pc_y_2d})")
    xv = X_pca[:, pc_x_idx]
    yv = X_pca[:, pc_y_idx]
    xvar = var_ratios[pc_x_idx]
    yvar = var_ratios[pc_y_idx]

    df_plot_2d          = pd.DataFrame({pc_x_2d: xv, pc_y_2d: yv})
    df_plot_2d['label'] = y_plot.values
    ul2d                = sorted(df_plot_2d['label'].unique())
    mpl_colors          = [plot_color_map.get(l, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) for i, l in enumerate(ul2d)]

    if legend_separate:
        fig_main, ax_main = plt.subplots(figsize=(8, 6))
        for lbl, col in zip(ul2d, mpl_colors):
            mask = df_plot_2d['label'] == lbl
            ax_main.scatter(df_plot_2d[mask][pc_x_2d], df_plot_2d[mask][pc_y_2d], c=col, label=lbl, s=50)
        ax_main.set_xlabel(f"{pc_x_2d} ({xvar:.1%})")
        ax_main.set_ylabel(f"{pc_y_2d} ({yvar:.1%})")
        ax_main.set_title("Static 2D PCA Plot")
        ax_main.grid(True, alpha=0.3)
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
        ax.set_xlabel(f"{pc_x_2d} ({xvar:.1%})")
        ax.set_ylabel(f"{pc_y_2d} ({yvar:.1%})")
        ax.set_title("Static 2D PCA Plot")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)
elif show_2d:
    st.warning("Need at least 2 features for 2D plot.")

# ── 2. 3D PCA Plot ───────────────────────────────────────────────────────────
if show_3d and n_total_pcs >= 3:
    st.subheader("3D PCA Plot (Interactive: Rotate/Zoom with Mouse)")
    if is_precomputed:
        X_pca_3d   = X_pca[:, :3]
        explained_3d = var_ratios[:3]
    else:
        pca_3d     = PCA(n_components=3)
        X_pca_3d   = pca_3d.fit_transform(X_scaled)
        explained_3d = pca_3d.explained_variance_ratio_
    df_plot        = pd.DataFrame(X_pca_3d, columns=['PC1','PC2','PC3'])
    df_plot['label'] = y_plot.values
    unique_labels_3d = sorted(df_plot['label'].unique())
    color_seq_3d   = [plot_color_map.get(l, DEFAULT_COLORS[i%len(DEFAULT_COLORS)]) for i,l in enumerate(unique_labels_3d)]

    fig_3d = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='label',
                            color_discrete_map=plot_color_map)
    fig_3d.update_traces(marker=dict(size=5))
    if legend_separate:
        fig_3d.update_layout(showlegend=False)
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        fig_3d.update_layout(title="Interactive 3D PCA Plot",
                              scene=dict(
                                  xaxis_title=f"PC1 ({explained_3d[0]:.1%})",
                                  yaxis_title=f"PC2 ({explained_3d[1]:.1%})",
                                  zaxis_title=f"PC3 ({explained_3d[2]:.1%})"
                              ))
        st.plotly_chart(fig_3d, use_container_width=True)
elif show_3d:
    st.warning("Need at least 3 features for 3D plot.")

# ── 3. Scree Plot ─────────────────────────────────────────────────────────────
if show_scree:
    st.subheader("Scree Plot: Variance Explained")
    if is_precomputed:
        st.warning("Using equal variance assumption for pre-computed PCs.")
    var_ratio   = var_ratios[:n_scree] * 100
    fig_scree   = make_subplots(specs=[[{"secondary_y": False}]])
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
    st.info(f"Total variance explained by shown PCs: {np.sum(var_ratios[:n_scree]):.1%} (≥99% reached at PC{n_99})")

# ── 4. Factor Loadings Plot ──────────────────────────────────────────────────
if show_loadings:
    if pca_full is None:
        st.warning("Loadings not available for pre-computed PC mode.")
    else:
        # ── Part A: Top-3 PCs grouped plot (original behaviour) ──────────────
        st.subheader("Factor Loadings Plot (Top 3 PCs)")
        max_pcs      = min(3, n_total_pcs)
        valid_indices = [i for i in range(max_pcs) if var_ratios[i] > 0]
        num_valid     = len(valid_indices)
        if num_valid == 0:
            st.warning("No PCs with >0% variance.")
        else:
            st.info(f"Showing loadings for {num_valid} valid PCs (out of top 3)")
            loadings_top3 = pd.DataFrame(
                pca_full.components_[valid_indices],
                columns=X.columns,
                index=[f'PC{i+1}' for i in valid_indices]
            )
            loadings_top3_abs = loadings_top3.abs()

            if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                fig_top3 = go.Figure()
                bar_colors   = px.colors.qualitative.Set3[:num_valid]
                max_loadings = loadings_top3_abs.max(axis=0)
                sorted_vars  = max_loadings.sort_values(ascending=False).index
                width        = 0.25
                for i, pc in enumerate(loadings_top3.index):
                    pc_data = loadings_top3_abs.loc[pc].loc[sorted_vars]
                    fig_top3.add_trace(go.Bar(
                        y=pc_data.values, x=sorted_vars,
                        name=pc, marker_color=bar_colors[i],
                        width=width, base=0, offsetgroup=i
                    ))
                fig_top3.update_layout(
                    barmode='group', height=400, showlegend=True,
                    title="Loadings: Grouped Bar Graph (Abs Values, Top 3 PCs)",
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
                    loadings_melt, x='Variable', y='Loading', color='PC',
                    markers=False,
                    title="Loadings: Connected Line Plot (Abs Values, Top 3 PCs)",
                    labels={'Variable':'Factors/Variables','Loading':'Loading Magnitude'}
                )
                fig_top3.update_traces(line=dict(width=2, dash='solid'))
                fig_top3.update_xaxes(tickangle=45, tickfont=dict(size=9))
                if len(original_vars) > 50:
                    st.warning("Many variables (>50)—zoom/pan the plot for details.")

            st.plotly_chart(fig_top3, use_container_width=True)
            st.subheader("Loadings Table (Top 3 PCs)")
            st.dataframe(loadings_top3)

        # ── Part B: Single-PC slider plot ─────────────────────────────────────
        st.subheader("Factor Loadings – Single PC Explorer")
        sel_pc_num = st.slider(
            "Select PC to explore its loadings",
            min_value=1, max_value=min(10, n_total_pcs), value=1,
            help="Use this slider to inspect any individual PC's factor loadings."
        )
        sel_pc_idx = sel_pc_num - 1
        sel_pc_label = f"PC{sel_pc_num}"

        if sel_pc_idx >= n_total_pcs or var_ratios[sel_pc_idx] == 0:
            st.warning(f"{sel_pc_label} has zero or no variance—cannot display loadings.")
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
                    st.warning("Many variables (>50)—zoom/pan for details.")

            st.plotly_chart(fig_single, use_container_width=True)
            st.subheader(f"Loadings Table – {sel_pc_label}")
            st.dataframe(loadings_row.to_frame(name=sel_pc_label))

# ── Download buttons ─────────────────────────────────────────────────────────
st.subheader("Download PCA Results")
col1, col2 = st.columns(2)
with col1:
    if is_precomputed:
        X_pca_save = X_pca[:, :num_save_pcs]
    else:
        pca_save   = PCA(n_components=num_save_pcs)
        X_pca_save = pca_save.fit_transform(X_scaled)
    df_scores  = pd.DataFrame(X_pca_save, columns=[f'PC{i+1}' for i in range(num_save_pcs)])
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

# ── Clustering ───────────────────────────────────────────────────────────────
if run_kmeans and X_pca_2d_global is not None:
    st.header("Clustering Results")
    if auto_optimize_k:
        inertias = []
        for k_i in range(1, 11):
            km = KMeans(n_clusters=k_i, random_state=42, n_init=10)
            km.fit(X_pca_2d_global)
            inertias.append(km.inertia_)
        diffs  = np.diff(inertias)
        diffs2 = np.diff(diffs)
        n_clusters = np.argmin(diffs2) + 2
        st.info(f"Auto-optimized K: {n_clusters}")
    kmeans        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels= kmeans.fit_predict(X_pca_2d_global)
    df_cluster    = pd.DataFrame(X_pca_2d_global, columns=['PC1','PC2'])
    df_cluster['cluster'] = cluster_labels.astype(str)
    fig_cluster   = px.scatter(df_cluster, x='PC1', y='PC2', color='cluster',
                                title=f"K-Means Clustering (k={n_clusters}) on PC1 vs PC2",
                                color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_cluster, use_container_width=True)
    if show_elbow:
        st.subheader("Elbow Plot")
        inertias = [KMeans(n_clusters=k_i, random_state=42, n_init=10).fit(X_pca_2d_global).inertia_
                    for k_i in range(1, 11)]
        fig_elbow = px.line(x=range(1,11), y=inertias, markers=True, title="Elbow Plot for Optimal K")
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
        centroids    = kmeans.cluster_centers_
        df_centroids = pd.DataFrame(centroids, columns=['PC1','PC2'])
        df_centroids['cluster'] = range(n_clusters)
        fig_profile = px.bar(df_centroids.melt(id_vars='cluster'), x='cluster', y='value',
                              color='variable', barmode='group', title="Cluster Centroids on PC1 and PC2")
        st.plotly_chart(fig_profile)

# ── Classification ────────────────────────────────────────────────────────────
st.header("Classification Results")

if n_total_pcs < n_pcs_for_classification:
    st.error(f"Not enough PCs. Selected {n_pcs_for_classification}, only {n_total_pcs} exist.")
else:
    X_class = X_pca[:, :n_pcs_for_classification]

    if label_mode == "Default Labels":
        y_selected   = y_global
        title_suffix = " (Multi-class)"
    else:
        if not selected_for_a or not selected_for_b:
            st.warning("Select groups to run combined classification.")
            st.stop()
        mask_group_a  = y_global.isin(selected_for_a)
        mask_group_b  = y_global.isin(selected_for_b)
        mask_selected = mask_group_a | mask_group_b
        X_class       = X_class[mask_selected]
        y_selected    = np.where(mask_group_a[mask_selected], 0, 1)
        title_suffix  = ""

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

    # ── Train selected DA model ──────────────────────────────────────────────
    if run_da:
        if da_type == "LDA":
            if optimize_da:
                param_grid_da = {'solver': ['svd','lsqr','eigen']}
                da_grid = GridSearchCV(LDA(), param_grid_da, cv=min(5,len(y_train_enc)))
                da_grid.fit(X_train, y_train_enc)
                best_da       = da_grid.best_estimator_
                best_params_da= da_grid.best_params_
            else:
                best_da = LDA(); best_da.fit(X_train, y_train_enc)
                best_params_da = {'solver':'svd'}
            st.write(f"**LDA Parameters:** {best_params_da} | Using **{n_pcs_for_classification} PCs**")

        elif da_type == "QDA":
            if optimize_da:
                param_grid_da = {'reg_param': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]}
                da_grid = GridSearchCV(QDA(), param_grid_da, cv=min(5,len(y_train_enc)))
                da_grid.fit(X_train, y_train_enc)
                best_da        = da_grid.best_estimator_
                best_params_da = da_grid.best_params_
            else:
                # Use slight regularization by default to prevent singular covariance issues
                best_da = QDA(reg_param=0.001)
                best_da.fit(X_train, y_train_enc)
                best_params_da = {'reg_param': 0.001}
            st.write(f"**QDA Parameters:** {best_params_da} | Using **{n_pcs_for_classification} PCs**")

        else:  # GaussianNB
            if optimize_da:
                param_grid_da = {'var_smoothing': np.logspace(0, -9, num=10)}
                da_grid = GridSearchCV(GaussianNB(), param_grid_da, cv=min(5,len(y_train_enc)))
                da_grid.fit(X_train, y_train_enc)
                best_da        = da_grid.best_estimator_
                best_params_da = da_grid.best_params_
            else:
                best_da = GaussianNB(); best_da.fit(X_train, y_train_enc)
                best_params_da = {'var_smoothing': 1e-9}
            st.write(f"**GaussianNB Parameters:** {best_params_da} | Using **{n_pcs_for_classification} PCs**")

        y_pred_da = best_da.predict(X_test)
        acc_da    = accuracy_score(y_test_enc, y_pred_da)

        st.subheader(f"{da_type} Confusion Matrix{title_suffix}")
        cm_da     = confusion_matrix(y_test_enc, y_pred_da)
        fig_cm_da = px.imshow(cm_da, text_auto=True, x=list(unique_y), y=list(unique_y),
                               color_continuous_scale='Blues',
                               title=f"{da_type} Confusion Matrix{title_suffix}")
        st.plotly_chart(fig_cm_da, use_container_width=True)
        st.write(f"**Accuracy:** {acc_da:.2f}")

        # ── Classic 2D decision boundary (requires exactly 2 PCs) ────────────
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
            st.subheader(f"{da_type} Ellipse Plot (JMP-style, {ellipse_std:.0f}σ confidence)")
            fig_ell = go.Figure()
            # Map encoded labels back to original class names
            if label_mode == "Default Labels":
                y_for_ellipse = y_global.values[mask_include.values] if mask_include is not None else y_global.values
                # re-index to match X_class
                y_for_ellipse = y_selected
            else:
                y_for_ellipse = y_selected

            for enc_val, cls_name in enumerate(unique_y):
                mask_c = (y_encoded == enc_val)
                Xc     = X_class[mask_c, :2]
                if len(Xc) < 3:
                    continue
                col_hex = color_map_hex.get(str(cls_name), DEFAULT_COLORS[enc_val % len(DEFAULT_COLORS)])

                # Scatter points
                fig_ell.add_trace(go.Scatter(
                    x=Xc[:, 0], y=Xc[:, 1], mode='markers',
                    marker=dict(color=col_hex, size=7, opacity=0.8),
                    name=str(cls_name)
                ))

                # Confidence ellipse via parametric curve
                ep = confidence_ellipse_params(Xc[:,0], Xc[:,1], n_std=ellipse_std)
                if ep:
                    mx, my, w, h, angle = ep
                    theta  = np.linspace(0, 2*np.pi, 200)
                    cos_a  = np.cos(np.radians(angle))
                    sin_a  = np.sin(np.radians(angle))
                    ex     = (w/2)*np.cos(theta)*cos_a - (h/2)*np.sin(theta)*sin_a + mx
                    ey     = (w/2)*np.cos(theta)*sin_a + (h/2)*np.sin(theta)*cos_a + my
                    fig_ell.add_trace(go.Scatter(
                        x=ex, y=ey, mode='lines',
                        line=dict(color=col_hex, width=2, dash='solid'),
                        showlegend=False
                    ))

            fig_ell.update_layout(
                title=f"{da_type} Confidence Ellipses ({ellipse_std:.0f}σ) on PC1 vs PC2",
                xaxis_title="PC1", yaxis_title="PC2", height=550,
                legend_title="Class"
            )
            st.plotly_chart(fig_ell, use_container_width=True)

    # ── KNN ──────────────────────────────────────────────────────────────────
    if run_knn:
        if optimize_knn:
            param_grid_knn = {'n_neighbors': range(1, min(21, len(y_train_enc)//2 + 1))}
            knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=min(5,len(y_train_enc)))
            knn_grid.fit(X_train, y_train_enc)
            best_knn       = knn_grid.best_estimator_
            best_params_knn= knn_grid.best_params_
            best_k         = best_params_knn['n_neighbors']
            st.write(f"**Optimized KNN Parameters:** {best_params_knn} | Using **{n_pcs_for_classification} PCs**")
        else:
            best_knn = KNeighborsClassifier(n_neighbors=k)
            best_knn.fit(X_train, y_train_enc)
            best_params_knn = {'n_neighbors': k}
            best_k          = k
            st.write(f"**KNN Parameters:** {best_params_knn} | Using **{n_pcs_for_classification} PCs**")

        y_pred_knn = best_knn.predict(X_test)
        acc_knn    = accuracy_score(y_test_enc, y_pred_knn)
        st.subheader(f"KNN Confusion Matrix{title_suffix}")
        cm_knn    = confusion_matrix(y_test_enc, y_pred_knn)
        knn_title = f"KNN Confusion Matrix{title_suffix}"
        if label_mode != "Default Labels":
            knn_title += f" (k={best_k})"
        fig_cm_knn = px.imshow(cm_knn, text_auto=True, x=list(unique_y), y=list(unique_y),
                                color_continuous_scale='Blues', title=knn_title)
        st.plotly_chart(fig_cm_knn, use_container_width=True)
        st.write(f"**Accuracy:** {acc_knn:.2f}")

        if n_pcs_for_classification == 2:
            st.subheader(f"KNN Decision Boundary{title_suffix}")
            try:
                from mlxtend.plotting import plot_decision_regions
                knn_db_title = f'KNN Decision Boundary{title_suffix}'
                if label_mode != "Default Labels":
                    knn_db_title += f' (k={best_k})'
                fig_knn, ax_knn = plt.subplots(figsize=(8, 6))
                plot_decision_regions(X_class, y_encoded, clf=best_knn, legend=2, ax=ax_knn)
                ax_knn.set_xlabel('PC1'); ax_knn.set_ylabel('PC2')
                ax_knn.set_title(knn_db_title)
                st.pyplot(fig_knn); plt.close(fig_knn)
            except Exception as e:
                st.warning(f"KNN decision boundary plot failed: {e}")
        else:
            st.info(f"Decision boundary plot only available for exactly 2 PCs (currently {n_pcs_for_classification}).")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("v9 – Custom hex colors, selectable PC loadings, PC axis selector, LDA/QDA fit test, 3D DA plot, ellipse plot, QDA bug fix.")
