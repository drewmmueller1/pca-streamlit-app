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
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score, mean_squared_error
from mlxtend.plotting import plot_decision_regions
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Title and instructions
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

# Sidebar mode selection
with st.sidebar.expander("Mode Selection", expanded=False):
    use_precomputed = st.checkbox("Use pre-computed PC scores", value=False)

# File uploaders
uploaded_file = st.file_uploader("Upload feature CSV", type="csv")
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
    # Simplify labels
    df['label'] = df['label'].astype(str).str.split('_').str[0]
    st.info(f"Simplified labels: Unique classes now {df['label'].nunique()}")
    # Prepare data
    X = df.drop('label', axis=1).select_dtypes(include=[np.number])
    y = df['label']
    if X.empty:
        st.error("No numerical PC columns found. Ensure your CSV has columns like PC1, PC2, etc.")
        st.stop()
    # No preprocessing or scaling for pre-computed PCs
    X_scaled = X.values
    X_pca = X_scaled
    n_total_pcs = X_pca.shape[1]
    var_ratios = np.full(n_total_pcs, 1.0 / n_total_pcs)
    pca_full = None
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
    # Data prep options
    with st.sidebar.expander("Data Prep Options", expanded=False):
        transpose_data = st.checkbox("Transpose Dataset (if samples are in columns)", value=False, help="Swaps rows and columns. Use if your data has samples as columns and features as rows (e.g., wavenumbers in first column).")
        # Preprocessing options
        preprocess_option = st.radio(
            "Select preprocessing:",
            ['SNV', 'Z-score'], index=1
        )
        # Normalization options
        normalize_option = st.radio(
            "Select normalization (applied after preprocessing, before PCA):",
            ['None', 'Min-Max (per sample)', 'Sum to 1 (per sample)', 'L2 Norm (sqrt(sum squares) per sample)'], index=0
        )
    if transpose_data:
        # Assume first col is features (e.g., wavenumber), rest are samples
        if df.shape[1] < 2:
            st.error("Dataset too narrow for transpose. Need at least 2 columns.")
            st.stop()
        features = df.iloc[:, 0].values # First col as feature names (e.g., wavenumbers)
        data = df.iloc[:, 1:].T # Transpose the data part: rows=samples, columns=features
        data.columns = features # Set columns to original first col values
        # Generate labels from original column names (samples)
        sample_names = df.columns[1:]
        data['label'] = [name.split('_')[0] for name in sample_names] # Prefix as label
        df = data.reset_index(drop=True) # Reset index to 0,1,...
        st.success(f"Dataset transposed: Samples as rows ({df.shape[0]}), features as columns ({df.shape[1]-1}). Labels generated from prefixes.")
    else:
        if 'label' not in df.columns:
            st.error("CSV must have a 'label' column for coloring points. Add it and re-upload (or enable transpose to auto-generate).")
            st.stop()
    # Simplify labels
    df['label'] = df['label'].astype(str).str.split('_').str[0]
    st.info(f"Simplified labels: Unique classes now {df['label'].nunique()}")
    # Prepare data for PCA
    X = df.drop('label', axis=1).select_dtypes(include=[np.number]) # Numerical features only
    y = df['label']
    if X.empty:
        st.error("No numerical columns found for PCA. Ensure your CSV has numeric feature columns.")
        st.stop()
    # Apply preprocessing
    X_processed = X.copy()
    if preprocess_option == 'SNV':
        for i in range(X_processed.shape[0]):
            row_mean = np.mean(X_processed.iloc[i])
            row_std = np.std(X_processed.iloc[i])
            if row_std > 0:
                X_processed.iloc[i] = (X_processed.iloc[i] - row_mean) / row_std
            else:
                st.warning(f"Row {i+1} has zero variance—SNV skipped for it.")
    elif preprocess_option == 'Z-score':
        scaler = StandardScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed), columns=X.columns, index=X.index)
    X = X_processed
    st.success(f"Preprocessing applied: {preprocess_option}")
    # Apply normalization
    if normalize_option != 'None':
        if normalize_option == 'Min-Max (per sample)':
            X_normalized = X.copy()
            for i in range(X_normalized.shape[0]):
                row_min = np.min(X_normalized.iloc[i])
                row_max = np.max(X_normalized.iloc[i])
                if row_max > row_min:
                    X_normalized.iloc[i] = (X_normalized.iloc[i] - row_min) / (row_max - row_min)
                else:
                    st.warning(f"Row {i+1} has constant values—Min-Max skipped for it.")
            st.success("Min-Max normalization applied (per sample).")
        elif normalize_option == 'Sum to 1 (per sample)':
            X_normalized = X.copy()
            for i in range(X_normalized.shape[0]):
                row_sum = np.sum(X_normalized.iloc[i])
                if row_sum > 0:
                    X_normalized.iloc[i] = X_normalized.iloc[i] / row_sum
                else:
                    st.warning(f"Row {i+1} has zero sum—normalization skipped for it.")
            st.success("Sum to 1 normalization applied (per sample).")
        elif normalize_option == 'L2 Norm (sqrt(sum squares) per sample)':
            X_normalized = X.copy()
            for i in range(X_normalized.shape[0]):
                row_l2 = np.sqrt(np.sum(X_normalized.iloc[i]**2))
                if row_l2 > 0:
                    X_normalized.iloc[i] = X_normalized.iloc[i] / row_l2
                else:
                    st.warning(f"Row {i+1} has zero L2 norm—normalization skipped for it.")
            st.success("L2 Norm normalization applied (per sample).")
        X = X_normalized
    # Conditional scaling only if not Z-score
    if preprocess_option != 'Z-score':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values # Already standardized
    is_precomputed = False

# Data Filtering
st.subheader("Data Filtering")
unique_labels = sorted(y.unique())
excluded_labels = st.multiselect("Labels to exclude", unique_labels, default=[])
mask_include = ~y.isin(excluded_labels)
X_scaled = X_scaled[mask_include]
y = y[mask_include]

# Compute PCA on filtered data
if is_precomputed:
    X_pca = X_scaled
    n_total_pcs = X_pca.shape[1]
    var_ratios = np.full(n_total_pcs, 1.0 / n_total_pcs) if n_total_pcs > 0 else np.array([])
    pca_full = None
else:
    pca_full = PCA()
    X_pca = pca_full.fit_transform(X_scaled)
    n_total_pcs = X_pca.shape[1]
    var_ratios = pca_full.explained_variance_ratio_

# Store global variables
y_global = y

# Sidebar options
with st.sidebar.expander("Plot Options", expanded=False):
    show_2d = st.checkbox("Show 2D PCA Plot (Static)", value=True)
    legend_separate = st.checkbox("Show legend in separate figure for PCA plots", value=False)
    show_3d = st.checkbox("Show 3D PCA Plot (Interactive)", value=True)
    show_scree = st.checkbox("Show Scree Plot", value=True)
    if show_scree:
        cum_var = np.cumsum(var_ratios)
        n_99 = np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else n_total_pcs
        n_999 = np.argmax(cum_var >= 0.999) + 1 if np.any(cum_var >= 0.999) else n_total_pcs
        n_scree = st.slider("Number of PCs to Show in Scree Plot", 1, n_999, n_99)
    else:
        n_scree = n_total_pcs
    show_loadings = st.checkbox("Show Loadings Plot (Top 3 PCs)", value=True)
    if show_loadings and not is_precomputed:
        loadings_type = st.selectbox("Loadings Plot Type", ["Bar Graph (Discrete, e.g., GCMS)", "Connected Scatterplot (Continuous, e.g., Spectroscopy)"], index=0)
    else:
        loadings_type = "Bar Graph (Discrete, e.g., GCMS)"

with st.sidebar.expander("Download Options", expanded=False):
    num_save_pcs = st.slider("Number of PCs to Save", 1, min(10, n_total_pcs), 3)

# === NEW: Classification Options + Input PCs ===
with st.sidebar.expander("Classification Options", expanded=False):
    run_da = st.checkbox("Run Discriminant Analysis")
    if run_da:
        da_type = st.selectbox("Discriminant Analysis Type", ["LDA", "QDA", "GaussianNB"], index=0)
        optimize_da = st.checkbox("Optimize Discriminant Analysis parameters")
    run_knn = st.checkbox("Run K-Nearest Neighbors (KNN)")
    optimize_knn = st.checkbox("Optimize KNN parameters") if run_knn else False
    if run_knn and not optimize_knn:
        k = st.slider("K value", 1, 20, 5)
    else:
        k = 5
    run_kmeans = st.checkbox("Run K-Means Clustering")
    if run_kmeans:
        auto_optimize_k = st.checkbox("Auto-optimize K", value=False)
        if not auto_optimize_k:
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
        show_elbow = st.checkbox("Show Elbow Plot", value=True)
        show_silhouette = st.checkbox("Show Silhouette Plot", value=True)
        show_cluster_profile = st.checkbox("Show Cluster Profile Plots", value=True)
    else:
        auto_optimize_k = False
        show_elbow = False
        show_silhouette = False
        show_cluster_profile = False
        n_clusters = 3

# New expander for number of PCs in classification
with st.sidebar.expander("Classification Input Options", expanded=True):
    max_pcs_for_class = min(10, n_total_pcs)
    n_pcs_for_classification = st.slider(
        "Number of PCs used for Classification",
        min_value=1,
        max_value=max_pcs_for_class,
        value=2,
        help="Number of principal components fed into LDA / QDA / KNN. "
             "Decision boundary plots are only available when exactly 2 PCs are selected."
    )

# Label Configuration
st.subheader("Label Configuration")
label_mode = st.radio("Label Mode", ["Default Labels", "Combined Groups"], index=0)
y_plot = y.copy()
if label_mode == "Combined Groups":
    if n_total_pcs >= 2:
        unique_classes = sorted(y_global.unique())
        selected_for_a = st.multiselect("Select labels for Group A", unique_classes, default=unique_classes[:1])
        selected_for_b = st.multiselect("Select labels for Group B", unique_classes, default=unique_classes[1:2])
        rename_a = st.text_input("Rename Group A (optional)", value=f"Group A ({', '.join(selected_for_a)})")
        rename_b = st.text_input("Rename Group B (optional)", value=f"Group B ({', '.join(selected_for_b)})")
        apply_to_plots = st.checkbox("Use combined labels for plots", value=True)
        if apply_to_plots and selected_for_a and selected_for_b:
            y_plot = y_plot.replace({label: rename_a for label in selected_for_a})
            y_plot = y_plot.replace({label: rename_b for label in selected_for_b})

# ====================== PLOTS (2D, 3D, Scree, Loadings) ======================
# (The plotting code remains mostly the same – only minor updates for consistency)

if show_2d and n_total_pcs >= 2:
    st.subheader("2D PCA Plot (PC1 vs PC2)")
    if is_precomputed:
        X_pca_2d = X_pca[:, :2]
        explained_2d = var_ratios[:2]
    else:
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        explained_2d = pca_2d.explained_variance_ratio_
    # ... (rest of 2D plot code remains unchanged)

# 3D Plot, Scree Plot, Loadings Plot – keep as in your original code

# ====================== CLASSIFICATION RESULTS ======================
st.header("Classification Results")

if n_total_pcs < n_pcs_for_classification:
    st.error(f"Not enough PCs available. You selected {n_pcs_for_classification} but only {n_total_pcs} exist.")
else:
    # Use selected number of PCs
    X_class = X_pca[:, :n_pcs_for_classification]
    
    if label_mode == "Default Labels":
        y_selected = y_global
        title_suffix = " (Multi-class)"
    else:
        if not selected_for_a or not selected_for_b:
            st.warning("Select groups for Combined Groups mode.")
            st.stop()
        mask_group_a = y_global.isin(selected_for_a)
        mask_group_b = y_global.isin(selected_for_b)
        mask_selected = mask_group_a | mask_group_b
        X_class = X_class[mask_selected]
        y_selected = np.where(mask_group_a[mask_selected], 0, 1)
        title_suffix = ""

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_selected)
    unique_y = le.classes_

    # Train/test split
    split_data = st.checkbox("Split into train/test sets", value=False)
    if split_data:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train_enc, y_test_enc = train_test_split(
            X_class, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train_enc, y_test_enc = X_class, X_class, y_encoded, y_encoded

    # ------------------- Discriminant Analysis -------------------
    if run_da:
        if da_type == "LDA":
            if optimize_da:
                param_grid_da = {'solver': ['svd', 'lsqr', 'eigen']}
                da_grid = GridSearchCV(LDA(), param_grid_da, cv=5)
                da_grid.fit(X_train, y_train_enc)
                best_da = da_grid.best_estimator_
                best_params_da = da_grid.best_params_
                st.write(f"**Optimized LDA Parameters:** {best_params_da} | **{n_pcs_for_classification} PCs**")
            else:
                best_da = LDA()
                best_da.fit(X_train, y_train_enc)
                best_params_da = {'solver': 'svd'}
                st.write(f"**LDA Parameters:** {best_params_da} | **{n_pcs_for_classification} PCs**")

        elif da_type == "QDA":
            if optimize_da:
                param_grid_da = {'reg_param': [0.0, 0.1, 0.5, 1.0]}
                da_grid = GridSearchCV(QDA(), param_grid_da, cv=5)
                da_grid.fit(X_train, y_train_enc)
                best_da = da_grid.best_estimator_
                best_params_da = da_grid.best_params_
                st.write(f"**Optimized QDA Parameters:** {best_params_da} | **{n_pcs_for_classification} PCs**")
            else:
                best_da = QDA()
                best_da.fit(X_train, y_train_enc)
                best_params_da = {'reg_param': 0.0}
                st.write(f"**QDA Parameters:** {best_params_da} | **{n_pcs_for_classification} PCs**")

        else:  # GaussianNB
            if optimize_da:
                param_grid_da = {'var_smoothing': np.logspace(0, -9, num=10)}
                da_grid = GridSearchCV(GaussianNB(), param_grid_da, cv=5)
                da_grid.fit(X_train, y_train_enc)
                best_da = da_grid.best_estimator_
                best_params_da = da_grid.best_params_
                st.write(f"**Optimized GaussianNB Parameters:** {best_params_da} | **{n_pcs_for_classification} PCs**")
            else:
                best_da = GaussianNB()
                best_da.fit(X_train, y_train_enc)
                best_params_da = {'var_smoothing': 1e-9}
                st.write(f"**GaussianNB Parameters:** {best_params_da} | **{n_pcs_for_classification} PCs**")

        y_pred_da = best_da.predict(X_test)
        acc_da = accuracy_score(y_test_enc, y_pred_da)
        st.subheader(f"{da_type} Confusion Matrix{title_suffix}")
        cm_da = confusion_matrix(y_test_enc, y_pred_da)
        fig_cm_da = px.imshow(cm_da, text_auto=True, x=unique_y, y=unique_y,
                              color_continuous_scale='Blues', title=f"{da_type} Confusion Matrix{title_suffix}")
        st.plotly_chart(fig_cm_da, use_container_width=True)
        st.write(f"**Accuracy:** {acc_da:.2f}")

        # Decision boundary only when using exactly 2 PCs
        if n_pcs_for_classification == 2:
            st.subheader(f"{da_type} Decision Boundary{title_suffix}")
            fig_da, ax_da = plt.subplots(figsize=(8, 6))
            plot_decision_regions(X_class, y_encoded, clf=best_da, legend=2, ax=ax_da)
            ax_da.set_xlabel('PC1')
            ax_da.set_ylabel('PC2')
            ax_da.set_title(f'{da_type} Decision Boundary{title_suffix} ({n_pcs_for_classification} PCs)')
            st.pyplot(fig_da)
        else:
            st.info(f"Decision boundary plot is only available when using exactly 2 PCs (currently using {n_pcs_for_classification}).")

    # ------------------- KNN -------------------
    if run_knn:
        if optimize_knn:
            param_grid_knn = {'n_neighbors': range(1, min(21, len(y_train_enc)//2 + 1))}
            knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
            knn_grid.fit(X_train, y_train_enc)
            best_knn = knn_grid.best_estimator_
            best_params_knn = knn_grid.best_params_
            st.write(f"**Optimized KNN Parameters:** {best_params_knn} | **{n_pcs_for_classification} PCs**")
        else:
            best_knn = KNeighborsClassifier(n_neighbors=k)
            best_knn.fit(X_train, y_train_enc)
            best_params_knn = {'n_neighbors': k}
            st.write(f"**KNN Parameters:** {best_params_knn} | **{n_pcs_for_classification} PCs**")

        y_pred_knn = best_knn.predict(X_test)
        acc_knn = accuracy_score(y_test_enc, y_pred_knn)
        st.subheader(f"KNN Confusion Matrix{title_suffix}")
        cm_knn = confusion_matrix(y_test_enc, y_pred_knn)
        fig_cm_knn = px.imshow(cm_knn, text_auto=True, x=unique_y, y=unique_y,
                               color_continuous_scale='Blues', title=f"KNN Confusion Matrix{title_suffix}")
        st.plotly_chart(fig_cm_knn, use_container_width=True)
        st.write(f"**Accuracy:** {acc_knn:.2f}")

        if n_pcs_for_classification == 2:
            st.subheader(f"KNN Decision Boundary{title_suffix}")
            fig_knn, ax_knn = plt.subplots(figsize=(8, 6))
            plot_decision_regions(X_class, y_encoded, clf=best_knn, legend=2, ax=ax_knn)
            ax_knn.set_xlabel('PC1')
            ax_knn.set_ylabel('PC2')
            ax_knn.set_title(f'KNN Decision Boundary{title_suffix} (k={best_params_knn["n_neighbors"]})')
            st.pyplot(fig_knn)
        else:
            st.info(f"Decision boundary plot is only available when using exactly 2 PCs (currently using {n_pcs_for_classification}).")

# K-Means Clustering section (unchanged)
if run_kmeans and n_total_pcs >= 2:
    # ... (your original KMeans code using X_pca_2d_global or update to use first 2 PCs)

st.markdown("---")
st.caption("Updated version: Number of PCs for classification is now adjustable.")
