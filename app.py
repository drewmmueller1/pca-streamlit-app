import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_decision_regions
# Title and instructions
st.title("PCA Visualization App for Lab Data")
st.markdown("""
Upload a CSV file with:
- For features: Numerical columns for features (e.g., measurements).
- Replicate measurements must have same name or seperateed by _. For example, "Sample1_1","Sample1_2","Sample2_1","Sample2_2".
- Data can be uploaded row-wise or column-wise with the following options:
  - If row-wise: Each row is a sample, and the first column is a 'label' column. Column 1 row 1 must be labeled "label"
  - If column-wise: Each column is a sample, and the first row is a 'label' row. Column 1 must be the variables (i.e. wavelengths).
- For pre-computed PCs: Columns named PC1, PC2, ..., and a 'label' column. Enable the checkbox in the sidebar to use this mode and upload the PC CSV.
""")
# Sidebar mode selection
st.sidebar.header("Mode Selection")
use_precomputed = st.sidebar.checkbox("Use pre-computed PC scores", value=False)
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
    st.sidebar.header("Data Prep Options")
    transpose_data = st.sidebar.checkbox("Transpose Dataset (if samples are in columns)", value=False, help="Swaps rows and columns. Use if your data has samples as columns and features as rows (e.g., wavenumbers in first column).")
    # Preprocessing options
    preprocess_option = st.sidebar.radio(
        "Select preprocessing:",
        ['SNV', 'Z-score'], index=1
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
        # SNV: per sample (row) normalization
        for i in range(X_processed.shape[0]):
            row_mean = np.mean(X_processed.iloc[i])
            row_std = np.std(X_processed.iloc[i])
            if row_std > 0:
                X_processed.iloc[i] = (X_processed.iloc[i] - row_mean) / row_std
            else:
                st.warning(f"Row {i+1} has zero variance—SNV skipped for it.")
    elif preprocess_option == 'Z-score':
        # Z-score: feature-wise (StandardScaler)
        scaler = StandardScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed), columns=X.columns, index=X.index)
    X = X_processed
    st.success(f"Preprocessing applied: {preprocess_option}")
    # Conditional scaling only if not Z-score (to avoid double scaling)
    if preprocess_option != 'Z-score':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values # Already standardized
    # Compute full PCA for reuse
    pca_full = PCA()
    X_pca = pca_full.fit_transform(X_scaled)
    n_total_pcs = X_pca.shape[1]
    var_ratios = pca_full.explained_variance_ratio_
    is_precomputed = False
# Common code after data preparation
# Store X_pca_2d for classification
if n_total_pcs >= 2:
    if is_precomputed:
        X_pca_2d_global = X_pca[:, :2]
    else:
        pca_2d_global = PCA(n_components=2)
        X_pca_2d_global = pca_2d_global.fit_transform(X_scaled)
    y_global = y
else:
    X_pca_2d_global = None
    y_global = None
# Sidebar options (toggles + save slider + loadings type)
st.sidebar.header("Plot Options")
show_2d = st.sidebar.checkbox("Show 2D PCA Plot (Static)", value=True)
show_3d = st.sidebar.checkbox("Show 3D PCA Plot (Interactive)", value=True)
show_scree = st.sidebar.checkbox("Show Scree Plot", value=True)
show_loadings = st.sidebar.checkbox("Show Loadings Plot (Top 3 PCs)", value=True)
if show_loadings and not is_precomputed:
    loadings_type = st.sidebar.selectbox("Loadings Plot Type", ["Bar Graph (Discrete, e.g., GCMS)", "Connected Scatterplot (Continuous, e.g., Spectroscopy)"], index=0)
else:
    loadings_type = "Bar Graph (Discrete, e.g., GCMS)" # Default if not shown
st.sidebar.header("Download Options")
num_save_pcs = st.sidebar.slider("Number of PCs to Save", 1, min(10, n_total_pcs), 3)
# Classification options in sidebar
st.sidebar.header("Classification Options")
run_lda = st.sidebar.checkbox("Run Linear Discriminant Analysis (LDA)")
optimize_lda = st.sidebar.checkbox("Optimize LDA parameters") if run_lda else False
run_knn = st.sidebar.checkbox("Run K-Nearest Neighbors (KNN)")
optimize_knn = st.sidebar.checkbox("Optimize KNN parameters") if run_knn else False
if run_knn and not optimize_knn:
    k = st.sidebar.slider("K value", 1, 20, 5)
else:
    k = 5 # Default
# Label combination options
st.subheader("Label Management")
if X_pca_2d_global is not None:
    unique_classes = sorted(y_global.unique())
    # Multi-select for Group A and Group B
    selected_for_a = st.multiselect("Select labels for Group A", unique_classes, default=unique_classes[:1])
    selected_for_b = st.multiselect("Select labels for Group B", unique_classes, default=unique_classes[1:2])
   
    # Rename options if combining
    rename_a = st.text_input("Rename Group A (optional)", value=f"Group A ({', '.join(selected_for_a)})")
    rename_b = st.text_input("Rename Group B (optional)", value=f"Group B ({', '.join(selected_for_b)})")
   
    if not selected_for_a or not selected_for_b:
        st.warning("Select at least one label for each group.")
    y_plot = y.copy()
    if selected_for_a and selected_for_b:
        y_plot = y_plot.replace(selected_for_a, rename_a)
        y_plot = y_plot.replace(selected_for_b, rename_b)
else:
    y_plot = y
    selected_for_a, selected_for_b, rename_a, rename_b = [], [], "Group A", "Group B"
# Dataset split options
st.subheader("Dataset Split Options")
split_data = st.checkbox("Split into train/test sets")
if split_data:
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
else:
    test_size = 0
# 1. 2D PCA Plot (Static, first 2 PCs)
if show_2d and n_total_pcs >= 2:
    st.subheader("2D PCA Plot (PC1 vs PC2)")
    if is_precomputed:
        X_pca_2d = X_pca[:, :2]
        explained_2d = var_ratios[:2]
    else:
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        explained_2d = pca_2d.explained_variance_ratio_
    df_plot_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
    df_plot_2d['label'] = y_plot
    # Matplotlib for static plot
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = df_plot_2d['label'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    for label in unique_labels:
        mask = df_plot_2d['label'] == label
        ax.scatter(df_plot_2d[mask]['PC1'], df_plot_2d[mask]['PC2'],
                   c=[color_map[label]], label=label, s=50)
    ax.set_xlabel(f"PC1 ({explained_2d[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained_2d[1]:.1%})")
    ax.set_title("Static 2D PCA Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
elif show_2d:
    st.warning("Need at least 2 features for 2D plot.")
# 2. 3D PCA Plot (Interactive, FIXED to first 3 PCs only—no options to change)
if show_3d and n_total_pcs >= 3:
    st.subheader("3D PCA Plot (Interactive: Rotate/Zoom with Mouse)")
    if is_precomputed:
        X_pca_3d = X_pca[:, :3]
        explained_3d = var_ratios[:3]
    else:
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        explained_3d = pca_3d.explained_variance_ratio_
    df_plot = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_plot['label'] = y_plot
    fig_3d = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='label',
                           color_discrete_sequence=px.colors.qualitative.Set1)
    fig_3d.update_traces(marker=dict(size=5))
    fig_3d.update_layout(title="Interactive 3D PCA Plot (Fixed to PC1-PC3)",
                         scene=dict(
                             xaxis_title=f"PC1 ({explained_3d[0]:.1%})",
                             yaxis_title=f"PC2 ({explained_3d[1]:.1%})",
                             zaxis_title=f"PC3 ({explained_3d[2]:.1%})"
                         ))
    st.plotly_chart(fig_3d, use_container_width=True)
elif show_3d:
    st.warning("Need at least 3 features for 3D plot.")
# 3. Scree Plot (Dynamic: >=99% var + 2 more PCs)
if show_scree:
    st.subheader("Scree Plot: Variance Explained")
    if is_precomputed:
        st.warning("Using equal variance assumption for pre-computed PCs.")
    # Find min n for >=99% cum var
    cum_var = np.cumsum(var_ratios)
    n_99 = np.argmax(cum_var >= 0.99) + 1 if np.any(cum_var >= 0.99) else n_total_pcs
    n_scree = min(n_99 + 2, n_total_pcs)
    # Use var_ratios directly
    var_ratio = var_ratios[:n_scree] * 100 # % variance
    cum_var_scree = np.cumsum(var_ratio)
    # Create subplot: bar for %var, line for cumulative
    fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
    # Bar: % variance per PC
    fig_scree.add_trace(
        go.Bar(x=[f'PC{i+1}' for i in range(n_scree)], y=var_ratio,
               name='% Variance', marker_color='lightblue'),
        secondary_y=False
    )
    # Add % labels above bars
    for i, v in enumerate(var_ratio):
        fig_scree.add_annotation(x=f'PC{i+1}', y=v, text=f'{v:.1f}%', showarrow=False,
                                 yshift=10, font=dict(size=10))
    # Line: Cumulative % variance
    fig_scree.add_trace(
        go.Scatter(x=[f'PC{i+1}' for i in range(n_scree)], y=cum_var_scree,
                   mode='lines+markers', name='Cumulative % Variance', line=dict(color='red', dash='dash')),
        secondary_y=True
    )
    fig_scree.update_layout(title=f"Scree Plot (Showing {n_scree} PCs: ≥99% + 2 more)",
                            xaxis_title="Principal Components",
                            yaxis_title="% Variance Explained", yaxis2_title="Cumulative % Variance")
    fig_scree.update_yaxes(range=[0, max(var_ratio.max(), cum_var_scree[-1]) * 1.1], secondary_y=False)
    fig_scree.update_yaxes(range=[0, 100], secondary_y=True)
    st.plotly_chart(fig_scree, use_container_width=True)
    # Total variance info
    st.info(f"Total variance explained by shown PCs: {cum_var_scree[-1]:.1f}% (≥99% reached at PC{n_99})")
# 4. Factor Loadings Plot (Toggle between Bar and Connected Scatterplot)
if show_loadings:
    if pca_full is None:
        st.warning("Loadings not available for pre-computed PC mode.")
    else:
        st.subheader("Factor Loadings Plot (Top 3 PCs)")
        # First 3 PCs
        max_pcs = min(3, n_total_pcs)
        var_ratios_top = var_ratios[:max_pcs]
        # Filter valid PCs (>0% var)
        valid_indices = [i for i in range(max_pcs) if var_ratios_top[i] > 0]
        num_valid = len(valid_indices)
        if num_valid == 0:
            st.warning("No PCs with >0% variance.")
        else:
            st.info(f"Showing loadings for {num_valid} valid PCs (out of top 3)")
            # Subset loadings (use abs for magnitude)
            loadings = pd.DataFrame(pca_full.components_[valid_indices],
                                    columns=X.columns,
                                    index=[f'PC{i+1}' for i in valid_indices])
            loadings_abs = loadings.abs()
            if loadings_type == "Bar Graph (Discrete, e.g., GCMS)":
                # Vertical grouped bars (variables on x)
                fig_loadings = go.Figure()
                colors = px.colors.qualitative.Set3[:num_valid]
                # Sort variables by max abs loading (descending) for bars
                max_loadings = loadings_abs.max(axis=0)
                sorted_vars = max_loadings.sort_values(ascending=False).index
                # Width and offset for grouped bars
                width = 0.25
                for i, pc in enumerate(loadings.index):
                    pc_data = loadings_abs.loc[pc].loc[sorted_vars]
                    x_pos = np.arange(len(sorted_vars)) + (i - (num_valid - 1) / 2) * width
                    fig_loadings.add_trace(go.Bar(y=pc_data.values, x=sorted_vars,
                                                  name=pc, marker_color=colors[i], width=width,
                                                  base=0, offsetgroup=i))
                fig_loadings.update_layout(barmode='group',
                                           height=400, showlegend=True,
                                           title="Loadings: Grouped Bar Graph (Abs Values)",
                                           xaxis_title="Variables",
                                           yaxis_title="Loading Magnitude")
                fig_loadings.update_xaxes(tickangle=45, tickfont=dict(size=9))
            else: # Connected Scatterplot (Continuous, e.g., Spectroscopy)
                # Prepare for line plot: Melt to long format, preserve original variable order
                loadings_melt = loadings_abs.reset_index().melt(id_vars='index', var_name='Variable', value_name='Loading')
                loadings_melt['PC'] = loadings_melt['index'] # Use PC name as color/group
                # Original order for continuous (e.g., wavelengths)
                original_vars = X.columns.tolist()
                loadings_melt['Variable'] = pd.Categorical(loadings_melt['Variable'], categories=original_vars, ordered=True)
                loadings_melt = loadings_melt.sort_values(['PC', 'Variable'])
                # Line plot: X=Variable, Y=Loading, color=PC, connected lines per PC, no markers
                fig_loadings = px.line(loadings_melt, x='Variable', y='Loading', color='PC',
                                       markers=False,
                                       title="Loadings: Connected Line Plot (Abs Values)",
                                       labels={'Variable': 'Factors/Variables', 'Loading': 'Loading Magnitude'})
                fig_loadings.update_traces(line=dict(width=2, dash='solid')) # Continuous solid lines
                fig_loadings.update_xaxes(tickangle=45, tickfont=dict(size=9))
                if len(original_vars) > 50:
                    st.warning("Many variables (>50)—zoom/pan the plot for details in spectroscopy data.")
            st.plotly_chart(fig_loadings, use_container_width=True)
            # Show loadings table
            st.subheader("Loadings Table (Top 3 PCs)")
            st.dataframe(loadings)
# Download buttons (always available after upload, but use num_save_pcs)
st.subheader("Download PCA Results")
col1, col2 = st.columns(2)
with col1:
    # PC Scores (transformed data)
    if is_precomputed:
        X_pca_save = X_pca[:, :num_save_pcs]
    else:
        pca_save = PCA(n_components=num_save_pcs)
        X_pca_save = pca_save.fit_transform(X_scaled)
    df_scores = pd.DataFrame(X_pca_save, columns=[f'PC{i+1}' for i in range(num_save_pcs)])
    df_scores['label'] = y # Use simplified labels
    csv_scores = df_scores.to_csv(index=False)
    st.download_button("Download PC Scores CSV", csv_scores, "pc_scores.csv", "text/csv")
with col2:
    if pca_full is not None:
        # Loadings
        loadings_save = pd.DataFrame(pca_full.components_[:num_save_pcs],
                                     columns=X.columns,
                                     index=[f'PC{i+1}' for i in range(num_save_pcs)])
        csv_loadings = loadings_save.to_csv(index=True)
        st.download_button("Download Loadings CSV", csv_loadings, "pca_loadings.csv", "text/csv")
    else:
        st.info("Loadings not available for pre-computed mode.")
st.info(f"Downloads include top {num_save_pcs} PCs.")
# Classification section (outputs in main body)
st.header("Classification Results")
if X_pca_2d_global is not None and (run_lda or run_knn) and selected_for_a and selected_for_b:
    # Filter data for selected groups
    mask_group_a = y_global.isin(selected_for_a)
    mask_group_b = y_global.isin(selected_for_b)
    mask_selected = mask_group_a | mask_group_b
    X_selected = X_pca_2d_global[mask_selected]
    y_selected = np.where(mask_group_a[mask_selected], 0, 1) # Binary: 0 for Group A, 1 for Group B
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size, random_state=42, stratify=y_selected
        )
    else:
        X_train, X_test, y_train, y_test = X_selected, X_selected, y_selected, y_selected
    if run_lda:
        if optimize_lda:
            param_grid_lda = {'solver': ['svd', 'lsqr', 'eigen']}
            lda_grid = GridSearchCV(LDA(), param_grid_lda, cv=5)
            lda_grid.fit(X_train, y_train)
            best_lda = lda_grid.best_estimator_
            best_params_lda = lda_grid.best_params_
            st.write(f"**Optimized LDA Parameters:** {best_params_lda}")
        else:
            best_lda = LDA()
            best_lda.fit(X_train, y_train)
            best_params_lda = {'solver': 'svd'}
            st.write(f"**LDA Parameters:** {best_params_lda}")
        y_pred_lda = best_lda.predict(X_test)
        acc_lda = accuracy_score(y_test, y_pred_lda)
        st.subheader("LDA Confusion Matrix")
        cm_lda = confusion_matrix(y_test, y_pred_lda)
        fig_cm_lda = px.imshow(cm_lda, text_auto=True, x=[rename_a, rename_b], y=[rename_a, rename_b],
                               color_continuous_scale='Blues', title="LDA Confusion Matrix")
        st.plotly_chart(fig_cm_lda, use_container_width=True)
        st.write(f"**Accuracy:** {acc_lda:.2f}")
        # LDA decision boundary using plot_decision_regions
        st.subheader("LDA Decision Boundary")
        fig_lda, ax_lda = plt.subplots(figsize=(8, 6))
        plot_decision_regions(X_selected, y_selected, clf=best_lda, legend=2, ax=ax_lda)
        ax_lda.set_xlabel('PC1')
        ax_lda.set_ylabel('PC2')
        ax_lda.set_title('LDA Decision Boundary')
        st.pyplot(fig_lda)
    if run_knn:
        if optimize_knn:
            param_grid_knn = {'n_neighbors': range(1, min(21, len(y_train)//2 + 1))}
            knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
            knn_grid.fit(X_train, y_train)
            best_knn = knn_grid.best_estimator_
            best_params_knn = knn_grid.best_params_
            best_k = best_params_knn['n_neighbors']
            st.write(f"**Optimized KNN Parameters:** {best_params_knn}")
        else:
            best_knn = KNeighborsClassifier(n_neighbors=k)
            best_knn.fit(X_train, y_train)
            best_params_knn = {'n_neighbors': k}
            best_k = k
            st.write(f"**KNN Parameters:** {best_params_knn}")
        y_pred_knn = best_knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        st.subheader("KNN Confusion Matrix")
        cm_knn = confusion_matrix(y_test, y_pred_knn)
        fig_cm_knn = px.imshow(cm_knn, text_auto=True, x=[rename_a, rename_b], y=[rename_a, rename_b],
                               color_continuous_scale='Blues', title=f"KNN (k={best_k}) Confusion Matrix")
        st.plotly_chart(fig_cm_knn, use_container_width=True)
        st.write(f"**Accuracy:** {acc_knn:.2f}")
        # KNN decision boundary using plot_decision_regions
        st.subheader("KNN Decision Boundary")
        fig_knn, ax_knn = plt.subplots(figsize=(8, 6))
        plot_decision_regions(X_selected, y_selected, clf=best_knn, legend=2, ax=ax_knn)
        ax_knn.set_xlabel('PC1')
        ax_knn.set_ylabel('PC2')
        ax_knn.set_title(f'KNN (k={best_k}) Decision Boundary')
        st.pyplot(fig_knn)
elif X_pca_2d_global is None:
    st.warning("Need at least 2 PCs for classification visualization.")
else:
    st.info("Enable LDA or KNN in the sidebar and select groups to run classification.")
# Footer
st.markdown("---")
st.caption("Reusable for any dataset. Built with Streamlit & scikit-learn. Questions? Ask Drew!")
