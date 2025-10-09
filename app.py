import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import io
from PIL import Image

# Streamlit app title
st.title("Spectral Data PCA and Classification App")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Data shape:", df.shape)
    st.write("First few rows:")
    st.dataframe(df.head())

    # Assume first column is 'wavenumber'
    wavenumber_col = df.columns[0]
    data_cols = df.columns[1:]
    
    # Extract unique sample names (assuming groups of 50 replicates per sample as in notebook)
    group_labels = []
    for sample_idx in range(len(data_cols) // 50):
        sample_name = data_cols[sample_idx * 50].split('_')[0]  # e.g., '1224535M65'
        group_labels.extend([sample_name] * 50)
    unique_samples = list(dict.fromkeys(group_labels))  # Unique sample names
    st.write("Detected unique samples (classes):", unique_samples)

    # Preprocessing option
    preprocess_option = st.radio(
        "Select preprocessing:",
        ['None', 'SNV', 'Z-score', 'SNV then Z-score']
    )

    # Prepare data
    X = df[data_cols].values.T  # Shape: (n_samples, n_features)
    wavenumbers = df[wavenumber_col].values

    # Apply preprocessing
    X_processed = X.copy()
    if preprocess_option == 'SNV' or preprocess_option == 'SNV then Z-score':
        # SNV: per sample (row) normalization
        for i in range(X_processed.shape[0]):
            row_mean = np.mean(X_processed[i])
            row_std = np.std(X_processed[i])
            if row_std == 0:
                X_processed[i] = 0  # Handle zero std
            else:
                X_processed[i] = (X_processed[i] - row_mean) / row_std
    if preprocess_option == 'Z-score' or preprocess_option == 'SNV then Z-score':
        # Z-score: feature-wise (StandardScaler)
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed)

    # Run PCA
    pca = PCA()
    scores = pca.fit_transform(X_processed)
    loadings = pca.components_.T
    variance_explained = pca.explained_variance_ratio_ * 100

    # Determine number of components
    cumulative_variance = np.cumsum(variance_explained)
    n_components = np.where(cumulative_variance >= 99)[0][0] + 2

    # Save PCA scores
    pca_df = pd.DataFrame(
        scores[:, :n_components],
        columns=[f'PC{i+1} ({variance_explained[i]:.2f}%)' for i in range(n_components)],
        index=data_cols
    )
    csv_buffer = io.StringIO()
    pca_df.to_csv(csv_buffer)
    st.download_button(
        label="Download PCA Scores CSV",
        data=csv_buffer.getvalue(),
        file_name=f"PCA_Scores_{uploaded_file.name}",
        mime="text/csv"
    )

    # PCA Plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA Scores Plot (PC1 vs PC2)")
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = unique_samples
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            mask = np.array(group_labels) == label
            ax.scatter(scores[mask, 0], scores[mask, 1], c=[color], label=label, alpha=0.7)
        ax.set_xlabel(f'PC1 ({variance_explained[0]:.2f}% variance)')
        ax.set_ylabel(f'PC2 ({variance_explained[1]:.2f}% variance)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Scree Plot")
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(range(1, n_components + 1), variance_explained[:n_components], alpha=0.7)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{variance_explained[i]:.2f}%',
                    ha='center', va='bottom')
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Variance Explained (%)')
        ax.set_title('Scree Plot (Up to 99% Cumulative Variance + 1 Component)')
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("PCA Loadings Plot (PC1, PC2, PC3)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavenumbers, loadings[:, 0], label='PC1 Loadings')
    ax.plot(wavenumbers, loadings[:, 1], label='PC2 Loadings')
    ax.plot(wavenumbers, loadings[:, 2], label='PC3 Loadings')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Loadings')
    ax.set_title('PCA Loadings Plot (PC1, PC2, PC3)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Classification options
    st.header("Classification (LDA / KNN)")
    split_data = st.checkbox("Split into train/test sets")
    if split_data:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    else:
        test_size = 0

    # Select two classes for binary classification
    class1 = st.selectbox("Select Class 1", unique_samples)
    class2_options = [c for c in unique_samples if c != class1]
    class2 = st.selectbox("Select Class 2", class2_options)

    # Filter data for selected classes
    mask_class1 = np.array(group_labels) == class1
    mask_class2 = np.array(group_labels) == class2
    mask_selected = mask_class1 | mask_class2
    X_selected = scores[mask_selected, :2]  # Use PC1 and PC2 for 2D viz
    y_selected = np.array([0 if mask_class1[i] else 1 for i in range(len(group_labels)) if mask_selected[i]])

    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size, random_state=42, stratify=y_selected
        )
    else:
        X_train, X_test, y_train, y_test = X_selected, X_selected, y_selected, y_selected

    run_lda = st.checkbox("Run Linear Discriminant Analysis (LDA)")
    run_knn = st.checkbox("Run K-Nearest Neighbors (KNN)")
    if run_knn:
        k = st.slider("K value", 1, 15, 5)

    # Confusion matrices and boundaries
    if run_lda or run_knn:
        col3, col4 = st.columns(2)
        
        # Plot setup for boundaries
        h = 0.02  # Step size in mesh
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

    if run_lda:
        lda = LDA()
        lda.fit(X_train, y_train)
        y_pred_lda = lda.predict(X_test)

        with col3:
            st.subheader("LDA Confusion Matrix")
            cm_lda = confusion_matrix(y_test, y_pred_lda)
            disp_lda = ConfusionMatrixDisplay(confusion_matrix=cm_lda, display_labels=[class1, class2])
            disp_lda.plot(cmap=plt.cm.Blues)
            st.pyplot(disp_lda.figure_)

        # LDA boundary plot
        Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_lda = Z_lda.reshape(xx.shape)

        fig_lda, ax_lda = plt.subplots(figsize=(8, 6))
        ax_lda.contourf(xx, yy, Z_lda, alpha=0.8, cmap=plt.cm.RdYlBu)
        ax_lda.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='k')
        ax_lda.set_xlabel('PC1')
        ax_lda.set_ylabel('PC2')
        ax_lda.set_title('LDA Decision Boundary')
        st.pyplot(fig_lda)

    if run_knn:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        with col4:
            st.subheader("KNN Confusion Matrix")
            cm_knn = confusion_matrix(y_test, y_pred_knn)
            disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=[class1, class2])
            disp_knn.plot(cmap=plt.cm.Blues)
            st.pyplot(disp_knn.figure_)

        # KNN boundary plot
        Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_knn = Z_knn.reshape(xx.shape)

        fig_knn, ax_knn = plt.subplots(figsize=(8, 6))
        ax_knn.contourf(xx, yy, Z_knn, alpha=0.8, cmap=plt.cm.RdYlBu)
        ax_knn.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='k')
        ax_knn.set_xlabel('PC1')
        ax_knn.set_ylabel('PC2')
        ax_knn.set_title(f'KNN (k={k}) Decision Boundary')
        st.pyplot(fig_knn)

else:
    st.info("Please upload a CSV file to proceed.")
