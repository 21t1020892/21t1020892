import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from openml.datasets import get_dataset
import time

# Load MNIST dataset from OpenML
@st.cache_data
def load_mnist(sample_size):
    dataset = get_dataset(554)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X, y = X.iloc[:sample_size], y.iloc[:sample_size]
    return X, y.astype(int)

# Function to display sample images
def show_sample_images(X, y, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
        axes[i].set_title(y.iloc[i])
        axes[i].axis('off')
    st.pyplot(fig)

# Function to apply PCA
def apply_pca(X, n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    return X_pca, explained_variance

# Function to apply t-SNE
def apply_tsne(X, n_components, perplexity):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

# Streamlit UI
st.title("PCA & t-SNE Dimensionality Reduction on MNIST")

# User input for number of samples and dimensions
sample_size = st.sidebar.slider("Number of MNIST Samples", 1000, 70000, 5000, step=1000)
num_dimensions = st.sidebar.slider("Number of Dimensions to Reduce To", 2, 100, 50)

# Load data
X, y = load_mnist(sample_size)
st.write("Data loaded successfully! Shape:", X.shape)

# Show sample images
show_sample_images(X, y)

# Tabs for PCA and t-SNE
selected_tab = st.sidebar.radio("Choose Method", ["PCA", "t-SNE"])

if selected_tab == "PCA":
    st.header("Principal Component Analysis (PCA)")
    n_components_pca = num_dimensions
    
    with mlflow.start_run():
        start_time = time.time()
        X_pca, explained_variance = apply_pca(X, n_components_pca)
        mlflow.log_param("pca_n_components", n_components_pca)
        mlflow.log_metric("pca_explained_variance", explained_variance)
        end_time = time.time()
        mlflow.log_metric("execution_time", end_time - start_time)
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.5)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        plt.colorbar(scatter)
        st.pyplot(fig)
        mlflow.end_run()

elif selected_tab == "t-SNE":
    st.header("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
    n_components_tsne = st.radio("Number of t-SNE Components", [2, 3], index=0)
    perplexity = st.slider("t-SNE Perplexity", 5, 50, 30)
    
    with mlflow.start_run():
        start_time = time.time()
        X_pca, _ = apply_pca(X, num_dimensions)  # Reduce to chosen dimensions for better performance
        X_tsne = apply_tsne(X_pca, n_components_tsne, perplexity)
        mlflow.log_param("tsne_n_components", n_components_tsne)
        mlflow.log_param("tsne_perplexity", perplexity)
        end_time = time.time()
        mlflow.log_metric("execution_time", end_time - start_time)
        
        fig, ax = plt.subplots()
        if n_components_tsne == 2:
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.5)
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
        else:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='tab10', alpha=0.5)
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.set_zlabel("t-SNE Component 3")
        
        plt.colorbar(scatter)
        st.pyplot(fig)
        mlflow.end_run()