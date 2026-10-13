"""
Feature collection and visualization utilities for ADA-SSR.

The functions `collect_feature` and `visualize_tsne` are from the Transfer Learning Library (TLlib)

The LLE, UMAP, and Isomap visualization functions were added for ADA-SSR.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import umap

from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from torch.utils.data import DataLoader


# ============================================================================
# Functions from Transfer Learning Library (TLlib)
# ============================================================================

def collect_feature(
    data_loader: DataLoader,
    feature_extractor: nn.Module,
    device: torch.device,
    max_num_features=None,
) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor`
    to collect features.

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device): Device used for feature extraction.
        max_num_features (int): The max number of features to return.

    Returns:
        Features in shape
        (min(len(data_loader), max_num_features * mini-batch size),
        :math:`|\\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break

            #inputs = data[0].to(device)
            #feature = feature_extractor(inputs).cpu()
            inputs = data[0]

            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]  # weak view pour l'analyse

            inputs = inputs.to(device)
            feature = feature_extractor(inputs).cpu()
            all_features.append(feature)

    return torch.cat(all_features, dim=0)




def visualize_tsne(source_feature, target_feature, filename,
                       source_color="red", target_color="blue"):

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Convert tensors to NumPy
    source = source_feature.detach().cpu().numpy()
    target = target_feature.detach().cpu().numpy()

    # Combine features: source first, target second
    features = np.concatenate([source, target], axis=0)

    # Verify input features
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf values.")

    # Normalize features
    features = StandardScaler().fit_transform(features)

    # Reduce dimensionality before t-SNE
    features = PCA(n_components=50, random_state=33).fit_transform(features)

    # Apply t-SNE
    X_tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        learning_rate=200,
        random_state=33
    ).fit_transform(features)

    # Separate source and target projections
    n_source = len(source)
    source_tsne = X_tsne[:n_source]
    target_tsne = X_tsne[n_source:]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(source_tsne[:, 0], source_tsne[:, 1],
               c=source_color, s=8, alpha=0.6, label="Source")

    ax.scatter(target_tsne[:, 0], target_tsne[:, 1],
               c=target_color, s=8, alpha=0.6, label="Target")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"t-SNE saved successfully: {filename}")

# ============================================================================
# ADA-SSR visualization utilities
# ============================================================================

def visualize_lle(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    filename: str,
    source_color="r",
    target_color="b",
):
    """
    Visualize features from different domains using
    Locally Linear Embedding (LLE).

    Args:
        source_feature (tensor): Source features in shape (minibatch, F).
        target_feature (tensor): Target features in shape (minibatch, F).
        filename (str): File name used to save the LLE visualization.
        source_color (str): Color of the source features. Default: 'r'.
        target_color (str): Color of the target features. Default: 'b'.
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # Map features to 2D using LLE.
    X_lle = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=10,
    ).fit_transform(features)

    # Domain labels: 1 represents source, 0 represents target.
    domains = np.concatenate(
        (
            np.ones(len(source_feature)),
            np.zeros(len(target_feature)),
        )
    )

    # Visualization.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.scatter(
        X_lle[:, 0],
        X_lle[:, 1],
        c=domains,
        cmap=col.ListedColormap([target_color, source_color]),
        s=20,
    )

    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    plt.close(fig)


def visualize_umap(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    filename: str,
    source_color="r",
    target_color="b",
):
    """
    Visualize features from different domains using UMAP.

    Args:
        source_feature (tensor): Source features in shape (minibatch, F).
        target_feature (tensor): Target features in shape (minibatch, F).
        filename (str): File name used to save the UMAP visualization.
        source_color (str): Color of the source features. Default: 'r'.
        target_color (str): Color of the target features. Default: 'b'.
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # Map features to 2D using UMAP.
    X_umap = umap.UMAP(
        n_components=2,
        random_state=33,
    ).fit_transform(features)

    # Domain labels: 1 represents source, 0 represents target.
    domains = np.concatenate(
        (
            np.ones(len(source_feature)),
            np.zeros(len(target_feature)),
        )
    )

    # Visualization.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=domains,
        cmap=col.ListedColormap([target_color, source_color]),
        s=20,
    )

    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    plt.close(fig)


def visualize_isomap(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    filename: str,
    source_color="r",
    target_color="b",
):
    """
    Visualize features from different domains using Isomap.

    Args:
        source_feature (tensor): Source features in shape (minibatch, F).
        target_feature (tensor): Target features in shape (minibatch, F).
        filename (str): File name used to save the Isomap visualization.
        source_color (str): Color of the source features. Default: 'r'.
        target_color (str): Color of the target features. Default: 'b'.
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # Map features to 2D using Isomap.
    X_isomap = Isomap(
        n_components=2,
    ).fit_transform(features)

    # Domain labels: 1 represents source, 0 represents target.
    domains = np.concatenate(
        (
            np.ones(len(source_feature)),
            np.zeros(len(target_feature)),
        )
    )

    # Visualization.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.scatter(
        X_isomap[:, 0],
        X_isomap[:, 1],
        c=domains,
        cmap=col.ListedColormap([target_color, source_color]),
        s=20,
    )

    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    plt.close(fig)