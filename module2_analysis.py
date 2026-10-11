"""Prepare and analyze the Module 2 clustering and PCA dataset.

This script creates the unlabeled numeric model input, reproducible result
tables, and static figures used by the Streamlit website. Survey response
codes are reverse-coded so that larger values consistently mean more of the
named construct (more support, stronger connection, or greater independence).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "module2"
ASSETS.mkdir(parents=True, exist_ok=True)

SOURCE_FEATURES = [
    "prepared_for_independence",
    "advice_finances",
    "advice_career",
    "advice_mental_health",
    "advice_physical_health",
    "advice_friendships",
    "relies_on_parents_emotionally",
    "financial_independence",
    "relationship_rating",
    "true_self_with_parent",
    "parent_relies_on_child_emotionally",
    "knows_parent_beyond_parent_role",
]

DISPLAY_NAMES = [
    "Preparation",
    "Financial Advice",
    "Career Advice",
    "Mental Health Advice",
    "Physical Health Advice",
    "Friendship Advice",
    "Emotional Reliance",
    "Financial Independence",
    "Relationship Quality",
    "Authenticity",
    "Parent Relies on Young Adult",
    "Knows Parent as Person",
]

SAGE = "#6F9F8C"
TEAL = "#2A7F78"
DARK = "#183F3A"
CREAM = "#FAF7EF"
CORAL = "#E58D72"
GOLD = "#D6AE55"


def save_figure(fig, filename):
    fig.savefig(ASSETS / filename, dpi=180, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)


def prepare_data():
    source = pd.read_csv(ROOT / "pew_young_adults_clean_numeric.csv")
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(source[SOURCE_FEATURES])

    # Original Pew codes run from 1 (most) to 5 (least). Reverse-coding makes
    # every variable easier to compare: 5 now means more of the construct.
    prepared = pd.DataFrame(6 - imputed, columns=DISPLAY_NAMES)
    prepared.to_csv(ROOT / "module2_unlabeled_numeric_data.csv", index=False)
    return source, prepared


def analyze(prepared, source):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(prepared)

    k_rows = []
    k_models = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = model.fit_predict(scaled)
        k_models[k] = (model, labels)
        k_rows.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(scaled, labels),
                "inertia": model.inertia_,
            }
        )
    k_scores = pd.DataFrame(k_rows)
    k_scores.to_csv(ROOT / "module2_kmeans_scores.csv", index=False)
    best_k = int(k_scores.loc[k_scores["silhouette_score"].idxmax(), "k"])
    kmeans_labels = k_models[best_k][1]

    cosine_distances = pdist(scaled, metric="cosine")
    hierarchy = linkage(cosine_distances, method="average")
    h_rows = []
    h_labels_by_k = {}
    for k in range(2, 7):
        labels = fcluster(hierarchy, t=k, criterion="maxclust") - 1
        h_labels_by_k[k] = labels
        h_rows.append(
            {
                "k": k,
                "cosine_silhouette": silhouette_score(
                    scaled, labels, metric="cosine"
                ),
            }
        )
    h_scores = pd.DataFrame(h_rows)
    h_scores.to_csv(ROOT / "module2_hierarchical_scores.csv", index=False)
    h_best_k = int(
        h_scores.loc[h_scores["cosine_silhouette"].idxmax(), "k"]
    )
    hierarchical_labels = h_labels_by_k[h_best_k]

    pca = PCA()
    pca_scores = pca.fit_transform(scaled)
    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(DISPLAY_NAMES) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(ROOT / "module2_pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=DISPLAY_NAMES,
        columns=[f"PC{i}" for i in range(1, len(DISPLAY_NAMES) + 1)],
    )
    loadings.rename_axis("feature").reset_index().to_csv(
        ROOT / "module2_pca_loadings.csv", index=False
    )

    results = pd.DataFrame(
        {
            "respondent_id": source["respondent_id"],
            "PC1": pca_scores[:, 0],
            "PC2": pca_scores[:, 1],
            "PC3": pca_scores[:, 2],
            "kmeans_cluster": kmeans_labels + 1,
            "hierarchical_cluster": hierarchical_labels + 1,
        }
    )
    results.to_csv(ROOT / "module2_model_results.csv", index=False)

    profile = prepared.assign(cluster=kmeans_labels + 1).groupby("cluster").mean()
    profile.rename_axis("cluster").reset_index().to_csv(
        ROOT / "module2_cluster_profiles.csv", index=False
    )

    agreement = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    summary = pd.DataFrame(
        {
            "metric": [
                "best_kmeans_k",
                "best_kmeans_silhouette",
                "best_hierarchical_k",
                "best_hierarchical_cosine_silhouette",
                "adjusted_rand_index",
                "pc1_variance",
                "pc2_variance",
                "pc1_pc2_cumulative_variance",
            ],
            "value": [
                best_k,
                k_scores.loc[
                    k_scores["k"].eq(best_k), "silhouette_score"
                ].iloc[0],
                h_best_k,
                h_scores.loc[
                    h_scores["k"].eq(h_best_k), "cosine_silhouette"
                ].iloc[0],
                agreement,
                pca.explained_variance_ratio_[0],
                pca.explained_variance_ratio_[1],
                pca.explained_variance_ratio_[:2].sum(),
            ],
        }
    )
    summary.to_csv(ROOT / "module2_analysis_summary.csv", index=False)
    return hierarchy, loadings


def create_sample_table(prepared, filename, title):
    sample = prepared.head(6).iloc[:, :7]
    fig, ax = plt.subplots(figsize=(13, 3.4), facecolor=CREAM)
    ax.axis("off")
    table = ax.table(
        cellText=sample.round(0).astype(int).values,
        colLabels=sample.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C8DDD4")
        if row == 0:
            cell.set_facecolor(SAGE)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F4FAF7" if row % 2 else "#E7F2ED")
            cell.set_text_props(color=DARK)
    ax.set_title(title, fontsize=15, color=DARK, weight="bold", pad=14)
    save_figure(fig, filename)


def create_dendrogram(hierarchy):
    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor=CREAM)
    dendrogram(
        hierarchy,
        truncate_mode="lastp",
        p=35,
        leaf_rotation=75,
        leaf_font_size=8,
        color_threshold=0.65 * hierarchy[:, 2].max(),
        above_threshold_color="#6A7E77",
        ax=ax,
    )
    ax.set_title(
        "Hierarchical Clustering with Cosine Distance and Average Linkage",
        color=DARK,
        fontsize=15,
        weight="bold",
    )
    ax.set_xlabel("Merged respondent groups (truncated for readability)")
    ax.set_ylabel("Cosine distance")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DCE8E3", alpha=0.7)
    fig.tight_layout()
    save_figure(fig, "hierarchical_dendrogram.png")


def create_clustering_concepts():
    rng = np.random.default_rng(14)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=CREAM)
    centers = np.array([[-2, -1], [0.2, 2], [2.3, -0.5]])
    colors = [TEAL, CORAL, GOLD]
    for center, color in zip(centers, colors):
        pts = rng.normal(center, [0.55, 0.48], size=(34, 2))
        axes[0].scatter(pts[:, 0], pts[:, 1], color=color, alpha=0.75, s=32)
        axes[0].scatter(*center, marker="*", s=240, color=DARK, edgecolor="white")
    axes[0].set_title("Partitional clustering", color=DARK, weight="bold")
    axes[0].text(
        0.5,
        -0.13,
        "K-Means assigns every observation to one of k groups.",
        ha="center",
        transform=axes[0].transAxes,
        color="#506A64",
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].spines[:].set_visible(False)

    small = linkage(rng.normal(size=(12, 3)), method="average")
    dendrogram(small, ax=axes[1], no_labels=True, color_threshold=2.0)
    axes[1].set_title("Hierarchical clustering", color=DARK, weight="bold")
    axes[1].text(
        0.5,
        -0.13,
        "A dendrogram records how observations merge into nested groups.",
        ha="center",
        transform=axes[1].transAxes,
        color="#506A64",
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.suptitle("Two Ways to Discover Groups", color=DARK, fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    save_figure(fig, "clustering_methods.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=CREAM)
    axes[0].scatter([0, 3], [0, 2], s=110, color=[TEAL, CORAL])
    axes[0].plot([0, 3], [0, 2], color=DARK, linestyle="--", linewidth=2)
    axes[0].text(1.5, 1.25, "straight-line distance", ha="center", color=DARK)
    axes[0].set_title("Euclidean distance", color=DARK, weight="bold")
    axes[0].set_xlim(-0.7, 3.7)
    axes[0].set_ylim(-0.7, 2.8)

    origin = np.array([0, 0])
    for vec, color in [(np.array([3.0, 1.0]), TEAL), (np.array([2.1, 2.2]), CORAL)]:
        axes[1].arrow(*origin, *vec, color=color, width=0.04, head_width=0.18)
    angle = np.linspace(np.arctan2(1, 3), np.arctan2(2.2, 2.1), 40)
    axes[1].plot(1.2 * np.cos(angle), 1.2 * np.sin(angle), color=DARK, linewidth=2)
    axes[1].text(1.25, 0.75, "angle", color=DARK)
    axes[1].set_title("Cosine distance", color=DARK, weight="bold")
    axes[1].set_xlim(-0.4, 3.6)
    axes[1].set_ylim(-0.4, 2.8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    fig.suptitle("Distance Determines What 'Similar' Means", color=DARK, fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "distance_metrics.png")


def create_pca_concepts():
    rng = np.random.default_rng(21)
    cloud = rng.multivariate_normal([0, 0], [[3.4, 2.5], [2.5, 2.4]], 220)

    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=CREAM)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=23, alpha=0.38, color=SAGE)
    ax.arrow(0, 0, 3.7, 2.8, color=TEAL, width=0.035, head_width=0.22, length_includes_head=True)
    ax.arrow(0, 0, -1.6, 2.1, color=CORAL, width=0.035, head_width=0.22, length_includes_head=True)
    ax.text(3.15, 3.05, "PC1 eigenvector\n(most variance)", color=TEAL, weight="bold")
    ax.text(-2.5, 2.35, "PC2 eigenvector\n(next variance)", color=CORAL, weight="bold")
    ax.axhline(0, color="#C8D8D2", linewidth=1)
    ax.axvline(0, color="#C8D8D2", linewidth=1)
    ax.set_title("Eigenvectors Define the New PCA Axes", color=DARK, fontsize=17, weight="bold")
    ax.set_xlabel("Original variable direction 1")
    ax.set_ylabel("Original variable direction 2")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "pca_eigenvectors.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=CREAM)
    axes[0].scatter(cloud[:, 0], cloud[:, 1], s=22, alpha=0.45, color=SAGE)
    axes[0].set_title("Many correlated variables", color=DARK, weight="bold")
    axes[0].set_xlabel("Variable A")
    axes[0].set_ylabel("Variable B")
    direction = np.array([0.8, 0.6])
    projection = cloud @ direction
    axes[1].scatter(projection, np.zeros_like(projection), s=25, alpha=0.45, color=TEAL)
    axes[1].set_title("Compressed onto principal component 1", color=DARK, weight="bold")
    axes[1].set_xlabel("PC1 score")
    axes[1].set_yticks([])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Dimensionality Reduction Preserves the Strongest Pattern", color=DARK, fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "pca_dimensionality_reduction.png")


def create_independence_pathways_image():
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=CREAM)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    center = FancyBboxPatch(
        (4.55, 1.75), 2.9, 1.45,
        boxstyle="round,pad=0.25,rounding_size=0.18",
        facecolor=TEAL, edgecolor="none"
    )
    ax.add_patch(center)
    ax.text(6, 2.47, "Young-adult\nindependence", ha="center", va="center", color="white", fontsize=17, weight="bold")
    nodes = [
        (1.25, 3.8, "Housing", SAGE),
        (3.3, 4.35, "Education", GOLD),
        (8.7, 4.35, "Employment", CORAL),
        (10.75, 3.8, "Finances", "#6E88A8"),
        (2.15, 0.65, "Emotional\nsupport", "#9B7AA5"),
        (9.85, 0.65, "Family\nconnection", "#5C9278"),
    ]
    for x, y, label, color in nodes:
        circ = Circle((x, y), 0.72, facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", color="white", weight="bold", fontsize=11)
        start = (x + (0.6 if x < 6 else -0.6), y + (-0.25 if y > 2.5 else 0.25))
        end = (4.65 if x < 6 else 7.35, 2.7 if y > 2.5 else 2.1)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, color="#78958A", linewidth=1.8))
    ax.set_title("Independence Develops Through Interconnected Life Pathways", color=DARK, fontsize=18, weight="bold", pad=14)
    save_figure(fig, "independence_pathways.png")


def main():
    source, prepared = prepare_data()
    hierarchy, _ = analyze(prepared, source)
    create_sample_table(
        prepared,
        "clustering_data_sample.png",
        "Sample of the Unlabeled Numeric Clustering Data",
    )
    create_sample_table(
        prepared,
        "pca_data_sample.png",
        "Sample of the Numeric PCA Input Data",
    )
    create_dendrogram(hierarchy)
    create_clustering_concepts()
    create_pca_concepts()
    create_independence_pathways_image()
    print("Module 2 data, results, and figures created successfully.")


if __name__ == "__main__":
    main()
