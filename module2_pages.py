"""Streamlit pages for Module 2: clustering and PCA."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "module2"
GITHUB = "https://github.com/swejal22/parent-child-independence-project/blob/main"
PALETTE = ["#2A7F78", "#E58D72", "#D6AE55", "#6E88A8", "#9B7AA5"]

FEATURE_SHORT_NAMES = {
    "Preparation": "Preparation",
    "Financial Advice": "Financial advice",
    "Career Advice": "Career advice",
    "Mental Health Advice": "Mental-health advice",
    "Physical Health Advice": "Physical-health advice",
    "Friendship Advice": "Friendship advice",
    "Emotional Reliance": "Emotional reliance",
    "Financial Independence": "Financial independence",
    "Relationship Quality": "Relationship quality",
    "Authenticity": "Authenticity",
    "Parent Relies on Young Adult": "Parent relies on child",
    "Knows Parent as Person": "Knows parent as person",
}


@st.cache_data
def load_module2_results():
    data = pd.read_csv(ROOT / "module2_unlabeled_numeric_data.csv")
    k_scores = pd.read_csv(ROOT / "module2_kmeans_scores.csv")
    h_scores = pd.read_csv(ROOT / "module2_hierarchical_scores.csv")
    model_results = pd.read_csv(ROOT / "module2_model_results.csv")
    profiles = pd.read_csv(ROOT / "module2_cluster_profiles.csv")
    variance = pd.read_csv(ROOT / "module2_pca_explained_variance.csv")
    loadings = pd.read_csv(ROOT / "module2_pca_loadings.csv").set_index("feature")
    summary = pd.read_csv(ROOT / "module2_analysis_summary.csv").set_index("metric")["value"]
    return data, k_scores, h_scores, model_results, profiles, variance, loadings, summary


def _chart_layout(fig, height=500):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=30, r=30, t=75, b=45),
        font=dict(color="#183F3A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
    )
    return fig


def render_clustering_page():
    data, k_scores, h_scores, results, profiles, _, _, summary = load_module2_results()
    best_k = int(summary["best_kmeans_k"])
    best_h_k = int(summary["best_hierarchical_k"])

    st.title("Clustering")
    st.caption(
        "Discovering relationship and independence patterns without using a target label"
    )

    st.header("Overview")
    st.markdown(
        """
        Clustering is an **unsupervised-learning** method that searches for groups of
        observations with similar feature patterns. It does not begin with known class
        labels or a correct answer. Instead, the algorithm uses distances among young
        adults' responses to discover structure. This project uses clustering to ask
        whether combinations of parental advice, emotional connection, relationship
        quality, preparation, authenticity, and financial independence form recurring
        profiles within the Pew sample.

        **Partitional clustering** divides the data directly into a chosen number of
        non-overlapping groups. K-Means is partitional: it repeatedly assigns each
        observation to the nearest centroid and updates the centroids until the solution
        stabilizes. K-Means uses **Euclidean distance** and therefore works best after
        standardization. **Hierarchical clustering** begins with individual observations
        and progressively merges the most similar groups. Its dendrogram preserves the
        nested history of those merges. The hierarchical analysis below uses
        **cosine distance**, as required, with average linkage. Cosine distance emphasizes
        similarity in response patterns rather than only straight-line magnitude.
        """
    )

    left, right = st.columns(2)
    with left:
        st.image(
            ASSETS / "clustering_methods.png",
            caption="Partitional clustering forms a fixed set of groups, while hierarchical clustering records nested merges.",
            use_container_width=True,
        )
    with right:
        st.image(
            ASSETS / "distance_metrics.png",
            caption="Euclidean and cosine distance define similarity in different ways.",
            use_container_width=True,
        )

    st.header("Data Prep")
    st.markdown(
        """
        Clustering requires **unlabeled numeric data** because the goal is to discover
        groups rather than predict a known outcome. Respondent IDs, survey weights, and
        demographic category codes were excluded from the model input. Twelve 1–5
        relationship measures were retained. The original Pew scale placed the strongest
        response at 1, so the variables were reverse-coded: a larger value now consistently
        means more preparation, advice, connection, relationship quality, authenticity, or
        independence. The small number of unanswered items was median-imputed, and all
        features were standardized before distance calculations so each measure contributed
        on a comparable scale. No cluster labels appear in the input file.
        """
    )
    st.image(
        ASSETS / "clustering_data_sample.png",
        caption="A small preview of the prepared numeric model input; the full file contains 1,495 rows and 12 features.",
        use_container_width=True,
    )
    st.markdown(
        f"""
        - [Prepared unlabeled numeric dataset]({GITHUB}/module2_unlabeled_numeric_data.csv)
        - [Original coded Pew dataset]({GITHUB}/pew_young_adults_clean_numeric.csv)
        """
    )

    st.header("Code")
    st.markdown(
        f"""
        The analysis was completed in Python using pandas, NumPy, scikit-learn,
        SciPy, Matplotlib, and Seaborn. The notebook tests K-Means with
        $k=2,3,4,5,6$, calculates silhouette scores, creates a PCA projection,
        and performs average-linkage hierarchical clustering with cosine distance.

        - [Clustering analysis notebook]({GITHUB}/03_Clustering_Analysis.ipynb)
        - [Reproducible Module 2 analysis script]({GITHUB}/module2_analysis.py)
        """
    )

    st.header("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Respondents", f"{len(data):,}")
    m2.metric("Numeric features", data.shape[1])
    m3.metric("Best K-Means k", best_k)
    m4.metric("K-Means silhouette", f"{summary['best_kmeans_silhouette']:.3f}")

    st.subheader("1. Selecting k with the silhouette method")
    fig = px.line(
        k_scores,
        x="k",
        y="silhouette_score",
        markers=True,
        title="K-Means Silhouette Scores for Five Candidate Solutions",
        labels={"k": "Number of clusters (k)", "silhouette_score": "Average silhouette score"},
    )
    fig.update_traces(line_color=PALETTE[0], line_width=4, marker_size=11)
    fig.add_scatter(
        x=[best_k],
        y=[summary["best_kmeans_silhouette"]],
        mode="markers+text",
        marker=dict(color=PALETTE[1], size=17),
        text=["best tested k"],
        textposition="top center",
        showlegend=False,
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(_chart_layout(fig, 460), use_container_width=True)
    st.markdown(
        f"""
        Five values were compared: **k = 2, 3, 4, 5, and 6**. The largest average
        silhouette score is **{summary['best_kmeans_silhouette']:.3f} at k = {best_k}**,
        so the two-cluster solution provides the clearest separation among the tested
        choices. The score is positive but moderate, meaning the groups capture a real
        tendency while still overlapping. Young-adult family relationships should
        therefore be interpreted as continua rather than rigid types.
        """
    )

    st.subheader("2. K-Means clusters in a two-component view")
    cluster_names = {
        1: "Lower reported advice and connection",
        2: "Higher reported advice and connection",
    }
    scatter_data = results.copy()
    scatter_data["Cluster profile"] = scatter_data["kmeans_cluster"].map(cluster_names)
    fig = px.scatter(
        scatter_data,
        x="PC1",
        y="PC2",
        color="Cluster profile",
        opacity=0.62,
        color_discrete_sequence=PALETTE[:2],
        title=f"K-Means Results Projected onto Two Principal Components (k = {best_k})",
        labels={"PC1": "Principal component 1", "PC2": "Principal component 2"},
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.3, color="white")))
    st.plotly_chart(_chart_layout(fig, 560), use_container_width=True)
    st.markdown(
        """
        The two clusters separate mainly along the first principal component, which
        summarizes overall family connection and support. The boundary is gradual rather
        than perfectly sharp, consistent with the silhouette result. The visualization is
        a two-dimensional projection of the full 12-feature analysis; K-Means itself was
        fitted using all standardized features.
        """
    )

    st.subheader("3. What distinguishes the K-Means groups?")
    profile_long = profiles.melt(
        id_vars="cluster", var_name="Measure", value_name="Average response"
    )
    profile_long["Cluster profile"] = profile_long["cluster"].map(cluster_names)
    fig = px.bar(
        profile_long,
        x="Measure",
        y="Average response",
        color="Cluster profile",
        barmode="group",
        color_discrete_sequence=PALETTE[:2],
        title="Average Prepared Responses by Cluster",
        labels={"Average response": "Average score (1–5; higher means more)"},
    )
    fig.update_xaxes(tickangle=-38)
    fig.update_yaxes(range=[1, 5])
    st.plotly_chart(_chart_layout(fig, 600), use_container_width=True)
    st.markdown(
        """
        The higher-connection group reports more parental advice across every topic,
        stronger emotional reliance, greater relationship quality, more authenticity,
        better knowledge of the parent as a person, and greater preparation for adult
        independence. The lower-connection group reports slightly greater financial
        independence despite lower scores on most relationship measures. This suggests
        that financial separation and emotional connection are related but distinct parts
        of becoming independent.
        """
    )

    st.subheader("4. Hierarchical clustering with cosine distance")
    st.image(
        ASSETS / "hierarchical_dendrogram.png",
        caption="The dendrogram is truncated to the last 35 merged groups so its branching structure remains readable.",
        use_container_width=True,
    )
    fig = px.line(
        h_scores,
        x="k",
        y="cosine_silhouette",
        markers=True,
        title="Cosine Silhouette Scores for Hierarchical Cuts",
        labels={"k": "Number of clusters", "cosine_silhouette": "Cosine silhouette score"},
    )
    fig.update_traces(line_color=PALETTE[2], line_width=4, marker_size=11)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(_chart_layout(fig, 430), use_container_width=True)
    st.markdown(
        f"""
        Cutting the cosine-distance hierarchy at **k = {best_h_k}** gives the highest
        tested cosine silhouette score (**{summary['best_hierarchical_cosine_silhouette']:.3f}**).
        Hierarchical clustering therefore supports the same broad two-group structure as
        K-Means. The Adjusted Rand Index between the two assignments is
        **{summary['adjusted_rand_index']:.3f}**, indicating substantial—not perfect—agreement.
        The difference is expected because K-Means uses Euclidean distance to centroids,
        while the hierarchy uses average linkage over cosine distances.
        """
    )

    st.header("Conclusions")
    st.markdown(
        """
        Both methods identify a broad contrast between young adults reporting stronger
        family advice and emotional connection and those reporting less of these forms of
        involvement. At the same time, the moderate separation and visible overlap show
        that independence cannot be reduced to two fixed family types. Financial autonomy
        can coexist with either close or limited parental connection, and high parental
        support does not automatically imply low preparation for adulthood. The most useful
        discovery is therefore a relationship pattern—not a judgment: emotional support,
        advice, authenticity, and financial independence represent connected but different
        dimensions of the transition to adulthood.
        """
    )


def render_pca_page():
    data, _, _, results, _, variance, loadings, summary = load_module2_results()

    st.title("Principal Component Analysis")
    st.caption(
        "Reducing twelve related measures to a smaller set of interpretable dimensions"
    )

    st.header("Overview")
    st.markdown(
        """
        Principal Component Analysis (PCA) is an unsupervised dimensionality-reduction
        method. A dataset with many related variables can be difficult to visualize,
        summarize, and model because people cannot directly inspect a 12-dimensional
        space. Correlated variables can also repeat similar information. PCA replaces the
        original variables with new, uncorrelated axes called **principal components**.
        The first component captures the greatest possible variance; each later component
        captures the greatest remaining variance while staying perpendicular to the
        earlier components.

        **Eigenvectors** determine the directions of the new axes, and their associated
        **eigenvalues** quantify how much variance those directions explain. Variable
        loadings show how strongly each original measure contributes to a component.
        Scikit-learn computes this transformation efficiently through Singular Value
        Decomposition (SVD), which is mathematically connected to the eigen-decomposition
        of the covariance matrix. Dimensionality reduction can reveal broad patterns,
        reduce redundancy, support visualization, and simplify later models. Its tradeoff
        is information loss: a two-component view is useful only when the percentage of
        retained variance is stated clearly.
        """
    )

    left, right = st.columns(2)
    with left:
        st.image(
            ASSETS / "pca_eigenvectors.png",
            caption="Eigenvectors define the component directions; eigenvalues rank them by explained variance.",
            use_container_width=True,
        )
    with right:
        st.image(
            ASSETS / "pca_dimensionality_reduction.png",
            caption="Projection compresses correlated dimensions while preserving the strongest pattern.",
            use_container_width=True,
        )

    st.header("Data Prep")
    st.markdown(
        """
        PCA requires numeric features measured across the same observations. It is
        sensitive to scale, so the same 12 reverse-coded relationship and independence
        variables used for clustering were median-imputed and standardized to mean zero
        and standard deviation one. Respondent IDs, survey weights, labels, and nominal
        demographic codes were excluded. Standardization ensures that the components
        represent correlation patterns rather than differences in measurement scale.
        """
    )
    st.image(
        ASSETS / "pca_data_sample.png",
        caption="Sample of the numeric PCA input before standardization; the model uses all 1,495 rows and 12 features.",
        use_container_width=True,
    )
    st.markdown(
        f"- [Prepared numeric PCA dataset]({GITHUB}/module2_unlabeled_numeric_data.csv)"
    )

    st.header("Code")
    st.markdown(
        f"""
        PCA was completed in Python with pandas, NumPy, scikit-learn, Matplotlib,
        and Seaborn. The notebook standardizes the features, fits all components,
        calculates explained variance, inspects loadings, and creates a scree plot,
        biplot, and loading heatmap.

        - [PCA analysis notebook]({GITHUB}/04_PCA_Analysis.ipynb)
        - [Reproducible Module 2 analysis script]({GITHUB}/module2_analysis.py)
        """
    )

    st.header("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original dimensions", data.shape[1])
    m2.metric("PC1 variance", f"{summary['pc1_variance']:.1%}")
    m3.metric("PC2 variance", f"{summary['pc2_variance']:.1%}")
    m4.metric("PC1 + PC2", f"{summary['pc1_pc2_cumulative_variance']:.1%}")

    st.subheader("1. Explained variance and the number of components")
    fig = go.Figure()
    fig.add_bar(
        x=variance["component"],
        y=variance["explained_variance_ratio"] * 100,
        name="Individual variance",
        marker_color=PALETTE[0],
    )
    fig.add_scatter(
        x=variance["component"],
        y=variance["cumulative_variance"] * 100,
        name="Cumulative variance",
        mode="lines+markers",
        line=dict(color=PALETTE[1], width=4),
        marker=dict(size=9),
    )
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#6A7E77",
        annotation_text="80% reference",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Scree Plot and Cumulative Explained Variance",
        xaxis_title="Principal component",
        yaxis_title="Variance explained (%)",
        barmode="overlay",
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(_chart_layout(fig, 520), use_container_width=True)
    st.markdown(
        f"""
        PC1 explains **{summary['pc1_variance']:.1%}** of the standardized variation and
        PC2 explains **{summary['pc2_variance']:.1%}**. Together they retain
        **{summary['pc1_pc2_cumulative_variance']:.1%}**, so a two-dimensional view captures
        more than half of the dataset's structure. Seven components are required to pass
        80% cumulative variance. The sharp decline after PC1 shows that the measures share
        a strong common pattern, but the remaining components preserve meaningful nuances.
        """
    )

    st.subheader("2. Relationship between variables and the first two components")
    plot_points = results.sample(n=min(900, len(results)), random_state=42)
    fig = px.scatter(
        plot_points,
        x="PC1",
        y="PC2",
        opacity=0.28,
        title="PCA Biplot: Respondents and Variable Directions",
        labels={"PC1": "PC1: family connection and support", "PC2": "PC2: autonomy with connection"},
        color_discrete_sequence=[PALETTE[0]],
    )
    combined = (loadings["PC1"].abs() + loadings["PC2"].abs()).nlargest(9).index
    arrow_scale = 4.5
    for feature in combined:
        x = float(loadings.loc[feature, "PC1"] * arrow_scale)
        y = float(loadings.loc[feature, "PC2"] * arrow_scale)
        fig.add_annotation(
            x=x,
            y=y,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=FEATURE_SHORT_NAMES[feature],
            showarrow=True,
            arrowhead=3,
            arrowwidth=1.6,
            arrowcolor=PALETTE[1],
            font=dict(size=10, color="#183F3A"),
        )
    fig.add_hline(y=0, line_color="#C8D8D2", line_width=1)
    fig.add_vline(x=0, line_color="#C8D8D2", line_width=1)
    st.plotly_chart(_chart_layout(fig, 650), use_container_width=True)
    st.markdown(
        """
        Arrows pointing in similar directions represent positively related variables;
        arrows pointing in opposite directions represent contrasts. PC1 receives strong
        contributions from parental advice, emotional reliance, relationship quality,
        and authenticity, so it represents broad **family connection and support**.
        PC2 is especially associated with financial independence, authenticity,
        relationship quality, and knowing a parent as a person. Several advice measures
        point in the opposite PC2 direction, creating an **autonomy-with-connection**
        contrast. Arrow length reflects how strongly a variable contributes to this
        two-component view.
        """
    )

    st.subheader("3. Loading patterns across the first four components")
    loading_view = loadings[["PC1", "PC2", "PC3", "PC4"]].copy()
    loading_view.index = [FEATURE_SHORT_NAMES[x] for x in loading_view.index]
    fig = px.imshow(
        loading_view,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
        text_auto=".2f",
        title="Variable Loadings on the First Four Principal Components",
        labels={"x": "Principal component", "y": "Original variable", "color": "Loading"},
    )
    st.plotly_chart(_chart_layout(fig, 620), use_container_width=True)
    st.markdown(
        """
        The heatmap makes the component structure explicit. PC1 draws broadly from the
        relationship and advice variables. PC2 emphasizes financial independence and
        relational authenticity. PC3 contrasts the parent's emotional reliance on the
        young adult with preparation and financial independence, revealing reciprocity
        as another dimension not fully visible in the first two components. Positive and
        negative signs indicate direction; because the sign of an eigenvector can be
        reversed without changing the solution, interpretation should focus on relative
        patterns rather than whether a sign is inherently good or bad.
        """
    )

    st.header("Conclusions")
    st.markdown(
        """
        PCA shows that the twelve measures do not describe one simple scale of dependence
        versus independence. A large shared dimension reflects family advice, emotional
        reliance, relationship quality, and authenticity, while a second dimension
        distinguishes financial autonomy from frequent parental guidance. The results
        support the project's central idea that becoming independent can occur alongside
        continued emotional closeness. A young adult may be financially self-sufficient
        and still maintain a strong parent relationship, or may receive less advice
        without feeling fully prepared. Reducing the data to principal components makes
        these broad patterns visible while preserving the caution that nearly half of the
        variation lies beyond the first two dimensions.
        """
    )
