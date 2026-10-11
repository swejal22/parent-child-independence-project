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
        Clustering is an **unsupervised learning** approach that looks for clusters of
        observations with similar feature combinations. Clustering does not start with
        classes or answers. Rather, the technique calculates distances between respondents
        of young adults. This research project applies clustering to determine whether 
        combinations of parental advice, emotional bond,relationship quality, preparedness, 
        authenticity, and financial independence recur in the Pew dataset.

        **Partitional clustering** directly splits the data into a desired number of clusters.
        K-means falls under partitional clustering, which continually classifies the data
        points into their closest centroid and then re-computes them until the process
        converges. K-means utilizes **Euclidean distance** and, hence, is best applied
        after standardization. **Hierarchical clustering** starts by treating every point
        as an independent cluster and then merges the closest clusters step by step. This
        method creates a dendrogram that captures the sequence of clusters being
        merged. The following hierarchical clustering utilizes **cosine distance**, as
        required, and average linkage.
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
        Clustering needs **numeric data without labels** since the intention is to find
        the hidden groups instead of predicting an outcome. Respondent IDs, survey
        weights, and demographic categories were not used in the analysis. Twelve scales
        measuring relationships in range of 1–5 were selected for clustering. In the
        original Pew research scale 1 represented the strongest reaction, therefore,
        the data was reverse-transformed: higher scores indicate more preparation,
        advice, contact, relationship quality, authenticity, and independence. Missing
        values on few questions were filled in with medians, and all the measures
        were scaled to be compared equally during the distance calculation. There are
        no labels for clusters in the input file.
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
        The analysis was done with Python programming language through pandas, NumPy, scikit-learn, 
        SciPy, Matplotlib, and Seaborn packages. The notebook runs K-Means with k = 2,3,4,5,6, 
        finds silhouette scores, does PCA projection, and hierarchical clustering with cosine 
        distance by average linkage.

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
        The five values are **k = 2, 3, 4, 5, and 6**. The highest average 
        silhouette score is **{summary['best_kmeans_silhouette']:.3f} for k = {best_k}**, 
        meaning that the two cluster analysis shows the most distinct separation. 
        The score is positive and moderate, indicating that the clusters describe an underlying 
        trend but have some overlap. Thus, young-adult family relations should be considered 
        continua and not rigid types.
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
        These two groups are primarily separated on the basis of the first principal component, 
        which is the indicator of the general family ties and connections. 
        The division is not strictly set, but gradual, similarly to the silhouette score result. 
        This graph is two-dimensional projection of the full 12 features analysis; K-means was 
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
        The high connection group reports receiving more parental advice on all topics,
        having greater emotional dependence, better relationship quality, more
        authenticity, better understanding of the individuality of their parents, and more
        readiness for adulthood. The low connection group has slightly higher levels of
        financial independence while scoring lower on all relationship measures.
        This indicates that there is a connection but also a difference between these two
        processes of becoming independent.
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
        The point where the cosine-distance hierarchy is cut at **k = {best_h_k}** produces 
        the maximum cosine silhouette score of **{summary['best_hierarchical_cosine_silhouette']:.3f}** 
        out of all of those that were tested. Therefore, hierarchical clustering also confirms the 
        same general two-cluster structure that was found using K-Means. The Adjusted Rand Index 
        between the two clusterings is **{summary['adjusted_rand_index']:.3f}**. 
        The discrepancy results from the fact that K-Means relies on Euclidean distance from centroids.
        """
    )

    st.header("Conclusions")
    st.markdown(
        """
        Both approaches point out a clear difference between young adults who perceive more
        parental involvement, such as more family support and communication, and those who
        perceive less of it. Still, a middle distance and an overlap indicate that
        independence cannot be limited to two types of families only. There could be a
        balance of financial independence and closer or distant parent-child relations.
        Moreover, parental support cannot be considered as a reason for poor preparation
        for adult life. The most valuable finding here is a pattern, not a dichotomy –
        emotional support, advice, authenticity and financial independence are different but
        connected aspects of growing up.
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
        Principal Component Analysis (PCA) is a dimension reduction technique that is
        unsupervised in nature. A data set with multiple correlated variables is hard to
        represent, analyze, and understand because humans cannot perceive a 12-dimensional
        space directly. Correlated variables might also carry redundant information.
        PCA changes the variables in the original data set into new uncorrelated axes
        called **principal components**, where the first principal component explains the
        most variance and each subsequent one explains the maximum variance after being
        orthogonal to all previous ones.

        **Eigen-vectors** define the direction of the new axis while the **Eigen-values**
        provide the amount of variance explained by those directions. The loadings of
        variables tell us about the contribution of the original variables towards
        any component. Sklearn makes use of SVD (Singular Value Decomposition) that
        has a mathematical relationship with the eigen-decomposition of the covariance
        matrix. Information loss is the drawback of dimensionality reduction as it helps
        in finding out the overall patterns and reduces redundancies. It only makes sense
        when the percentage of variance retained is mentioned explicitly.
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
        PCA necessitates numeric features that have been observed on the same set of observations. 
        This technique is scale dependent; therefore, the 12 reversed relationship and independence
        variables that were used for clustering were standardized by using the median
        imputation method, which ensured a zero mean and unit standard deviation.
        Responder IDs, weights, labels, and categorical demographic variables were
        removed from the analysis.
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
        The PCA analysis was conducted using Python and the following libraries: pandas, NumPy, 
        scikit-learn, Matplotlib, and Seaborn. The notebook will standardize features, fit components, 
        calculate explained variance, examine loadings, and generate a scree plot, biplot, and loading heatmap.

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
        PC1 accounts for **{summary['pc1_variance']:.1%}** of the standardized variability,
        while PC2 accounts for **{summary['pc2_variance']:.1%}**. Both together
        account for **{summary['pc1_pc2_cumulative_variance']:.1%}**, meaning that the two
        dimensional representation captures more than half of the data variability.
        Seven components are needed to capture 80% cumulative variance. The steep fall-off
        from PC1 means that there is high commonality among the variables, but the
        additional components capture important differences.
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
        Arrows which point in a similar direction are indicative of positively correlated variables;
        arrows which point in opposite directions signify differences. PC1 is highly influenced by parental advice,
        emotional dependence, relationship, and authenticity; therefore, it denotes family connectivity. PC2 is
        mainly characterized by financial independence, authenticity, relationship, and knowledge of parents.
        Several measures of advice show an opposition to PC2; thus, there exists an **autonomy versus connection**
        difference. The length of arrows denotes the extent of influence on the two component model.
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
        Component Structure Heatmap. Component 1 is based largely on relationship and advice.
        Component 2 is focused on financial independence and relational authenticity.
        Component 3 involves reciprocity between the parents' dependence on the
        young adult and the preparedness of the young adult. This shows that
        reciprocity was also one of the dimensions which could not be seen in the
        previous components. The direction of the relationships is indicated by the
        positive and negative signs. The sign of the eigenvector is arbitrary,
        and thus the interpretation should be based on the relative patterns.
        """
    )

    st.header("Conclusions")
    st.markdown(
        """
        The PCA indicates that the twelve indicators do not reflect one unidimensional construct
        regarding dependence vs. independence. The prominent common factor is related to
        advice from the family, emotional dependence, relationship quality, and
        authenticity, while the second factor divides financial independence from
        parental advising. The findings confirm the core assumption of the project,
        namely that independence does not necessarily exclude the maintenance of close
        emotions. A young person can be independent in a financial aspect but at the same
        time have good relationships with his/her parents, or can have less advice, yet be
        far from being ready.
        """
    )
