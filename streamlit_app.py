import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Parent–Child Bonds and Independence",
    page_icon="🏡",
    layout="wide"
)

# Website styling
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #274C4B;
        }

        .project-subtitle {
            color: #526D6C;
            font-size: 1.15rem;
            margin-bottom: 2rem;
        }

        .future-section {
            padding: 2rem;
            background-color: #F3F7F6;
            border-left: 5px solid #4F7C78;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    pew_data = pd.read_csv("pew_young_adults_eda_labeled.csv")
    census_data = pd.read_csv(
        "census_acs_2024_living_arrangements_clean.csv"
    )
    return pew_data, census_data


def introduction():
    st.title("Parent–Child Bonds and Independence")
    st.markdown(
        """
        <p class="project-subtitle">
        Exploring communication, emotional support, living arrangements,
        and financial independence among young adults in the United States.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "The complete introduction, topic image, and ten research "
        "questions will be added in Module 1."
    )


def data_prep_eda():
    st.title("Data Preparation and Exploratory Data Analysis")

    pew_data, census_data = load_data()

    st.markdown(
        """
        This section documents the data sources, collection process,
        cleaning decisions, summary statistics, and exploratory
        visualizations used to understand the experiences of young adults.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Pew respondents", f"{len(pew_data):,}")
    col2.metric("Pew variables", pew_data.shape[1])
    col3.metric("States and D.C.", len(census_data))

    st.subheader("Preview of the cleaned Pew dataset")
    st.dataframe(pew_data.head(), use_container_width=True)

    st.subheader("Preview of the cleaned Census dataset")
    st.dataframe(census_data.head(), use_container_width=True)


def future_page(title, module):
    st.title(title)
    st.markdown(
        f"""
        <div class="future-section">
            <h3>Coming in {module}</h3>
            <p>
            This section will contain an overview of the method, prepared
            data, code links, analysis, visual results, and interpretation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def clustering():
    future_page("Clustering", "Module 2")


def pca():
    future_page("PCA", "Module 2")


def naive_bayes():
    future_page("Naive Bayes", "Module 3")


def decision_trees():
    future_page("Decision Trees", "Module 3")


def svms():
    future_page("Support Vector Machines", "Module 4")


def regression():
    future_page("Regression", "Module 5")


def neural_networks():
    future_page("Neural Networks", "Module 5")


def conclusions():
    st.title("Conclusions")
    st.markdown(
        """
        <div class="future-section">
            <h3>Final conclusions will be added in Module 5</h3>
            <p>
            This section will present the project’s main findings in
            accessible, non-technical language.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


pages = [
    st.Page(introduction, title="Introduction"),
    st.Page(data_prep_eda, title="DataPrep_EDA"),
    st.Page(clustering, title="Clustering"),
    st.Page(pca, title="PCA"),
    st.Page(naive_bayes, title="NaiveBayes"),
    st.Page(decision_trees, title="DecTrees"),
    st.Page(svms, title="SVMs"),
    st.Page(regression, title="Regression"),
    st.Page(neural_networks, title="NN"),
    st.Page(conclusions, title="Conclusions")
]

navigation = st.navigation(pages, position="top")
navigation.run()
