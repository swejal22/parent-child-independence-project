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
    st.subheader(
        "Communication, emotional support, living arrangements, "
        "and financial independence among young adults in the United States"
    )

    st.markdown("""
    The transition from adolescence to adulthood involves major changes in housing,
    education, employment, finances, and family relationships. For many young adults,
    becoming independent is a gradual process rather than a single event. Some leave
    the parental home for college or employment, while others remain at home or return
    after living elsewhere. According to the
    [U.S. Census Bureau](https://www.census.gov/library/stories/2024/05/living-arrangements.html),
    living with parents was the most common arrangement among Americans ages 18 to 24
    in 2022. The Census Bureau also reported that 57 percent of young men and 55 percent
    of young women in this age group lived in a parental home. Housing costs, educational
    expenses, employment opportunities, and cultural expectations can all shape these
    arrangements. Living with parents does not necessarily mean that a young adult lacks
    maturity or responsibility. It may provide stability while a person completes school,
    begins a career, saves money, or manages an unexpected difficulty. Understanding
    independence therefore requires attention to both material circumstances and the
    relationships connecting young adults with their parents.
    """)

    st.image(
        "https://www.census.gov/content/dam/Census/library/stories/2024/05/"
        "living-arrangements/living-arrangements.jpg",
        caption=(
            "Living arrangements differ across stages of young adulthood. "
            "Source: U.S. Census Bureau."
        ),
        use_container_width=True
    )

    st.markdown("""
    Parent–child relationships also continue to change after children reach adulthood.
    Parents may provide advice, emotional reassurance, housing, transportation, or
    financial assistance while young adults develop greater responsibility for their
    own decisions. Young adults may value that support while also wanting privacy,
    autonomy, and recognition as adults. Differences in expectations about money,
    communication, household responsibilities, careers, and romantic relationships can
    create tension. The
    [American Psychological Association](https://www.apa.org/news/podcasts/speaking-of-psychology/parent-adult-children-relationships)
    notes that later transitions into marriage, independent households, and financial
    self-sufficiency have influenced relationships between parents and their adult
    children. Frequent communication may strengthen some relationships, although its
    meaning can depend on the quality and purpose of the interaction. Emotional support
    can encourage confidence, while unwanted advice or excessive involvement may feel
    restrictive. Experiences can also differ according to age, gender, education,
    income, cultural background, and living arrangement. Examining these differences
    helps clarify how families balance connection with autonomy. A broader understanding
    of these relationships may help families, educators, counselors, and communities
    support healthier transitions into adulthood.
    """)

    st.header("Questions Guiding the Study")

    questions = [
        "How is communication frequency related to young adults’ ratings of their relationships with their parents?",
        "Do young adults who receive greater emotional support feel more prepared for independence?",
        "How do relationship ratings differ between young adults who live with parents and those who do not?",
        "How is financial independence associated with current living arrangements?",
        "Do patterns of parental communication differ between adults ages 18–29 and adults ages 30–34?",
        "How do experiences of parental support vary across gender groups?",
        "Does education level relate to financial independence or living arrangements?",
        "How do relationship quality and the ability to be one’s true self with a parent relate?",
        "How do living arrangements for young adults vary across U.S. states and regions?",
        "Which combination of communication, emotional support, financial circumstances, and demographics is most strongly associated with independence?"
    ]

    for number, question in enumerate(questions, start=1):
        st.markdown(f"{number}. {question}")


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
