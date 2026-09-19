import pandas as pd
import plotly.express as px
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
    "parent-young-adult-illustration.png",
    caption=(
        "Family support and connection can remain important "
        "as young adults develop greater independence."
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
    census_raw = pd.read_csv("census_acs_2024_raw.csv")
    pew_numeric = pd.read_csv("pew_young_adults_clean_numeric.csv")

    # Structural skips and unanswered survey questions are made explicit for EDA.
    pew_eda = pew_data.fillna("Not applicable / no response")

    st.markdown("""
    This section documents the collection, preparation, cleaning, and exploration
    of two datasets related to young adulthood and independence in the United
    States. The first dataset comes from the Pew Research Center's American Trends
    Panel Wave 137 and describes communication, emotional support, relationship
    quality, financial independence, education, employment, and living
    arrangements among young adults. The second dataset was collected through the
    U.S. Census Bureau American Community Survey API and provides state-level
    estimates of living arrangements for adults ages 18–34. Together, these
    sources connect personal experiences reported in a national survey with the
    broader geographic patterns reported by the Census Bureau.
    """)

    st.header("Data Sources")

    st.markdown("""
    **Pew Research Center**

    - [Parents, Young Adult Children and the Transition to Adulthood](https://www.pewresearch.org/social-trends/2024/01/25/parents-relationship-with-their-young-adult-children/)
    - [Pew coded numeric data](https://github.com/swejal22/parent-child-independence-project/blob/main/pew_young_adults_clean_numeric.csv)
    - [Pew readable EDA data](https://github.com/swejal22/parent-child-independence-project/blob/main/pew_young_adults_eda_labeled.csv)

    The Pew dataset was selected because it directly measures relationship quality,
    communication, emotional support, financial independence, and living
    arrangements. The original survey contained 4,512 respondents and 196
    variables. It was filtered to 1,495 respondents ages 18–34 who were in contact
    with at least one living parent, and 30 relevant variables were retained.

    **U.S. Census Bureau**

    - [American Community Survey API](https://www.census.gov/data/developers/data-sets/acs-1year.html)
    - [Raw Census API data](https://github.com/swejal22/parent-child-independence-project/blob/main/census_acs_2024_raw.csv)
    - [Clean Census data](https://github.com/swejal22/parent-child-independence-project/blob/main/census_acs_2024_living_arrangements_clean.csv)
    - [Complete Python notebook](https://github.com/swejal22/parent-child-independence-project/blob/main/01_Load_and_Filter_Pew_Data.ipynb)

    The 2024 ACS one-year API was used to collect state-level estimates for people
    ages 18–34 who lived alone, lived with a spouse, lived with an unmarried
    partner, or lived as a child of the householder. Puerto Rico was excluded so
    the final dataset represents the 50 states and Washington, D.C.
    """)

    st.subheader("Census API Endpoint")

    st.code(
        "https://api.census.gov/data/2024/acs/acs1?"
        "get=NAME,B09021_008E,B09021_009E,B09021_010E,"
        "B09021_011E,B09021_012E,B09021_013E,B09021_014E"
        "&for=state:*",
        language="text"
    )

    st.markdown("""
    Python's `requests` package sent a GET request to the endpoint. The JSON
    response was converted into a pandas DataFrame, checked for a successful HTTP
    status, validated, cleaned, and saved as CSV files. The API key was entered
    securely during execution and was not stored in the notebook or repository.
    """)

    st.header("Data Cleaning and Preparation")

    st.markdown("""
    The Pew data was filtered using the survey's young-adult sample indicator.
    Unrelated columns were removed, technical variable names were replaced with
    meaningful names, and survey codes were converted into readable labels.
    Refusal codes were changed to missing values before structural skips and
    unanswered questions were assigned the explicit label **Not applicable / no
    response** for exploratory analysis. This approach prevents missing responses
    from being mistaken for valid opinions while preserving the number of survey
    participants. Duplicate respondent identifiers were checked, categorical
    values were reviewed, and demographic codes were translated into readable
    age, gender, education, race and ethnicity, income, marital-status, and region
    labels.

    The Census API response initially used technical ACS variable names and
    returned 52 geographic records, including Puerto Rico. Columns were renamed,
    count variables were converted to numeric form, Puerto Rico was excluded, and
    duplicate states and missing values were checked. Four living-arrangement
    percentages were calculated using each state's total population ages 18–34.
    A validation calculation confirmed that the component categories equaled the
    reported total for every retained state. The resulting Census dataset contains
    51 rows and 15 variables.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pew respondents", f"{len(pew_eda):,}")
    col2.metric("Pew variables", pew_eda.shape[1])
    col3.metric("States and D.C.", len(census_data))
    col4.metric(
        "Census missing values",
        int(census_data.isna().sum().sum())
    )

    st.header("Raw and Cleaned Data Comparison")

    st.subheader("Raw Census API response")
    st.dataframe(census_raw.head(), use_container_width=True)

    st.markdown("""
    The raw Census response uses technical variable names such as `B09021_008E`
    and stores the geographic identifier separately from the state name. These
    names are accurate for the API but are difficult for a general audience to
    interpret.
    """)

    st.subheader("Cleaned Census dataset")
    st.dataframe(census_data.head(), use_container_width=True)

    st.markdown("""
    The cleaned version uses descriptive names and includes calculated
    percentages for easier state comparisons. Puerto Rico was removed, numeric
    types were verified, and the category totals were validated against the
    reported population totals.
    """)

    st.subheader("Coded Pew extract")
    st.dataframe(pew_numeric.head(), use_container_width=True)

    st.markdown("""
    The coded Pew extract retains numerical survey categories that are useful for
    later machine-learning models. Without a codebook, however, values such as
    1, 2, and 3 are not immediately meaningful to a website reader.
    """)

    st.subheader("Readable Pew EDA dataset")
    st.dataframe(pew_eda.head(), use_container_width=True)

    st.markdown("""
    The EDA version replaces survey codes with meaningful category labels while
    preserving the same respondents and variables. It is used below because the
    charts can be interpreted without repeatedly consulting the survey codebook.
    """)

    st.header("Exploratory Visualizations")

    def show_count_chart(number, column, title, color, explanation):
        counts = (
            pew_eda[column]
            .astype(str)
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Respondents")
        )

        fig = px.bar(
            counts,
            x="Respondents",
            y="Category",
            orientation="h",
            title=f"Figure {number}. {title}",
            color_discrete_sequence=[color],
            text="Respondents"
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(
            template="plotly_white",
            yaxis={"categoryorder": "total ascending"},
            height=430,
            margin=dict(l=20, r=40, t=70, b=40),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(explanation)

    show_count_chart(
        1,
        "relationship_rating",
        "Young Adults' Ratings of Their Relationships with Parents",
        "#2A9D8F",
        """
        Most respondents describe their parental relationship positively, although
        the ratings are not identical across the sample. The smaller negative
        categories remain important because they identify young adults whose family
        experiences differ from the dominant pattern.
        """
    )

    show_count_chart(
        2,
        "text_frequency",
        "Frequency of Text Communication with Parents",
        "#457B9D",
        """
        Text communication is frequent for many young adults who do not live with
        their parents. The not-applicable group largely reflects survey routing for
        respondents living with parents rather than an ordinary unanswered value.
        """
    )

    show_count_chart(
        3,
        "financial_independence",
        "Level of Financial Independence",
        "#E76F51",
        """
        Financial independence exists on a continuum rather than as a simple
        independent-or-dependent division. The distribution allows later analyses
        to examine whether financial circumstances are related to relationship
        quality and living arrangements.
        """
    )

    show_count_chart(
        4,
        "lives_with_parents",
        "Current Living Arrangement with Parents",
        "#F4A261",
        """
        The survey includes both young adults who live with parents and those who
        maintain a separate household. This distinction provides an important basis
        for comparing communication, support, and independence.
        """
    )

    show_count_chart(
        5,
        "prepared_for_independence",
        "Perceived Preparation for Independence",
        "#6A4C93",
        """
        Respondents report different levels of preparation for becoming independent
        adults. These differences may reflect family support, education, employment,
        financial resources, and earlier opportunities to make decisions.
        """
    )

    show_count_chart(
        6,
        "advice_finances",
        "Frequency of Receiving Financial Advice from Parents",
        "#3A86FF",
        """
        Parental financial advice remains common after children reach adulthood.
        The chart also shows that the level of parental involvement varies
        considerably across young adults.
        """
    )

    show_count_chart(
        7,
        "true_self_with_parent",
        "Ability to Be One's True Self with a Parent",
        "#8338EC",
        """
        Many respondents report being able to act authentically around a parent,
        although this experience is not universal. Comfort with self-expression may
        be an important indicator of emotional closeness and relationship quality.
        """
    )

    show_count_chart(
        8,
        "employment_status",
        "Employment Status of Young Adult Respondents",
        "#00A896",
        """
        Full-time employment is the most common status in this sample, but
        part-time work and nonemployment are also represented. Employment can affect
        income, housing choices, and the timing of financial independence.
        """
    )

    st.subheader(
        "Figure 9. Relationship Ratings by Whether Young Adults Live with Parents"
    )

    relationship_living = pd.crosstab(
        pew_eda["lives_with_parents"].astype(str),
        pew_eda["relationship_rating"].astype(str),
        normalize="index"
    ).mul(100).reset_index()

    relationship_living = relationship_living.melt(
        id_vars="lives_with_parents",
        var_name="Relationship rating",
        value_name="Percent"
    )

    fig9 = px.bar(
        relationship_living,
        x="lives_with_parents",
        y="Percent",
        color="Relationship rating",
        barmode="stack",
        labels={"lives_with_parents": "Living arrangement"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig9.update_layout(
        template="plotly_white",
        height=500,
        yaxis_title="Percent of respondents",
        xaxis_title="Living arrangement",
        legend_title="Relationship rating"
    )

    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("""
    The stacked percentages compare relationship ratings without allowing the
    larger living-arrangement group to dominate the result. Differences between
    the bars indicate whether living with parents is associated with a different
    pattern of relationship evaluations.
    """)

    st.subheader(
        "Figure 10. States with the Highest Percentage of Young Adults Living Alone"
    )

    top_states = (
        census_data.nlargest(10, "pct_lives_alone")
        .sort_values("pct_lives_alone")
    )

    fig10 = px.bar(
        top_states,
        x="pct_lives_alone",
        y="state_name",
        orientation="h",
        labels={
            "pct_lives_alone": "Young adults living alone (%)",
            "state_name": "State"
        },
        color="pct_lives_alone",
        color_continuous_scale="Teal"
    )

    fig10.update_layout(
        template="plotly_white",
        height=500,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig10, use_container_width=True)

    st.markdown("""
    The percentage of adults ages 18–34 living alone varies noticeably across
    states, showing that independence is also shaped by geographic conditions.
    Differences may reflect housing costs, employment opportunities, migration,
    population composition, and regional norms.
    """)

    st.header("EDA Summary")

    st.markdown("""
    The initial exploration shows that most young adults describe positive
    relationships with their parents, but communication, emotional openness,
    financial independence, employment, and living arrangements vary across the
    sample. The Census results further demonstrate that young-adult living
    arrangements are not geographically uniform. These patterns provide a
    foundation for the clustering, principal component analysis, classification,
    regression, and neural-network sections that will be added in later modules.
    """)


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
