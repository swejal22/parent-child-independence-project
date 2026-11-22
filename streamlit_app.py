import pandas as pd
import plotly.express as px
import streamlit as st
from module2_pages import render_clustering_page, render_pca_page

st.set_page_config(
    page_title="Parent–Child Bonds and Independence",
    page_icon="🌿",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Light sage navigation toolbar */
    header[data-testid="stHeader"] {
        background-color: #A8CCB9 !important;
        border-bottom: 3px solid #78A48F !important;
    }

    /* Navigation labels and toolbar icons */
    header[data-testid="stHeader"] p,
    header[data-testid="stHeader"] span,
    header[data-testid="stHeader"] svg {
        color: #18352F !important;
        fill: #18352F !important;
    }

    /* Selected page */
    header[data-testid="stHeader"] a[aria-current="page"] {
        background-color: #F2F7F3 !important;
        border-radius: 9px !important;
    }

    header[data-testid="stHeader"] a[aria-current="page"] p {
        color: #18352F !important;
        font-weight: 700 !important;
    }

    /* Hover effect */
    header[data-testid="stHeader"] a:hover {
        background-color: #91BAA6 !important;
        border-radius: 9px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
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
    relationships connecting young adults with their parents.These changing pathways 
    demonstrate why independence should be understood as a gradual and highly individual 
    transition rather than a single universal milestone.
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

    st.image(
        "assets/module2/independence_pathways.png",
        caption=(
            "Housing, education, employment, finances, emotional support, and family "
            "connection all contribute to the transition toward independence."
        ),
        use_container_width=True,
    )

    st.markdown("""
    Independence also carries different meanings for different families and communities.
    Young adults are often expected to make their own decisions while still maintaining
    relationships with the people who helped raise them. These expectations can become
    difficult when economic reality does not match traditional ideas about the timing of
    adulthood. Rising housing costs, educational debt, unstable employment, and health
    challenges may delay moving out or becoming fully financially self-sufficient. Parents
    often respond by extending practical and emotional support beyond the teenage years.
    This support may give young adults room to develop confidence, complete an education,
    or recover from a setback. At the same time, too much involvement can create conflict
    when advice begins to feel like control. Healthy families therefore need ways to balance
    connection with boundaries, support with responsibility, and guidance with respect for
    adult choices. Communities, educators, and counselors can also help by recognizing that
    there is no single correct path into adulthood. A clearer understanding of these varied
    experiences can encourage more realistic and compassionate conversations about what
    independence means in the United States today.
    """)

    st.markdown(""" 
    Following the identification of trends in parent-young adult relationships using clustering analysis and PCA, 
    this research work proceeds towards supervised machine learning techniques. The following stage in this process 
    will involve the application of Naïve Bayes and Decision Trees analysis for investigating the features that could 
    be correlated with higher levels of independence among young adults. These tools will allow for an investigation of 
    the association between financial independence, cohabitation status, and parents’ participation and the different 
    levels of independence.
    """)
    st.markdown("""
    To further analyze the connection between parents' engagement
    and independence of their young adults in the USA, this project
    will employ such machine learning approaches as linear
    regression, logistic regression, and AdaBoost. Linear regression
    will assist in analyzing the association between various
    economic, demographic, and familial characteristics and
    quantitative measures of independence, whereas logistic
    regression will predict the probability of young adults' financial
    independence. Also, AdaBoost algorithm involves training of many
    weak learners on those observations misclassified by previous
    algorithms. Using this approach alongside already implemented
    Naïve Bayes and Decision Tree models, we can find interesting
    patterns, assess their predictive ability, and better understand.
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
    - [Raw Pew young-adult subset](https://github.com/swejal22/parent-child-independence-project/blob/main/pew_young_adults_raw_subset.csv)
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
    - [Census API collection and cleaning notebook](https://github.com/swejal22/parent-child-independence-project/blob/main/02_Census_API_Fetch.ipynb)
    
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
    st.subheader("Code and Reproducibility")

    st.markdown("""
    The Python workflows used for data collection and preparation are available
    in the following notebooks:

    - [Pew data filtering notebook](https://github.com/swejal22/parent-child-independence-project/blob/main/01_Load_and_Filter_Pew_Data.ipynb)
    - [Census API collection and cleaning notebook](https://github.com/swejal22/parent-child-independence-project/blob/main/02_Census_API_Fetch.ipynb)

    The analysis was completed in Python using pandas, NumPy, requests, and
    Plotly.
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

        # ---------------------------------------------------------
    # Figure 1: Horizontal bar chart
    # ---------------------------------------------------------
    relationship_counts = (
        pew_eda["relationship_rating"]
        .astype(str)
        .value_counts()
        .rename_axis("Relationship rating")
        .reset_index(name="Respondents")
        .sort_values("Respondents")
    )

    fig1 = px.bar(
        relationship_counts,
        x="Respondents",
        y="Relationship rating",
        orientation="h",
        title="Figure 1. Young Adults' Ratings of Their Relationships with Parents",
        color_discrete_sequence=["#2A9D8F"],
        text="Respondents"
    )

    fig1.update_traces(textposition="outside")
    fig1.update_layout(
        template="plotly_white",
        height=450,
        showlegend=False
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    Most respondents describe their parental relationships positively, although
    the ratings are not identical across the sample. The smaller negative
    categories identify young adults whose family experiences differ from the
    dominant pattern.
    """)

    # ---------------------------------------------------------
        # ---------------------------------------------------------
    # Shared colors and helper
    # ---------------------------------------------------------

    sage_palette = [
        "#2F7668",
        "#5F9988",
        "#86B5A3",
        "#A8CCB9",
        "#D3E5DA",
        "#E9A178"
    ]

    def applicable_counts(column, label_map=None, order=None):
        values = pew_eda[column].astype(str).str.strip()

        values = values[
            ~values.str.lower().isin([
                "not applicable / no response",
                "not applicable/no response",
                "nan",
                "none"
            ])
        ]

        if label_map:
            values = values.replace(label_map)

        counts = (
            values.value_counts()
            .rename_axis("Category")
            .reset_index(name="Respondents")
        )

        counts["Percent"] = (
            counts["Respondents"] / counts["Respondents"].sum() * 100
        )

        if order:
            counts["Category"] = pd.Categorical(
                counts["Category"],
                categories=order,
                ordered=True
            )
            counts = counts.sort_values("Category")

        return counts


    # ---------------------------------------------------------
    # FIGURE 2 — Horizontal bars with shortened labels
    # ---------------------------------------------------------

    st.subheader("Figure 2. Frequency of Text Communication with Parents")

    text_labels = {
        "At least once a day": "Daily",
        "A few times a week": "A few times weekly",
        "A few times a month": "A few times monthly",
        "Once a month": "Once monthly",
        "Less than once a month": "Less than monthly"
    }

    text_order = [
        "Daily",
        "A few times weekly",
        "A few times monthly",
        "Once monthly",
        "Less than monthly",
        "Never"
    ]

    text_counts = applicable_counts(
        "text_frequency",
        label_map=text_labels,
        order=text_order
    )

    fig2 = px.bar(
        text_counts,
        x="Percent",
        y="Category",
        orientation="h",
        text="Percent",
        color="Percent",
        color_continuous_scale=[
            [0, "#CFE2D7"],
            [1, "#2F7668"]
        ],
        labels={
            "Percent": "Applicable respondents (%)",
            "Category": ""
        }
    )

    fig2.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig2.update_layout(
        template="plotly_white",
        height=440,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "array", "categoryarray": text_order[::-1]},
        margin=dict(l=25, r=70, t=20, b=50),
        xaxis_range=[0, text_counts["Percent"].max() * 1.18]
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    Daily and weekly texting account for a substantial share of communication
    among respondents who live separately from their parents. Survey-routing
    responses were excluded so that the chart describes actual communication
    frequency rather than living-arrangement eligibility.
    """)


    # ---------------------------------------------------------
    # FIGURE 3 — Ordered horizontal financial-independence scale
    # ---------------------------------------------------------

    st.subheader("Figure 3. Level of Financial Independence")

    financial_order = [
        "Completely",
        "Mostly",
        "Somewhat",
        "A little",
        "Not at all"
    ]

    financial_counts = applicable_counts(
        "financial_independence",
        order=financial_order
    )

    fig3 = px.bar(
        financial_counts,
        x="Percent",
        y="Category",
        orientation="h",
        text="Percent",
        color="Category",
        color_discrete_sequence=[
            "#2F7668",
            "#5F9988",
            "#86B5A3",
            "#D8B46A",
            "#E58B73"
        ],
        labels={
            "Percent": "Respondents (%)",
            "Category": ""
        }
    )

    fig3.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig3.update_layout(
        template="plotly_white",
        height=410,
        showlegend=False,
        yaxis={
            "categoryorder": "array",
            "categoryarray": financial_order[::-1]
        },
        margin=dict(l=25, r=70, t=20, b=50),
        xaxis_range=[0, financial_counts["Percent"].max() * 1.18]
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    Financial independence forms a gradual scale instead of a simple division
    between dependence and independence. The distribution provides a basis for
    examining how financial circumstances correspond with housing and parental
    support.
    """)


    # ---------------------------------------------------------
    # FIGURE 4 — Living arrangements as percentages
    # ---------------------------------------------------------

    st.subheader("Figure 4. Current Living Arrangement with Parents")

    living_labels = {
        "Does not live with a parent": "Lives separately",
        "Lives with all parents": "Lives with all parents",
        "Lives with at least one parent": "Lives with one parent"
    }

    living_order = [
        "Lives separately",
        "Lives with all parents",
        "Lives with one parent"
    ]

    living_counts = applicable_counts(
        "lives_with_parents",
        label_map=living_labels,
        order=living_order
    )

    fig4 = px.bar(
        living_counts,
        x="Category",
        y="Percent",
        text="Percent",
        color="Category",
        color_discrete_sequence=[
            "#2F7668",
            "#7EAD98",
            "#E9A178"
        ],
        labels={
            "Percent": "Respondents (%)",
            "Category": "Living arrangement"
        }
    )

    fig4.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig4.update_layout(
        template="plotly_white",
        height=440,
        showlegend=False,
        xaxis_tickangle=0,
        margin=dict(l=25, r=35, t=20, b=60),
        yaxis_range=[0, living_counts["Percent"].max() * 1.15]
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    Most respondents maintain a household separate from their parents, while a
    meaningful minority live with one or both parents. These categories provide
    a clearer basis for comparing communication, support, and relationship
    experiences.
    """)


    # ---------------------------------------------------------
    # FIGURE 5 — 100% stacked relationship profile
    # ---------------------------------------------------------

    st.subheader(
        "Figure 5. Relationship Ratings Across Levels of Preparation "
        "for Independence"
    )

    relationship_order = [
        "Poor",
        "Fair",
        "Good",
        "Very good",
        "Excellent"
    ]

    preparation_order = [
        "Not at all",
        "Not much",
        "Some",
        "A fair amount",
        "A great deal"
    ]

    figure5_data = pew_eda[
        ["relationship_rating", "prepared_for_independence"]
    ].astype(str)

    figure5_data = figure5_data[
        ~figure5_data["relationship_rating"].str.lower().isin(
            ["not applicable / no response", "nan", "none"]
        )
        &
        ~figure5_data["prepared_for_independence"].str.lower().isin(
            ["not applicable / no response", "nan", "none"]
        )
    ]

    figure5_table = pd.crosstab(
        figure5_data["prepared_for_independence"],
        figure5_data["relationship_rating"],
        normalize="index"
    ).mul(100)

    figure5_table = figure5_table.reindex(
        index=preparation_order,
        columns=relationship_order,
        fill_value=0
    ).reset_index()

    figure5_long = figure5_table.melt(
        id_vars="prepared_for_independence",
        var_name="Relationship rating",
        value_name="Percent"
    )

    fig5 = px.bar(
        figure5_long,
        x="Percent",
        y="prepared_for_independence",
        color="Relationship rating",
        orientation="h",
        category_orders={
            "prepared_for_independence": preparation_order,
            "Relationship rating": relationship_order
        },
        color_discrete_map={
            "Poor": "#D97967",
            "Fair": "#E9A178",
            "Good": "#A8CCB9",
            "Very good": "#5F9988",
            "Excellent": "#2F7668"
        },
        labels={
            "Percent": "Relationship-rating composition (%)",
            "prepared_for_independence": "Preparation for independence"
        }
    )

    fig5.update_layout(
        template="plotly_white",
        height=480,
        barmode="stack",
        legend_title="Relationship rating",
        margin=dict(l=25, r=25, t=20, b=50)
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    Each bar shows the relationship-rating composition within one level of
    preparation for independence. The growing share of positive ratings helps
    illustrate whether stronger perceived preparation accompanies stronger
    parent–child relationships.
    """)


    # ---------------------------------------------------------
    # FIGURE 6 — Positive relationship ratings by arrangement
    # ---------------------------------------------------------

    st.subheader(
        "Figure 6. Positive Parent–Child Relationship Ratings "
        "by Living Arrangement"
    )

    figure6_data = pew_eda[
        ["lives_with_parents", "relationship_rating"]
    ].astype(str)

    figure6_data = figure6_data[
        ~figure6_data["lives_with_parents"].str.lower().isin(
            ["not applicable / no response", "nan", "none"]
        )
        &
        ~figure6_data["relationship_rating"].str.lower().isin(
            ["not applicable / no response", "nan", "none"]
        )
    ]

    figure6_data["Living arrangement"] = (
        figure6_data["lives_with_parents"].replace(living_labels)
    )

    figure6_data["Positive relationship"] = (
        figure6_data["relationship_rating"]
        .isin(["Excellent", "Very good"])
    )

    figure6_summary = (
        figure6_data
        .groupby("Living arrangement")["Positive relationship"]
        .mean()
        .mul(100)
        .reset_index(name="Positive ratings")
    )

    figure6_summary["Living arrangement"] = pd.Categorical(
        figure6_summary["Living arrangement"],
        categories=living_order,
        ordered=True
    )

    figure6_summary = figure6_summary.sort_values("Living arrangement")

    fig6 = px.bar(
        figure6_summary,
        x="Living arrangement",
        y="Positive ratings",
        text="Positive ratings",
        color="Living arrangement",
        color_discrete_sequence=[
            "#2F7668",
            "#7EAD98",
            "#E9A178"
        ],
        labels={
            "Positive ratings":
                "Respondents rating relationship Excellent or Very good (%)",
            "Living arrangement": ""
        }
    )

    fig6.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig6.update_layout(
        template="plotly_white",
        height=440,
        showlegend=False,
        yaxis_range=[0, 100],
        margin=dict(l=25, r=30, t=20, b=60)
    )

    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("""
    This comparison focuses on the percentage of respondents who describe their
    parental relationship as either excellent or very good. Presenting one clear
    measure makes differences among living arrangements easier to interpret than
    the earlier multi-category stacked chart.
    """)


    # ---------------------------------------------------------
    # FIGURE 7 — Employment-status donut
    # ---------------------------------------------------------

    st.subheader("Figure 7. Employment Status of Young Adult Respondents")

    employment_counts = applicable_counts("employment_status")

    fig7 = px.pie(
        employment_counts,
        names="Category",
        values="Respondents",
        hole=0.50,
        color="Category",
        color_discrete_sequence=[
            "#2F7668",
            "#86B5A3",
            "#E9A178"
        ]
    )

    fig7.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Respondents: %{value}<br>"
            "Share: %{percent}<extra></extra>"
        )
    )

    fig7.update_layout(
        template="plotly_white",
        height=450,
        legend_title="Employment status",
        margin=dict(l=20, r=20, t=20, b=30),
        annotations=[
            dict(
                text="Employment<br>status",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#18352F")
            )
        ]
    )

    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    Full-time employment is the most common status in the sample, while
    part-time employment and nonemployment remain meaningfully represented.
    Employment may influence income, housing options, and the timing of
    financial independence.
    """)


        # ---------------------------------------------------------
    # FIGURE 8 — U.S. state map
    # ---------------------------------------------------------

    st.subheader(
        "Figure 8. Percentage of Young Adults Living Alone Across U.S. States"
    )

    state_abbreviations = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY"
    }

    figure8_data = census_data[
        ["state_name", "pct_lives_alone"]
    ].dropna().copy()

    figure8_data["State code"] = (
        figure8_data["state_name"].map(state_abbreviations)
    )

    figure8_data = figure8_data.dropna(subset=["State code"])

    fig8 = px.choropleth(
        figure8_data,
        locations="State code",
        locationmode="USA-states",
        color="pct_lives_alone",
        scope="usa",
        hover_name="state_name",
        hover_data={
            "State code": False,
            "pct_lives_alone": ":.1f"
        },
        labels={
            "pct_lives_alone": "Living alone (%)"
        },
        color_continuous_scale=[
            [0.00, "#DDECE4"],
            [0.35, "#A8CCB9"],
            [0.70, "#5F9988"],
            [1.00, "#245F55"]
        ]
    )

    fig8.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title="Living<br>alone (%)",
            thickness=18,
            len=0.70
        )
    )

    fig8.update_geos(
        bgcolor="rgba(0,0,0,0)",
        lakecolor="#E8F2EC",
        showlakes=True
    )

    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("""
    Darker states have a higher percentage of young adults ages 18–34 living
    alone, while lighter states have a lower percentage. The geographic pattern
    demonstrates that living arrangements differ across the country and may be
    influenced by housing costs, employment opportunities, migration, and
    regional norms.
    """)

    # ---------------------------------------------------------
    # Figure 9: Scatterplot
    # ---------------------------------------------------------
    fig9 = px.scatter(
        census_data,
        x="pct_lives_alone",
        y="pct_child_of_householder",
        hover_name="state_name",
        title=(
            "Figure 9. Living Alone Compared with Living as "
            "a Child of the Householder"
        ),
        labels={
            "pct_lives_alone": "Young adults living alone (%)",
            "pct_child_of_householder":
                "Young adults recorded as child of householder (%)"
        },
        color="pct_lives_with_spouse",
        size="pct_lives_with_unmarried_partner",
        color_continuous_scale="Teal"
    )

    fig9.update_traces(
        marker=dict(
            line=dict(width=1, color="#FFFFFF"),
            opacity=0.85
        )
    )

    fig9.update_layout(
        template="plotly_white",
        height=520,
        coloraxis_colorbar_title="Living with<br>spouse (%)"
    )

    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("""
    Each point represents one state and compares two different young-adult
    living arrangements. The figure helps reveal whether states with higher
    rates of living alone tend to have lower rates of young adults living in
    a parent's household.
    """)

    # ---------------------------------------------------------
    # Figure 10: Ranked horizontal bar chart
    # ---------------------------------------------------------
    top_states = (
        census_data.nlargest(10, "pct_lives_alone")
        .sort_values("pct_lives_alone")
    )

    fig10 = px.bar(
        top_states,
        x="pct_lives_alone",
        y="state_name",
        orientation="h",
        title=(
            "Figure 10. States with the Highest Percentage of "
            "Young Adults Living Alone"
        ),
        labels={
            "pct_lives_alone": "Young adults living alone (%)",
            "state_name": "State"
        },
        color="pct_lives_alone",
        color_continuous_scale="Teal",
        text="pct_lives_alone"
    )

    fig10.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig10.update_layout(
        template="plotly_white",
        height=520,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig10, use_container_width=True)

    st.markdown("""
    The percentage of young adults living alone varies noticeably across
    states, demonstrating that independence also has a geographic dimension.
    Housing costs, employment opportunities, migration, and regional norms may
    contribute to these differences.
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
    render_clustering_page()


def pca():
    render_pca_page()


def naive_bayes():
    st.title("Naïve Bayes Classification")

    st.markdown(
        "### Understanding Young Adult Independence "
        "Through Supervised Learning"
    )

    st.header("1. Overview")

    st.subheader("What is Naïve Bayes?")
    st.markdown("""
    The Naïve Bayes classifier is an example of supervised machine learning technique which is applied to solve 
    classification problems. It is based on the Bayes’ rule which is designed to compute probabilities of outcomes
    on the basis of available information. The name of the algorithm means that it assumes independence of predictors
    in case of known class. However, it may not necessarily hold true in real-life situations. Naïve Bayes uses
    probabilities estimated from training sets to categorize new instances. This algorithm is applied in many fields 
    including but not limited to email spam filtering and sentiment analysis.
    """)

    st.subheader("Multinomial Naïve Bayes")
    st.markdown("""
    The Multinomial Naïve Bayes algorithm is a form of Naïve Bayes that is built for nonnegative feature data, 
    for instance, the number of times a word appears in a document. The algorithm calculates the probability of 
    each feature in relation to each class and then uses the probabilities in predicting the class an object belongs to. 
    The algorithm has been found to be useful in classification of documents, e-mails among others. 
    Even though it is best suited for count data, it may also be used on some nonnegative numerical features.
    """)

    st.subheader("Bernoulli Naïve Bayes")
    st.markdown("""
    The next variant of the Naïve Bayes algorithm is Bernoulli Naïve Bayes, which utilizes binary features that are typically 
    denoted by 0 and 1. It does not consider the frequency of the occurrence of something but the presence or absence of a 
    specific feature. For instance, if we conduct a survey among youth regarding the receipt of help from parents, one variable 
    may be whether the person receives such help or not, either yes or no (1 or 0). Bernoulli Naïve Bayes can then make use of 
    this information for classification purposes.
    """)

    st.subheader("Application to Our Research")
    st.markdown("""
   Here, one might choose to employ Naïve Bayes in the investigation of patterns related to independence in young adults in America. 
   Features like financial help, housing situation, interaction with parents, and parent involvement would prove to be helpful in
   classification. If there is an appropriate independence-related feature in the data set, it is possible to train the model to 
   classify it into categories depending on certain chosen input features. In this way, it would be possible to learn about the 
   combinations of features that lead to particular results. Finally, the performance of the model could be analyzed through the 
   use of a separate test data set, a confusion matrix, and accuracy.
    """)


    # -----------------------------------------------------
    # 2. DATA PREPARATION
    # -----------------------------------------------------
    st.header("2. Data Preparation")
    st.markdown("""
    For the Naïve Bayes classifier, the dataset was preprocessed through the use of survey data about parent-young adult relationship 
    and financial independence in the US. The original dataset had 1,495 records and 30 attributes. Following the selection of six input 
    attributes and the output attribute which is financial independence, 17 observations having missing values were dropped and there 
    remained 1,478 observations. These attributes include financial support from parents, housing, parent involvement, satisfaction 
    in communication, employment status, and age category. Because these attributes had categorical data, one-hot encoding was used to 
    make 18 binary attributes for the Multinomial Naïve Bayes Classifier.

    The data after cleaning was split into 80% training set (1,182 data points) and 20% testing set (296 data points) based on stratified
    sampling to preserve the same proportion of the five financial independence classes. The training set is used for developing the 
    classifier, whereas the test set is used to determine the performance of the classifier. The two datasets are kept separate, and there
    are no overlaps at all in order to avoid data leakage. Despite being a classifier which works best on count data, the one-hot encoded 
    survey data provide nonnegative input to this classification problem.
     """)

    st.subheader("Financial Independence Categories")

    category_labels = pd.DataFrame({
        "Code": [1, 2, 3, 4, 5],
        "Financial Independence Level": [
            "Completely independent",
            "Mostly independent",
            "Somewhat independent",
            "A little independent",
            "Not at all independent"
        ]
    })

    st.dataframe(category_labels, hide_index=True)

    st.subheader("Labeled Dataset")

    labeled_data = pd.read_csv("nb_labeled_data.csv")

    st.dataframe(labeled_data.head(), use_container_width=True)
    st.image(
        "nb_labeled_sample.png",
        caption="Sample of the labeled Naïve Bayes dataset",
        use_container_width=True
    )

    st.markdown(
        "[View the labeled dataset on GitHub]"
        "(https://github.com/swejal22/parent-child-independence-project/blob/main/nb_labeled_data.csv)"
    )

    st.subheader("Training and Testing Sets")

    train_data = pd.read_csv("nb_training_data.csv")
    test_data = pd.read_csv("nb_testing_data.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training Data Sample**")
        st.dataframe(train_data.head(), use_container_width=True)
        st.image(
            "nb_training_sample.png",
            caption="Naïve Bayes Training Dataset Sample",
            use_container_width=True
        )

    with col2:
        st.markdown("**Testing Data Sample**")
        st.dataframe(test_data.head(), use_container_width=True)
        st.image(
            "nb_testing_sample.png",
            caption="Naïve Bayes Testing Dataset Sample",
            use_container_width=True
        )

    st.image(
        "nb_train_test_split.png",
        caption="Training and testing dataset sizes",
        use_container_width=True
    )

    st.markdown(
        "[Training dataset]"
        "(https://github.com/swejal22/parent-child-independence-project/blob/main/nb_training_data.csv)"
        " | "
        "[Testing dataset]"
        "(https://github.com/swejal22/parent-child-independence-project/blob/main/nb_testing_data.csv)"
    )

    # -----------------------------------------------------
    # 3. CODE
    # -----------------------------------------------------
    st.header("3. Python Implementation")

    st.markdown(
        "[View Naïve Bayes Analysis Notebook on GitHub]"
        "(https://github.com/swejal22/parent-child-independence-project/blob/main/05_Naive_Bayes_Analysis.ipynb)"
    )

    # -----------------------------------------------------
    # 4. RESULTS
    # -----------------------------------------------------
    st.header("4. Results")
    st.markdown("""
    The Multinomial Naïve Bayes classifier yielded an accuracy rate of 59.12% with respect to the test data set, which correctly 
    identified 175 out of 296 observations. The confusion matrix further reveals that the classifier was most effective in identifying 
    the observations within Category 1, which successfully classified 128 out of 157 observations. This can be contrasted against the 
    fact that the classifier did not perform well when it came to Categories 3, 4, and 5, whereby fewer observations were correctly 
    identified by the classifier. One probable reason could be the skewed nature of the categories, whereby Category 1 had significantly
    more observations than the rest.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Accuracy", "59.12%")
    col2.metric("Training Records", "1,182")
    col3.metric("Testing Records", "296")

    st.subheader("Confusion Matrix")

    st.image(
        "nb_confusion_matrix.png",
        caption="Multinomial Naïve Bayes classification results",
        use_container_width=True
    )

    # -----------------------------------------------------
    # 5. CONCLUSIONS
    # -----------------------------------------------------
    st.header("5. Conclusions")
    st.markdown("""
    From the results of the Multinomial Naïve Bayes analysis, we get the first insights into the possibility of predicting 
    financial independence categories based on various factors. With the help of the data on parental financial aid, living conditions,
    parental participation, communication satisfaction, employment, and age group, we managed to obtain the following result in 
    terms of the overall accuracy – 59.12%.

    From the confusion matrix, it can be observed that the classifier performs relatively well when predicting the category with the 
    highest probability in the sample. However, the results are poorer when predicting other categories due to class imbalance. 
    It is thus possible to conclude that the selected variables do not fully describe the factors responsible for the financial 
    independence of the population. Moreover, Naïve Bayes assumes that the features are independent; however, the factors being 
    considered are not. For example, financial support and living situation can affect each other.
    """)




def decision_trees():
    st.title("Decision Tree Classification")

    st.markdown(
        "#### Understanding Financial Independence "
        "Through Decision Tree Models"
    )

    # 1. OVERVIEW
    st.header("1. Overview")

    st.subheader("What is a Decision Tree?")
    st.markdown("""
    Decision Tree is a supervised learning method that is utilized in classification and regression problems. 
    This method divides the data set in smaller subsets on the basis of specific attributes and conditions. 
    A Decision Tree starts with root nodes that have the whole data set. Branches are created through making decisions 
    at each node. Decision Tree helps in understanding how various attributes affect classification decisions because of 
    its structure. For this research project, Decision Trees will be used to categorize young people into various financial 
    independence categories.
    """)

    st.subheader("Decision Tree Structure")

    st.image(
        "dt_tree_1.png",
        caption="Decision Tree using Gini impurity (depth 3)",
        use_container_width=True
    )

    st.image(
        "dt_tree_3.png",
        caption="Decision Tree using entropy (depth 5)",
        use_container_width=True
    )

    st.subheader("Gini Impurity")
    st.markdown("""
    Gini impurity is a measure of how mixed the classes are in the node of a Decision Tree. 
    Gini impurity of a node is zero if all observations of the node belong to the same class. 
    The more classes are mixed, the higher is Gini impurity. When fitting the tree, the algorithm checks 
    potential splits and usually picks the split which causes maximum decrease in weighted Gini impurity.
    """)

    st.subheader("Entropy")
    st.markdown("""
    Entropy is also an indicator which is used in order to calculate uncertainty in Decision Tree nodes. 
    In cases when all observations pertain to one class only, entropy will be equal to zero. As the number 
    of classes is distributed evenly among observations, entropy becomes greater. With the help of entropy, 
    Decision Trees can select better splits to create more homogeneous groups.
    """)

    st.subheader("Information Gain")
    st.markdown("""
    The Information Gain is the amount of reduction of the entropy after splitting the parent node into child nodes. 
    The formula to calculate the information gain is the entropy of the parent node minus the weighted average entropy 
    of the child nodes. Higher the information gain, lower is the uncertainty due to a split. When entropy is used as 
    a splitting function, the Decision Trees will always select the highest information gain split.
    """)

    st.subheader("Example of a Decision Tree Split")

    col1, col2 = st.columns(2)
    col1.metric("Parent Gini", "0.5000")
    col2.metric("Child Weighted Gini", "0.4167")

    st.metric("Gini Reduction", "0.0833")
    st.metric("Information Gain", "0.1245 bits")

    st.markdown("""
    Let us imagine a sample consisting of ten observations where five belong to classes A and B respectively. The Gini 
    impurity and entropy for the parent node will be 0.5000 and 1.0000 bits correspondingly. Imagine that we split our 
    sample into two parts: in the first part, there will be three observations of Class A and one observation of Class B; 
    in the second part, there will be two observations of Class A and four observations of Class B. As a result of splitting, 
    the weighted Gini impurity equals 0.4167, which is the decrease by 0.0833. Entropy also will be reduced up to 0.8755 bits, 
    resulting in information gain of 0.1245 bits.
    """)

    st.subheader("Why Can Different Trees Be Created?")
    st.markdown("""
    There can be multiple Decision Trees built using one particular dataset as there is more than one way to split observations 
    according to their features. The criteria that we use for splitting, such as the maximal depth of the tree, the available features, 
    or the minimal number of observations necessary to perform a split, will give us various Decision Trees. Using different thresholds 
    for numerical features will also give different Decision Tree results. If there are no constraints set, trees will become more and 
    more complex and will start to memorize the training data patterns, which will lead to overfitting. Constraining tree depth and 
    complexity is important for achieving this balance.
    For the numerical features that can take continuous values, there can be an infinite number of potential split points to consider. 
    But on the finite dataset, there will always be a finite set of such split points.
    """)
    
    # -----------------------------------------------------
    # 2. DATA PREPARATION
    # -----------------------------------------------------
    st.header("2. Data Preparation")

    st.markdown("""
    In the case of Decision Tree classification, the same data set used for Naïve Bayes algorithm was used to preserve consistency between 
    both machine learning algorithms. Initially, there were 1,495 observations and 30 variables in the data set. However, after choosing 6 
    features as inputs and financial independence as the output variable, 17 data points with missing values were deleted. Hence, 
    there remained 1,478 observations. The chosen six features were parent’s support, accommodation, parental influence, communication 
    satisfaction, employment status, and age group. In order to represent these categorical variables numerically, one-hot encoding 
    technique was used to create 18 binary variables.

    The dataset after cleaning was split into 80% training set which had 1,182 observations and 20% testing set which had 296 observations. 
    Stratified sampling technique was employed in order to have about an equal distribution of the five categories of financial independence
    in both sets. The training set was used in order to build three decision tree models using varying splitting criteria and maximum depth, 
    while the test set was left aside to evaluate their performances. There was no overlap between the two sets at all in order to avoid data
    leakage.
    """)
    st.subheader("Financial Independence Categories")

    category_labels = pd.DataFrame({
        "Code": [1, 2, 3, 4, 5],
        "Financial Independence Level": [
            "Completely independent",
            "Mostly independent",
            "Somewhat independent",
            "A little independent",
            "Not at all independent"
        ]
    })

    st.dataframe(category_labels, hide_index=True)

    

    st.subheader("Labeled Dataset")

    labeled_data = pd.read_csv("dt_labeled_data.csv")
    st.dataframe(labeled_data.head(), use_container_width=True)

    st.image(
        "dt_labeled_sample.png",
        caption="Sample of the labeled Decision Tree dataset",
        use_container_width=True
    )

    st.markdown(
        "[View labeled dataset on GitHub]"
        "(https://github.com/swejal22/parent-child-independence-project/blob/main/dt_labeled_data.csv)"
    )

    st.subheader("Training and Testing Datasets")

    train_data = pd.read_csv("dt_training_data.csv")
    test_data = pd.read_csv("dt_testing_data.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training Dataset**")
        st.dataframe(train_data.head(), use_container_width=True)
        st.image("dt_training_sample.png",
                 use_container_width=True)

    with col2:
        st.markdown("**Testing Dataset**")
        st.dataframe(test_data.head(), use_container_width=True)
        st.image("dt_testing_sample.png",
                 use_container_width=True)

    st.image(
        "dt_train_test_split.png",
        caption="Stratified 80/20 training and testing split",
        use_container_width=True
    )

    repo = "https://github.com/swejal22/parent-child-independence-project/blob/main/"

    st.markdown(
        f"[Training dataset]({repo}dt_training_data.csv)"
        " | "
        f"[Testing dataset]({repo}dt_testing_data.csv)"
    )

    # -----------------------------------------------------
    # 3. PYTHON CODE
    # -----------------------------------------------------
    st.header("3. Python Implementation")

    st.markdown(
        "[View Decision Tree Analysis Notebook on GitHub]"
        f"({repo}06_Decision_Tree_Analysis.ipynb)"
    )

    # -----------------------------------------------------
    # 4. RESULTS
    # -----------------------------------------------------
    st.header("4. Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Tree Accuracy", "59.12%")
    col2.metric("Training Records", "1,182")
    col3.metric("Testing Records", "296")

    st.subheader("Comparison of Three Decision Trees")

    comparison = pd.read_csv("dt_model_comparison.csv")
    st.dataframe(comparison, use_container_width=True)

    st.image(
        "dt_model_comparison.png",
        caption="Accuracy, balanced accuracy, and Macro F1",
        use_container_width=True
    )

    st.subheader("Tree 1: Gini — Maximum Depth 3")

    st.image("dt_tree_1.png", use_container_width=True)
    st.image("dt_confusion_matrix_1.png",
             use_container_width=True)

    st.subheader("Tree 2: Gini — Maximum Depth 5")

    st.image("dt_tree_2.png", use_container_width=True)
    st.image("dt_confusion_matrix_2.png",
             use_container_width=True)

    st.subheader("Tree 3: Entropy — Maximum Depth 5")

    st.image("dt_tree_3.png", use_container_width=True)
    st.image("dt_confusion_matrix_3.png",
             use_container_width=True)

    st.subheader("Feature Importance")

    st.image(
        "dt_feature_importance.png",
        caption="Important features identified by Tree 2",
        use_container_width=True
    )

    st.markdown("""
    Three Decision Tree models were created using various types 
    of splitting criterion and maximum depth of the tree to 
    analyze the financial independence classification in young adults. 
    In the first case, a decision tree was built using the Gini criterion 
    with the maximum depth of 3, which yielded 58.45% accuracy. In the second case, 
    a decision tree was built using the Gini criterion with the maximum depth of 5, 
    which yielded the best result with 59.12% accuracy and macro F1 score 0.4074. In the last case, 
    the decision tree was created using entropy criterion with the maximum depth of 5, which resulted in 58.45% accuracy.

    From the confusion matrices above, one can note that all the three algorithms were able to perform better when it came to 
    predicting Category 1, compared to the other less prevalent financial independence categories. This could be attributed 
    to the unbalanced nature of the observations of the five target variables. In the feature importance of the second tree, 
    it is clear that financial aid from parents and living situation were the two important predictors for the model. 
    Some predictions depended on the employment status, satisfaction with communication, and age group.
    """)

    # -----------------------------------------------------
    # 5. CONCLUSIONS
    # -----------------------------------------------------
    st.header("5. Conclusions")

    st.markdown("""
    The Decision Tree analysis brought about additional information regarding how certain variables may be linked to financial 
    independence of young people in the US. With the use of such variables as financial assistance, housing, parental involvement, 
    satisfaction with communication, employment status, and age groups, it became possible to reveal the relationships in order to 
    be able to classify the cases into financial independence categories. Out of all three models created, the Gini Decision Tree 
    with maximum depth of 5 was the most accurate one, as its accuracy rate was 59.12%.

    While the predictive power of the Decision Tree models was shown, their performance as a whole was quite moderate. 
    It was also evident from the analysis of the confusion matrix that the models performed better on classification of frequently 
    occurring classes as compared to infrequent ones, which implies that class imbalance might have negatively impacted the results. 
    However, in comparison with the Naïve Bayes classifier, the best-performing Decision Tree showed the same level of accuracy, 
    but a higher macro F1-score. This result implies that financial independence has complex dependencies that might not be captured 
    using the chosen variables.
    """)




def regression():
    st.title("Regression")

    st.header("1. Overview")

    st.markdown("""
    Regression analysis is a type of supervised learning method used
    to examine the associations between independent predictors and a dependent variable.
    For this study, linear regression and logistic regression models have been used
    to analyze factors related to young adult independence in the USA.
    """)

    st.subheader("Linear Regression")

    st.markdown("""
    Linear regression makes prediction for a numeric target variable
    using a linear approach between the independent variables
    and the dependent variable. Linear regression attempts to
    minimize the sum of squared differences between the observed
    and predicted values.

    In this research, linear regression will help explore the relationship between
    demographic, economic, and family-based variables
    and a quantitative indicator of independence.
    """)

    st.latex(r"\hat{y}=\beta_0+\beta_1x_1+\cdots+\beta_px_p")

    st.markdown("""
    In this case, the intercept is the prediction of the
    outcome when all predictors are zero, whereas each coefficient
    is the prediction of the change in the outcome by one unit of
    its respective predictor, while all others are held constant.
    """)
    
    st.image(
         "linear_regression_overview.png",
          caption="Figure 1: Linear regression with a fitted line and residuals."
    )
    st.subheader("Logistic Regression")

    st.markdown("""
    As opposed to linear regression, logistic regression is employed for
    classification purposes. Binary logistic regression determines the
    probability of belonging to one of two categories.

    In the model, first a linear score is calculated,
    after which the sigmoid function is used
    to convert that score to probability.
    """)

    st.latex(r"z=\beta_0+\beta_1x_1+\cdots+\beta_px_p")

    st.latex(r"P(Y=1|X)=\frac{1}{1+e^{-z}}")

    st.markdown("""
    By using a threshold for classification as 0.5, we assign all the 
    probabilities greater than or equal to 0.5 to class 1 and all those 
    less than 0.5 to class 0.

    Logistic regression is a useful tool that can be applied
    for predicting the financial independence of young people by
    using certain predictor variables.
    """)
    st.image(
         "logistic_regression_overview.png",
          caption="Figure 2: Sigmoid function and the binary classification threshold."
    )
    st.subheader("Linear Regression vs. Logistic Regression")

    st.table({
        "Feature": [
            "Purpose",
            "Target Variable",
            "Output",
            "Function",
            "Evaluation"
        ],
        "Linear Regression": [
            "Numerical prediction",
            "Continuous",
            "Numerical value",
            "Linear function",
            "RMSE, MAE, R-squared"
        ],
        "Logistic Regression": [
            "Classification",
            "Binary categorical",
            "Probability and class",
            "Sigmoid function",
            "Accuracy, confusion matrix"
        ]
    })
    st.header("2. Data Prep")

    st.markdown("""
    The data that were used in conducting the regression analysis have been obtained from
    the Pew Research Center survey on parent-child relationship in the United States.

    Prepared data have 1,478 observations,
    six predictor variables, and two target
    variables. There is no missing value in the dataset.
    """)

    st.subheader("Predictor Variables")

    st.markdown("""
    - Received financial help
    - Living arrangements with parents
    - Parental involvement level
    - Communication satisfaction
    - Employment status
    - Age group
    """)

    st.subheader("Target Variables")

    st.markdown("""
    Two target variables were created from the original
    financial independence survey responses.

    **Linear Regression:** The codes for the financial
    independence variable have been inverted, such that the
    new numerical score is between 1 and 5,
    with higher scores indicating greater financial
    independence.

    **Logistic Regression:** The five different categories of
    respondents have been collapsed into two categories.
    Those who are completely or mostly financially
    independent have class 1, while the other groups fall
    under class 0.
    """)

    st.table({
        "Survey Response": [
            "Completely",
            "Mostly",
            "Somewhat",
            "A little",
            "Not at all"
        ],
        "Linear Target": [5, 4, 3, 2, 1],
        "Logistic Target": [1, 1, 0, 0, 0]
    })

    repo_url = (
        "https://github.com/swejal22/"
        "parent-child-independence-project/blob/main/"
    )

    st.subheader("Labeled Dataset")

    st.image(
        "regression_labeled_sample.png",
        caption="Sample of the prepared labeled dataset."
    )

    st.markdown(
        f"[View labeled dataset (CSV)]"
        f"({repo_url}regression_labeled_data.csv)"
    )

    st.subheader("Training and Testing Sets")

    st.markdown("""
    The dataset was split into 80% training set and
    20% test set by stratified sampling, random_state=42.

    Stratification ensured that the distribution of binary
    financial independence categories was maintained in both
    datasets.

    There is no overlap between the training and testing
    sets, which means that no row is present in both. It
    avoids any data leakage and helps evaluate the models
    on unseen rows.

    All columns related to the target variable were
    dropped from the predictor variables.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Observations", "1,478")
    col2.metric("Training Set", "1,182")
    col3.metric("Testing Set", "296")

    st.image(
        "regression_training_sample.png",
        caption="Sample of the training dataset."
    )

    st.markdown(
        f"[View training dataset (CSV)]"
        f"({repo_url}regression_training_data.csv)"
    )

    st.image(
        "regression_testing_sample.png",
        caption="Sample of the testing dataset."
    )

    st.markdown(
        f"[View testing dataset (CSV)]"
        f"({repo_url}regression_testing_data.csv)"
    )

    st.subheader("Categorical Feature Encoding")

    st.markdown("""
    The initial predictors were coded using numerical
    codes, but these codes were categories from the
    surveys, not continuous variables.

    The predictors were transformed into twelve binary
    variables using one-hot encoding of the six initial
    predictors, where one reference category per each predictor
    was removed.

    The fit method of the encoder was used only for the
    training set and then applied for the testing set.
    """)

    st.image(
        "regression_before_encoding.png",
        caption="Original categorical predictors before encoding."
    )

    st.image(
        "regression_after_encoding.png",
        caption="Twelve binary predictors after one-hot encoding."
    )

    st.markdown(
        f"[View encoded training dataset (CSV)]"
        f"({repo_url}regression_encoded_training_data.csv)"
    )

    st.markdown(
        f"[View encoded testing dataset (CSV)]"
        f"({repo_url}regression_encoded_testing_data.csv)"
    )

    st.header("3. Code")

    st.markdown("""
    Both Linear Regression and Logistic Regression were built using 
    Python and scikit-learn.

    Pandas and NumPy were utilized in data preparation,
    while visualization was done using Matplotlib and Seaborn.

    The entire code includes the following aspects: data preparation, 
    feature encoding, training, evaluating the models, interpreting the coefficients,
    and comparing Logistic Regression with Multinomial Naive Bayes..
    """)

    st.markdown(
        f"[View complete Regression Python notebook]"
        f"({repo_url}07_Regression_Analysis.ipynb)"
    )

    st.header("4. Results")

    st.subheader("Linear Regression Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", "0.8788")
    col2.metric("MAE", "0.5815")
    col3.metric("R-squared", "0.5025")

    st.markdown("""
    The Linear Regression model was able to attain an RMSE of
    0.8788, an MAE of 0.5815, and an R-squared score
    of 0.5025 using the test data.

    On average, the differences between the predicted and
    actual financial independence scores are approximately 0.58
    points on a scale of 1 to 5.

    The R-squared score means that the variance explained
    by the Linear Regression model is 50.25%.
    """)

    st.image(
        "regression_actual_vs_predicted.png",
        caption="Actual versus predicted financial independence scores."
    )

    st.markdown("""
    The red dotted line is for perfect prediction.
    Predicted values far away from this line mean
    that the error is bigger.

    The regression model usually gives higher
    prediction values for higher actual values.
    But the regression model usually overestimates
    independence for the respondents with the lowest
    actual value.

    There are some predicted values that go beyond 5
    because Linear Regression does not limit the
    predicted values to the actual scale.
    """)

    st.subheader("Linear Regression Coefficient Analysis")

    st.image(
        "regression_linear_coefficients.png",
        caption="Linear Regression coefficients relative to reference categories."
    )

    st.markdown("""
    **Part-Time Work (-0.5181):**

    The youth working part-time are expected to have
    financial independence levels that are about 0.52
    points less than those of young adults who work full-time,
    all else being equal.

    **Not Working for Payment (-0.5838):**

    The youth not currently working for payment are expected
    to have financial independence levels that are about 0.58
    points less than those of young adults who work full-time,
    all else being equal.

    **Age Groups (+0.2223):**

    The youth aged between 30 and 34 are expected to have
    financial independence levels that are about 0.22 points
    more than those aged between 18 and 29, all else being equal.

    These coefficients are predictive and not causal.
    """)

    st.markdown(
        f"[View Linear Regression coefficients (CSV)]"
        f"({repo_url}regression_linear_coefficients.csv)"
    )

    st.subheader("Logistic Regression Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Accuracy", "86.82%")
    col2.metric("Correct Predictions", "257 / 296")
    col3.metric("Majority Baseline", "76.01%")

    st.image(
        "regression_logistic_confusion_matrix.png",
        caption="Confusion matrix for binary Logistic Regression."
    )

    st.markdown("""
    The Logistic Regression model identified
    257 out of 296 testing samples with an
    accuracy of 86.82%.

    In the confusion matrix, there are 45 correct
    classifications of lower independence and 212
    correct classifications of high independence.

    There are 26 lower-independence cases that have been
    falsely identified as high independence cases and
    13 high-independence cases that have been falsely
    identified as lower independence cases.

    The recall rate for the high independence class is
    about 94%, whereas that for the lower independence
    class is 63%.

    Thus, the model is better at classifying the majority
    class, and the accuracy is not enough to characterize its performance.
    """)

    st.subheader("Logistic Regression Odds Ratios")

    st.latex(r"\text{Odds Ratio}=e^{\beta}")

    st.image(
        "regression_logistic_odds_ratios.png",
        caption="Estimated odds ratios for financial independence."
    )

    st.markdown("""
    **Part-time Employment:**

    The coefficient is -1.3301, with an odds ratio
    of 0.2645. Young adults employed in part-time jobs
    have about 73.55% reduced predicted odds of being financially
    independent compared to young adults in full-time employment,
    all else being equal.

    **Not Currently Employed for Pay:**

    The coefficient is -1.4531, with an odds ratio
    of 0.2338. Young adults who are not currently employed
    for pay have about 76.62% reduced predicted odds of being
    financially independent compared to young adults in full-time
    employment.

    **Age Group:**

    The coefficient is 0.5176, with an odds ratio
    of 1.6780. Young adults aged 30-34 have about 67.80%
    increased predicted odds of being financially independent
    than young adults aged 18-29 years old.

    Odds change is not the same as probability change.
    """)

    st.markdown(
        f"[View Logistic Regression coefficients (CSV)]"
        f"({repo_url}regression_logistic_coefficients.csv)"
    )

    st.subheader("Comparison with Naive Bayes")

    st.markdown("""
    The first Multinomial Naive Bayes Classifier built
    in Module 3 has an accuracy rate of 59.12% for
    classification into five groups of financial independence.

    As Logistic Regression classifies into two groups,
    the accuracies cannot be compared directly.

    Therefore, Multinomial Naive Bayes was trained again,
    using the same binary output and the same observations
    for training and testing as Logistic Regression did.
    Each classification algorithm used the respective encoding.
    """)

    st.table({
        "Model": [
            "Majority-Class Baseline",
            "Binary Naive Bayes",
            "Logistic Regression"
        ],
        "Test Accuracy": [
            "76.01%",
            "86.49%",
            "86.82%"
        ]
    })

    st.image(
        "regression_model_comparison.png",
        caption="Test accuracy comparison using the same binary target."
    )

    st.markdown("""
    Logistic Regression was slightly more accurate
    than binary Naive Bayes, having a marginally greater
    accuracy by 0.34 percent.

    Nevertheless, there was only a single observation more
    that Logistic Regression classified correctly than Naive
    Bayes.

    Naive Bayes had a better recall rate for lower
    financial independence cases (69%) compared to
    Logistic Regression (63%).

    The two models were better than the baseline model,
    yet none of them showed dominance over another.
    """)

    st.markdown(
        f"[View model comparison results (CSV)]"
        f"({repo_url}regression_model_comparison.csv)"
    )

    st.markdown(
        f"[View all regression evaluation metrics (CSV)]"
        f"({repo_url}regression_evaluation_metrics.csv)"
    )

    st.header("5. Conclusions")

    st.markdown("""
    The results obtained from the regression
    analysis shed light on the relationship between
    the socioeconomic and family-related factors and
    financial independence of young adults in the
    United States.

    The Linear Regression model explains about
    half of the variance in the financial
    independence scores. Employment, living
    arrangements, financial assistance, and age were
    found to predict financial independence.

    Young adults employed part time or without
    current paid work had lower predicted
    financial independence scores than the full-time
    workers. Older young adults had higher
    predicted financial independence scores.

    Logistic Regression model reached an accuracy
    of 86.82%, and binary Naive Bayes model - 86.49%.
    Both models show the practicality of the
    predictors chosen.

    The results suggest that the financial
    independence is related to multiple
    socioeconomic and family-related variables but
    not just one.

    Nevertheless, there are some limitations that
    should be addressed. Linear Regression model
    treats an ordinal variable as a numeric one,
    and the binary classification ignores the
    original five categories of the survey. There
    is also a problem of class imbalance, and no
    survey weights were used in the models.

    Thus, the above results are only predictive
    associations of the analyzed sample data.
    """)

def adaboost():
    st.title("AdaBoost")

    st.header("1. Overview")

    st.markdown("""
      AdaBoost, also referred to as Adaptive Boosting, is
      an ensemble machine learning algorithm that utilizes
      a number of weak classifiers to form a strong classifier.

      The application of AdaBoost algorithm in this project
      is for classifying young adults into two groups:
      those with high financial independence and those with
      lower financial independence.
    """)

    st.subheader("Boosting vs. Bagging")

    st.markdown("""
    Bagging and boosting are two ensemble methods,
    though they are different in training the individual models.

    The bagging method trains multiple models separately
    through bootstrapping and makes predictions by
    voting or averaging.

    The boosting method trains multiple models one after another
    with the next model concentrating on instances wrongly classified by the previous models.
    """)

    st.table({
        "Feature": [
            "Model Training",
            "Data Sampling",
            "Model Dependency",
            "Final Prediction",
            "Main Objective"
        ],
        "Bagging": [
            "Independent",
            "Bootstrap samples",
            "Independent models",
            "Majority vote or average",
            "Reduce variance"
        ],
        "Boosting": [
            "Sequential",
            "Weighted observations (AdaBoost)",
            "Dependent models",
            "Weighted voting (AdaBoost)",
            "Improve predictive performance"
        ]
    })
    st.image(
        "adaboost_bagging_vs_boosting.png",
        caption="Figure 1: Comparison of bagging and boosting."
    )
    st.subheader("Decision Stumps")

    st.markdown("""
    A decision stump consists of a decision tree that has the maximum
    depth of one. It comprises only one decision and two leaves.

    Though a single decision stump does not have good prediction
    capacity, AdaBoosting uses several stumps to build a good classifier.
    """)

    st.subheader("How AdaBoost Works")

    st.markdown("""
    **Step 1: Initialize observation weights**

    Initially, all training observations receive equal
    weights.
    """)

    st.latex(r"w_i=\frac{1}{N}")

    st.markdown("""
    **Step 2: Train a weak learner**

    A decision stump is trained using the weighted
    training observations.

    **Step 3: Calculate weighted classification error**

    The weighted error is calculated by adding the
    weights of incorrectly classified observations.
    """)

    st.latex(
        r"\epsilon_t=\sum_{i=1}^{N}"
        r"w_i I(y_i\neq h_t(x_i))"
    )

    st.markdown("""
    **Step 4: Calculate the learner weight (alpha)**

    Alpha represents the weight assigned to each base classifier.
    The better the performance of a learner,
    the more weight it gets.

    In the classical binary AdaBoost formulation:
    """)

    st.latex(
        r"\alpha_t=\frac{1}{2}"
        r"\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)"
    )

    st.markdown("""
    **Step 5: Update observation weights**

    Misclassified samples carry relatively high
    weights, while correctly classified samples carry low
    relative weights.

    The weights are normalized such that they sum to one.
    This makes it possible for later learners to concentrate
    on hard samples.
    """)

    st.latex(
        r"w_i^{(t+1)}="
        r"\frac{w_i^{(t)}e^{-\alpha_t y_i h_t(x_i)}}{Z_t}"
    )

    st.markdown("""
    In this classical formulation, class labels and
    stump predictions are represented as -1 and +1,
    and Z is the normalization factor.

    **Step 6: Final weighted prediction**

    Predictions from all weak learners are aggregated using weighted voting. 
    Those learners having larger alphas contribute more to the final decision process.
    """)

    st.latex(
        r"H(x)=\operatorname{sign}"
        r"\left(\sum_{t=1}^{T}\alpha_t h_t(x)\right)"
    )

    st.markdown("""
    These represent the equations for classical binary AdaBoost.
    scikit-learn implements binary AdaBoost by a formulation
    called SAMME that is equivalent to the following.

    There are two main hyperparameters that are used: the
    number of estimators and the learning rate. The number
    of estimators represents the maximum number of weak
    learners, while the learning rate represents the contribution
    of each learner.
    These parameters will be tuned using validation
    data rather than the final testing dataset.
    """)
    st.image(
        "adaboost_workflow.png",
        caption="Figure 2: AdaBoost sequential learning, observation reweighting, and weighted voting."
    )
    
    st.header("2. Data Prep")

    st.markdown("""
    For training the AdaBoost model, the dataset from the
    Pew Research Center on associations between parents
    and young people in America is used.

    The preprocessed dataset has 1,478 observations
    and 6 independent variables. The initial dependent
    variable about financial independence has been
    transformed into two categories for classification.

    Category 1 includes respondents who are fully or mostly
    financially independent. Category 0 consists of respondents
    who are somewhat, slightly, or not at all financially independent.
    """)

    repo_url = (
        "https://github.com/swejal22/"
        "parent-child-independence-project/blob/main/"
    )

    st.subheader("Labeled Dataset")

    st.image(
        "adaboost_labeled_sample.png",
        caption="Sample of the labeled dataset used for AdaBoost."
    )

    st.markdown(
        f"[View labeled dataset (CSV)]"
        f"({repo_url}adaboost_labeled_data.csv)"
    )

    st.subheader("Training and Testing Sets")

    st.markdown("""
    The data were partitioned into 80% training data and 20%
    testing data through stratified sampling with random_state=42.

    There are 1,182 samples in the training data and 296 samples in the testing data.

    These two sets are disjoint in that a single person cannot be found in both the training and testing data. 
    This ensures that there is no data leakage and enables the evaluation of the model on data not seen during the training process.

    Hyperparameter optimization was conducted using five-fold cross validation on the training data set..
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Observations", "1,478")
    col2.metric("Training Set", "1,182")
    col3.metric("Testing Set", "296")

    st.image(
        "adaboost_training_sample.png",
        caption="Sample of the training dataset."
    )

    st.markdown(
        f"[View training dataset (CSV)]"
        f"({repo_url}adaboost_training_data.csv)"
    )

    st.image(
        "adaboost_testing_sample.png",
        caption="Sample of the testing dataset."
    )

    st.markdown(
        f"[View testing dataset (CSV)]"
        f"({repo_url}adaboost_testing_data.csv)"
    )

    st.subheader("Categorical Feature Encoding")

    st.markdown("""
    These six predictors are coded using numeric values
    on the surveys but they are actually categories
    and not continuous values.

    One hot encoding is applied on these six
    categorical predictors to generate 18 binary
    predictors.

    All categories are kept since decision stump
    classifiers do not need to keep any reference
    category to address multicollinearity.

    Encoder was fitted on training data alone
    and included in model pipeline during cross-validation
    to avoid leakage.
    """)

    st.image(
        "adaboost_before_encoding.png",
        caption="Six categorical predictors before one-hot encoding."
    )

    st.image(
        "adaboost_after_encoding.png",
        caption="Eighteen binary predictors after one-hot encoding."
    )

    st.markdown(
        f"[View encoded training dataset (CSV)]"
        f"({repo_url}adaboost_encoded_training_data.csv)"
    )

    st.markdown(
        f"[View encoded testing dataset (CSV)]"
        f"({repo_url}adaboost_encoded_testing_data.csv)"
    )

    st.header("3. Code")

    st.markdown("""
    Python programming language and scikit-learn package were utilized in creating and analyzing
    the AdaBoost classification model.

    The process includes the preparation of the binary outcome variable,
    splitting the dataset, using one-hot encoding,
    fitting decision stumps, tuning parameters,
    and assessment of the final model.

    The pipeline was used for combining preprocessing
    with AdaBoost classification model. Five-fold stratified cross-validation
    assisted in finding the optimal number of estimators and learning rate without using
    the final testing dataset.

    The full Google Colab Notebook with Python implementation and analysis is presented below.
    Also, the original visualizations and comparison with Decision Tree
    from Module 3 are included.
    """)

    st.markdown(
        f"[View Complete AdaBoost Python Notebook]"
        f"({repo_url}08_AdaBoost_Analysis.ipynb)"
    )

    st.subheader("Python Libraries")

    st.code(
        """from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder""",
        language="python"
    )

    st.header("4. Results")

    st.subheader("Initial AdaBoost Model")

    st.markdown("""
    AdaBoost Classifier
    The first AdaBoost classifier was built by using 50 stumps with a learning rate of 1.0.

    Cross-validation of five folds was done for estimating the performance of the classifier.

    Accuracy achieved by the classifier is 84.60% with a standard deviation of 1.45%..
    """)

    col1, col2 = st.columns(2)

    col1.metric("Initial Validation Accuracy", "84.60%")
    col2.metric("Cross-Validation Folds", "5")

    st.subheader("Hyperparameter Tuning")

    st.markdown("""
    The grid search method known as GridSearchCV was used to analyze 35 different sets
    of estimators and learning rates.

    The optimal set consisted of 200 estimators
    and a learning rate of 0.5.

    The optimal set had an accuracy score of 85.03%
    when using cross-validation, which is 0.42% higher than the original model.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Estimators", "200")
    col2.metric("Best Learning Rate", "0.5")
    col3.metric("Validation Accuracy", "85.03%")

    st.image(
        "adaboost_validation_accuracy.png",
        caption="Mean five-fold validation accuracy across boosting rounds and learning rates."
    )

    st.markdown("""
    From the validation results, an increase in
    the number of estimators does not necessarily
    lead to an improvement in prediction.

    Small learning rates often need more boosting
    iterations to have competitive accuracy.

    From the set-up explored, 200 estimators
    with learning rate of 0.5 generated the best
    mean validation accuracy.
    """)

    st.markdown(
        f"[View Hyperparameter Tuning Results (CSV)]"
        f"({repo_url}adaboost_hyperparameter_results.csv)"
    )

    st.subheader("Final AdaBoost Classification Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Testing Accuracy", "87.16%")
    col2.metric("Correct Predictions", "258 / 296")
    col3.metric("Majority-Class Baseline", "76.01%")

    st.markdown("""
    The trained AdaBoost classifier had an accuracy rate of 87.16% for classifying 258
    out of 296 unseen data points.

    This outperformed the majority class baseline of
    76.01% by around 11.15 percentage points.

    The model classified 46 data points belonging to low
    financial independence category and 212 data points
    belonging to high financial independence category.
    """)

    st.image(
        "adaboost_confusion_matrix.png",
        caption="Confusion matrix for the tuned AdaBoost classifier."
    )

    st.markdown("""
     The confusion matrix indicates that 25 people
     with low financial independence have been
     incorrectly classified as having high independence.

     Thirteen people with high financial independence
     have been incorrectly classified as having low 
     financial independence.

     The recall rate was approximately 65% for low
     independence class and 94% for the high
     independence class.

     This indicates that AdaBoost performed better
     at identifying respondents in the majority class.
    """)

    st.subheader("Comparison with Module 3 Decision Tree")

    st.markdown("""
    The initial Decision Tree model from Module 3
    categorized people into five groups of financial independence.

    Since AdaBoost classifies into two categories, the initial
    accuracy for the five categories Decision Tree of 59.12%
    cannot be compared to the accuracy of AdaBoost.

    Thus, the best Decision Tree model, which uses Gini index
    and has a max_depth of five, was trained on the same
    data used by AdaBoost.
    """)

    st.table({
        "Model": [
            "Binary Decision Tree",
            "Tuned AdaBoost"
        ],
        "Test Accuracy": [
            "85.14%",
            "87.16%"
        ],
        "Correct Predictions": [
            252,
            258
        ],
        "Lower Independence Recall": [
            "58%",
            "65%"
        ]
    })

    st.image(
        "adaboost_model_comparison.png",
        caption="Testing accuracy comparison between the binary Decision Tree and AdaBoost."
    )

    st.markdown("""
    The accuracy rate obtained using the Adaboost algorithm was 87.16%,
    while the Decision Tree obtained an accuracy rate of 85.14%.

    An increase of about
    2.03% was realized from the Adaboost algorithm.

    Adaboost correctly classified six more respondents and had improved
    recall for low financial independence.

     The recall of both models for respondents with high financial independence
     was about 94%.

      In summary, the AdaBoost algorithm performed marginally better than
      the single Decision Tree algorithm.
    """)

    st.markdown(
        f"[View Model Comparison Results (CSV)]"
        f"({repo_url}adaboost_model_comparison.csv)"
    )

    st.markdown(
        f"[View AdaBoost Evaluation Metrics (CSV)]"
        f"({repo_url}adaboost_evaluation_metrics.csv)"
    )

    st.header("5. Conclusions")

    st.markdown("""
    This study aimed to investigate whether the AdaBoost algorithm
    would be able to forecast financial independence in
    young Americans through their demographic, socioeconomic,
    and family-related factors.

    The resulting model obtained an accuracy of 87.16%
    when applied to the test set, making 258 correct predictions
    out of 296. The value exceeded the accuracy of the
    majority-class predictor at 76.01%.

    The hyperparameter optimization indicated that 200
    estimators and a learning rate of 0.5 delivered the best
    average cross-validation accuracy of 85.03%.

    Compared to the binary Decision Tree algorithm,
    which had an accuracy rate of 85.14%, AdaBoost
    was more accurate by about 2.03%.

    Yet, it is important to note that the model
    behaved differently based on the two financial
    independence classes. It managed to classify
    around 94% of respondents who had high financial
    independence but could not classify only about
    65% of respondents who had low independence.

    There are some limitations of the model that
    need to be noted. First of all, there were
    five financial independence classes that were.

    Class imbalance is also present in the data as
    about 76% of participants fall into the high
    level of independence group. Moreover,
    survey weights were not used while building
    the models.

    This study reports on associations and not
    causation and should not be considered
    a national sample.

    In conclusion, using AdaBoost, ensemble
    modeling was shown to be useful in finding
    the associations between the studied variables
    related to financial independence. The results
    also demonstrate the significance of such
    factors as employment, family support,
    and living conditions for understanding.
    """)
def gradient_boosting():
    future_page("GradientBoosting", "Module 5")


def xgboost():
    future_page("XGBoost", "Module 5")


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
    st.Page(regression, title="Regression"),
    st.Page(adaboost, title="AdaBoost"),
    st.Page(gradient_boosting, title="GradientBoosting"),
    st.Page(xgboost, title="XGBoost"),
    st.Page(conclusions, title="Conclusions")
]

navigation = st.navigation(pages, position="top")
navigation.run()
