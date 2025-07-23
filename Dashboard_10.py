import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import re
import os
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="Search Query Analysis Dashboard", layout="wide", page_icon="🔍")

# Custom styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #D4F1F4;
    }
    .css-1d391kg, .css-1d391kg * {
        color: #004E64 !important;
    }
    .stButton>button {
        background-color: #004E64;
        color: white;
        border-radius: 5px;
    }
    .stExpander {
        background-color: #F0F8FF;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data(show_spinner=True)
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        logger.info("Dataset loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Initialize session state
if "filtered" not in st.session_state:
    file_path = "sampled_search_data.csv"
    df = load_data(file_path)
    if df is None:
        st.stop()
    st.session_state.df = df
    st.session_state.filtered = df
    st.session_state.sample_size = 50000  # Reduced for performance

# Always assign from session state
df = st.session_state.df
df_filtered = st.session_state.filtered

# Verify expected columns
expected_columns = ['Date', 'Sub Category', 'Class', 'normalized_query', 'CTR', 'ATCR', 'CR', 'Count', 'Rev', 'Month',
                    'day_of_week', 'week_number', 'revenue_per_search', 'conversions', 'conversion_value', 'language',
                    'processed_query', 'misspelling_type', 'match_confidence']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    logger.error(f"Missing columns: {missing_columns}")
    st.stop()

# Preprocessing for visualization
@st.cache_data
def preprocess_data(df, sample_size):
    df = df.copy()
    df['query_length'] = df['normalized_query'].astype(str).apply(len)
    df['CTR'] = df['CTR'] * 100  # Convert to percentage
    df['CR'] = df['CR'] * 100    # Convert to percentage
    df['ATCR'] = df['ATCR'] * 100  # Convert to percentage
    sample_size = min(sample_size, len(df))
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        logger.info(f"Using sampled dataset for visualizations ({sample_size} rows)")
        st.info(f"Using sampled dataset for visualizations ({sample_size} rows)")
    else:
        df_sample = df
        logger.info("Using full dataset for visualizations")
        st.info("Using full dataset for visualizations")
    return df, df_sample

# Sidebar filters
st.sidebar.title("🔎 Filters")
with st.sidebar.expander("📍 Refine your view"):
    language_options = ['All'] + sorted(df['language'].unique().tolist())
    selected_languages = st.multiselect("🌐 Select Language(s)", language_options, default=['All'])
    subcategory_options = ['All'] + sorted(df['Sub Category'].unique().tolist())
    selected_subcategories = st.multiselect("🧴 Select Sub Category(ies)", subcategory_options, default=['All'])
    month_options = ['All'] + sorted(df['Month'].unique().tolist(), key=lambda x: ['Apr', 'May', 'Jun', 'Jul'].index(x))
    selected_months = st.multiselect("📅 Select Month(s)", month_options, default=['All'])
    sample_size = st.slider("🔢 Sample Size for Visualizations", 10000, 200000, st.session_state.sample_size, 10000)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Apply Filters"):
            df_filtered = df.copy()
            if 'All' not in selected_languages:
                df_filtered = df_filtered[df_filtered['language'].isin(selected_languages)]
            if 'All' not in selected_subcategories:
                df_filtered = df_filtered[df_filtered['Sub Category'].isin(selected_subcategories)]
            if 'All' not in selected_months:
                df_filtered = df_filtered[df_filtered['Month'].isin(selected_months)]
            st.session_state.filtered = df_filtered
            st.session_state.sample_size = sample_size
            st.success("Filters applied successfully!")
    with col2:
        if st.button("🔄 Reset Filters"):
            st.session_state.filtered = df
            st.session_state.sample_size = 50000
            st.success("Filters reset to default!")

# Apply preprocessing
df_filtered, df_filtered_sample = preprocess_data(st.session_state.filtered, st.session_state.sample_size)

# Tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Univariate", "📈 Bivariate", "🔍 Categorical", "📅 Temporal", "📋 Summary & Export"])

# Visualization definitions with insights
color_sequence = px.colors.qualitative.D3

@st.cache_data
def get_word_freq(data, column):
    all_words = ' '.join(data[column]).lower()
    word_freq = Counter(re.findall(r'\w+', all_words))
    return pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])

vis_definitions = [
    (1, "Top 20 Most Frequent Search Queries", lambda df, df_sample: px.bar(
        df['normalized_query'].value_counts().head(20).reset_index(name='count'),
        x='normalized_query', y='count', color='count',
        title='Top 20 Most Frequent Search Queries',
        labels={'normalized_query': 'Query', 'count': 'Count'},
        color_discrete_sequence=color_sequence
    ), "Univariate", "Shows the most common search queries. **Benefit**: Identify popular products (e.g., 'قطرات ادول') to prioritize inventory or marketing campaigns."),
    (2, "Language Distribution", lambda df, df_sample: px.pie(
        df, names='language', title='Language Distribution of Queries',
        color_discrete_sequence=px.colors.qualitative.Plotly
    ), "Univariate", "Displays the proportion of queries by language. **Benefit**: Target marketing efforts to dominant languages (e.g., Arabic) for better reach."),
    (3, "Count Distribution", lambda df, df_sample: px.histogram(
        df_sample, x='Count', nbins=50, title='Distribution of Query Counts',
        color_discrete_sequence=color_sequence, log_y=True
    ).update_layout(xaxis_title='Search Count', yaxis_title='Frequency (Log Scale)'), "Univariate", "Shows how often queries are searched. **Benefit**: Identify frequently searched terms to optimize search engine performance."),
    (4, "Revenue Distribution", lambda df, df_sample: px.histogram(
        df_sample, x='Rev', nbins=50, title='Distribution of Revenue per Query (SAR)',
        color_discrete_sequence=color_sequence, log_y=True
    ).update_layout(xaxis_title='Revenue (SAR)', yaxis_title='Frequency (Log Scale)'), "Univariate", "Shows revenue spread per query. **Benefit**: Pinpoint high-revenue queries for targeted promotions."),
    (5, "Conversion Rate Distribution", lambda df, df_sample: px.histogram(
        df_sample, x='CR', nbins=50, title='Distribution of Conversion Rate (%)',
        color_discrete_sequence=color_sequence
    ).update_layout(xaxis_title='Conversion Rate (%)'), "Univariate", "Displays conversion rate spread. **Benefit**: Identify queries with high conversion potential for ad focus."),
    (6, "Query Length Distribution", lambda df, df_sample: px.histogram(
        df_sample, x='query_length', nbins=30, title='Distribution of Query Length (Characters)',
        color_discrete_sequence=color_sequence
    ), "Univariate", "Shows query length distribution. **Benefit**: Optimize search for short vs. long queries to improve user experience."),
    (7, "Most Common Words in Queries", lambda df, df_sample: px.bar(
        get_word_freq(df_sample, 'normalized_query'), x='word', y='count',
        title='Most Common Words in Search Queries',
        color='count', color_continuous_scale='Mint'
    ), "Univariate", "Identifies frequent words in queries. **Benefit**: Refine search suggestions or keywords for SEO."),
    (8, "Top 20 Queries by CTR", lambda df, df_sample: px.bar(
        df.sort_values('CTR', ascending=False).head(20),
        x='normalized_query', y='CTR', color='CTR',
        title='Top 20 Queries by CTR (%)',
        color_continuous_scale='Greens'
    ).update_layout(xaxis_title='Query', yaxis_title='CTR (%)'), "Univariate", "Highlights queries with high click-through rates. **Benefit**: Focus on high-engagement queries for ad campaigns."),
    (9, "Top 20 Queries by Revenue", lambda df, df_sample: px.bar(
        df.sort_values('Rev', ascending=False).head(20),
        x='normalized_query', y='Rev', color='Rev',
        title='Top 20 Queries by Revenue (SAR)',
        color_continuous_scale='Greens'
    ).update_layout(xaxis_title='Query', yaxis_title='Revenue (SAR)'), "Univariate", "Shows queries driving the most revenue. **Benefit**: Prioritize these queries for inventory and promotions."),
    (10, "Low Revenue but High Count Queries", lambda df, df_sample: px.bar(
        df[df['Count'] > df['Count'].median()].sort_values('Rev').head(20),
        x='normalized_query', y='Rev', color='Count',
        title='Low Revenue but High Count Queries (SAR)',
        color_continuous_scale='Teal'
    ).update_layout(xaxis_title='Query', yaxis_title='Revenue (SAR)'), "Univariate", "Identifies popular but low-revenue queries. **Benefit**: Optimize these queries to increase conversions."),
    (11, "Count vs Revenue", lambda df, df_sample: px.density_heatmap(
        df_sample, x='Count', y='Rev', z='CTR', histfunc='avg',
        title='Count vs Revenue (Color = Avg CTR, SAR)',
        color_continuous_scale='Blues', nbinsx=50, nbinsy=50
    ).update_layout(xaxis_title='Search Count', yaxis_title='Revenue (SAR)', coloraxis_colorbar_title='Avg CTR (%)'), "Bivariate", "Shows relationship between search frequency, revenue, and CTR. **Benefit**: Identify high-frequency, low-revenue areas for optimization."),
    (12, "CTR vs Conversion Rate", lambda df, df_sample: px.density_heatmap(
        df_sample, x='CTR', y='CR', z='Count', histfunc='sum',
        title='CTR vs Conversion Rate (Color = Count, SAR)',
        color_continuous_scale='Viridis', nbinsx=50, nbinsy=50
    ).update_layout(xaxis_title='CTR (%)', yaxis_title='Conversion Rate (%)', coloraxis_colorbar_title='Count'), "Bivariate", "Explores CTR and conversion rate with query frequency. **Benefit**: Target high-CTR, high-conversion zones for maximum ROI."),
    (13, "Query Length vs CTR", lambda df, df_sample: px.scatter(
        df_sample, x='query_length', y='CTR', color='CR', size='Count',
        title='Query Length vs CTR (Size = Count, Color = CR)',
        color_continuous_scale='Teal', opacity=0.5
    ).update_layout(xaxis_title='Query Length (Characters)', yaxis_title='CTR (%)', coloraxis_colorbar_title='Conversion Rate (%)'), "Bivariate", "Examines query length impact on CTR and CR. **Benefit**: Optimize search for specific query lengths."),
    (14, "Average CTR by Language", lambda df, df_sample: px.bar(
        df.groupby('language')['CTR'].mean().reset_index(),
        x='language', y='CTR', color='CTR',
        title='Average CTR by Query Language',
        color_continuous_scale='Blues'
    ).update_layout(xaxis_title='Language', yaxis_title='Average CTR (%)'), "Categorical", "Shows CTR by language. **Benefit**: Tailor campaigns to high-CTR languages like Arabic."),
    (15, "Count vs CTR", lambda df, df_sample: px.density_heatmap(
        df_sample, x='Count', y='CTR', z='Rev', histfunc='avg',
        title='Count vs CTR (Color = Avg Revenue, SAR)',
        color_continuous_scale='Mint', nbinsx=50, nbinsy=50
    ).update_layout(xaxis_title='Search Count', yaxis_title='CTR (%)', coloraxis_colorbar_title='Revenue (SAR)'), "Bivariate", "Analyzes search frequency and CTR with revenue. **Benefit**: Identify high-traffic, low-CTR queries for improvement."),
    (16, "Revenue vs Conversion Rate", lambda df, df_sample: px.density_heatmap(
        df_sample, x='Rev', y='CR', z='Count', histfunc='sum',
        title='Revenue vs Conversion Rate (Color = Count, SAR)',
        color_continuous_scale='RdYlBu', nbinsx=50, nbinsy=50
    ).update_layout(xaxis_title='Revenue (SAR)', yaxis_title='Conversion Rate (%)', coloraxis_colorbar_title='Count'), "Bivariate", "Explores revenue and conversion rate with query frequency. **Benefit**: Focus on high-conversion, high-revenue zones."),
    (17, "Average Conversion Rate by Language", lambda df, df_sample: px.bar(
        df.groupby('language')['CR'].mean().reset_index(),
        x='language', y='CR', color='CR',
        title='Average Conversion Rate by Language',
        color_continuous_scale='Greens'
    ).update_layout(xaxis_title='Language', yaxis_title='Average Conversion Rate (%)'), "Categorical", "Shows conversion rates by language. **Benefit**: Prioritize high-conversion languages for marketing."),
    (18, "CTR Distribution by Language", lambda df, df_sample: px.box(
        df_sample, x='language', y='CTR', color='language',
        title='CTR Distribution by Language',
        color_discrete_sequence=px.colors.qualitative.Plotly
    ).update_layout(xaxis_title='Language', yaxis_title='CTR (%)'), "Categorical", "Shows CTR variability by language. **Benefit**: Identify languages with inconsistent CTR for search improvements."),
    (19, "Query Length vs Count", lambda df, df_sample: px.scatter(
        df_sample, x='query_length', y='Count', color='CTR', size='CR',
        title='Query Length vs Count',
        color_continuous_scale='Teal', opacity=0.5
    ).update_layout(xaxis_title='Query Length (Characters)', yaxis_title='Search Count', coloraxis_colorbar_title='CTR (%)'), "Bivariate", "Examines query length and search frequency with CTR/CR. **Benefit**: Optimize for popular query lengths."),
    (20, "Top 20 Queries by Conversion Rate", lambda df, df_sample: px.bar(
        df.sort_values('CR', ascending=False).head(20),
        x='normalized_query', y='CR', color='CR',
        title='Top 20 Queries by Conversion Rate',
        color_continuous_scale='Teal'
    ).update_layout(xaxis_title='Query', yaxis_title='Conversion Rate (%)'), "Univariate", "Highlights high-conversion queries. **Benefit**: Target these queries for sales campaigns."),
    (21, "Language Split in Top CTR Queries", lambda df, df_sample: px.histogram(
        df[df['CTR'] > df['CTR'].median()], x='language',
        title='Language Split in Top CTR Queries', color='language',
        color_discrete_sequence=px.colors.qualitative.Plotly
    ).update_layout(xaxis_title='Language', yaxis_title='Count'), "Categorical", "Shows language distribution for high-CTR queries. **Benefit**: Focus on dominant languages for ad targeting."),
    (22, "CTR Spread Across Languages", lambda df, df_sample: px.box(
        df_sample, x='language', y='CTR', color='language',
        title='CTR Spread Across Languages',
        color_discrete_sequence=px.colors.qualitative.Plotly
    ).update_layout(xaxis_title='Language', yaxis_title='CTR (%)'), "Categorical", "Compares CTR variability across languages. **Benefit**: Improve search for languages with low CTR."),
    (23, "Top Performing Queries (CTR > 75th Percentile)", lambda df, df_sample: px.scatter(
        df[df['CTR'] > df['CTR'].quantile(0.75)], x='Count', y='Rev', size='CR', color='language',
        title='Top Performing Queries (CTR > 75th Percentile, SAR)', opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Plotly
    ).update_layout(xaxis_title='Search Count', yaxis_title='Revenue (SAR)'), "Bivariate", "Shows high-CTR queries with revenue and conversion. **Benefit**: Prioritize these for maximum ROI."),
    (24, "Revenue by Query Length", lambda df, df_sample: px.box(
        df_sample.assign(length_bucket=pd.cut(df_sample['query_length'], bins=[0, 10, 20, 30, 100], labels=['Short', 'Medium', 'Long', 'Very Long'])),
        x='length_bucket', y='Rev', color='length_bucket', title='Revenue by Query Length (SAR)',
        color_discrete_sequence=px.colors.qualitative.D3
    ).update_layout(xaxis_title='Query Length Bucket', yaxis_title='Revenue (SAR)'), "Categorical", "Shows revenue by query length. **Benefit**: Optimize search for high-revenue query lengths."),
    (25, "High Count, Low CTR Queries", lambda df, df_sample: px.bar(
        df[(df['Count'] > df['Count'].median()) & (df['CTR'] < df['CTR'].median())].head(20),
        x='normalized_query', y='CTR', color='Rev',
        title='High Count, Low CTR Queries (SAR)',
        color_continuous_scale='YlGn'
    ).update_layout(xaxis_title='Query', yaxis_title='CTR (%)'), "Univariate", "Identifies popular but low-engagement queries. **Benefit**: Improve search results for these queries to boost CTR."),
    (26, "CTR Trend by Query Length Bucket", lambda df, df_sample: px.line(
        df.assign(length_bucket=pd.cut(df['query_length'], bins=[0, 10, 20, 30, 100], labels=['Short', 'Medium', 'Long', 'Very Long']))
        .groupby('length_bucket')['CTR'].mean().reset_index(),
        x='length_bucket', y='CTR', markers=True,
        title='CTR Trend Across Query Length Buckets',
        color_discrete_sequence=color_sequence
    ).update_layout(xaxis_title='Query Length Bucket', yaxis_title='Average CTR (%)'), "Categorical", "Shows CTR trends by query length. **Benefit**: Tailor search for optimal query lengths."),
    (28, "CTR Spread by Query Length", lambda df, df_sample: px.box(
        df_sample.assign(length_bucket=pd.cut(df_sample['query_length'], bins=[0, 10, 20, 30, 100], labels=['Short', 'Medium', 'Long', 'Very Long'])),
        x='length_bucket', y='CTR', color='length_bucket',
        title='CTR Spread by Query Length',
        color_discrete_sequence=px.colors.qualitative.D3
    ).update_layout(xaxis_title='Query Length Bucket', yaxis_title='CTR (%)'), "Categorical", "Shows CTR variability by query length. **Benefit**: Improve search for inconsistent lengths."),
    (29, "Revenue vs CTR by Language", lambda df, df_sample: px.density_heatmap(
        df_sample, x='CTR', y='Rev', z='Count', histfunc='sum', facet_col='language',
        title='Revenue vs CTR by Language (Color = Count, SAR)',
        color_continuous_scale='Blues', nbinsx=50, nbinsy=50
    ).update_layout(xaxis_title='CTR (%)', yaxis_title='Revenue (SAR)', coloraxis_colorbar_title='Count'), "Bivariate", "Compares CTR and revenue by language. **Benefit**: Target high-revenue languages for campaigns."),
    (30, "Low Revenue but High CTR Queries", lambda df, df_sample: px.bar(
        df[(df['Rev'] < df['Rev'].median()) & (df['CTR'] > df['CTR'].median())].sort_values('CTR', ascending=False).head(15),
        x='normalized_query', y='CTR', color='Rev',
        title='Low Revenue but High CTR Queries (SAR)',
        color_continuous_scale='Mint'
    ).update_layout(xaxis_title='Query', yaxis_title='CTR (%)'), "Univariate", "Shows high-CTR but low-revenue queries. **Benefit**: Optimize these for higher conversions."),
    (31, "Revenue by Sub Category and Class", lambda df, df_sample: px.sunburst(
        df, path=['Sub Category', 'Class'], values='Rev',
        title='Revenue Breakdown by Sub Category and Class (SAR)',
        color='Rev', color_continuous_scale='Blues'
    ), "Categorical", "Breaks down revenue by category hierarchy. **Benefit**: Focus inventory on high-revenue subcategories like Baby Diapers."),
    (34, "Query Language Trends Across CTR Levels", lambda df, df_sample: px.bar(
        df.assign(ctr_bins=pd.qcut(df['CTR'], 4, labels=['Low', 'Med', 'High', 'Very High']))
        .groupby(['ctr_bins', 'language']).size().reset_index(name='count'),
        x='ctr_bins', y='count', color='language', barmode='group',
        title='Query Language Trends Across CTR Levels',
        color_discrete_sequence=px.colors.qualitative.Plotly
    ).update_layout(xaxis_title='CTR Bucket', yaxis_title='Count'), "Categorical", "Shows language distribution across CTR levels. **Benefit**: Target high-CTR language segments."),
    (35, "CTR Trend by Month", lambda df, df_sample: px.line(
        df.groupby('Month')['CTR'].mean().reset_index().assign(
            Month=pd.Categorical(df.groupby('Month')['CTR'].mean().reset_index()['Month'],
                                categories=['Apr', 'May', 'Jun', 'Jul'], ordered=True)
        ).sort_values('Month'),
        x='Month', y='CTR', markers=True,
        title='Average CTR by Month', color_discrete_sequence=color_sequence
    ).update_layout(xaxis_title='Month', yaxis_title='Average CTR (%)'), "Temporal", "Shows CTR trends over months. **Benefit**: Plan campaigns for high-CTR months like Jul."),
    (36, "Revenue by Day of Week", lambda df, df_sample: px.bar(
        df.groupby('day_of_week')['Rev'].sum().reset_index().assign(
            day_of_week=pd.Categorical(df.groupby('day_of_week')['Rev'].sum().reset_index()['day_of_week'],
                                      categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], ordered=True)
        ).sort_values('day_of_week'),
        x='day_of_week', y='Rev', color='Rev',
        title='Total Revenue by Day of Week (SAR)',
        color_continuous_scale='Greens'
    ).update_layout(xaxis_title='Day of Week', yaxis_title='Total Revenue (SAR)'), "Temporal", "Shows revenue by day of week. **Benefit**: Schedule promotions for high-revenue days."),
    (37, "Conversion Rate by Week Number", lambda df, df_sample: px.line(
        df.groupby('week_number')['CR'].mean().reset_index(),
        x='week_number', y='CR', markers=True,
        title='Average Conversion Rate by Week Number',
        color_discrete_sequence=color_sequence
    ).update_layout(xaxis_title='Week Number', yaxis_title='Average Conversion Rate (%)'), "Temporal", "Shows conversion trends by week. **Benefit**: Plan weekly sales strategies."),
    (38, "Misspelling Type Impact on CTR", lambda df, df_sample: px.bar(
        df.groupby('misspelling_type')['CTR'].mean().reset_index(),
        x='misspelling_type', y='CTR', color='CTR',
        title='Average CTR by Misspelling Type',
        color_continuous_scale='Greens'
    ).update_layout(xaxis_title='Misspelling Type', yaxis_title='Average CTR (%)'), "Categorical", "Shows CTR impact of misspellings. **Benefit**: Improve search algorithms for misspelled queries."),
    (39, "Top 20 Sub Categories by Conversion Value", lambda df, df_sample: px.bar(
        df.groupby('Sub Category')['conversion_value'].mean().reset_index().sort_values('conversion_value', ascending=False).head(20),
        x='Sub Category', y='conversion_value', color='conversion_value',
        title='Top 20 Sub Categories by Average Conversion Value (SAR)',
        color_continuous_scale='Greens'
    ).update_layout(xaxis_title='Sub Category', yaxis_title='Average Conversion Value (SAR)'), "Categorical", "Shows high-value subcategories. **Benefit**: Focus on high-conversion categories like Baby Diapers."),
    (40, "CTR vs ATCR by Language", lambda df, df_sample: px.scatter(
        df_sample, x='CTR', y='ATCR', color='language', size='Count',
        title='CTR vs ATCR by Language (Size = Count)', opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Plotly
    ).update_layout(xaxis_title='CTR (%)', yaxis_title='ATCR (%)'), "Bivariate", "Compares CTR and ATCR by language. **Benefit**: Optimize search for languages with high ATCR."),
    (42, "Revenue per Conversion by Sub Category", lambda df, df_sample: px.bar(
        df.groupby('Sub Category').apply(lambda x: x['Rev'].sum() / x['conversions'].sum() if x['conversions'].sum() > 0 else 0).reset_index(name='rev_per_conversion').sort_values('rev_per_conversion', ascending=False).head(20),
        x='Sub Category', y='rev_per_conversion', color='rev_per_conversion',
        title='Top 20 Sub Categories by Revenue per Conversion (SAR)',
        color_continuous_scale='Teal'
    ).update_layout(xaxis_title='Sub Category', yaxis_title='Revenue per Conversion (SAR)'), "Categorical", "Shows revenue per conversion by subcategory. **Benefit**: Prioritize high-value conversion categories."),
    (44, "Misspelling Impact on Revenue by Language", lambda df, df_sample: px.bar(
        df.groupby(['misspelling_type', 'language'])['Rev'].mean().reset_index(),
        x='misspelling_type', y='Rev', color='language', barmode='group',
        title='Average Revenue by Misspelling Type and Language (SAR)',
        color_discrete_sequence=px.colors.qualitative.Plotly
    ).update_layout(xaxis_title='Misspelling Type', yaxis_title='Average Revenue (SAR)'), "Categorical", "Shows revenue impact of misspellings by language. **Benefit**: Enhance search for misspelled queries in key languages."),
    (45, "Monthly Revenue by Sub Category", lambda df, df_sample: px.line(
        df.groupby(['Month', 'Sub Category'])['Rev'].sum().reset_index().assign(
            Month=pd.Categorical(df.groupby(['Month', 'Sub Category'])['Rev'].sum().reset_index()['Month'],
                                categories=['Apr', 'May', 'Jun', 'Jul'], ordered=True)
        ),
        x='Month', y='Rev', color='Sub Category',
        title='Monthly Revenue by Sub Category (SAR)',
        color_discrete_sequence=px.colors.qualitative.D3
    ).update_layout(xaxis_title='Month', yaxis_title='Revenue (SAR)'), "Temporal", "Shows revenue trends by subcategory over months. **Benefit**: Plan seasonal promotions for high-revenue categories.")
]

# Render visualizations with progress bar
with st.spinner("Rendering visualizations..."):
    for tab, tab_name in [(tab1, "Univariate"), (tab2, "Bivariate"), (tab3, "Categorical"), (tab4, "Temporal")]:
        with tab:
            st.header(f"{tab_name} Visualizations")
            # Table of contents
            vis_list = [f"VIS {num}: {title}" for num, title, _, cat, _ in vis_definitions if cat == tab_name]
            st.markdown(f"**Visualizations in this tab**: {', '.join(vis_list)}")
            for vis_num, vis_title, vis_func, vis_category, vis_insight in vis_definitions:
                if vis_category == tab_name:
                    try:
                        with st.expander(f"VIS {vis_num}: {vis_title}"):
                            st.subheader(vis_title)
                            fig = vis_func(df_filtered, df_filtered_sample)
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown(f"**Insight**: {vis_insight}")
                    except Exception as e:
                        st.error(f"Error rendering {vis_title}: {str(e)}")
                        logger.error(f"Error in {vis_title}: {str(e)}")

# Summary & Export tab
with tab5:
    st.header("📅 Dataset Summary & Export")
    st.write(f"**Total Rows**: {len(df_filtered)}")
    st.write(f"**Filters Applied**: Languages={', '.join(selected_languages)}, Sub Categories={', '.join(selected_subcategories)}, Months={', '.join(selected_months)}")
    st.write(f"**Sample Size**: {st.session_state.sample_size}")
    st.write("**Summary Statistics**:")
    stats = df_filtered[['CTR', 'CR', 'Rev', 'Count', 'ATCR', 'revenue_per_search', 'conversions', 'conversion_value']].describe()
    st.dataframe(stats.rename(columns={'Rev': 'Revenue (SAR)', 'revenue_per_search': 'Revenue per Search (SAR)', 'conversion_value': 'Conversion Value (SAR)'}), use_container_width=True)
    st.write("**Filtered Data Preview**:")
    st.dataframe(df_filtered.head(10), use_container_width=True)
    st.download_button("⬇ Download Filtered Data", df_filtered.rename(columns={'Rev': 'Revenue (SAR)', 'revenue_per_search': 'Revenue per Search (SAR)', 'conversion_value': 'Conversion Value (SAR)'}).to_csv(index=False), file_name='filtered_search_data.csv')
    st.success("Download the filtered data for further analysis.")

# Instructions
st.markdown("""
### Instructionsb
1. Use the sidebar to select multiple languages, sub categories, and months.
2. Adjust the sample size for faster visualization rendering (default: 50,000).
3. Click **Apply Filters** to update or **Reset Filters** to revert to the full dataset.
4. Navigate through tabs (Univariate, Bivariate, Categorical, Temporal, Summary & Export).
5. Expand each visualization to see its business insight.
6. Download the filtered dataset as a CSV from the Summary & Export tab.
7. Hover over charts for details, and use Plotly's tools to zoom or pan.
""")
