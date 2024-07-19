import streamlit as st
import numpy as np
from spotify import SpotifyAPI
from streamlit.logger import get_logger
import data_wrangling
import plotting

# Set page configuration
st.set_page_config(
    page_title="General Plots",
    page_icon="📈",
    layout="centered",
)

# Initialize logger
logger = get_logger(__name__)

# Initialize Spotify API if not in session state
if 'spotify' not in st.session_state:
    st.session_state.spotify = SpotifyAPI()

# Read data
glz_df = data_wrangling.read_data()

# Title and introduction
st.title("General Plots for Galgalatz Charts Analysis")
st.markdown(
    """
    Explore the dynamic trends in the Galgalatz charts over the past decade. Use the filters below to customize your view.
    """
)



# Filter selection section
st.subheader("Filter Selection")
col1, col2 = st.columns(2, gap='medium')

with col1:
    market_labels = {
        None: 'All Markets',
        'IL': 'Israel',
        'INTL': 'International',
    }
    market = st.selectbox(
        'Market',
        [None, 'IL', 'INTL'],
        key='market',
        format_func=lambda x: market_labels[x],
    )

    rank = st.slider(
        "Max Rank",
        1,
        10,
        5,
        1,
        help='Will only filter for songs ranked better than this number (1 is the best)'
    )

with col2:
    min_date, max_date = data_wrangling.get_date_range(glz_df)
    filter_type = st.radio(
        "Choose Filter Type",
        ('year', 'date'),
        format_func=lambda x: 'Year' if x == 'year' else 'Date Range',
        index=1,
        key='filter_field',
    )

    if st.session_state['filter_field'] == 'year':
        date = None
        year = st.selectbox(
            'Select Year',
            list(range(min_date.year, max_date.year + 1)),
            help='Select a year to filter data',
        )
    else:
        year = None
        date = st.date_input(
            "Select a Date Range to Filter Data",
            (min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD",
            help='Select a date range to filter data'
        )
        try:
            start_date, end_date = date
            date = np.datetime64(start_date), np.datetime64(end_date)
        except ValueError:
            st.error("You must pick a start and end date")
            st.stop()

# Plot artist stats and top artists with songs
st.subheader("Artist Impact")
st.plotly_chart(plotting.plot_artist_stats(market, year, rank, date))

st.subheader("Hottest Artists")
st.plotly_chart(plotting.plot_top_artists_with_songs(market, year, rank, date))



top_artist = plotting.text_plots(glz_df)[1]

st.plotly_chart(top_artist)


unique_artists = plotting.text_plots(glz_df)[0]

st.plotly_chart(unique_artists)


