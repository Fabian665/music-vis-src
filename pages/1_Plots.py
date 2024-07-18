import streamlit as st
import numpy as np
from spotify import SpotifyAPI
from streamlit.logger import get_logger
import data_wrangling
import plotting
st.set_page_config(
    page_title="General Plots",
    page_icon="📈",
    layout="centered",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

logger = get_logger(__name__)

if 'spotify' not in st.session_state:
    st.session_state.spotify = SpotifyAPI()

glz_df = data_wrangling.read_data()

st.plotly_chart(plotting.plot_scatter_song_length(glz_df))

st.header('Filter Selection')
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

    rank = st.slider("Max rank", 1, 10, 5, 1, help='Will only filter for songs ranked better than this number (1 is the best)')

with col2:
    min_date, max_date = data_wrangling.get_date_range(glz_df)

    filter_type = st.radio(
        "Choose filter type",
        ('year', 'date'),
        format_func=lambda x: 'Year' if x == 'year' else 'Date Range',
        index=1,
        key='filter_field',
    )

    logger.debug(f"Filter type: {st.session_state['filter_field']}")
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
            "Select a date range to filter data",
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


st.plotly_chart(plotting.plot_artist_stats(market, year, rank, date))
st.plotly_chart(plotting.plot_top_artists_with_songs(market, year, rank, date))
