import data_wrangling
import plotting
from datetime import timedelta, date
import numpy as np
from spotify import SpotifyAPI
import streamlit as st
st.set_page_config(
    page_title="Song Mania",
    page_icon="🎶",
    initial_sidebar_state="collapsed",
)

glz_df = data_wrangling.read_data()


# Title and introduction
st.title("Song Mania: Unveiling the Top Hits of Galgalatz")
st.markdown(
    """
    Welcome to Song Mania! 🎵 Dive into the world of the top hits that have dominated the Galgalatz charts over the past decade.
    Discover the trends in the music and find out how these elements have evolved over the years.
    """
)


# New section: Yearly Top Song Analysis
st.subheader("Yearly Top Song Analysis")

# Sidebar filter options
market_labels = {
    'IL': 'Israel',
    'INTL': 'International',
}
market = st.selectbox(
    'Market',
    ['IL', 'INTL'],
    key='market',
    format_func=lambda x: market_labels[x],
)

criteria = st.radio(
    "Criteria",
    (1, 10),
    format_func=lambda x: 'Number One Weeks' if x == 1 else 'Total Weeks on Billboard',
)

sort_by = st.radio(
    "Sort By",
    ('year', 'criteria'),
    format_func=lambda x: 'Year' if x == 'year' else 'Criteria',
)


st.plotly_chart(plotting.plot_top_hits(glz_df, market, criteria, sort_by))


# Plot song duration over time
st.subheader("Song Duration Over Time")
st.plotly_chart(plotting.plot_scatter_song_length(glz_df))


mode_distribution_df = data_wrangling.mode_distribution(glz_df)

st.subheader("Major vs. Minor over the Years")
st.plotly_chart(plotting.plot_mode_distribution(mode_distribution_df))


time_signature_df = data_wrangling.time_signature_distribution(glz_df)
_, _, top_song, time_signature = plotting.text_plots(glz_df)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(time_signature)

with col2:
    st.plotly_chart(top_song)
