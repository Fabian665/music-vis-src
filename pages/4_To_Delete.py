import data_wrangling
import plotting
import streamlit as st
st.set_page_config(
    page_title="Statistics",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="collapsed",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

glz_df = data_wrangling.read_data()

time_signature_df = data_wrangling.time_signature_distribution(glz_df)
mode_distribution_df = data_wrangling.mode_distribution(glz_df)


col1, col2, col3, col4 = st.columns(4)
unique_artists, top_artist, top_song, time_signature = plotting.text_plots(glz_df)
with col1:
    st.plotly_chart(time_signature)
with col2:
    st.plotly_chart(top_song)
with col3:
    st.plotly_chart(top_artist)
with col4:
    st.plotly_chart(unique_artists)

col1, col2 = st.columns(2)
with col1:
    # st.plotly_chart(plotting.plot_time_signature(time_signature_df))
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
    st.plotly_chart(plotting.plot_genre_trends(glz_df, market))

with col2:
    st.empty()
    st.plotly_chart(plotting.plot_mode_distribution(mode_distribution_df))

