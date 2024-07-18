import data_wrangling
import plotting
from datetime import timedelta, date
import numpy as np
from spotify import SpotifyAPI
import streamlit as st
st.set_page_config(
    page_title="Top 10",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)


if 'spotify' not in st.session_state:
    st.session_state.spotify = SpotifyAPI()

st.header('Galgalaz Top 10 Visualized')
# explain about bump charts

glz_df = data_wrangling.read_data()

min_date_data, max_date_data = data_wrangling.get_date_range(glz_df)

market_labels = {
    None: 'All Markets',
    'IL': 'Israel',
    'INTL': 'International',
}
market = st.selectbox(
    'Market',
    ['IL', 'INTL'],
    key='market',
    format_func=lambda x: market_labels[x],
)
st.write("Select a date range to filter data")
with st.form('date_range'):
    col1, col2, col3 = st.columns(3)
    with col1:
        years = list(range(min_date_data.year, max_date_data.year + 1))
        year = st.selectbox(
            'Select Year',
            years,
            index=len(years) - 1,
        )
    with col2:
        month = st.selectbox(
            'Select Month',
            list(range(1, 13)),
            index=max_date_data.month - 1,
        )
    with col3:
        day = st.selectbox(
            'Select Day',
            list(range(1, 32)),
            index=max_date_data.day - 1,
        )

    chosen_date = np.datetime64(date(year=year, month=month, day=day))
    last_before = glz_df[glz_df['date'] <= chosen_date]['date'].max()
    min_date = last_before - timedelta(weeks=7)
    date = np.datetime64(min_date), np.datetime64(chosen_date)
    st.form_submit_button('Plot Week')


st.plotly_chart(plotting.plot_bumpchart(glz_df, market, None, 10, date))
