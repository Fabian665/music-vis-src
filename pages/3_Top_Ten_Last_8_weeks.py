import data_wrangling
import plotting
from datetime import timedelta
from spotify import SpotifyAPI
import streamlit as st
st.set_page_config(layout="wide")


if 'spotify' not in st.session_state:
    st.session_state.spotify = SpotifyAPI()

st.header('Galgalaz Top 10 Visualized')
# explain about bump charts

glz_df = data_wrangling.read_data()

_, max_date = data_wrangling.get_date_range(glz_df)
min_date = max_date - timedelta(weeks=7)
date = min_date, max_date

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

st.plotly_chart(plotting.plot_bumpchart(glz_df, market, None, 10, date))
