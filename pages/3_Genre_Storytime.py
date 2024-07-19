import streamlit as st
import spotify
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import data_wrangling
import plotting
st.set_page_config(
    page_title="Genre Feature Compare",
    page_icon="🎵",
    layout="centered",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

logger = get_logger(__name__)

features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']


split_feature_names = {
    'year': {},
    'month': {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December',
    },
    'market': {
        'IL': 'Israel',
        'INTL': 'International',
    },
    'key': {
        -1: 'N/A',
        0: 'C',
        1: 'C♯, D♭',
        2: 'D',
        3: 'D♯, E♭',
        4: 'E',
        5: 'F',
        6: 'F♯, G♭',
        7: 'G',
        8: 'G♯, A♭',
        9: 'A',
        10: 'A♯, B♭',
        11: 'B',
    },
    'None': {
        None: 'Total mean',
    },
}

if 'spotify' not in st.session_state:
    st.session_state.spotify = spotify.SpotifyAPI()
if 'queried_song' not in st.session_state:
    st.session_state.queried_song = {'name': None, 'artist': None, 'album': None, 'release_date': None, 'preview_url': None, 'image_url': None, 'features': None}

with st.form('compare_songs'):
    st.header('Compare Songs')
    song_name = st.text_input('Enter song name', 'HUMBLE.', key='song_name')
    artist_name = st.text_input('Enter artist name', key='song_artist')
    album_name = st.text_input('Enter album name', key='song_album')
    submitted = st.form_submit_button('Search')

    if submitted:
        result_track, result_features = spotify.search_track(song_name, artist_name, album_name)
        if result_track['tracks']['items']:
            track = result_track['tracks']['items'][0]
            audio_features = result_features[0]
            st.session_state.queried_song['name'] = track['name']
            st.session_state.queried_song['artist'] = track['artists'][0]['name']
            st.session_state.queried_song['album'] = track['album']['name']
            st.session_state.queried_song['release_date'] = track['album']['release_date']
            st.session_state.queried_song['preview_url'] = track['preview_url']
            st.session_state.queried_song['image_url'] = track['album']['images'][0]['url']
            st.session_state.queried_song['features'] = np.array([audio_features[feature] for feature in features])
        else:
            st.session_state.queried_song['name'] = None
            st.session_state.queried_song['artist'] = None
            st.session_state.queried_song['album'] = None
            st.session_state.queried_song['release_date'] = None
            st.session_state.queried_song['preview_url'] = None
            st.session_state.queried_song['image_url'] = None
            st.session_state.queried_song['features'] = None

    if st.session_state.queried_song['name']:
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.queried_song['image_url'], width=200, use_column_width=True)
        with col2:
            st.write(f"Song name: {st.session_state.queried_song['name']}")
            st.write(f"Artist: {st.session_state.queried_song['artist']}")
            st.write(f"Album: {st.session_state.queried_song['album']}")
            st.write(f"Release date: {st.session_state.queried_song['release_date']}")
        if st.session_state.queried_song['preview_url']:
            url = st.session_state.queried_song['preview_url']
            spotify.play_audio(url, volume=0.2)
    else:
        st.write('Song not found')

split_feature = 'simplified_artist_genres'

glz_df = data_wrangling.read_data()

genres = st.multiselect(
    'Select genres to compare',
    ['pop', 'rock', 'punk', 'metal', 'mizrahi', 'mediterranean', 'hip hop', 'rap', 'blues', 'r&b', 'funk', 'soul', 'reggaeton', 'folk', 'country', 'dance', 'edm', 'trance', 'indie'],
    ['pop', 'country', 'soul', 'edm'],
    format_func=lambda genre: genre.title(),
    max_selections=5
)


graph = plotting.polar_graph(genres, split_feature, features)
st.plotly_chart(graph, use_container_width=True)


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