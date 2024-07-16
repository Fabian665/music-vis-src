import streamlit as st
from spotify import SpotifyAPI
import streamlit.components.v1 as components
from streamlit.logger import get_logger
from ast import literal_eval
import pandas as pd
import numpy as np
import plotly.graph_objects as go

logger = get_logger(__name__)

@st.cache_data
def read_data():
    glz_df = pd.read_csv('/data/galgalaz_expanded.csv')

    glz_df['artist_genres'] = glz_df['artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    glz_df['simplified_artist_genres'] = glz_df['simplified_artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    glz_df['simplified_artist_israeli_genres'] = glz_df['simplified_artist_israeli_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    return glz_df

@st.cache_data()
def genre_mask(genre):
    ret = glz_df[split_feature].apply(predicate, args=(genre, ))
    return ret

def predicate(artist_genres, genre):
    return genre in artist_genres if isinstance(artist_genres, list) else False

@st.cache_data()
def split_data(genres, split_feature, output_features):
    if split_feature == 'None':
        return {None: abs(glz_df[output_features].values)}
    
    dfs = (glz_df[genre_mask(genre)] for genre in genres)
    
    genre_feature_values = {genre: abs(df[output_features].values) for df, genre in zip(dfs, genres)}
    return genre_feature_values

@st.cache_data()
def data_scale_values(data_slices):
    if len(data_slices) == 1:
        key = next(iter(data_slices))
        return data_slices[key].min(axis=0), data_slices[key].max(axis=0)
    min_values = np.minimum.reduce([features_values.min(axis=0) for features_values in data_slices.values()])
    max_values = np.maximum.reduce([features_values.max(axis=0) for features_values in data_slices.values()])
    return min_values, max_values

st.cache_data()
def get_mean_of_features(data):
    return data.mean(axis=0)

# List of features to include in the radar chart
features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
features_repeated = features + [features[0]]

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
    st.session_state.spotify = SpotifyAPI()
if 'queried_song' not in st.session_state:
    st.session_state.queried_song = {'name': None, 'artist': None, 'album': None, 'release_date': None, 'preview_url': None, 'image_url': None, 'features': None}

@st.cache_data
def search_track(song_name, artist_name=None, album_name=None):
    track_res = st.session_state.spotify.search_track(song_name, artist_name, album_name)
    track_uri = track_res['tracks']['items'][0]['uri']
    features_res = st.session_state.spotify.get_features(track_uri)
    return track_res, features_res

def play_audio(url, volume=0.5):
    # Create a unique key for the component based on the URL
    component_key = str(hash(url))

    # HTML code for the audio player with volume control
    html = f"""
        <audio id="{component_key}" controls autoplay>
            <source src="{url}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        <script>
            var audio = document.getElementById("{component_key}");
            audio.volume = {volume};
        </script>
    """

    # Create the Streamlit component
    components.html(html, height=50)

with st.form('compare_songs'):
    st.header('Compare Songs')
    song_name = st.text_input('Enter song name', 'HUMBLE.', key='song_name')
    artist_name = st.text_input('Enter artist name', key='song_artist')
    album_name = st.text_input('Enter album name', key='song_album')
    submitted = st.form_submit_button('Search')

    if submitted:
        result_track, result_features = search_track(song_name, artist_name, album_name)
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
            play_audio(url, volume=0.2)
    else:
        st.write('Song not found')

split_feature = 'simplified_artist_genres'

glz_df = read_data()

genres = st.multiselect(
    'Select genres to compare',
    ['pop', 'rock', 'punk', 'metal', 'mizrahi', 'mediterranean', 'hip hop', 'rap', 'blues', 'r&b', 'funk', 'soul', 'reggaeton', 'folk', 'country', 'dance', 'edm', 'trance', 'indie'],
    ['pop', 'country', 'soul', 'edm'],
    format_func=lambda genre: genre.title(),
    max_selections=5
)

def polar_graph(genres, split_feature, output_features):
    data_slices = split_data(genres, split_feature, output_features)

    min_values, max_values = data_scale_values(data_slices)

    if st.session_state.queried_song['features'] is not None:
        res = abs(st.session_state.queried_song['features'])
        min_values = np.minimum(min_values, res)
        max_values = np.maximum(max_values, res)
        res = (res - min_values) / (max_values - min_values)
    else:
        res = None

    data_slices = {value: ((get_mean_of_features(features_values) - min_values) / (max_values - min_values)) for value, features_values in data_slices.items()}

    # Create radar charts for IL and INTL
    fig = go.Figure()

    for value, features_values in data_slices.items():
        features_trace_values = features_values
        features_trace_values =  np.concatenate((features_trace_values, np.array([features_trace_values[0]])))
        name = value.title()
        fig.add_trace(go.Scatterpolar(
            r=features_trace_values,
            theta=features_repeated,
            name=name,
            # fill='toself',
        ))

    if res is not None:
        res_trace_values = np.concatenate((res, np.array([res[0]])))
        song_name = st.session_state.queried_song['name']
        artist_name = st.session_state.queried_song['artist']
        fig.add_trace(go.Scatterpolar(
            r=res_trace_values,
            theta=features_repeated,
            name=f'{song_name} by {artist_name}',
            line=dict(color='red'),
            marker=dict(color='red'),
            # fill='toself',
        ))

    # fig.update_traces(fill='toself')

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Radar Chart of Mean Feature Values for IL and INTL'
    )

    return fig

graph = polar_graph(genres, split_feature, features)
st.plotly_chart(graph, use_container_width=True)