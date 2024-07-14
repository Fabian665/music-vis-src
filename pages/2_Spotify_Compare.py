import streamlit as st
from spotify import SpotifyAPI
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go


glz_df = pd.read_csv('/data/galgalaz_expanded.csv')

# List of features to include in the radar chart
features = ['track_danceability', 'track_energy', 'track_valence', 'track_tempo', 'track_loudness']
features_names = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
features_repeated = features + [features[0]]


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
            st.session_state.queried_song['features'] = np.array([audio_features[feature] for feature in features_names])
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
            st.write(f"Song name: {track['name']}")
            st.write(f"Artist: {track['artists'][0]['name']}")
            st.write(f"Album: {track['album']['name']}")
            st.write(f"Release date: {track['album']['release_date']}")
        if track['preview_url']:
            url = track['preview_url']
            play_audio(url, volume=0.2)
    else:
        st.write('Song not found')

glz_df['date'] = pd.to_datetime(glz_df['date'])

split_feature = 'market'
split_feature_unique_values = glz_df[split_feature].unique()

# Ensure the 'date' column is in datetime format

# Separate data based on split feature
data_slices = {value: abs(glz_df[glz_df[split_feature] == value][features].values) for value in split_feature_unique_values}

min_values = np.minimum.reduce([features_values.min(axis=0) for features_values in data_slices.values()])
max_values = np.maximum.reduce([features_values.max(axis=0) for features_values in data_slices.values()])

if st.session_state.queried_song['features'] is not None:
    res = abs(st.session_state.queried_song['features'])
    min_values = np.minimum(min_values, res)
    max_values = np.maximum(max_values, res)
    res = (res - min_values) / (max_values - min_values)
else:
    res = None

data_slices = {value: ((features_values - min_values) / (max_values - min_values)) for value, features_values in data_slices.items()}

# Create radar charts for IL and INTL
fig = go.Figure()

for value, features_values in data_slices.items():
    features_trace_values = features_values.mean(axis=0)
    features_trace_values =  np.concatenate((features_trace_values, np.array([features_trace_values[0]])))
    print(features_trace_values)
    print(features_repeated)
    fig.add_trace(go.Scatterpolar(
        r=features_trace_values,
        theta=features_repeated,
        name=value,
    ))

if res is not None:
    res_trace_values = np.concatenate((res, np.array([res[0]])))
    fig.add_trace(go.Scatterpolar(
        r=res_trace_values,
        theta=features_repeated,
        name='Res',
    ))

fig.update_traces(fill='toself')

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

fig