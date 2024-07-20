import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import streamlit.components.v1 as components


class SpotifyAPI:
    def __init__(self):
        if 'spotipy' not in st.secrets:
            # for local development
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
        else:
            # for deployment
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=st.secrets['spotipy']['client_id'],
                client_secret=st.secrets['spotipy']['client_secret']
            ))

    def search_track(self, song_name: str, artist_name: str = None, album_name: str = None):
        query = f'{song_name} {artist_name} {album_name}'
        return self.sp.search(query, type='track', limit=1, market='IL')
    
    def get_features(self, uri: str):
        return self.sp.audio_features(uri)

    def get_artists_images(self, artists_uris):
        images = {}
        response = self.sp.artists(artists_uris)
        for artist in response['artists']:
            images[artist['name']] = artist['images'][-1]['url']
        return images
    
    def get_songs_images(self, songs_uris):
        images = {}
        response = self.sp.tracks(songs_uris)
        for song_uri, song in zip(songs_uris, response['tracks']):
            images[song_uri] = song['album']['images'][-2]['url']
        return images
    

@st.cache_data
def search_track(song_name, artist_name=None, album_name=None):
    track_res = st.session_state.spotify.search_track(song_name, artist_name, album_name)
    track_uri = track_res['tracks']['items'][0]['uri']
    features_res = st.session_state.spotify.get_features(track_uri)
    return track_res, features_res

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