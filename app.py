import streamlit as st
from spotify import SpotifyAPI


if 'spotify' not in st.session_state:
    st.session_state.spotify = SpotifyAPI()
if 'queried_song' not in st.session_state:
    st.session_state.queried_song = {'name': None, 'artist': None, 'album': None, 'release_date': None, 'preview_url': None}

@st.cache_data
def search_track(song_name, artist_name=None, album_name=None):
    return st.session_state.spotify.search_track(song_name, artist_name, album_name)

with st.form('search_song'):
    song_name = st.text_input('Enter song name', 'HUMBLE.')
    artist_name = st.text_input('Enter artist name')
    album_name = st.text_input('Enter album name')
    submitted = st.form_submit_button('Search song')

if submitted:
    result = search_track(song_name, artist_name, album_name)
    if result['tracks']['items']:
        track = result['tracks']['items'][0]
        st.session_state.queried_song['name'] = track['name']
        st.session_state.queried_song['artist'] = track['artists'][0]['name']
        st.session_state.queried_song['album'] = track['album']['name']
        st.session_state.queried_song['release_date'] = track['album']['release_date']
        st.session_state.queried_song['preview_url'] = track['preview_url']
    else:
        st.session_state.queried_song['name'] = None
        st.session_state.queried_song['artist'] = None
        st.session_state.queried_song['album'] = None
        st.session_state.queried_song['release_date'] = None
        st.session_state.queried_song['preview_url'] = None
        
if st.session_state.queried_song['name']:
    st.write(f"Song name: {track['name']}")
    st.write(f"Artist: {track['artists'][0]['name']}")
    st.write(f"Album: {track['album']['name']}")
    st.write(f"Release date: {track['album']['release_date']}")
    if track['preview_url']:
        st.audio(track['preview_url'], format='audio/mp3', autoplay=True)
else:
    st.write('Song not found')
