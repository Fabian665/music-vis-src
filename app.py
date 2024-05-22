import streamlit as st
from spotify import SpotifyAPI
import streamlit.components.v1 as components


if 'spotify' not in st.session_state:
    st.session_state.spotify = SpotifyAPI()
if 'queried_song1' not in st.session_state:
    st.session_state.queried_song1 = {'name': None, 'artist': None, 'album': None, 'release_date': None, 'preview_url': None}
if 'queried_song2' not in st.session_state:
    st.session_state.queried_song2 = {'name': None, 'artist': None, 'album': None, 'release_date': None, 'preview_url': None}

@st.cache_data
def search_track(song_name, artist_name=None, album_name=None):
    return st.session_state.spotify.search_track(song_name, artist_name, album_name)

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
    col1, col2 = st.columns(2)
    with col1:
        st.header('Song #1')
        song_name1 = st.text_input('Enter song name', 'HUMBLE.', key='song1_name')
        artist_name1 = st.text_input('Enter artist name', key='song1_artist')
        album_name1 = st.text_input('Enter album name', key='song1_album')
    with col2:
        st.header('Song #2')
        song_name2 = st.text_input('Enter song name', 'I Like the Way You Kiss Me', key='song2_name')
        artist_name2 = st.text_input('Enter artist name', key='song2_artist')
        album_name2 = st.text_input('Enter album name', key='song2_album')
    submitted = st.form_submit_button('Search')

    if submitted:
        result1 = search_track(song_name1, artist_name1, album_name1)
        result2 = search_track(song_name2, artist_name2, album_name2)
        if result1['tracks']['items'] and result2['tracks']['items']:
            track1 = result1['tracks']['items'][0]
            st.session_state.queried_song1['name'] = track1['name']
            st.session_state.queried_song1['artist'] = track1['artists'][0]['name']
            st.session_state.queried_song1['album'] = track1['album']['name']
            st.session_state.queried_song1['release_date'] = track1['album']['release_date']
            st.session_state.queried_song1['preview_url'] = track1['preview_url']
            st.session_state.queried_song1['image_url'] = track1['album']['images'][0]['url']

            track2 = result2['tracks']['items'][0]
            st.session_state.queried_song2['name'] = track2['name']
            st.session_state.queried_song2['artist'] = track2['artists'][0]['name']
            st.session_state.queried_song2['album'] = track2['album']['name']
            st.session_state.queried_song2['release_date'] = track2['album']['release_date']
            st.session_state.queried_song2['preview_url'] = track2['preview_url']
            st.session_state.queried_song2['image_url'] = track2['album']['images'][0]['url']
        else:
            st.session_state.queried_song1['name'] = None
            st.session_state.queried_song1['artist'] = None
            st.session_state.queried_song1['album'] = None
            st.session_state.queried_song1['release_date'] = None
            st.session_state.queried_song1['preview_url'] = None
            st.session_state.queried_song1['image_url'] = None

            st.session_state.queried_song2['name'] = None
            st.session_state.queried_song2['artist'] = None
            st.session_state.queried_song2['album'] = None
            st.session_state.queried_song2['release_date'] = None
            st.session_state.queried_song2['preview_url'] = None
            st.session_state.queried_song2['image_url'] = None
            
    if st.session_state.queried_song1['name']:
        with col1:
            col11, col12 = st.columns(2)
            with col11:
                st.image(st.session_state.queried_song1['image_url'], width=200, use_column_width=True)
            with col12:
                st.write(f"Song name: {track1['name']}")
                st.write(f"Artist: {track1['artists'][0]['name']}")
                st.write(f"Album: {track1['album']['name']}")
                st.write(f"Release date: {track1['album']['release_date']}")
        with col2:
            col21, col22 = st.columns(2)
            with col21:
                st.image(st.session_state.queried_song2['image_url'], width=200, use_column_width=True)
            with col22:
                st.write(f"Song name: {track2['name']}")
                st.write(f"Artist: {track2['artists'][0]['name']}")
                st.write(f"Album: {track2['album']['name']}")
                st.write(f"Release date: {track2['album']['release_date']}")
        if track1['preview_url'] or track2['preview_url']:
            url = track1['preview_url'] if track1['preview_url'] else track2['preview_url']
            # st.audio(url, format='audio/mp3', autoplay=True)
            play_audio(url, volume=0.2)
    else:
        st.write('Song not found')
