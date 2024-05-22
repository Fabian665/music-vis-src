import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyAPI:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

    def search_track(self, song_name: str, artist_name: str = None, album_name: str = None):
        query = f'track:{song_name}'
        if artist_name:
            query += f' artist:{artist_name}'
        if album_name:
            query += f' album:{album_name}'
        return self.sp.search(query, type='track', limit=1)