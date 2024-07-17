import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyAPI:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

    def search_track(self, song_name: str, artist_name: str = None, album_name: str = None):
        # query = f'track:{song_name}'
        # if artist_name:
        #     query += f' artist:{artist_name}'
        # if album_name:
        #     query += f' album:{album_name}'
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