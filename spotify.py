import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyAPI:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

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