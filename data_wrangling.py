import streamlit as st
import numpy as np
import pandas as pd
from ast import literal_eval
from numpy.polynomial import Polynomial
from PIL import Image, ImageDraw
import requests
from io import BytesIO


@st.cache_data
def read_data():
    df = pd.read_csv('/data/galgalaz_expanded.csv')

    df['artist_genres'] = df['artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['simplified_artist_genres'] = df['simplified_artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['simplified_artist_israeli_genres'] = df['simplified_artist_israeli_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['date'] = pd.to_datetime(df['date'])
    df['duration_dt'] = pd.to_datetime(df['duration'], unit='ms')
    df['year'] = df['date'].dt.year
    return df

@st.cache_data
def genre_mask(df, genre):
    ret = df['artist_genres'].apply(predicate, args=(genre, ))
    return ret

def predicate(artist_genres, genre):
    return genre in artist_genres if isinstance(artist_genres, list) else False

@st.cache_data
def split_data(df, genres, split_feature, output_features):
    if split_feature == 'None':
        return {None: abs(df[output_features].values)}
    
    dfs = (df[genre_mask(genre)] for genre in genres)
    
    genre_feature_values = {genre: abs(df[output_features].values) for df, genre in zip(dfs, genres)}
    return genre_feature_values

@st.cache_data
def get_distinct_songs(df):
    distinct_songs = df.sort_values(by='date').drop_duplicates(subset='track_name', keep='first')
    X = distinct_songs.date.apply(lambda x: x.timestamp()).values
    Y = distinct_songs.duration_dt.apply(lambda x: x.timestamp())
    polynomial = Polynomial.fit(X, Y, 1)
    return distinct_songs, polynomial

@st.cache_data
def get_artist_song_count(df):
    artist_song_count = (
        df.groupby("main_artist_name")
        #   .agg(unique_tracks='nunique', chart_appearances='count')
        .agg({
            'main_artist': 'first',
            'track_name': ['nunique', 'count'],
        })
        .reset_index()
    )
    artist_song_count.columns = ['Artist', 'main_artist', 'unique_tracks', 'chart_appearances']
    artist_song_count['ratio'] = artist_song_count.apply(lambda x: x['chart_appearances'] / x['unique_tracks'], axis=1)
    del(artist_song_count['chart_appearances'])
    return artist_song_count

@st.cache_data
def get_track_count(df):
  track_count = (
      df.groupby(["main_artist_name", "track_name"])
      .size()
      .reset_index(name="track_count")
      .sort_values(by="track_count", ascending=False)
  )
  return track_count

@st.cache_data
def filter_dataframe(df, *args):
    filtered_df = df.copy()

    # Iterate through the filter arguments
    mask = np.all([get_filter_mask(df, arg) for arg in args], axis=0)
    filtered_df = filtered_df[mask]

    return filtered_df

@st.cache_data
def get_filter_mask(df, arg):
    col_name, criterion = arg
    if criterion is None:
        return np.ones(df.shape[0], dtype=bool)
    elif col_name == 'date':
        return (df['date'] >= criterion[0]) & (df['date'] <= criterion[1])
    elif col_name == 'rank':
        return df['rank'] <= criterion
    elif col_name == 'market':
        return df['market'] == criterion
    else:
        return df[col_name] == criterion

@st.cache_data
def search_track(song_name, artist_name=None, album_name=None):
    return st.session_state.spotify.search_track(song_name, artist_name, album_name)

@st.cache_data
def circle_image(image_url):
    response = requests.get(image_url).content
    image = Image.open(BytesIO(response))

    size = min(image.size)

    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    image.thumbnail((size, size))

    image = image.crop((0, 0, size, size))
    image.putalpha(mask) 

    return image

@st.cache_data
def get_date_range(df):
    return df['date'].agg(['min', 'max'])

@st.cache_data
def generate_bump_data(df, market, year, rank, date):
    filtered_data = filter_dataframe(
        df,
        ('market', market),
        ('year', year),
        ('rank', rank),
        ('date', date),
    )
    return filtered_data.pivot(columns=['track_name', 'track_uri', 'main_artist_name'], index=['date'], values='rank')

def score_bumpchart(col):
    col = col.to_numpy()
    date_penalize = 1.35 ** np.arange(len(col))
    return (-col + 11) * date_penalize
