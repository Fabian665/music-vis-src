import streamlit as st
import numpy as np
import pandas as pd
from ast import literal_eval
from numpy.polynomial import Polynomial
from PIL import Image, ImageDraw
import requests
from io import BytesIO
try:
    from st_files_connection import FilesConnection
except ModuleNotFoundError:
    pass


@st.cache_data(show_spinner=False)
def read_data():
    try:
        df = pd.read_csv('/data/galgalaz_expanded.csv')
    except FileNotFoundError:
        conn = st.connection('gcs', type=FilesConnection)
        df = conn.read("music-vis-data/galgalaz_2024_07_18.csv", input_format='csv')

    df['artist_genres'] = df['artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['simplified_artist_genres'] = df['simplified_artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['simplified_artist_israeli_genres'] = df['simplified_artist_israeli_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['date'] = pd.to_datetime(df['date'])
    df['duration_dt'] = pd.to_datetime(df['duration'], unit='ms')
    df['year'] = df['date'].dt.year
    return df

@st.cache_data(show_spinner=False)
def genre_mask(df, genre):
    ret = df['artist_genres'].apply(predicate, args=(genre, ))
    return ret

def predicate(artist_genres, genre):
    return genre in artist_genres if isinstance(artist_genres, list) else False

@st.cache_data(show_spinner=False)
def split_data(df, genres, split_feature, output_features):
    if split_feature == 'None':
        return {None: abs(df[output_features].values)}
    
    dfs = (df[genre_mask(genre)] for genre in genres)
    
    genre_feature_values = {genre: abs(df[output_features].values) for df, genre in zip(dfs, genres)}
    return genre_feature_values

@st.cache_data(show_spinner=False)
def get_distinct_songs(df):
    distinct_songs = drop_duplicate_songs(df)
    X = distinct_songs.date.apply(lambda x: x.timestamp()).values
    Y = distinct_songs.duration_dt.apply(lambda x: x.timestamp())
    polynomial = Polynomial.fit(X, Y, 1)
    return distinct_songs, polynomial

@st.cache_data(show_spinner=False)
def get_artist_song_count(df):
    artist_song_count = (
        df.groupby("main_artist_name")
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

@st.cache_data(show_spinner=False)
def get_track_count(df):
  track_count = (
      df.groupby(["main_artist_name", "track_name"])
      .size()
      .reset_index(name="track_count")
      .sort_values(by="track_count", ascending=False)
  )
  return track_count

@st.cache_data(show_spinner=False)
def filter_dataframe(df, *args):
    filtered_df = df.copy()

    # Iterate through the filter arguments
    mask = np.all([get_filter_mask(df, arg) for arg in args], axis=0)
    filtered_df = filtered_df[mask]

    return filtered_df

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def search_track(song_name, artist_name=None, album_name=None):
    return st.session_state.spotify.search_track(song_name, artist_name, album_name)

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def get_date_range(df):
    return df['date'].agg(['min', 'max'])

@st.cache_data(show_spinner=False)
def generate_bump_data(df, market, date):
    filtered_data = filter_dataframe(
        df,
        ('market', market),
        ('date', date),
    )
    return filtered_data.pivot(columns=['track_name', 'track_uri', 'main_artist_name'], index=['date'], values='rank')

def score_bumpchart(col):
    col = col.to_numpy()
    date_penalize = 1.35 ** np.arange(len(col))
    return (-col + 11) * date_penalize


@st.cache_data(show_spinner=False)
def time_signature_distribution(df):
    time_signature_counts = drop_duplicate_songs(df)['time_signature'].value_counts()

    # Create a new DataFrame to categorize time signatures into '4/4' and 'Other'
    time_signature_distribution = pd.DataFrame({
        'Category': ['4/4', 'Other'],
        'Count': [
            time_signature_counts.get(4, 0),
            time_signature_counts.drop(labels=4, errors='ignore').sum()
        ]
    })

    total_songs = time_signature_distribution['Count'].sum()
    time_signature_distribution['Percentage'] = (time_signature_distribution['Count'] / total_songs) * 100
    return time_signature_distribution


@st.cache_data(show_spinner=False)
def drop_duplicate_songs(df):
    return df.sort_values(by='date').drop_duplicates(subset='track_uri', keep='first')


@st.cache_data(show_spinner=False)
def mode_distribution(df):
    mode_distribution = drop_duplicate_songs(df).groupby(['year', 'market', 'mode']).size().reset_index(name='count')

    mode_labels = {0: 'Minor', 1: 'Major'}
    mode_distribution['mode'] = mode_distribution['mode'].map(mode_labels)

    mode_distribution['proportion'] = mode_distribution.groupby(['year', 'market'])['count'].transform(lambda x: x / x.sum())
    return mode_distribution


@st.cache_data(show_spinner=False)
def genre_trends(df, market):
    market_data = filter_dataframe(
        df,
        ('market', market),
    )

    # Drop rows where 'simplified_artist_genres' is not a list
    market_data = market_data[market_data['simplified_artist_genres'].apply(lambda x: isinstance(x, list))]

    # Explode the simplified_artist_genres column to create a row for each genre
    exploded_genres_df = market_data.explode('simplified_artist_genres')

    # Count the number of songs per genre per year
    genre_trends = exploded_genres_df.groupby(['year', 'simplified_artist_genres']).size().reset_index(name='count')

    # Calculate the total count of all genres per year
    total_genre_counts_per_year = genre_trends.groupby('year')['count'].sum().reset_index(name='total_genre_count')

    # Merge the dataframes to get the total count for each year
    genre_trends = pd.merge(genre_trends, total_genre_counts_per_year, on='year')

    # Calculate the relative proportion of each genre using the total genre counts
    genre_trends['proportion'] = genre_trends['count'] / genre_trends['total_genre_count']

    # Filter to top 6 genres by total count
    top_genres = genre_trends.groupby('simplified_artist_genres')['count'].sum().nlargest(6).index
    genre_trends['simplified_artist_genres'] = genre_trends['simplified_artist_genres'].apply(
        lambda x: x if x in top_genres else 'Other'
    )

    # Recalculate the proportion for the modified genre categories
    genre_trends = genre_trends.groupby(['year', 'simplified_artist_genres']).agg({'count': 'sum', 'total_genre_count': 'first'}).reset_index()
    genre_trends['proportion'] = genre_trends['count'] / genre_trends['total_genre_count']

    return genre_trends


@st.cache_data(show_spinner=False)
def text_stats(df):
    # Calculate total number of unique artists
    total_unique_artists = df['main_artist_name'].nunique()

    # Calculate the top artist by cumulative weeks on chart
    artist_weeks_on_chart = df.groupby('main_artist_name')['date'].nunique().reset_index(name='weeks_on_chart')
    top_artist = artist_weeks_on_chart.nlargest(1, 'weeks_on_chart').iloc[0]

    # Filter the data to get only the songs that ranked number one
    number_one_hits = df[df['rank'] == 1]

    # Calculate the top song by cumulative weeks at number one
    song_weeks_at_number_one = number_one_hits.groupby('track_name')['date'].nunique().reset_index(name='weeks_at_number_one')
    top_song = song_weeks_at_number_one.nlargest(1, 'weeks_at_number_one').iloc[0]
    
    time_signatures = df.drop_duplicates('track_uri')['time_signature'].value_counts()
    time_signatures = time_signatures.loc[4] / time_signatures.sum()

    return total_unique_artists, top_artist, top_song, time_signatures

@st.cache_data()
def genre_mask(df, genre, split_feature):
    ret = df[split_feature].apply(predicate, args=(genre, ))
    return ret

def predicate(artist_genres, genre):
    return genre in artist_genres if isinstance(artist_genres, list) else False

@st.cache_data()
def split_data(df, genres, split_feature, output_features):
    if split_feature == 'None':
        return {None: abs(df[output_features].values)}
    
    dfs = (df[genre_mask(df, genre, split_feature)] for genre in genres)
    
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

@st.cache_data
def get_mean_of_features(data):
    return data.mean(axis=0)

@st.cache_data(show_spinner=False)
def top_hits_data(glz_df, market, rank, sort_by):
    if rank == 1:
        column_name = 'Weeks at Number One'
    elif rank == 10:
        column_name = 'Total Weeks in Billboard'
    
    market_data = filter_dataframe(glz_df, ('market', market), ('rank', rank))

    weeks_at_number_one = market_data.groupby(['year', 'track_name', 'main_artist_name']).size().reset_index(name=column_name)
    top_hits = weeks_at_number_one.groupby('year').apply(lambda x: x.nlargest(1, column_name)).reset_index(drop=True)

    top_hits['track_name_wrapped'] = top_hits['track_name']
    if sort_by == 'year':
        top_hits = top_hits.sort_values(by='year', ascending=True)
    else:
        top_hits = top_hits.sort_values(by=column_name, ascending=False)
    return top_hits, column_name
