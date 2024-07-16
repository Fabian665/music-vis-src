import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from plotly.colors import n_colors
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from ast import literal_eval
from datetime import datetime, timedelta


@st.cache_data
def read_data():
    glz_df = pd.read_csv('/data/galgalaz_expanded.csv')

    glz_df['artist_genres'] = glz_df['artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    glz_df['simplified_artist_genres'] = glz_df['simplified_artist_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    glz_df['simplified_artist_israeli_genres'] = glz_df['simplified_artist_israeli_genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    glz_df['date'] = pd.to_datetime(glz_df['date'])
    glz_df['duration_dt'] = pd.to_datetime(glz_df['duration'], unit='ms')
    glz_df['year'] = glz_df['date'].dt.year
    return glz_df

@st.cache_data()
def genre_mask(genre):
    ret = glz_df['artist_genres'].apply(predicate, args=(genre, ))
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
def get_distinct_songs():
    return glz_df.sort_values(by='date').drop_duplicates(subset='track_name', keep='first')

@st.cache_data()
def get_polynomial():
    X = distinct_songs.date.apply(lambda x: x.timestamp()).values
    Y = distinct_songs.duration_dt.apply(lambda x: x.timestamp())
    return Polynomial.fit(X, Y, 1)

glz_df = read_data()

# Filter to only distinct songs by their first appearance
distinct_songs = get_distinct_songs()

p = get_polynomial()


def plot_scatter_song_length():

    fig = go.Figure()

    # Create Scatter trace
    fig.add_trace(go.Scatter(
        x=distinct_songs['date'],
        y=distinct_songs['duration_dt'],
        mode='markers',
        name='Song Length'
    ))

    x_range = distinct_songs['date'].agg(['min', 'max'])
    y_range = [datetime.fromtimestamp(p(x)) for x in x_range.apply(lambda x: x.timestamp()).tolist()]
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        line=dict(width=5)
    ))

    # Update layout
    fig.update_layout(
        title='Song Length Over Time',
        xaxis_title='Date',
        yaxis_title='Song Length (minutes:seconds)',
        xaxis_tickformatstops = [
            dict(dtickrange=[604800000, "M1"], value="%d/%m/%y"),
            dict(dtickrange=["M1", "M12"], value="%b %Y"),
            dict(dtickrange=["Y1", None], value="%Y")
        ],
        yaxis=dict(
            tickformat='%M:%S',
            hoverformat='%M:%S',
            range=[datetime(1970, 1, 1, 0, 0), distinct_songs['duration_dt'].max() + timedelta(seconds=15)]
        ),
        template='plotly_white',
    )

    fig

plot_scatter_song_length()