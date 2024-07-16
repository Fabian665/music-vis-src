import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from numpy.polynomial import Polynomial
from ast import literal_eval
from datetime import datetime, timedelta
from spotify import SpotifyAPI
from PIL import Image, ImageDraw
import requests
from io import BytesIO


if 'spotify' not in st.session_state:
    st.session_state.spotify = SpotifyAPI()

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

@st.cache_data
def genre_mask(genre):
    ret = glz_df['artist_genres'].apply(predicate, args=(genre, ))
    return ret

def predicate(artist_genres, genre):
    return genre in artist_genres if isinstance(artist_genres, list) else False

@st.cache_data
def split_data(genres, split_feature, output_features):
    if split_feature == 'None':
        return {None: abs(glz_df[output_features].values)}
    
    dfs = (glz_df[genre_mask(genre)] for genre in genres)
    
    genre_feature_values = {genre: abs(df[output_features].values) for df, genre in zip(dfs, genres)}
    return genre_feature_values

@st.cache_data
def get_distinct_songs(glz_df):
    distinct_songs = glz_df.sort_values(by='date').drop_duplicates(subset='track_name', keep='first')
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
def filter_dataframe(df, *args):
    filtered_df = df.copy()

    # Iterate through the filter arguments
    mask = np.all([get_filter_mask(df, arg) for arg in args], axis=0)
    filtered_df = filtered_df[mask]

    return filtered_df

@st.cache_data
def get_filter_mask(df, arg):
    col_name, criterion = arg
    if col_name == 'date':
        return (df['date'] >= criterion[0]) & (df['date'] <= criterion[1])
    elif col_name == 'rank':
        return df['rank'] <= criterion
    elif col_name == 'market':
        if criterion in ['IL', 'INTL']:
            return df['market'] == criterion
        else:
            return np.ones(len(df), dtype=bool)
    else:
        return df[col_name] == criterion

@st.cache_data
def search_track(song_name, artist_name=None, album_name=None):
    return st.session_state.spotify.search_track(song_name, artist_name, album_name)


@st.cache_data
def filter_by_market(market):
    if market in ['IL', 'INTL']:
        return glz_df[glz_df['market'] == market]
    else:
        return glz_df

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

glz_df = read_data()


# p = get_polynomial()

@st.cache_data(show_spinner=False)
def plot_scatter_song_length(glz_df):
    distinct_songs, polynomial = get_distinct_songs(glz_df)

    fig = go.Figure()

    # Create Scatter trace
    fig.add_trace(go.Scatter(
        x=distinct_songs['date'],
        y=distinct_songs['duration_dt'],
        mode='markers',
        name='Song Length'
    ))

    x_range = distinct_songs['date'].agg(['min', 'max'])
    y_range = [datetime.fromtimestamp(polynomial(x)) for x in x_range.apply(lambda x: x.timestamp()).tolist()]
    
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

    return fig

@st.cache_data(show_spinner=False)
def plot_artist_stats(market, year, rank):
    market_data = filter_dataframe(
        glz_df,
        ('market', market),
        ('year', year),
        ('rank', rank)
    )

    top_ranked_songs = market_data[market_data['rank'] <= rank]

    # Calculate the number of weeks each song stays in the top 10
    plot_df = get_artist_song_count(top_ranked_songs)
    plot_df = plot_df.nlargest(10, 'unique_tracks')

    # Create dictionary of artists images
    uris = plot_df['main_artist'].tolist()
    artist_photos = st.session_state['spotify'].get_artists_images(uris)

    # Create a scatter plot for the number of different songs and average song time on the billboard
    fig = go.Figure()

    
    # Add dots for average weeks in top 10
    fig.add_trace(go.Scatter(
        x=plot_df['Artist'],
        y=plot_df['unique_tracks'],
        mode='markers+lines',
        name='Average Weeks in Top 10',
        marker=dict(color='blue', size=10)
    ))

    max_x = plot_df[['unique_tracks', 'ratio']].max().max()

    # Add dots for number of different songs with artist photos
    for index, row in plot_df.iterrows():
        if row['Artist'] in artist_photos:
            photo_url = artist_photos[row['Artist']]
            image = circle_image(photo_url)
            fig.add_layout_image(
                dict(
                    source=image,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=row['Artist'],
                    y=row['unique_tracks'],
                    sizex=max_x / 6.5,
                    sizey=max_x / 6.5,
                    sizing="contain",
                    layer="above"
                )
            )

    # Add dots for average weeks in top 10
    fig.add_trace(go.Scatter(
        x=plot_df['Artist'],
        y=plot_df['ratio'],
        mode='markers+lines',
        name='Average Weeks in Top 10',
        marker=dict(color='orange', size=10)
    ))

    # Update layout
    fig.update_layout(
        title=f'Number of Different Songs and Average Song Time in Billboard for Top 10 Artists ({market})',
        xaxis=dict(title='Artist'),
        yaxis=dict(title='Count / Weeks in Top 10', range=[0, 1.15 * max_x]),
        template='plotly_white',
        showlegend=True
    )

    return fig

st.plotly_chart(plot_scatter_song_length(glz_df))
st.plotly_chart(plot_artist_stats('INTL', 2018, 5))
