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
def get_distinct_songs(glz_df):
    distinct_songs = glz_df.sort_values(by='date').drop_duplicates(subset='track_name', keep='first')
    X = distinct_songs.date.apply(lambda x: x.timestamp()).values
    Y = distinct_songs.duration_dt.apply(lambda x: x.timestamp())
    polynomial = Polynomial.fit(X, Y, 1)
    return distinct_songs, polynomial

# @st.cache_data()
# def get_polynomial(distinctint_songs):
#     X = distinct_songs.date.apply(lambda x: x.timestamp()).values
#     Y = distinct_songs.duration_dt.apply(lambda x: x.timestamp())
#     return Polynomial.fit(X, Y, 1)

@st.cache_data()
def get_top_10_artists(market):
    market_data = filter_by_market(market)
    # Calculate the number of weeks each song stays in the top 10
    market_data['week_count'] = market_data.groupby('track_name')['date'].transform(lambda x: x.nunique())
    # Calculate the total number of appearances for each artist
    artist_appearances = market_data.groupby('main_artist').size().reset_index(name='count')
    return artist_appearances.nlargest(10, 'count')['main_artist']

@st.cache_data
def search_track(song_name, artist_name=None, album_name=None):
    return st.session_state.spotify.search_track(song_name, artist_name, album_name)


@st.cache_data()
def filter_by_market(market):
    if market in ['IL', 'INTL']:
        return glz_df[glz_df['market'] == market]
    else:
        return glz_df

@st.cache_data()
def circle_image(image_url):
    response = requests.get(image_url).content
    image = Image.open(BytesIO(response))

    # Find the smaller dimension for a square crop
    size = min(image.size)

    # Create a new square image to hold the circle
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    # Resize the original image to the square size
    image.thumbnail((size, size))

    # Crop the original image using the square mask
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
def plot_artist_stats(market):
    market_data = filter_by_market(market)

    # Calculate the number of weeks each song stays in the top 10
    market_data['week_count'] = market_data.groupby('track_name')['date'].transform(lambda x: x.nunique())

    # Calculate the total number of appearances for each artist
    artist_appearances = market_data.groupby('main_artist').size().reset_index(name='count')

    # Get the top 10 artists with the most appearances
    top_10_artists = artist_appearances.nlargest(10, 'count')['main_artist']

    # Filter the data to include only the top 10 artists
    top_10_df = market_data[market_data['main_artist'].isin(top_10_artists)]

    # Drop duplicates to get one entry per song per artist
    duration_df = top_10_df.drop_duplicates(subset=['track_name', 'main_artist'])

    # Calculate the average weeks in top 10 for each artist
    average_weeks = duration_df.groupby('main_artist').agg({'main_artist_name': 'first', 'week_count': 'mean'}).reset_index()
    average_weeks.columns = ['URI', 'Artist', 'Average Weeks in Top 10']

    # Calculate the number of different songs each artist has had on the billboard
    unique_songs = duration_df.groupby('main_artist').agg({'main_artist_name': 'first', 'track_name': 'nunique'}).reset_index()
    unique_songs.columns = ['URI', 'Artist', 'Number of Different Songs']

    # Merge the dataframes
    merged_df = pd.merge(average_weeks, unique_songs, on='Artist')
    merged_df = merged_df.sort_values(by='Number of Different Songs', ascending=False)

    # Create dictionary of artists images
    uris = top_10_artists.tolist()
    artist_photos = st.session_state['spotify'].get_artists_images(uris)

    # Create a scatter plot for the number of different songs and average song time on the billboard
    fig = go.Figure()

    
    # Add dots for average weeks in top 10
    fig.add_trace(go.Scatter(
        x=merged_df['Artist'],
        y=merged_df['Number of Different Songs'],
        mode='markers+lines',
        name='Average Weeks in Top 10',
        marker=dict(color='blue', size=10)
    ))

    # Add dots for number of different songs with artist photos
    for index, row in merged_df.iterrows():
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
                    y=row['Number of Different Songs'],
                    sizex=3.5,
                    sizey=3.5,
                    sizing="contain",
                    layer="above"
                )
            )

    # Add dots for average weeks in top 10
    fig.add_trace(go.Scatter(
        x=merged_df['Artist'],
        y=merged_df['Average Weeks in Top 10'],
        mode='markers+lines',
        name='Average Weeks in Top 10',
        marker=dict(color='orange', size=10)
    ))

    # Update layout
    fig.update_layout(
        title=f'Number of Different Songs and Average Song Time in Billboard for Top 10 Artists ({market})',
        xaxis=dict(title='Artist'),
        yaxis=dict(title='Count / Weeks in Top 10', range=[0, merged_df[['Number of Different Songs', 'Average Weeks in Top 10']].max().max() + 2]),
        template='plotly_white',
        showlegend=True
    )

    return fig

st.plotly_chart(plot_scatter_song_length(glz_df))
st.plotly_chart(plot_artist_stats('all'))
