import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from numpy.polynomial import Polynomial
from ast import literal_eval
from datetime import datetime, timedelta
from spotify import SpotifyAPI
from PIL import Image, ImageDraw
import requests
from streamlit.logger import get_logger
from io import BytesIO

logger = get_logger(__name__)

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

@st.cache_data
def get_date_range(df):
    return df['date'].agg(['min', 'max'])

@st.cache_data(show_spinner=False)
def plot_scatter_song_length(glz_df):
    distinct_songs, polynomial = get_distinct_songs(glz_df)
    text = distinct_songs.apply(lambda x: f"{x['track_name']} by {x['main_artist_name']}", axis=1)

    fig = go.Figure()

    # Create Scatter trace
    fig.add_trace(go.Scatter(
        x=distinct_songs['date'],
        y=distinct_songs['duration_dt'],
        mode='markers',
        name='Song',
        text=text,  # Add song names for hover text
        hovertemplate='%{text}<br>(%{x}, %{y})'  # Customize hover template
    ))

    x_range = get_date_range(distinct_songs)
    y_range = [datetime.fromtimestamp(polynomial(x)) for x in x_range.apply(lambda x: x.timestamp()).tolist()]

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        line=dict(width=5),
        name='Trend Line'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Song Duration Over Time',
            'x': 0.5,  # Centering the title
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='Song Duration (minutes:seconds)',
        xaxis_tickformatstops=[
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

st.plotly_chart(plot_scatter_song_length(glz_df))

with st.form('filter_selection'):
    st.header('Filter Selection')
    market_labels = {
        None: 'All Markets',
        'IL': 'Isreal',
        'INTL': 'International',
    }
    market = st.selectbox(
        'Market',
        [None, 'IL', 'INTL'],
        key='market',
        format_func=lambda x: market_labels[x],
    )
    rank = st.slider("Max rank", 1, 10, 5, 1, help='Will only filter for songs ranked better than this number (1 is the best)')
    min_date, max_date = get_date_range(glz_df)
    date = st.date_input(
        "Select your vacation for next year",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD",
    )
    if isinstance(date, tuple):
        date = (np.datetime64(date[0]), np.datetime64(date[1]))
    else:
        date = np.datetime64(date)
    st.form_submit_button('Filter', help='Click to filter the data', use_container_width=True)


@st.cache_data(show_spinner=False)
def plot_artist_stats(market, year, rank, date):
    market_data = filter_dataframe(
        read_data(),
        ('market', market),
        ('year', year),
        ('rank', rank),
        ('date', date),
    )
    # top_ranked_songs = market_data[market_data['rank'] <= rank]

    # Calculate the number of weeks each song stays in the top 10
    plot_df = get_artist_song_count(market_data)
    plot_df = plot_df.nlargest(10, 'unique_tracks')

    # Create dictionary of artists images
    uris = plot_df['main_artist'].tolist()

    # DONOT DELETE
    artist_photos = st.session_state['spotify'].get_artists_images(uris)  # dashboard
    # artist_photos = get_artists_images(uris)  # colab

    # Create a scatter plot for the number of different songs and average song time on the billboard
    fig = go.Figure()


    # Add dots for average weeks in top 10
    fig.add_trace(go.Scatter(
        x=plot_df['Artist'],
        y=plot_df['unique_tracks'],
        mode='markers+lines',
        name='Songs on Billboard',
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
        name='Average Weeks on Billboard',
        marker=dict(color='orange', size=10),
        hovertemplate='%{y:.2f} weeks'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Artist Impact ({market})<br><sup>Number of Songs and Average Song Time on Billboard for Top 10 Artists</sup>',
            'x': 0.5,  # Centering the title and subtitle
            'xanchor': 'center'
        },
        xaxis=dict(title='Artist'),
        yaxis=dict(title='Weeks', range=[0, 1.15 * max_x]),
        template='plotly_white',
        showlegend=True
    )


    return fig

@st.cache_data(show_spinner=False)
def plot_top_artists_with_songs(market, year, rank, date):
    market_data = filter_dataframe(
        glz_df,
        ('market', market),
        ('year', year),
        ('rank', rank),
        ('date', date),
    )

    # Get the top 5 artists
    data = get_track_count(market_data)
    top_5_artists = set(data.groupby('main_artist_name')['track_count'].sum().nlargest(5, keep='all').index)
    top_5_data = data[data['main_artist_name'].isin(top_5_artists)]

    # Create a stacked bar chart for the top 5 artists and their songs
    fig = px.bar(top_5_data, x='main_artist_name', y='track_count', color='track_name',
                 labels={'main_artist_name': 'Artist', 'track_count': f'Number of Times Ranked {rank} or higher', 'track_name': 'Song'},
                 title=f"Leading Artists in {year} ({market})")

    # Update layout for better readability and sort from largest to smallest
    fig.update_layout(template='plotly_white', xaxis=dict(categoryorder='total descending', tickangle=45))

    return fig

year = None

st.plotly_chart(plot_artist_stats(market, year, rank, date))
st.plotly_chart(plot_top_artists_with_songs(market, year, rank, date))