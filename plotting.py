import data_wrangling
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd


@st.cache_data(show_spinner=False)
def plot_scatter_song_length(glz_df):
    distinct_songs, polynomial = data_wrangling.get_distinct_songs(glz_df)

    fig = go.Figure()

    # Create Scatter trace
    fig.add_trace(go.Scatter(
        x=distinct_songs['date'],
        y=distinct_songs['duration_dt'],
        mode='markers',
        name='Song',
        text=distinct_songs['track_name'],  # Add song names for hover text
        hovertemplate='%{text}<br>(%{x}, %{y})',  # Customize hover template
        marker=dict(color='#F6B8B8')  # First color for scatter points
    ))

    x_range = data_wrangling.get_date_range(distinct_songs)
    y_range = [datetime.fromtimestamp(polynomial(x)) for x in x_range.apply(lambda x: x.timestamp()).tolist()]

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        line=dict(width=5, color='#A89CFF'),  # Darker color for trend line
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
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        )
    )

    return fig

@st.cache_data(show_spinner=False)
def plot_artist_stats(market, year, rank, date):
    market_data = data_wrangling.filter_dataframe(
        data_wrangling.read_data(),
        ('market', market),
        ('year', year),
        ('rank', rank),
        ('date', date),
    )

    # Calculate the number of weeks each song stays in the top 10
    plot_df = data_wrangling.get_artist_song_count(market_data)
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
        mode='markers',
        name='Songs on Billboard',
        marker=dict(color='#F6B8B8', size=10)
    ))

    max_x = plot_df[['unique_tracks', 'ratio']].max().max()

    # Add dots for number of different songs with artist photos
    for index, row in plot_df.iterrows():
        if row['Artist'] in artist_photos:
            photo_url = artist_photos[row['Artist']]
            image = data_wrangling.circle_image(photo_url)
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
        marker=dict(color='#D7CCF6', size=10)
    ))

    # Determine the time period text
    if year:
        time_period = f"in {year}"
    else:
        start_date, end_date = date
        time_period = f"from {pd.to_datetime(start_date).strftime('%d.%m.%Y')} to {pd.to_datetime(end_date).strftime('%d.%m.%y')}"

    # Update layout
    fig.update_layout(
        title={
            'text': f'Artist Impact ({market})<br><sup>Number of Songs and Average Song Time on Billboard for Top 10 Artists {time_period}</sup>',
            'x': 0.5,  # Centering the title and subtitle
            'xanchor': 'center'
        },
        xaxis=dict(title='Artist'),
        yaxis=dict(title='Weeks', range=[0, 1.15 * max_x]),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        )
    )

    return fig


@st.cache_data(show_spinner=False)
def plot_top_artists_with_songs(market, year, rank, date):
    market_data = data_wrangling.filter_dataframe(
        data_wrangling.read_data(),
        ('market', market),
        ('year', year),
        ('rank', rank),
        ('date', date),
    )

    # Get the top 5 artists
    data = data_wrangling.get_track_count(market_data)
    top_artists = list(data.groupby('main_artist_name')['track_count'].sum().nlargest(5, keep='all').index)
    top_data = data[data['main_artist_name'].isin(top_artists)]

    # Determine y-axis title and subtitle
    if rank is None:
        yaxis_title = 'Times on Billboard'
    else:
        yaxis_title = f'Times Ranked #{rank} or Higher'

    if year:
        title_text = f"Hottest Artists in {year} ({market})"
        subtitle = f"Artists with most #{rank} or Higher Song Rankings and their Songs in {year}"
    else:
        start_date, end_date = date
        title_text = f"Hottest Artists from {pd.to_datetime(start_date).strftime('%d.%m.%Y')} to {pd.to_datetime(end_date).strftime('%d.%m.%Y')} ({market})"
        subtitle = f"Artists with Most #{rank} or Higher Song Rankings and their Songs from {pd.to_datetime(start_date).strftime('%d.%m.%Y')} to {pd.to_datetime(end_date).strftime('%d.%m.%Y')}"

    # Create a stacked bar chart for the top 5 artists and their songs
    fig = px.bar(top_data, x='main_artist_name', y='track_count', color_discrete_sequence=['#A89CFF'],
                 labels={'main_artist_name': 'Artist', 'track_count': yaxis_title, 'track_name': 'Song'},
                 title=f"{title_text}<br><sup>{subtitle}</sup>", hover_data=['track_name']
                 )

    # Update layout for better readability and sort from largest to smallest
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(categoryorder='total descending', tickangle=45),
        title={
            'text': f"{title_text}<br><sup>{subtitle}</sup>",
            'x': 0.5,  # Centering the title and subtitle
            'xanchor': 'center',
        },
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        )
    )

    return fig