import data_wrangling
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.colors as colors
import numpy as np



genre_color_map = {
    'pop': '#FFAB80',  # Orange
    'hip hop': '#FB8C8C',  # Red
    'mediterranean': '#A8E6CF',  # Green
    'mizrahi': '#D7CCF6',  # Purple
    'rock': '#B8DFF6',  # Blue
    'rap': '#FF8C8C',  # Pink
    'punk': '#FBB4B4',  # Light Pink
    'metal': '#A89CFF',  # Light Purple
    'blues': '#CCEFFF',  # Light Blue
    'r&b': '#F6CCF6',  # Light Lavender
    'funk': '#FFD3B6',  # Light Peach
    'soul': '#FFD6BB',  # Peach
    'reggaeton': '#CCFFEA',  # Light Green
    'folk': '#FFB4FF',  # Light Lavender
    'country': '#B0FF80',  # Light Green
    'dance': '#99FFC8',  # Light Green
    'edm': '#E89CFF',  # Light Purple
    'trance': '#B8E0FF',  # Light Blue
    'indie': '#A8E6F6',  # Light Blue
    'Other': '#CAB2D6'  # Light Purple
}

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

    # Adjust the title based on market value
    market_text = f"({market})" if market else ""
    title_text = f'Artist Impact {market_text}<br><sup>Number of Songs and Average Song Time on Billboard for Top 10 Artists {time_period}</sup>'

    # Update layout
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,  # Centering the title and subtitle
            'xanchor': 'center'
        },
        xaxis=dict(title='Artist'),
        yaxis=dict(title=None, range=[0, 1.15 * max_x]),
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


@st.cache_data(show_spinner=False)
def plot_bumpchart(df, market, date):
    pivot_ranks = data_wrangling.generate_bump_data(df, market, date)
    score = pivot_ranks.fillna(11).apply(data_wrangling.score_bumpchart).sum()
    score_index = score.index
    score = score.to_numpy()
    score -= score.min()
    score /= (score.max() - score.min())
    cmap = colors.sample_colorscale('viridis', score)
    cmap = dict(zip(score_index, cmap))

    fig = go.Figure()

    for index in pivot_ranks.columns:
        song, _, artist = index
        name = f'{song}, by {artist}'

        fig.add_trace(go.Scatter(
            x=pivot_ranks.index,
            y=pivot_ranks[index],
            mode='lines+markers',
            name=name,
            meta=name,
            hovertemplate='<br>%{meta}<br>Week:%{x}<br>Postion on chart:%{y}<extra></extra>',
            line=dict(width=4),
            marker=dict(
                size=10,
                color=cmap[index],
            ),
            connectgaps=False,
        ))

    artwork = st.session_state['spotify'].get_songs_images(list(pivot_ranks.columns.get_level_values(1)))

    for column in pivot_ranks.columns:
        if column[1] in artwork:
            latest_appearance = pivot_ranks[column].last_valid_index()
            photo_url = artwork[column[1]]
            image = data_wrangling.circle_image(photo_url)
            fig.add_layout_image(
                dict(
                    source=image,
                    name=f'{column[0]} by {column[1]}',
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=latest_appearance,
                    y=pivot_ranks.loc[latest_appearance][column],
                    sizex=8.64e7 * 1.8,
                    sizey=1,
                    sizing="contain",
                    layer="above"
                )
            )

    # Customize the chart's appearance
    fig.update_layout(
        title={
            'text': f'Weekly ranks from {pivot_ranks.index.min().date()} to {pivot_ranks.index.max().date()}',
            'x': 0,
            'xanchor': 'left',
            'font': dict(
                size=20,
            ),
        },
        xaxis_title='Date',
        xaxis=dict(
            tick0=pivot_ranks.index.min(),
            dtick=604800000,
        ),
        yaxis_title='Rank',
        yaxis=dict(
            autorange="reversed",
            dtick=1,
        ),
        height=800,
        template='plotly_white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=20,
        ),
        showlegend=False,
        dragmode=False,
    )

    return fig


@st.cache_data(show_spinner=False)
def plot_time_signature(df):
    fig = px.bar(df,
                y=[0,0],
                x='Percentage',
                orientation='h',
                color='Category',
                text='Category',
                color_discrete_sequence=['#A8E6CF', '#FFD3B6'],
                hover_name='Category',
                )

    fig.update_traces(hovertemplate='Percentage of Songs: %<br>%{x:.2f}%<extra></extra>'),
    fig.update_yaxes(showgrid=False, showticklabels=False)
    fig.update_layout(
        title={
            'text': 'Distribution of 4/4 vs Other Time Signatures',
            'x': 0.5,  # Centering the title
            'xanchor': 'center'
        },
        xaxis=dict(title='Percentage of Songs (%)', dtick=5, range=[0, 100]),
        xaxis_tickformat=".%",
        yaxis=dict(title=''),
        template='plotly_white',
        font=dict(size=16),
    )
    return fig


@st.cache_data(show_spinner=False)
def plot_mode_distribution(df):
    # Create a combined grouped bar chart
    fig = px.bar(df, x='year', y='proportion', color='mode',
                facet_col='market', barmode='stack',
                labels={'mode': 'Mode', 'proportion': 'Proportion', 'year': 'Year'},
                title=' Major vs. Minor Mode Distribution between IL and INTL over the Years',
                color_discrete_sequence=['#F6B8B8', '#B8DFF6'])

    # Update layout to center the title
    fig.update_layout(
        template='plotly_white',
        title={
            'text': 'Major vs. Minor over the Years<br><sup>Comparison of Mode Distribution by Year and Market</sup>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title='Year', dtick='Y2'),
    )

    return fig


@st.cache_data(show_spinner=False)
def plot_genre_trends(df, market):
    genre_trends = data_wrangling.genre_trends(df, market)
    # Create a stacked area chart for relative proportions
    fig = px.area(genre_trends, x='year', y='proportion', color='simplified_artist_genres',
                  title=f'Relative Popularity of Top 5 Genres Over Time ({market})',
                  labels={'proportion': 'Proportion of All Songs', 'simplified_artist_genres': 'Genre'},
                  hover_data={'proportion': ':.2f', 'simplified_artist_genres': True, 'year': None},
                  color_discrete_map=genre_color_map)

    # Centering the title
    fig.update_layout(
        template='plotly_white',
        title={
            'text': f'Relative Popularity of Top 5 Genres Over Time ({market})',
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode='x unified',
        xaxis=dict(title='Year', dtick='Y1'),
    )
    return fig


@st.cache_data(show_spinner=False)
def text_plots(df):
    total_unique_artists, top_artist_data, top_song_data, time_signatures = data_wrangling.text_stats(df)


    # Visualization for Total Number of Unique Artists
    unique_artists = go.Figure(go.Indicator(
        mode="number",
        value=total_unique_artists,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 60, 'color': '#FFAB80'}}
    ))

    # Visualization for Top Artist by Cumulative Weeks on Chart
    top_artist = go.Figure(go.Indicator(
        mode="number",
        value=top_artist_data['weeks_on_chart'],
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 60, 'color': '#FF8C8C'}}
    ))

    # Visualization for Top Song by Cumulative Weeks at Number One
    top_song = go.Figure(go.Indicator(
        mode="number",
        value=top_song_data['weeks_at_number_one'],
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 60, 'color': '#A8E6CF'}}
    ))

    # Visualization for Time signature distribution
    time_signature = go.Figure(go.Indicator(
        mode="number",
        value=time_signatures,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 60, 'color': '#A8E6CF'}},
    ))
    
    unique_artists.update_layout(title={'x': 0.5, 'xanchor': 'center', "text": "Total Number of Unique Artists"}, height=200)
    top_artist.update_layout(title={'x': 0.5, 'xanchor': 'center', "text": f"Top Artist: {top_artist_data['main_artist_name']}\nby Cumulative Weeks on Chart"}, height=200)
    top_song.update_layout(title={'x': 0.5, 'xanchor': 'center', "text": f"Top Song: {top_song_data['track_name']}\nby Cumulative Weeks at Number One"}, height=200)
    time_signature.update_layout(title={'x': 0.5, 'xanchor': 'center', "text": "Percentage of songs with 4/4 Time signature"}, height=200)
    time_signature.update_traces(number={'valueformat': '.2%'})

    # Display the figures
    return unique_artists, top_artist, top_song, time_signature

def polar_graph(genres, split_feature, output_features):
    # List of features to include in the radar chart
    features_repeated = output_features + [output_features[0]]

    glz_df = data_wrangling.read_data()
    data_slices = data_wrangling.split_data(glz_df, genres, split_feature, output_features)

    min_values, max_values = data_wrangling.data_scale_values(data_slices)

    if st.session_state.queried_song['features'] is not None:
        res = abs(st.session_state.queried_song['features'])
        min_values = np.minimum(min_values, res)
        max_values = np.maximum(max_values, res)
        res = (res - min_values) / (max_values - min_values)
    else:
        res = None

    data_slices = {value: ((data_wrangling.get_mean_of_features(features_values) - min_values) / (max_values - min_values)) for value, features_values in data_slices.items()}

    # Create radar charts for IL and INTL
    fig = go.Figure()

    for value, features_values in data_slices.items():
        features_trace_values = features_values
        features_trace_values =  np.concatenate((features_trace_values, np.array([features_trace_values[0]])))
        name = value.title()
        fig.add_trace(go.Scatterpolar(
            r=features_trace_values,
            theta=features_repeated,
            name=name,
            line=dict(color=genre_color_map[value], width=5),
            marker=dict(color=genre_color_map[value], size=10)
        ))

    if res is not None:
        res_trace_values = np.concatenate((res, np.array([res[0]])))
        song_name = st.session_state.queried_song['name']
        artist_name = st.session_state.queried_song['artist']
        fig.add_trace(go.Scatterpolar(
            r=res_trace_values,
            theta=features_repeated,
            name=f'{song_name} by {artist_name}',
            line=dict(color='red', width=4),
            marker=dict(color='red', size=8),
        ))


    # Update layout
    fig.update_layout(
        template='plotly_white',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        legend=dict(
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            orientation="h",
        ),
        margin=dict(t=5, l=0, r=0),
    )

    return fig