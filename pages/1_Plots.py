import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import n_colors
import numpy as np

glz_df = pd.read_csv('/data/galgalaz_expanded.csv')
glz_df['date'] = pd.to_datetime(glz_df['date'])
glz_df['year'] = glz_df['date'].dt.year
glz_df['track_duration'] = pd.to_datetime(glz_df['track_duration'], unit='ms')

data = dict(glz_df.groupby('year').track_duration.apply(np.array).sort_index(ascending=False))

# data = glz_df
colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')

fig = go.Figure()
for (year, data_line), color in zip(data.items(), colors):
    fig.add_trace(go.Violin(
        x=data_line,
        line_color=color,
        name=year,
    ))

fig.update_traces(
    orientation='h',
    side='positive',
    width=3,
    points=False,
    meanline_visible=True,
    hoveron='kde',
)
fig.update_layout(
    xaxis_showgrid=False,
    xaxis_zeroline=False,
    height=800,
    xaxis_tickformat='%M:%S',
    xaxis_hoverformat='%M:%S',
)
fig