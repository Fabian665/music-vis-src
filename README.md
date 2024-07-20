# Music-Vis-Src: Source Code for Music Data Visualization

This repository contains the source code for a Streamlit application that visualizes music data. It's part of a larger project that includes Docker containerization.

## Project Structure

The repository is organized as follows:

```
src/
├── .streamlit/
│   └── secrets.toml
├── pages/
│   ├── 1_All_About_The_Artist.py
│   ├── 2_Song_Mania.py
│   ├── 3_Genre_Storytime.py
│   └── 4_Top_10_Visualized.py
├── data_wrangling.py
├── Galgalaz_Vis.py
├── plotting.py
└── spotify.py
```

## File Descriptions

- `.streamlit`: This directory contains configurations files
  - `secrets.toml`: Not provided, contains API keys for GCP and Spotify.

- `pages/`: This directory contains the individual pages of the Streamlit app.
  - `1_All_About_The_Artist.py`: Provides detailed information about selected artists.
  - `2_Song_Mania.py`: Offers analysis and visualization of individual songs.
  - `3_Genre_Storytime.py`: Explores music genres and their characteristics.
  - `4_Top_10_Visualized.py`: Visualizes top 10 lists (e.g., songs, artists) in various categories.

- `data_wrangling.py`: Contains functions for cleaning and preprocessing music data.
- `Galgalaz_Vis.py`: The main file that sets up the Streamlit app and handles navigation between pages.
- `plotting.py`: Includes functions for creating various types of plots and visualizations.
- `spotify.py`: Handles interactions with the Spotify API, including data retrieval and authentication.

## Setup and Dependencies

To set up this project locally:

1. Clone the repository:
   ```
   git clone https://github.com/Fabian665/music-vis-src.git
   cd music-vis-src
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Spotify API credentials as environment variables:
   ```
   export SPOTIPY_CLIENT_ID='your-spotify-client-id'
   export SPOTIPY_CLIENT_SECRET='your-spotify-client-secret'
   ```

## Running the Application

To run the Streamlit app locally:

```
streamlit run src/Galgalaz_Vis.py
```

Navigate to the URL provided in the terminal (usually `http://localhost:80`) to view the app.

---

For the containerized version of this project, please visit our [Docker repository](https://github.com/Fabian665/music-vis).