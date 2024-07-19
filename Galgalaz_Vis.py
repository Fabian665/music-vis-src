import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Galgalatz Charts Visualization",
    page_icon="🎵",
)

# Welcome message
st.title("Welcome to Roey and Dvir's Galgalatz Charts Visualization! 👋")
st.markdown(
    """
    Welcome to our exciting journey through the musical landscape of the past decade! 🎶✨ We're thrilled to present our visualization project, showcasing the Galgalatz top ten songs chart from 2013 to 2024. Galgalatz, one of Israel's favorite radio stations, reflects the nation's musical tastes and trends. Our interactive dashboard invites you to explore how preferences have evolved, discover top-charting songs and artists, and uncover fascinating trends. Whether you're a music enthusiast, industry professional, or just curious, our project offers a wealth of insights and fun facts. Dive in and join us on this melodious adventure! 🎵🎉

    👈 Select a dashboard from the sidebar to explore various aspects of the Galgalatz charts.
    """
)

# Sidebar success message
st.sidebar.success("Select a dashboard from the sidebar.")

# Display the image
st.image("cover_image.webp", use_column_width=True)# Footer
st.markdown("---")
st.markdown(
    """
    **Created by Roey and Dvir** | Data from Galgalatz | Visualized with Streamlit
    """
)

# Adjusting font sizes with CSS
st.markdown(
    """
    <style>
    .css-18e3th9 {
        font-size: 1.2rem;
    }
    .css-1d391kg {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
