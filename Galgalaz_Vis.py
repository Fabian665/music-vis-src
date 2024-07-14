import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="📈",
)

st.write("# Dvir and Roey Galgalaz Charts Visualization! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a dashboard from the sidebar**
"""
)