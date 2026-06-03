import streamlit as st
st.write("Streamlit OK")
try:
    import plotly
    st.success(f"Plotly Installed: {plotly.__version__}")
except Exception as e:
    st.error(f"Plotly Installed: {e}")