import streamlit as st 

if "selected_pipeline" and "selected_data" and "selected_config" and 'selected_pipeline_config' not in st.session_state:
  st.switch_page("app.py")

st.write(st.session_state)