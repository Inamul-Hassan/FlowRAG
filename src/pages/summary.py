import streamlit as st 
import json

st.set_page_config(page_title="FlowRAG", page_icon="💬")

st.markdown("""
      <style>
          section[data-testid="stSidebar"][aria-expanded="true"]{
              display: none;
          }
      </style>
      """, unsafe_allow_html=True)

if "selected_pipeline" and "selected_data" and 'selected_pipeline_config' not in st.session_state:
  st.switch_page("app.py")

config = {}

config["pipline"] = st.session_state.selected_pipeline
config["data"] = st.session_state.selected_data
config["config"] = st.session_state.selected_pipeline_config

def save_config():
  with open('src/pages/user_config.json', 'w') as f:
    json.dump(config, f)

save_config()

st.write(config)

create_button = st.button("Create Pipeline")

if create_button:
  st.switch_page("pages/chat.py")