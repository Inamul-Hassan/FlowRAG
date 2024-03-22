import streamlit as st 

if "selected_pipeline" and "selected_data" and "selected_config" and 'selected_pipeline_config' not in st.session_state:
  st.switch_page("app.py")

config = {}

config["pipline"] = st.session_state.selected_pipeline
config["data"] = st.session_state.selected_data
config["config"] = st.session_state.selected_pipeline_config

# create a json file and store the configuration in st.session_state
import json
def save_config():
  with open('src/pages/user_config.json', 'w') as f:
    json.dump(config, f)

save_config()

# LOAD FROM A JSON FILE
# def load_json(file_path):
#   with open(file_path) as f:
#     data = json.load(f)
#   return data

# st.write(load_json('user_config.json'))

st.write(config)

create_button = st.button("Create Pipeline")

if create_button:
  st.switch_page("pages/chat.py")