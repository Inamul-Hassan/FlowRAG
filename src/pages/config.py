import streamlit as st

def onClick(selected_config):
  st.session_state.selected_config = selected_config


if "selected_pipeline" and "selected_data" not in st.session_state:
  st.switch_page("app.py")
  
st.info(f"Pipeline : {st.session_state.selected_pipeline}")

st.write("# Configuration")

llm_mapping = {
  "Gemini":["gemini-pro-1.0"],
  "OpenAI":["gpt3.5-turbo","gpt-4"]
}
embedding_models = ["embedding-001","embedding-002","embedding-003"]

st.write("## LLM")

llm_provider = st.selectbox(label="Select a llm provider",options=llm_mapping.keys(),index=0)
options = llm_mapping[llm_provider]
llm_model = st.selectbox(label="Select a llm model",options=options,index=0)
llm_api_key = st.text_input(label="API Key",type="password")

st.write("## Embedding")

embedding_model = st.selectbox(label="Select a embedding model",options=embedding_models,index=0)
embedding_api_key = st.text_input(label="API Key",type="password",placeholder="If its applicable for the selected model")

selected_config = {"llm_provider":llm_provider,"llm_model":llm_model,"llm_api_key":llm_api_key,"embedding_model":embedding_model,"embedding_api_key":embedding_api_key}

st.button("Next",key="config_submit_bt",on_click=onClick, args=(selected_config,),type="primary")

if "selected_config" in st.session_state:
  if not llm_api_key:
    st.error("Please provide an API key for the selected LLM model")
  else:
    st.switch_page("pages/pipeline.py")

with st.expander("Debug"):
  st.write(st.session_state)