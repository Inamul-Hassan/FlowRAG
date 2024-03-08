import streamlit as st

st.write("# Configuration")

llm_mapping = {
  "Gemini":["gemini-pro-1.0"],
  "OpenAI":["gpt3.5-turbo","gpt-4"]
}


st.write("## LLM")
with st.form(key="config-form"):
  llm_provider = st.selectbox(label="Select a llm provider",options=["Gemini","OpenAI"],index=0)
  options = llm_mapping[llm_provider]
  st.selectbox(label="Select a llm model",options=options,index=0)
  st.form_submit_button(label="Submit")
  
with st.expander("Debug"):
  st.write(st.session_state)