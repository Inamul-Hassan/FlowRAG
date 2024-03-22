import streamlit as st
import streamlit_antd_components as sac

st.set_page_config(page_title="FlowRAG", page_icon="💬")

st.markdown("""
      <style>
          section[data-testid="stSidebar"][aria-expanded="true"]{
              display: none;
          }
      </style>
      """, unsafe_allow_html=True)


if "selected_pipeline" and "selected_data" not in st.session_state:
  st.switch_page("app.py")
  
if 'finish_bt' in st.session_state:
    st.switch_page("pages/summary.py")
  

import json
with open('src/pages/config.json') as f:
    configuration = json.load(f)

pipeline = st.session_state.selected_pipeline

try:
    pipeline_config = configuration[pipeline]
except Exception as e:
    st.error("No configuration found")
    st.stop()
    
if "selected_pipeline_config" not in st.session_state:
    st.session_state.selected_pipeline_config = {}
    
def onClickForm(pipeline_step,configuration,isLast:bool):
    st.session_state.selected_pipeline_config[pipeline_step] = configuration
    print(isLast)
    if isLast:
        st.session_state.finish_bt = True       
    else:   
        change_to = tabs[tabs.index(st.session_state.tab_selected)+1]
        st.session_state.tab_selected = change_to
 

def display_config_form(pipeline_step:str,isLast:bool): 
    st.session_state.configuration = {}
    for key,value in pipeline_config[pipeline_step].items():
        try:
            val = int(value)
            st.session_state.configuration[key] = st.number_input(label = key, value = val)
        except:
            val = value
            st.session_state.configuration[key] = st.text_input(label = key, value = val)

    
    if isLast:
        st.button(label="Finish",on_click=onClickForm,args=(pipeline_step,st.session_state.configuration,True))
    else: 
        st.button(label="Next",on_click=onClickForm,args=(pipeline_step,st.session_state.configuration,False))
   

tabs = ["Transformation","Retrieval","Vector Store","Chat Store"]

if "tab_selected" not in st.session_state:
    st.session_state.tab_selected = "Transformation"
    
st.info(f"Pipeline : {st.session_state.selected_pipeline}")

st.markdown("# Configure Your Pipeline")

sac.steps(
    items=[
        sac.StepsItem(title='Transformation',disabled=True),
        sac.StepsItem(title='Retrieval',disabled=True),
        sac.StepsItem(title='Vector Store',disabled=True),
        sac.StepsItem(title='Chat Store', disabled=True),
    ], placement='vertical',index=tabs.index(st.session_state.tab_selected)
)



match st.session_state.tab_selected:
    case "Transformation":
        display_config_form("transform",False)
    case "Retrieval":
        display_config_form("retriever",False)
    case "Vector Store":
        display_config_form("storage",False)
    case "Chat Store":
        display_config_form("chat_history",True)
    case _:
        st.write("Not implemented yet")

