import streamlit as st
import streamlit_antd_components as sac
from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards  
from streamlit_card import card as cd
import streamlit_shadcn_ui as ui


st.title("FlowRAG")
st.caption("A plug and play tool to quickly setup your own advanced RAG pipeline.")
st.write("You can use this tool for two main purposes:")
st.markdown("- Quickly setup a chatbot based on a specfic RAG pipeline architechture(with limited customization) over your data.")
st.markdown("- Get an end-to-end implementation code of an advanced RAG pipeline and customize each indvidual components of it as per your wish.")

def onClick(event):
    st.session_state.selected_pipeline = event

# st.markdown("## RAG Pipeline")
# st.write("The main components of a RAG pipeline are,")

st.info("This is a developer centric tool. It is meant to be used by developers/enthusiasts to quickly setup a complex RAG pipeline for their data and run test/eval over them.")

st.markdown("## Select a pipeline")

st.markdown("### For Unstrucutured Data")

with st.expander(label="SubQuestionQueryEngine",expanded=False):
    st.write("Sub Question Querying can break down a complex question into to smaller sub question and retreieve relavent data from the provided sources/files(Even multiple files/sources).")
    st.markdown("#### The main components of the pipeline are,")
    tab_returend_index = sac.tabs([
        sac.TabsItem(label='Transform'),
        sac.TabsItem(label='Indexing'),
        sac.TabsItem(label='Vector Store'),
        sac.TabsItem(label='Query engine'),
        sac.TabsItem(label='Chat store'),
        sac.TabsItem(label='Sub question query engine'),
    ], return_index=True,key="sqqe_tab")

    st.button("Get Started",key="sqqe_submit_bt",on_click=onClick, args=("SubQuestionQueryEngine",),type="primary")
    
with st.expander(label="ReciprocalRerankFusionRetriever",expanded=False):
    st.write("@vishal")
    st.markdown("#### The main components of the pipeline are,")
    tab_returend_index = sac.tabs([
        sac.TabsItem(label='Transform'),
        sac.TabsItem(label='Indexing'),
        sac.TabsItem(label='Vector Store'),
        sac.TabsItem(label='Query engine'),
        sac.TabsItem(label='Chat store'),
        sac.TabsItem(label='Sub question query engine'),
    ], return_index=True,key="rrfr_tab")

    st.button("Get Started",key="rrfr_submit_bt",on_click=onClick, args=("ReciprocalRerankFusionRetriever",),type="primary")
# st.markdown("### For Structured Data")
    
if "selected_pipeline" in st.session_state:
    st.switch_page("pages/data.py")

# Debug
with st.expander("Debug"):
    st.write(st.session_state)
    # st.write(event)
    # st.write(card1)
    # st.write(card2)
    # st.write(card3)
