import streamlit as st
import streamlit_antd_components as sac
from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards  
from streamlit_card import card as cd
from pages import config
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

    st.button("Select",key="sqqe_submit_bt")
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

    st.button("Select",key="rrfr_submit_bt")
    
    

# with ui.card(key="card1"):
#     ui.element("span", children~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# =["Email"], className="text-black text-xl font-bold m-1", key="label1")
#     ui.element("input", key="email_input", placeholder="Your email")

#     ui.element("span", children=["User Name"], className="text-gray-400 text-sm font-medium m-1", key="label2")
#     ui.element("input", key="username_input", placeholder="Create a User Name")
#     ui.element("button", text="Submit", key="button", className="m-1")
    
# event=card(
#         title="Hello World!",
#         text="Some description",
#         key="card_sac_1",
        
#         # image="http://placekitten.com/300/250",
#         # url="https://www.google.com",
#     )
# cols = st.columns(3)
# with cols[0]:
#     card1 = ui.metric_card(title="Total Revenue", content="$45,231.89", description="+20.1% from last month", key="card1")
# with cols[1]:
#     card2 = ui.metric_card(title="Total Revenue", content="$45,231.89", description="+20.1% from last month", key="card2")
# with cols[2]:
#     card3 = ui.metric_card(title="Total Revenue", content="$45,231.89", description="+20.1% from last month", key="card3")

# col1, col2, col3 = st.columns(3)

# col1.metric(label="Gain", value=5000, delta=1000)
# col2.metric(label="Loss", value=5000, delta=-1000)
# col3.metric(label="No Change", value=5000, delta=0)

# style_metric_cards(background_color="#FFF")


# Debug
with st.expander("Debug"):
    st.write(st.session_state)
    # st.write(event)
    # st.write(card1)
    # st.write(card2)
    # st.write(card3)
