import streamlit as st

st.title("FlowRAG")
st.caption("A plug and play tool to quickly setup your own advanced RAG pipeline.")
st.write("You can use this tool for two main purposes:")
st.markdown("- Quickly setup a chatbot based on a specfic RAG pipeline architechture(with limited customization) over your data.")
st.markdown("- Get an end-to-end implementation code of an advanced RAG pipeline and customize each indvidual components of it as per your wish.")

def onClick(event):
    st.session_state.selected_pipeline = event


st.info("This is a developer centric tool. It is meant to be used by developers/enthusiasts to quickly setup a complex RAG pipeline for their data and run test/eval over them.")

st.markdown("## Select a pipeline")

st.markdown("### For Unstrucutured Data")

with st.expander(label="SubQuestionQuerying - recommended for multiple data sources",expanded=False):
    st.markdown("### Sub Question Query Engine")
    st.markdown("This strategy can be used to tackle the problem of answering a complex query using multiple data sources. It first breaks down the complex query into sub questions for each relevant data source, then gather all the intermediate reponses and synthesizes a final response.")
    st.markdown("Check out [Sub Question Query Engine](%s)" % r"https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html#")

    st.button("Get Started",key="sqqe_submit_bt",on_click=onClick, args=("SubQuestionQuerying",),type="primary")
    
with st.expander(label="ReciprocalRerankFusionRetriever - recommended for quality response",expanded=False):
    st.markdown("### Reciprocal Rerank Fusion Retriever")
    st.markdown("The retrieved nodes will be reranked according to the Reciprocal Rerank Fusion algorithm demonstrated in this [paper](%s). It provides an effecient method for rerranking retrieval results without excessive computation or reliance on external models." % r"https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf")
    st.markdown("Check out [Reciprocal Rerank Fusion Retriever](%s)" % r"https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion.html")

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
