import streamlit as st
import streamlit_antd_components as sac

if "selected_pipeline" not in st.session_state:
    st.switch_page("app.py")

st.info(f"Pipeline : {st.session_state.selected_pipeline}")

st.markdown("## Data")

st.write("- The first step in the creating your own RAG pipeline is to define all the data formats/sources that your pipeline should support.")
st.write("- FlowRAG supports multiple files/sources but you have configure the pipeline to handle all the selected format/source.")
st.divider()
st.markdown("## Data Source")
st.warning("⚠️ You don't be able to modify the data formats/sources later")
with st.form(key="is_data_source_selected"):
    st.write("Select all the required data formats/sources")
    selected = sac.chip(items=[
        sac.ChipItem(label='csv'),
        sac.ChipItem(label='docx'),
        sac.ChipItem(label='epub'),
        sac.ChipItem(label='html'),
        sac.ChipItem(label='ipynb'),
        sac.ChipItem(label='json'),
        sac.ChipItem(label='md'),
        sac.ChipItem(label='pdf'),
        sac.ChipItem(label='ppt'),
        sac.ChipItem(label='pptm'),
        sac.ChipItem(label='pptx'),
        sac.ChipItem(label='txt'),
        sac.ChipItem(label='webpage'),
    ], radius='md', color='blue', multiple=True)
    st.session_state.selected_data = selected
    st.form_submit_button(label='Next', type="primary")

if st.session_state["FormSubmitter:is_data_source_selected-Next"]:
    if not st.session_state.selected_data:
        st.error("Please select at least one data format/source")   
    else:
        st.switch_page("pages/config.py")
        pass
    # st.write(selected)
    # st.write(st.session_state)