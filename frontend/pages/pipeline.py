import streamlit as st
import streamlit_antd_components as sac

# load json from local file and convert it into a dictionary
import json
with open('E:\Python\Directory\FlowRAG-main\storage\config.json') as f:
    configuration = json.load(f)

st.markdown("# Pipeline")


def tranformation():
    file_formats = ["txt"]
    print(type(configuration))
    for format in file_formats:
        transformations = {}
        for tranform in configuration[format]["transformer"]:
            name, config = tranform['type'], tranform['config']
            transformations[name] = config
        st.selectbox(label="", options=transformations.keys(),
                     key=f"{format}_transformation", placeholder="Select a transformation", index=None)
        if st.session_state[f"{format}_transformation"]:
            with st.form(key=f"{format}_transformation_config"):
                for key, value in transformations[st.session_state[f"{format}_transformation"]].items():
                    st.text_input(label=key, value=value)
                st.form_submit_button(label="Submit")


# Tabs
tab_selected = sac.segmented(
    items=[
        sac.SegmentedItem(label='Transformation'),
        sac.SegmentedItem(label='Indexing'),
        sac.SegmentedItem(label='Retrieval'),
        sac.SegmentedItem(label="LLM Config"),
        sac.SegmentedItem(label="Output Parser"),
    ], align='start', use_container_width=True
)

st.session_state.tab_selected = tab_selected

match st.session_state.tab_selected:
    case "Transformation":
        tranformation()
    case _:
        st.write("Not implemented yet")


# Debug
with st.expander("Debug"):
    st.write(st.session_state)
