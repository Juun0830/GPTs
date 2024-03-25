'''
import streamlit as st

st.title("Hello, World!")

with st.sidebar:
    st.title("sidebar title")
    st.text_input("Input here")

tab_one, tab_two, tab_three = st.tabs(["A","B","C"])

with tab_one:
    st.write("a")

with tab_two:
    st.write("b")

with tab_three:
    st.write("c")
'''

import streamlit as st

st.set_page_config(
    page_title = "Full-Stack GPTs",
    page_icon = "😄"
)

st.title("Full-Stack GPTs")



