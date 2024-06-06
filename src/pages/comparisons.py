import streamlit as st

st.header("Comparisons between models , similatity, chunk-size")

option = st.selectbox(
    label='Select Comparison Element',
   options=("chunk", "similarity", "embedding model"),
   index=None,
   placeholder="Comparison by:",
)

input = st.text_input(label="input question ...")
answer = st.button("Answer")
