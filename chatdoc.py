import streamlit as st # used to create our UI frontend 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("Chat with Document")

loader = TextLoader("./constitution.txt")  # load text document
documents = loader.load()
print(documents) # print to ensure document loaded correctly.
