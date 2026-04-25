import streamlit as st # used to create our UI frontend 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

st.title("Chat with Document")

loader = TextLoader("./constitution.txt")  # load text document
documents = loader.load()
# print(documents) # print to ensure document loaded correctly.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

ollama.pull('nomic-embed-text')
embeddings = OllamaEmbeddings(model='nomic-embed-text')
vector_store = Chroma.from_documents(chunks,embeddings)

# to see the chunks
# st.write(chunks[0])
# st.write(chunks[1])

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model="llama3.2")
# a simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five different
    versions of the given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to help the user
    overcome some of the limitations of the distance-based similarity search. Provide these
    alternative questions separated by newlines. Original question: {question}""",
)
retriever = MultiQueryRetriever.from_llm( vector_store.as_retriever(), llm, prompt=QUERY_PROMPT )

# RAG prompt
template = """Answer the question based ONLY on the following context: {context} Question: {question} """

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = st.text_input('Input your question')
if question:
    res = chain.invoke(input=(question))
    st.write(res)
