import re

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

embeddings = OllamaEmbeddings(model='nomic-embed-text')
vector_store = Chroma.from_documents(chunks,embeddings)

# to see the chunks
# st.write(chunks[0])
# st.write(chunks[1])

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model="llama3.2:3b")
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
base_retriever = vector_store.as_retriever(search_kwargs={"k": 6})

retriever = MultiQueryRetriever.from_llm(
    base_retriever,
    llm,
    prompt=QUERY_PROMPT
)

STOP_WORDS = {
    "about",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "there",
    "their",
    "have",
    "with",
    "from",
    "that",
    "this",
}

def retrieve_documents(question):
    vector_docs = retriever.invoke(question)
    keywords = {
        word
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
        if word not in STOP_WORDS
    }

    keyword_matches = []
    for chunk in chunks:
        chunk_text = chunk.page_content.lower()
        score = sum(1 for keyword in keywords if keyword in chunk_text)
        if score:
            keyword_matches.append((score, chunk))

    keyword_matches.sort(key=lambda match: match[0], reverse=True)

    combined_docs = []
    seen_content = set()
    for doc in vector_docs + [chunk for _, chunk in keyword_matches[:3]]:
        if doc.page_content not in seen_content:
            combined_docs.append(doc)
            seen_content.add(doc.page_content)

    return combined_docs

# RAG prompt
template = """
Answer the question based ONLY on the following context.

If the answer is not in the context, say:
"I do not know based on the provided document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retrieve_documents, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = st.text_input('Input your question')
if question:
    docs = retrieve_documents(question)

    with st.expander("Retrieved context"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)

    res = chain.invoke(input=(question))
    st.write(res)
