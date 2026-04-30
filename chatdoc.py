import os
import re
import time

import streamlit as st # used to create our UI frontend 
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

st.title("Chat with Document")

OLLAMA_BASE_URL = "https://ollama.com"
CHAT_MODEL = "gpt-oss:120b"
EMBEDDING_MODEL = "models/gemini-embedding-001"
PINECONE_INDEX_NAME = "constitution-rag"
PINECONE_NAMESPACE = "constitution"
PINECONE_DIMENSION = 3072
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

def get_secret(name):
    value = os.getenv(name)
    if not value:
        try:
            value = st.secrets.get(name)
        except FileNotFoundError:
            value = None

    if not value:
        st.error(f"Add {name} to .streamlit/secrets.toml or your environment.")
        st.stop()
    return value.strip()


def get_ollama_api_key():
    api_key = get_secret("OLLAMA_API_KEY")
    if api_key.lower().startswith("bearer "):
        api_key = api_key.split(None, 1)[1].strip()
    return api_key


def get_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except FileNotFoundError:
            api_key = None
    if not api_key:
        api_key = get_secret("GOOGLE_API_KEY")
    return api_key.strip()


ollama_client_kwargs = {
    "headers": {"Authorization": f"Bearer {get_ollama_api_key()}"}
}

loader = TextLoader("./constitution.txt")  # load text document
documents = loader.load()
# print(documents) # print to ensure document loaded correctly.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=get_gemini_api_key(),
)


@st.cache_resource(show_spinner="Preparing Pinecone index...")
def get_pinecone_index():
    pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    index = pc.Index(PINECONE_INDEX_NAME)
    chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    chunk_ids = [f"constitution-{i}" for i in range(len(chunks))]
    vectors = [
        {
            "id": chunk_id,
            "values": chunk_vector,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "constitution.txt"),
                "chunk": i,
            },
        }
        for i, (chunk_id, chunk, chunk_vector) in enumerate(
            zip(chunk_ids, chunks, chunk_vectors)
        )
    ]
    index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
    return index


pinecone_index = get_pinecone_index()

# to see the chunks
# st.write(chunks[0])
# st.write(chunks[1])

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    client_kwargs=ollama_client_kwargs,
)
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


def get_search_queries(question):
    generated_queries = (QUERY_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question}
    )
    alternate_queries = [
        query.strip()
        for query in generated_queries.splitlines()
        if query.strip()
    ]
    return [question] + alternate_queries


def retrieve_vector_documents(question):
    docs = []
    seen_content = set()

    for search_query in get_search_queries(question):
        query_vector = embeddings.embed_query(search_query)
        results = pinecone_index.query(
            vector=query_vector,
            top_k=6,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )

        for match in results["matches"]:
            text = match["metadata"]["text"]
            if text not in seen_content:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": match["metadata"].get("source"),
                            "chunk": match["metadata"].get("chunk"),
                            "score": match["score"],
                        },
                    )
                )
                seen_content.add(text)

    return docs


def retrieve_documents(question):
    vector_docs = retrieve_vector_documents(question)
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
