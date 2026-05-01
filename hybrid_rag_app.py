import os
import re
import time

import streamlit as st # used to create our UI frontend 
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

st.title("Chat with the 1787 U.S. Constitution")

st.markdown(
    """
    <style>
    .debug-expander summary p {
        color: rgba(250, 250, 250, 0.45);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

OLLAMA_BASE_URL = "https://ollama.com"
CHAT_MODEL = "gpt-oss:120b"
JUDGE_MODEL = "llama-3.1-8b-instant"
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


def get_groq_api_key():
    return get_secret("GROQ_API_KEY")


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
from langchain_groq import ChatGroq

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    client_kwargs=ollama_client_kwargs,
)
judge_llm = ChatGroq(
    model=JUDGE_MODEL,
    api_key=get_groq_api_key(),
    temperature=0,
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
REFUSAL = "I do not know based on the provided document."


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


def format_docs_for_prompt(docs):
    return "\n\n".join(
        f"[Chunk {i}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def get_cited_chunk_numbers(answer, total_docs):
    cited_numbers = {
        int(match)
        for match in re.findall(r"[\[【]Chunk\s*(\d+)[\]】]", answer)
        if int(match) <= total_docs
    }
    return sorted(cited_numbers)


def is_refusal(answer):
    return REFUSAL in answer


def get_evidence_quotes(answer):
    return re.findall(r'Evidence:\s*["“]([^"”]+)["”]', answer, flags=re.IGNORECASE)


def put_evidence_after_answer(answer):
    parts = answer.split("\n\n", 1)
    if len(parts) == 2 and parts[0].lstrip().startswith("Evidence:"):
        return f"{parts[1]}\n\n{parts[0]}"
    return answer


judge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict evidence judge for a RAG app.

Use only the evidence quote. Do not use outside knowledge.
Consider synonyms and equivalent wording, such as "consent" and "approval".
Reject evidence that merely mentions related names, titles, or topics without
stating the exact fact or relationship needed to answer the question.

Return only YES or NO.""",
        ),
        (
            "human",
            """Question:
{question}

Answer:
{answer}

Evidence quote:
{evidence}

Does the evidence quote directly support the answer to the question?""",
        ),
    ]
)


def judge_evidence_support(answer, question):
    if is_refusal(answer):
        return "YES"

    evidence_quotes = get_evidence_quotes(answer)
    if not evidence_quotes:
        return "NO"

    evidence = "\n".join(evidence_quotes)
    judgment = (judge_prompt | judge_llm | StrOutputParser()).invoke(
        {
            "question": question,
            "answer": answer,
            "evidence": evidence,
        }
    )
    return judgment.strip().upper()


def enforce_evidence_support(answer, question):
    judgment = judge_evidence_support(answer, question)
    if judgment.startswith("YES"):
        return answer, judgment
    return REFUSAL, judgment

# RAG prompt
template = """
Answer the question based ONLY on the retrieved context below.

Follow this order:
1. First, find an exact quote from the context that directly answers the question.
2. If no exact quote directly answers the question, say exactly:
   "{refusal}"
3. If you find an exact quote, answer using only that quote.

Direct support means the quote states the exact fact or relationship needed to
answer the question. If a name or topic appears in the context, but the context
does not state the fact asked about, that is not enough support.

Do not use outside knowledge.
For any answer other than the refusal above, use this format:

Evidence: "short exact quote" [Chunk 1]

Answer: answer based only on that quote [Chunk 1]

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

question = st.text_input('Input your question')
if question:
    docs = retrieve_documents(question)
    context = format_docs_for_prompt(docs)
    res = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question, "refusal": REFUSAL}
    )
    res, judge_result = enforce_evidence_support(res, question)
    st.write(put_evidence_after_answer(res))

    cited_chunk_numbers = get_cited_chunk_numbers(res, len(docs))
    if not cited_chunk_numbers and not is_refusal(res):
        st.warning("No cited evidence was provided for this answer.")

    st.markdown('<div class="debug-expander">', unsafe_allow_html=True)
    if cited_chunk_numbers:
        with st.expander("Retrieved source chunks"):
            for chunk_number in cited_chunk_numbers:
                doc = docs[chunk_number - 1]
                st.markdown(f"### Chunk {chunk_number}")
                st.write(doc.page_content)

    with st.expander("Retrieved context"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)

    with st.expander("Evidence judge"):
        st.write(f"LLM-as-a-Judge result: {judge_result}")
    st.markdown("</div>", unsafe_allow_html=True)
