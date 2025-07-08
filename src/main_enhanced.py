import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

# Configuration

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")  # claude-sonnet-4-20250514
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")    #  "all-MiniLM-L6-v2" fast, 384‑dimensional HF model

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


if not ANTHROPIC_API_KEY:
    st.warning("Please set your ANTHROPIC_API_KEY in the environment before launching Streamlit.")

# Helper functions

def load_pdf_and_split(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ".", " ", ""],
    )
    return text_splitter.split_documents(raw_docs)


def build_vector_store(docs):
    """Create or update FAISS vector store from documents."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    return FAISS.from_documents(docs, embeddings)


def get_retrieval_chain(vstore):
    system_prompt = (
        "Your name is 'AskMyPDF Assistant'. You are an expert assistant that answers questions strictly based on the given PDF file context. "
        "If the answer is not contained in the context, reply with 'I don't know based on the provided document.', unless it is basic human interaction."
    )

    question_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""<system>{system_prompt}</system>
<user>
Context:
{{context}}

Question: {{question}}
</user>"""
    )

    llm = ChatAnthropic(
        model_name=CLAUDE_MODEL,
        temperature=0.2,
        max_tokens=1024,
        api_key=ANTHROPIC_API_KEY,
    )

    retriever = vstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": question_prompt,
        },
        return_source_documents=True,
    )

    return chain

# Streamlit UI

st.set_page_config("Ask My PDF")
st.markdown("## Ask My PDF")
st.markdown("----")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.session_state.get("file_name") != uploaded_file.name:
        with st.status("Processing Your File, This Might Take Some Time...", expanded=True) as status:
            st.write("Creating Temp File...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            # Vector store pipeline
            st.write("Loding File...")
            docs = load_pdf_and_split(tmp_path)
            st.write("Vectorizing...")
            vstore = build_vector_store(docs)
            st.write("Getting the final details done...")
            chain = get_retrieval_chain(vstore)

            st.session_state.chain = chain
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages.clear()
            status.update(
                label="Your Chat is Ready!", state="complete", expanded=False
            )

st.markdown("----")
# Show chat messages
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["is_user"] else "assistant"):
        st.write(msg["text"])

        # Show expander if assistant message has sources
        if not msg["is_user"] and "sources" in msg and msg["sources"]:
            with st.expander("See sources"):
                source_chunks = "\n\n".join(
                    f"**Chunk {i+1} (Page {src.metadata.get('page', '?')}):**\n{src.page_content}"
                    for i, src in enumerate(msg["sources"])
                )
                st.write(source_chunks)


# Chat input
if st.session_state.chain:
    user_input = st.chat_input("Ask something about the PDF")

    if user_input:
        # Display and store user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"text": user_input, "is_user": True})

        with st.spinner("Thinking..."):
            result = st.session_state.chain({"query": user_input})

        answer = result["result"]
        sources = result.get("source_documents", [])

        # Display and store assistant message
        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "text": answer,
            "is_user": False,
            "sources": sources  # store the list of source documents
        })

        # Optional: Show retrieval sources
        expander = st.expander("See sources")
        if sources:
                source_chunks = "\n\n".join(
                    f"**Chunk {i+1} (Page {src.metadata.get('page', '?')}):**\n{src.page_content}"
                    for i, src in enumerate(sources)
                )
                expander.write(source_chunks)

else:
    st.info("Upload a PDF to begin.")