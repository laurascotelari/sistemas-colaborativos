# Laura Ferré Scotelari - 12543436
import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from agent_rag import (
    build_llm,
    build_embeddings,
    build_vectorstore_from_pages,
    build_retriever,
    build_agent,
    load_pdf_pages,
    read_collab_document,
    write_collab_document,
    read_votes
)

USERS = ["Laura", "Pedro", "Maria", "Lucas"]

def get_shared_vectorstore_dir():
    return os.environ.get("RAG_VDB_DIR", "./vdb")

def ensure_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "selected_user" not in st.session_state:
        st.session_state.selected_user = USERS[0]
    if "k" not in st.session_state:
        st.session_state.k = 6
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o-mini"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0

def build_or_update_index(uploaded_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = tmp.name
    embeddings = build_embeddings()
    pages = load_pdf_pages(tmp_path)
    vs = build_vectorstore_from_pages(pages, embeddings, persist_directory=get_shared_vectorstore_dir(), collection_name="book")
    retriever = build_retriever(vs, k=st.session_state.k)
    os.unlink(tmp_path)
    return retriever

def main():
    st.set_page_config(page_title="RAG Collab (3C)", page_icon="📄")
    ensure_session_state()

    st.title("RAG Collaborative Workspace — 3C (Comunicação / Colaboração / Coordenação)")
    st.markdown("Simule múltiplos usuários perguntando e colaborando sobre um conjunto de PDFs (máx. 5).")

    with st.sidebar:
        st.header("User")
        st.session_state.selected_user = st.selectbox("Active user", USERS, index=USERS.index(st.session_state.selected_user))
        st.caption("Mensagens enviadas serão atribuídas a este usuário (simulado).")

        st.header("Documents")
        uploaded = st.file_uploader("Upload PDF (1 por vez)", type=["pdf"])
        if uploaded:
            if st.button("Build/Update Index", type="primary"):
                with st.spinner("Creating index..."):
                    st.session_state.retriever = build_or_update_index(uploaded.read(), uploaded.name)
                st.success("Index created / updated.")

        st.divider()
        st.header("Agent config")
        st.session_state.k = st.slider("Retrieve k chunks", min_value=2, max_value=10, value=st.session_state.k, step=1)
        st.session_state.model = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o"], index=0)
        st.session_state.temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=float(st.session_state.temperature), step=0.1)

        if st.button("(re)Create Agent"):
            if st.session_state.retriever is None:
                st.warning("Build an index first.")
            else:
                llm = build_llm(model=st.session_state.model, temperature=st.session_state.temperature)
                retriever = st.session_state.retriever
                retriever.search_kwargs["k"] = st.session_state.k
                st.session_state.agent = build_agent(retriever, llm)
                st.success("Agent is ready.")

    st.subheader("Conversation (shared)")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{msg.get('user','User')}**: {msg['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    prompt = st.chat_input(f"{st.session_state.selected_user} diz: ")
    if prompt:
        st.session_state.messages.append({"user": st.session_state.selected_user, "role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.selected_user}**: {prompt}")

        if st.session_state.agent is None:
            with st.chat_message("assistant"):
                st.warning("Create the agent before asking questions.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke({"messages": [{"type": "human","content": prompt}]})
                    answer = result["messages"][-1].content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    st.subheader("Shared collaborative document")
    st.markdown("O documento coletivo exibido abaixo é atualizado pela ferramenta `update_document_tool`.")
    doc = read_collab_document()
    if doc:
        st.text_area("Collaborative Document (read-only)", value=doc, height=220)
    else:
        st.write("_Documento vazio_")

    st.subheader("Coordenação / Votação")
    st.markdown("Veja resultados de votação (se houver):")
    votes = read_votes()
    st.write(votes)

if __name__ == "__main__":
    main()
