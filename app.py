# Laura Ferré Scotelari - 12543436

import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Carrega variaveis de ambiente do arquivo .env
load_dotenv()

# Importa funcoes especificas do modulo agent_rag
from agent_rag import (
    build_llm,  #constroi o modelo de linguagem
    build_embeddings,  #cria embeddings
    build_vectorstore_from_pages,  #cria um armazenamento vetorial a partir de paginas
    build_retriever,  #cria um mecanismo de recuperação de informações
    build_agent,  #cria o agente de IA
    load_pdf_pages,  #carrega paginas de um arquivo PDF
    read_collab_document,  #le o documento colaborativo
    write_collab_document,  #escreve no documento colaborativo
    read_votes  #le os votos registrados
)

USERS = ["Laura", "Pedro", "Maria", "Lucas"]

# obtem o diretorio compartilhado onde o indice vetorial e armazenado
def get_shared_vectorstore_dir():
    return os.environ.get("RAG_VDB_DIR", "./vdb")

# Função para garantir que o estado da sessão do Streamlit esteja inicializado corretamente
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

# Funcao para criar ou atualizar o indice vetorial a partir de um arquivo PDF enviado pelo usuario
def build_or_update_index(uploaded_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1] 
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_bytes) 
        tmp_path = tmp.name 
    embeddings = build_embeddings() 
    pages = load_pdf_pages(tmp_path) 
    vs = build_vectorstore_from_pages(pages, embeddings, persist_directory=get_shared_vectorstore_dir(), collection_name="book") 
    retriever = build_retriever(vs, k=st.session_state.k) 
    os.unlink(tmp_path)  # Remove o arquivo temporário
    return retriever

# Função principal que define a interface do Streamlit
def main():
    st.set_page_config(page_title="Sistemas Colaborativos", page_icon="📄") 
    ensure_session_state() 

    st.title("RAG - Sistemas Colaborativos")
    st.markdown("Cenário:  Após a entrada de novos membros, um time de devs. precisa revisar e sistematizar as **Definitions of Done (DoD)** e **Definitions of Ready (DoR)** do time")

    # Configurações da barra lateral
    with st.sidebar:
        st.header("Usuário")
        st.session_state.selected_user = st.selectbox("Usuário Atual", USERS, index=USERS.index(st.session_state.selected_user))  # Seleção do usuario atual
        st.caption("Mensagens enviadas serão atribuídas a este usuário (simulado).")

        st.header("Documentos")
        uploaded = st.file_uploader("Upload PDF (1 por vez)", type=["pdf"])  # Upload de arquivos PDF
        if uploaded:
            if st.button("Crie/Atualize index", type="primary"):
                with st.spinner("Criando index..."):
                    st.session_state.retriever = build_or_update_index(uploaded.read(), uploaded.name)  # Cria ou atualiza o indice vetorial
                st.success("Índice criado / atualizado.")

        st.divider()
        st.header("Configuração do Agente")
        st.session_state.k = st.slider("Recuperar k chunks", min_value=2, max_value=10, value=st.session_state.k, step=1)  # Configuração do nummero de chunks
        st.session_state.model = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o"], index=0)  # Seleção do modelo de LLM
        st.session_state.temperature = st.slider("Temperatura", min_value=0.0, max_value=2.0, value=float(st.session_state.temperature), step=0.1)  # Configuração da temperatura

        if st.button("(re)Criar Agente"):
            if "retriever" not in st.session_state or st.session_state.retriever is None:
                st.warning("Construa um índice primeiro (use 'Build/Update Index').")
            else:
                # Reconfigura o modelo LLM
                llm = build_llm(
                    model=st.session_state.model,
                    temperature=st.session_state.temperature
                )

                # Atualiza o retriever com o numero de chunks definido pelo usuario
                retriever = st.session_state.retriever
                retriever.search_kwargs["k"] = st.session_state.k

                # Cria o agente e armazena no session_state
                st.session_state.agent = build_agent(retriever, llm)

                st.success("Agent created and connected to the index.")

    # Seção de chat compartilhado
    st.subheader("Chat Compartilhado")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{msg.get('user','User')}**: {msg['content']}")  # Exibe mensagens do usuario
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])  # Exibe mensagens do assistente

    prompt = st.chat_input(f"{st.session_state.selected_user} diz: ")  # Entrada de texto para o chat
    if prompt:
        st.session_state.messages.append({"user": st.session_state.selected_user, "role": "user", "content": prompt})  # Adiciona a mensagem do usuario ao estado da sessão
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.selected_user}**: {prompt}")

        if st.session_state.agent is None:
            with st.chat_message("assistant"):
                st.warning("Crie o agente antes de fazer perguntas.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    # Invoca o agente para processar a mensagem
                    result = st.session_state.agent.invoke({"messages": [{"type": "human","content": prompt}]})  
                    answer = result["messages"][-1].content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    st.subheader("Documento Colaborativo")
    st.markdown("O documento coletivo exibido abaixo é atualizado pela ferramenta `update_document_tool`.")
    doc = read_collab_document()
    if doc:
        # Documento colaborativo
        st.text_area("Documento Colaborativo (somente leitura)", value=doc, height=220) 
    else:
        st.write("_Documento vazio_")

    st.subheader("Coordenação / Votação")
    st.markdown("Veja resultados de votação (se houver):")
    votes = read_votes()  
    st.write(votes)  
    
# Função principal
if __name__ == "__main__":
    main()
