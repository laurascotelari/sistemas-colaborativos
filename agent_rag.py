import os
import json
import math
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from operator import add as add_messages

# Define diretorios e arquivos utilizados para armazenamento de dados
VDB_DIR = os.environ.get("RAG_VDB_DIR", "./vdb")
DOC_STORE = os.path.join(VDB_DIR, "collab_doc.txt")
VOTES_FILE = os.path.join(VDB_DIR, "votes.json")

os.makedirs(VDB_DIR, exist_ok=True)

# Funcao para construir o modelo de linguagem 
def build_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return ChatOpenAI(model=model, temperature=temperature)

# Funcao para construir embeddings
def build_embeddings(model: str = "text-embedding-3-small"):
    return OpenAIEmbeddings(model=model)

#carrega paginas de um arquivo PDF
def load_pdf_pages(path: str):
    print(f"Loading PDF pages from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    loader = PyPDFLoader(file_path=path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")
    return pages

# criar um armazenamento vetorial a partir de paginas de documentos
def build_vectorstore_from_pages(pages, embeddings, persist_directory: str = VDB_DIR, collection_name: str = "pdfs"):
    print(f"Building vectorstore in {persist_directory} under collection '{collection_name}'")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Divide o texto em chunks
    chunks = splitter.split_documents(pages)
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings,
                               persist_directory=persist_directory,
                               collection_name=collection_name)
    return vs

# criar um mecanismo de recuperacao de informacoes (retriever)
def build_retriever(vectorstore, k: int = 6):
    print(f"Building retriever with k={k}")
    r = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return r

# le o conteudo do documento colaborativo
def read_collab_document() -> str:
    print(f"Reading collaborative document from {DOC_STORE}")
    if not os.path.exists(DOC_STORE):
        return ""
    with open(DOC_STORE, "r", encoding="utf-8") as f:
        return f.read()

# Funcao para escrever conteudo no documento colaborativo
def write_collab_document(content: str):
    with open(DOC_STORE, "w", encoding="utf-8") as f:
        f.write(content)

# Funcao para adicionar conteudo ao final do documento colaborativo
def append_collab_document(content: str):
    print("Appending to collaborative document.")
    existing = read_collab_document()
    new = existing + ("\n" if existing else "") + content
    write_collab_document(new)

# garante qe o arquivo de votos exista
def ensure_votes_file():
    if not os.path.exists(VOTES_FILE):
        with open(VOTES_FILE, "w", encoding="utf-8") as f:
            json.dump({"votes": {}}, f)

def read_votes():
    ensure_votes_file()
    with open(VOTES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Funcao para adicionar um voto a uma opcao especifica
def add_vote(key: str):
    print(f"Adding vote for key '{key}'.")
    data = read_votes()
    votes = data.get("votes", {})
    votes[key] = votes.get(key, 0) + 1  # Incrementa o contador de votos para a key
    data["votes"] = votes
    with open(VOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return votes[key]

@tool
def retriever_tool(query: str, retriever=None) -> str:
    """
    Busca no vectorstore (retriever.invoke) — o retriever e passado pela closure no agent builder.
    """
    print("Invoking retriever tool.")
    if retriever is None:
        return "No retriever configured."
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant info found."
    out = []
    for i, d in enumerate(docs):
        md = getattr(d, "metadata", {}) or {}
        page = md.get("page") or md.get("page_number") or "?"
        out.append(f"[Doc {i+1} | page {page}] {d.page_content[:800]}")
    return "\n\n".join(out)

@tool
def update_document_tool(new_text: str) -> str:
    """
    Atualiza (apenda) o documento colaborativo com um bloco novo.
    """
    print("Invoking update_document_tool.")
    append_collab_document(f"{new_text}")
    current = read_collab_document()
    return f"Document updated. Current length: {len(current)} chars."

@tool
def save_document_tool(filename: str = "collab_doc.txt") -> str:
    """
    Salva o documento (o arquivo ja esta persistido em DOC_STORE; aqui podemos copiar para filename)
    """
    print("Invoking save_document_tool.")
    try:
        content = read_collab_document()
        target = filename if filename.endswith(".txt") else f"{filename}.txt"
        with open(os.path.join(VDB_DIR, target), "w", encoding="utf-8") as f:
            f.write(content)
        return f"Document saved to {os.path.join(VDB_DIR, target)}"
    except Exception as e:
        return f"Error saving document: {str(e)}"

@tool
def summarize_tool(text: str, max_chars: int = 800) -> str:
    """
    Resuma um texto usando uma chamada curta ao LLM. Para simplicidade, chamamos o ChatOpenAI diretamente.
    """
    print("Invoking summarize_tool.")
    llm = build_llm()
    prompt = [
        SystemMessage(content="Voce e um assistente que gera resumos claros e curtos."),
        HumanMessage(content=f"Resuma em ate {max_chars} caracteres: {text}")
    ]
    res = llm.invoke(prompt)
    return res.content

@tool
def vote_tool(key: str) -> str:
    """
    Vota em uma opcao (key). Retorna o numero atual de votos.
    """
    print("Invoking vote_tool.")
    votes_count = add_vote(key)
    return f"Vote recorded for '{key}'. Total votes: {votes_count}"

def build_agent(retriever, llm):
    """
    Monta o StateGraph que controla o loop LLM <-> tools.
    Tools disponiveis: retriever_tool, update_document_tool, save_document_tool, summarize_tool, vote_tool
    """
    print("Building agent with tools.")

    def retriever_invoke(q: str):
        return retriever.invoke(q)

    @tool
    def _retriever_wrapper(query: str) -> str:
        """ Wrapper para passar o retriever via closure."""
        print("Invoking retriever wrapper tool.")
        return retriever_tool(query, retriever=retriever)

    tools = [_retriever_wrapper, update_document_tool, save_document_tool, summarize_tool, vote_tool]
    llm_with_tools = llm.bind_tools(tools)
    tools_dict = {t.name: t for t in tools}
    print(f"Available tools: {list(tools_dict.keys())}")

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    system_prompt = (
        "You are a collaborative assistant. Use the available tools when asked (search documents, summarize, update the shared document, "
        "save or vote). When you use a tool, create a tool call. Provide clear, short answers and cite sources when relevant."
    )

    # Funcao que chama o modelo de linguagem para processar o estado atual
    def call_llm(state: AgentState):
        # Combina a mensagem do sistema com as mensagens do estado atual
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        # Invoca o LLM com as mensagens fornecidas
        message = llm_with_tools.invoke(msgs)
        print("LLM response received.")
        print(f"LLM response content: {message.content}")
        return {"messages": [message]}

    # Funcao que executa acoes baseadas nas chamadas de ferramentas solicitadas pelo LLM
    def take_action(state: AgentState):
        last_message = state["messages"][-1]
        # Extrai as chamadas de ferramentas da última mensagem
        tool_calls = getattr(last_message, "tool_calls", [])
        results = []
        print(f"Processing {len(tool_calls)} tool calls.")

        # Processa cada chamada de ferramenta
        for call in tool_calls:
            tool_name = call["name"] 
            args = call.get("args", {}) or {}

            # Verifica se a ferramenta existe no dicionário de ferramentas
            if tool_name not in tools_dict:
                result = f"Tool {tool_name} not found."
            else:
                tool = tools_dict[tool_name]  # Recupera a ferramenta correspondente
                try:
                    # Invoca a ferramenta com os argumentos fornecidos
                    if hasattr(tool, "invoke"):
                        result = tool.invoke(args)
                    else:
                        result = tool.run(args)
                except Exception as e:
                    # Captura erros durante a invocação da ferramenta
                    result = f"Error invoking tool {tool_name}: {e}"

            # Adiciona o resultado da ferramenta como uma mensagem
            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=tool_name,
                    content=str(result),
                )
            )

        return {"messages": results}

    # Configura o grafo de estados para controlar o fluxo entre LLM e ferramentas
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)  # Adiciona o nó para chamar o LLM
    graph.add_node("tools", take_action)  # Adiciona o nó para executar ferramentas
    # Define as transições condicionais entre os nós
    graph.add_conditional_edges("llm", should_continue, {True: "tools", False: END})
    graph.add_edge("tools", "llm")  # Conecta o nó de ferramentas de volta ao LLM
    graph.set_entry_point("llm")  # Define o ponto de entrada inicial como o nó do LLM
    return graph.compile()  # Compila e retorna o grafo configurado
