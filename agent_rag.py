# agent_rag.py
import os
import json
import math
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from operator import add as add_messages

VDB_DIR = os.environ.get("RAG_VDB_DIR", "./vdb")
DOC_STORE = os.path.join(VDB_DIR, "collab_doc.txt")
VOTES_FILE = os.path.join(VDB_DIR, "votes.json")

os.makedirs(VDB_DIR, exist_ok=True)

def build_llm(model: str = "llama3.1", temperature: float = 0.7):
    return Ollama(model=model, temperature=temperature)

def build_embeddings(model: str = "nomic-embed-text"):
    return OllamaEmbeddings(model=model)


def load_pdf_pages(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    loader = PyPDFLoader(file_path=path)
    pages = loader.load()
    return pages

def build_vectorstore_from_pages(pages, embeddings, persist_directory: str = VDB_DIR, collection_name: str = "pdfs"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings,
                               persist_directory=persist_directory,
                               collection_name=collection_name)
    return vs

def build_retriever(vectorstore, k: int = 6):
    r = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return r

def read_collab_document() -> str:
    if not os.path.exists(DOC_STORE):
        return ""
    with open(DOC_STORE, "r", encoding="utf-8") as f:
        return f.read()

def write_collab_document(content: str):
    with open(DOC_STORE, "w", encoding="utf-8") as f:
        f.write(content)

def append_collab_document(content: str):
    existing = read_collab_document()
    new = existing + ("\n" if existing else "") + content
    write_collab_document(new)

def ensure_votes_file():
    if not os.path.exists(VOTES_FILE):
        with open(VOTES_FILE, "w", encoding="utf-8") as f:
            json.dump({"votes": {}}, f)

def read_votes():
    ensure_votes_file()
    with open(VOTES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def add_vote(key: str, user: str):
    data = read_votes()
    votes = data.get("votes", {})
    votes.setdefault(key, [])
    if user not in votes[key]:
        votes[key].append(user)
    data["votes"] = votes
    with open(VOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return votes[key]


@tool
def retriever_tool(query: str, retriever=None) -> str:
    """
    Busca no vectorstore (retriever.invoke) — o retriever é passado pela closure no agent builder.
    """
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
def update_document_tool(new_text: str, user: str = "anonymous") -> str:
    """
    Atualiza (apenda) o documento colaborativo com um bloco novo.
    """
    append_collab_document(f"{user}: {new_text}")
    current = read_collab_document()
    return f"Document updated by {user}. Current length: {len(current)} chars."

@tool
def save_document_tool(filename: str = "collab_doc.txt") -> str:
    """
    Salva o documento (o arquivo já está persistido em DOC_STORE; aqui podemos copiar para filename)
    """
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
    Summarize a text using a short LLM call. For simplicity we call a Ollama here synchronously.
    (In production you may want to route this through the same `llm.bind_tools` chain.)
    """
    llm = build_llm()
    prompt = [
        SystemMessage(content="Você é um assistente que gera resumos claros e curtos."),
        HumanMessage(content=f"Resuma em até {max_chars} caracteres: {text}")
    ]
    res = llm.invoke(prompt)
    return res.content

@tool
def vote_tool(key: str, user: str) -> str:
    """
    Vota em uma opção (key) por user. Retorna lista atual de votantes.
    """
    voters = add_vote(key, user)
    return f"Vote recorded for '{key}'. Current votes: {voters}"

def build_agent(retriever, llm):
    """
    Monta o StateGraph que controla o loop LLM <-> tools.
    Compatível com Ollama (sem bind_tools nem chamadas automáticas).
    """

    @tool
    def _retriever_wrapper(query: str) -> str:
        """Busca informações no vectorstore."""
        return retriever_tool(query, retriever=retriever)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    system_prompt = (
        "Você é um assistente colaborativo. Use as ferramentas disponíveis (buscar, resumir, atualizar documento, salvar, votar) "
        "quando for relevante. Sempre explique suas ações de forma breve e clara."
    )

    def call_llm(state: AgentState):
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        res = llm.invoke(msgs)
        return {"messages": [res]}

    def take_action(state: AgentState):
        last = state["messages"][-1]
        content = last.content.lower()

        result = None
        if "buscar" in content or "procurar" in content:
            q = content.replace("buscar", "").replace("procurar", "").strip()
            result = _retriever_wrapper.invoke(query=q)
        elif "resum" in content:
            result = summarize_tool.invoke(text=content)
        elif "atualiz" in content:
            result = update_document_tool.invoke(new_text=content)
        elif "salvar" in content:
            result = save_document_tool.invoke(filename="collab_doc.txt")
        elif "votar" in content:
            result = vote_tool.invoke(key="ideia", user="user")

        if result:
            return {"messages": [ToolMessage(tool_call_id="manual", name="tool_action", content=str(result))]}
        else:
            return {"messages": [ToolMessage(tool_call_id="none", name="no_action", content="Nenhuma ferramenta acionada.")]}        

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", take_action)
    graph.add_edge("llm", END)
    graph.set_entry_point("llm")
    return graph.compile()
