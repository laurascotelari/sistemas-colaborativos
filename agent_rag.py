# agent_rag.py
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

# === Configs simples ===
VDB_DIR = os.environ.get("RAG_VDB_DIR", "./vdb")
DOC_STORE = os.path.join(VDB_DIR, "collab_doc.txt")
VOTES_FILE = os.path.join(VDB_DIR, "votes.json")

os.makedirs(VDB_DIR, exist_ok=True)

# === Builders ===
def build_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return ChatOpenAI(model=model, temperature=temperature)

def build_embeddings(model: str = "text-embedding-3-small"):
    return OpenAIEmbeddings(model=model)

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
    Summarize a text using a short LLM call. For simplicity we call a ChatOpenAI here synchronously.
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
    Tools disponíveis: retriever_tool (recebe retriever via closure), update_document_tool, save_document_tool, summarize_tool, vote_tool
    """

    def retriever_invoke(q: str):
        return retriever.invoke(q)

    @tool
    def _retriever_wrapper(query: str) -> str:
        return retriever_tool(query, retriever=retriever)

    tools = [_retriever_wrapper, update_document_tool, save_document_tool, summarize_tool, vote_tool]
    llm_with_tools = llm.bind_tools(tools)
    tools_dict = {t.name: t for t in tools}

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    system_prompt = (
        "You are a collaborative assistant. Use the available tools when asked (search documents, summarize, update the shared document, "
        "save or vote). When you use a tool, create a tool call. Provide clear, short answers and cite sources when relevant."
    )

    def call_llm(state: AgentState):
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm_with_tools.invoke(msgs)
        return {"messages": [message]}

    def take_action(state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t["name"]
            args = t["args"] or {}
            if tool_name not in tools_dict:
                result = f"Tool {tool_name} not found."
            else:
                func = tools_dict[tool_name]

                try:
                    result = func.invoke(**args)
                except TypeError:
                    if "query" in args:
                        result = func.invoke(args["query"])
                    elif "text" in args:
                        result = func.invoke(args["text"])
                    elif "filename" in args:
                        result = func.invoke(args["filename"])
                    elif "key" in args and "user" in args:
                        result = func.invoke(args["key"], args["user"])
                    else:
                        result = "Tool invocation failed due to args mismatch."

            results.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=str(result)))
        return {"messages": results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "tools", False: END})
    graph.add_edge("tools", "llm")
    graph.set_entry_point("llm")
    return graph.compile()
