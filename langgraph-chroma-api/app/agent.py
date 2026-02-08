from typing import List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END

from app.tools import vector_search


class AgentState(TypedDict):
    messages: List[BaseMessage]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

tools = [
    Tool(
        name="vector_search",
        description="Search the vector database for relevant information",
        func=vector_search,
    )
]

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an expert assistant specialized in analyzing and summarizing information from OCR-extracted documents.
Your goal is to provide accurate, concise, and visually appealing answers based on the provided context.

Follow these formatting rules strictly:
1. **Reorganize** information into logical sections.
2. **Use Markdown Headers** (##, ###) for major sections.
3. **Use Bullet Points** and numbered lists for readability.
4. **Bold** key terms and dates.
5. If the context contains tables or structured data, represent it as a **Markdown Table**.
6. If the information is missing from the context, state that clearly.

Always aim for a professional and premium look in your responses."""


def agent_node(state: AgentState):
    # Prepare messages including system prompt if it's the first turn or if not already present
    messages = state["messages"]

    # Check if a system message is already present to avoid duplication
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    new_messages = []

    for call in last_message.tool_calls:
        if call["name"] == "vector_search":
            # Extract the query argument - handle both named and positional args
            args = call.get("args", {})
            if isinstance(args, dict):
                result = vector_search(query=args.get("query", ""))
            else:
                result = vector_search(query=str(args))
            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    return {"messages": state["messages"] + new_messages}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

agent_executor = graph.compile()
