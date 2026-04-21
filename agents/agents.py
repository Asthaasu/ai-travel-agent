import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator

load_dotenv()

from agents.tools import search_flights, search_hotels, send_travel_email

# ── State definition ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_email: str
    travel_plan: str
    awaiting_approval: bool

# ── LLMs ─────────────────────────────────────────────────────────────────────
tools = [search_flights, search_hotels, send_travel_email]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful AI Travel Agent. Your job is to:
1. Understand the user's travel request (destination, dates, preferences)
2. Search for available flights using the search_flights tool
3. Search for hotels using the search_hotels tool  
4. Compile a comprehensive travel plan
5. When the user approves, send the plan to their email using send_travel_email

Always be friendly and thorough. When presenting a travel plan, format it clearly with sections for flights, hotels, and tips.
After presenting the plan, ask the user if they'd like it emailed to them."""

# ── Node functions ────────────────────────────────────────────────────────────
def agent_node(state: AgentState):
    """Main agent that decides what to do next."""
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Router: decide whether to call tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ── Build the graph ───────────────────────────────────────────────────────────
def build_agent():
    workflow = StateGraph(AgentState)
    
    tool_node = ToolNode(tools)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# ── Public runner ─────────────────────────────────────────────────────────────
agent_graph = build_agent()

def run_agent(user_message: str, history: list, user_email: str = "") -> str:
    """Run one turn of the agent and return the response."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=user_message))
    
    state = {
        "messages": messages,
        "user_email": user_email,
        "travel_plan": "",
        "awaiting_approval": False
    }
    
    result = agent_graph.invoke(state)
    
    last_msg = result["messages"][-1]
    return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
