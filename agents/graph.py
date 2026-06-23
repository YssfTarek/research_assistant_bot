from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from database.tools import fetch_department_metrics
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash") # Matching main.py's default

# Initialize Model
model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# Bind tools to the model
tools = [fetch_department_metrics]
model_with_tools = model.bind_tools(tools)

# Core Prompt Instruction
sys_msg = SystemMessage(content="You are a helpful executive assistant that generates reports based on the latest department metrics.")


# ==================== 1. CUSTOM STATE OBJECT ====================
class ReportState(MessagesState):
    summary: str  # Retains a persistent context string across state iterations


# ==================== 2. SUMMARIZATION NODE ====================
async def summarization_node(state: ReportState):
    existing_summary = state.get("summary", "")
    
    # Retain the last 2 messages for active conversation context, summarize the older turns
    messages_to_summarize = state["messages"][:-2]

    summary_prompt = (
        f"Progressively summarize the conversation history below. Combine it cleanly with "
        f"the existing summary.\n\nExisting Summary: {existing_summary}\n\nNew messages:\n"
    )

    for m in messages_to_summarize:
        summary_prompt += f"{m.type}: {m.content}\n"

    # Safely wrapped in a list to prevent LangChain validation errors
    response = await model.ainvoke([HumanMessage(content=summary_prompt)])
    
    # Generate message deletion events targeting everything condensed
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    return {
        "summary": response.content,
        "messages": delete_messages
    }


# ==================== 3. AGENT EXECUTION NODE ====================
async def agent_node(state: ReportState):
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    # Mix running memory context back into system prompt dynamically
    if summary:
        contextual_sys_message = SystemMessage(
            content=f"{sys_msg.content}\n\nContext summary of prior turns: {summary}"
        )
        input_messages = [contextual_sys_message] + messages
    else:
        input_messages = [sys_msg] + messages
    
    response = await model_with_tools.ainvoke(input_messages)
    return {"messages": [response]}


# ==================== 4. CONDITIONAL LOOP ROUTER ====================
def post_agent_router(state: ReportState):
    """Intercepts turn termination to check if memory cleanup is required."""
    route = tools_condition(state)
    
    # If the model wants to call a tool, respect it and jump straight to the tool node
    if route == "tools":
        return "tools"
        
    # If the model wanted to finish, evaluate message list threshold bounds
    if len(state.get("messages", [])) > 6:
        return "summarize"
        
    return END


# ==================== 5. TOPOLOGY WIRING ====================
workflow = StateGraph(ReportState)

# Register Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summarize", summarization_node)

# Entry Connection: Always start with the agent
workflow.add_edge(START, "agent")

# Evaluate loops directly upon completion of agent execution cycles
workflow.add_conditional_edges(
    "agent",
    post_agent_router,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END
    }
)

# If the summarizer runs, its work is complete and it can exit safely until the next turn
workflow.add_edge("summarize", END)

# Tool executions pipe immediately back to the model to evaluate the result
workflow.add_edge("tools", "agent")

# CRITICAL: We stop here! Do NOT call workflow.compile() unless using Studio
#compiled_reporting_graph = workflow.compile() #uncomment when not using Studio