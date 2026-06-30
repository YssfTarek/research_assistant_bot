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


# ==================== 3. MULTI-AGENT EXECUTION NODES ====================
async def investigator_node(state: ReportState):
    """Agent 1: The Triage Specialist. Evaluates if database tools are required."""
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    investigator_sys = SystemMessage(content=(
        "You are an Elite Data Investigator. Look at the user's request.\n"
        "If they want fresh metrics, updates, or a data lookup, use your tools immediately.\n"
        "If they are just asking to edit layout, format text, or change wording, do not use tools; "
        "just pass the conversation forward."
    ))

    # Mix running memory context back into system prompt dynamically
    if summary:
        input_message = SystemMessage(content=f"{investigator_sys.content}\n\nPrior Context: {summary}")
        input_messages = [input_message] + messages
    else:
        input_messages = [investigator_sys] + messages
    
    response = await model_with_tools.ainvoke(input_messages)
    response.name = "investigator"
    return {"messages": [response]}

async def writer_node(state: ReportState):
    """Agent 2: The Content Artisan. Focuses purely on writing clean markdown."""
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    writer_sys = SystemMessage(content=(
        "You are a Master Content Artisan. Your task is to take the user's request and any prior context, "
        "and produce a clean, professional, beautifully structured executive markdown report. Do not add new data; just format and refine."
        "Do not include conversational filler, introductions, or signatures. Output raw markdown only."
    ))

    # Mix running memory context back into system prompt dynamically
    if summary:
        input_message = SystemMessage(content=f"{writer_sys.content}\n\nPrior Context: {summary}")
        input_messages = [input_message] + messages
    else:
        input_messages = [writer_sys] + messages
    
    response = await model.ainvoke(input_messages)
    response.name = "writer"
    return {"messages": [response]}


# ==================== 4. CONDITIONAL LOOP ROUTER ====================
def post_agent_router(state: ReportState):
    """Deterministic routing logic to coordinate our multi-agent workflow."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if tools_condition(state) == "tools":
        return "tools"
    
    if last_message and getattr(last_message, "name", "") == "investigator":
        return "writer"
    
    if len(messages) > 6:  # Arbitrary threshold to trigger summarization
        return "summarize"
    
    return END


# ==================== 5. TOPOLOGY WIRING ====================
workflow = StateGraph(ReportState)

# Register Nodes
workflow.add_node("investigator", investigator_node)
workflow.add_node("writer", writer_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summarize", summarization_node)

# Entry Connection: Always start with the agent
workflow.add_edge(START, "investigator")

# Tools always route their results right back back to the investigator 
workflow.add_edge("tools", "investigator")

# Apply our conditional router to the investigator node
workflow.add_conditional_edges("investigator", post_agent_router, {
    "tools": "tools",
    "writer": "writer",
})

# Apply our conditional router to the writer node so it can evaluate memory bounds
workflow.add_conditional_edges("writer", post_agent_router, {
    "summarize": "summarize",
    END: END
})

# If the summarizer runs, its work is complete and it can exit safely until the next turn
workflow.add_edge("summarize", END)

# CRITICAL: We stop here! Do NOT call workflow.compile() unless using Studio
#compiled_reporting_graph = workflow.compile() #uncomment when not using Studio