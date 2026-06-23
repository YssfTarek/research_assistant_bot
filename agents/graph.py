from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from database.tools import fetch_department_metrics
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
    temperature = 0,
    google_api_key = GOOGLE_API_KEY
)

tools = [fetch_department_metrics]
model_with_tools = model.bind_tools(tools)


sys_msg = SystemMessage(content="You are a helpful executive assistant that generates reports based on the latest department metrics.")

async def agent_node(state: MessagesState):
    response = await model_with_tools.ainvoke([sys_msg] + state["messages"])
    return {"messages": response}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))


workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
compiled_reporting_graph = workflow.compile(checkpointer=memory)

studio_reporting_graph = workflow.compile()