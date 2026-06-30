import os
import json
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from supabase import create_client, Client
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import RemoveMessage
from agents.graph import workflow  # Import the uncompiled workflow
from uuid import uuid4

load_dotenv()

# LangChain Tracing Setup
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")  # Default to gemini-3.5-flash if not set

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPA_API_KEY = os.getenv("SUPA_API_KEY")
SUPA_URL = os.getenv("SUPA_URL")
SUPA_MEM_URI = os.getenv("SUPA_MEM_URI")

# Initialize Model (Using standard stable Gemini production naming conventions)
model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

class Prompt(BaseModel):
    query: Optional[str] = None

# ==================== FASTAPI LIFESPAN MANAGER ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize the async checkpointer from your environment variable
    async with AsyncPostgresSaver.from_conn_string(SUPA_MEM_URI) as checkpointer:
        # 2. Run table checking/generation asynchronously
        await checkpointer.setup()
        
        # 3. Compile the graph with your async database memory layer
        app.state.graph = workflow.compile(checkpointer=checkpointer)
        
        # 4. Turn control over to FastAPI. The connection stays active!
        yield
# ==================================================================

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def say_hi():
    return {"message": "Hello, World!"} # Fixed malformed Set type into a Dict

@app.get("/api")
async def api_endpoint():
    return {"message": f"This is your google api key: {GOOGLE_API_KEY}."}

@app.post("/chat")
async def chat_with_google_genai(prompt: Prompt):
    messages = [
        ("human", f"{prompt.query}"),
    ]
    # Fixed: Swapped to await + ainvoke and added missing return statement
    response = await model.ainvoke(messages)
    return {"response": response.content}

@app.post("/test_supa")
async def test_supa():
    """Diagnostic endpoint to test raw connection credentials directly."""
    supabase: Client = create_client(SUPA_URL, SUPA_API_KEY)
    response = (
        supabase.table("exec_assistant_test")  # Fixed typo: 'assitant' -> 'assistant'
        .select("*")
        .execute()
    )
    return response.data

@app.post("/trigger_report")
async def handle_report_request(
    prompt: Prompt,
    x_session_id: Optional[str] = Header(None, description="Session ID for the request"),
):
    is_initial_click = x_session_id is None

    active_session_id  = x_session_id if not is_initial_click else str(uuid4())
    config = {"configurable": {"thread_id": active_session_id}}

    if is_initial_click or not prompt.query:
        final_query = "Generate executive report for present projects using department metrics."
    else:
        final_query = prompt.query

    graph = app.state.graph  # Access the compiled graph from app.state

    async def event_stream():
        yield f"data: {json.dumps({'session_id': active_session_id})}\n\n"

        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": final_query}]},
            version="v2",
            config=config
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    thought_chunk = ""
                    content_chunk = ""
                    if isinstance(chunk.content, list):
                        for item in chunk.content:
                            if isinstance(item, dict):
                                if item.get("type") == "thinking" or "thinking" in item:
                                    thought_chunk += item.get("thinking", item.get("text", ""))
                                elif "text" in item:
                                    content_chunk += item["text"]
                            else:
                                is_thinking = getattr(item, "type", None) == "thinking"
                                if is_thinking and hasattr(item, "text"):
                                    thought_chunk += item.text
                                elif hasattr(item, "text"):
                                    content_chunk += item.text
                    else:
                        content_chunk = str(chunk.content)
                    
                    if thought_chunk:
                        yield f"data: {json.dumps({'type': 'thought', 'content': thought_chunk})}\n\n"
                    if content_chunk:
                        yield f"data: {json.dumps({'type': 'report', 'content': content_chunk})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/clear_session")
async def clear_session(x_session_id: str = Header(...)):
    graph = app.state.graph
    config = {"configurable": {"thread_id": x_session_id}}
    
    current_state = await graph.aget_state(config)
    if not current_state.values:
        raise HTTPException(status_code=404, detail="Session not found or already empty.")
        
    existing_summary = current_state.values.get("summary", "")
    messages = current_state.values.get("messages", [])
    
    # Generate instructions to delete all messages from the checkpoints table array
    delete_commands = [RemoveMessage(id=m.id) for m in messages]
    
    await graph.aupdate_state(
        config,
        values={
            "messages": delete_commands,
            "summary": f"User logged out. Historical context: {existing_summary}"
        },
        as_node="agent"
    )
    return {"status": "success", "detail": "Messages purged. Summary retained for 24-hour cleanup window."}