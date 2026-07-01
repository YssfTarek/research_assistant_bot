import os
import json
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from supabase import create_client, Client
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import RemoveMessage
from agents.graph import workflow  # Import your clean uncompiled workflow

load_dotenv()

SUPA_MEM_URI = os.getenv("SUPA_MEM_URI")

class Prompt(BaseModel):
    query: Optional[str] = None  # Allows empty body for the initial trigger click

# ==================== FASTAPI LIFESPAN MANAGER ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Establish persistent connection pool wrapper inside Supabase
    # Use 'async with' to cleanly unpack the saver and let it auto-initialize
    async with AsyncPostgresSaver.from_conn_string(SUPA_MEM_URI) as pool_saver:
        # 1. Automatically run migrations/setup if needed
        await pool_saver.setup()
        
        # 2. Assign the cleanly setup checkpointer instance to FastAPI's state
        app.state.graph = workflow.compile(checkpointer=pool_saver)
        
        # 3. Yield control to keep the API server alive and listening
        yield
        
app = FastAPI(lifespan=lifespan)

@app.post("/trigger_report")
async def handle_report_request(
    prompt: Prompt,
    x_session_id: Optional[str] = Header(None, description="Session ID for the request"),
):
    # Establish lifecycle state: reuse existing session or generate a new one
    is_initial_click = x_session_id is None
    active_session_id = x_session_id if not is_initial_click else str(uuid4())
    config = {"configurable": {"thread_id": active_session_id}}
    
    # Auto-inject default prompt text if it's the very first automated click
    if is_initial_click or not prompt.query:
        final_query = "Generate executive report for present projects using department metrics."
    else:
        final_query = prompt.query

    graph = app.state.graph

    async def event_stream():
        # Let the client know what thread session id folder we are streaming into
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
                    
                    # Clean block parser for Gemini's structural outputs
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

                    # Stream clean data payloads back to the listener
                    if thought_chunk:
                        yield f"data: {json.dumps({'type': 'thought', 'content': thought_chunk})}\n\n"
                    if content_chunk:
                        yield f"data: {json.dumps({'type': 'report', 'content': content_chunk})}\n\n"
                        
    return StreamingResponse(event_stream(), media_type="text/event-stream")