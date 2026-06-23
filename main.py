import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from supabase import create_client, Client
from langchain_core.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from agents.graph import workflow  # Import the uncompiled workflow

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPA_API_KEY = os.getenv("SUPA_API_KEY")
SUPA_URL = os.getenv("SUPA_URL")

SUPA_MEM_URI = os.getenv("SUPA_MEM_URI")

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
    temperature = 0,
    google_api_key = GOOGLE_API_KEY
)

class Prompt(BaseModel):
    query: str

# ==================== FASTAPI LIFESPAN MANAGER ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize the async checkpointer from your environment variable
    async with AsyncPostgresSaver.from_conn_string(os.getenv("SUPA_MEM_URI")) as checkpointer:
        # 2. Run table checking/generation asynchronously
        await checkpointer.setup()
        
        # 3. Compile the graph with your new async database memory layer
        # Storing it in app.state makes it accessible across all route files
        app.state.graph = workflow.compile(checkpointer=checkpointer)
        
        # 4. Turn control over to FastAPI. The connection stays active!
        yield
# ==================================================================

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def say_hi():
    return {"Hello, World!"}

@app.get("/api")
async def api_endpoint():
    return {"message": f"This is your google api key: {GOOGLE_API_KEY}."}

@app.post("/chat")
async def chat_with_google_genai(prompt: Prompt):
    messages = [
        ("human", f"{prompt.query}"),
    ]
    response = model.invoke(messages)@tool
def get_supa_rows():
    supabase: Client = create_client(SUPA_URL, SUPA_API_KEY)
    response = (
        supabase.table("exec_assitant_test")
        .select("*")
        .execute()
    )
    return response.data



@app.post("/test_supa")
async def test_supa():
    supabase: Client = create_client(SUPA_URL, SUPA_API_KEY)
    response = (
        supabase.table("exec_assitant_test")
        .select("*")
        .execute()
    )
    return response.data

@app.post("/trigger_report")
async def handle_report_request(
    prompt: Prompt,
    x_session_id: str = Header(..., description="Session ID for the request"),
):
    config = {"configurable": {"thread_id": x_session_id}}

    graph = app.state.graph  # Access the compiled graph from app.state

    async def event_stream():
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": prompt.query}]},
            version= "v2",
            config=config
        ):

            if event["event"] == "on_chat_model_stream":
                chunk  = event["data"]["chunk"]
                if chunk.content:
                    raw_text = ""

                    if isinstance(chunk.content, list):
                        for item in chunk.content:
                            if isinstance(item, dict) and "text" in item:
                                raw_text += item["text"]
                            elif hasattr(item, "text"):
                                raw_text += item.text
                    else:
                        raw_text = str(chunk.content)

                    if raw_text:
                        yield f"data: {json.dumps({'text': raw_text})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream ")