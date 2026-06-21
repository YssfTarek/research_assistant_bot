from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from supabase import create_client, Client
from langchain_core.tools import tool

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPA_API_KEY = os.getenv("SUPA_API_KEY")
SUPA_URL = os.getenv("SUPA_URL")

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0,
    google_api_key = GOOGLE_API_KEY
)

class Prompt(BaseModel):
    query: str

app = FastAPI()

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



@app.post("/exec_assistant")
async def exec_assistant(prompt: Prompt):
    supabase: Client = create_client(SUPA_URL, SUPA_API_KEY)
    response = (
        supabase.table("exec_assitant_test")
        .select("*")
        .execute()
    )
    messages = [
        ("human", f"{prompt.query}"),
    ]
    response = model.invoke(messages)
    return response.content
    return response.content

@app.post("/test_supa")
async def test_supa():
    supabase: Client = create_client(SUPA_URL, SUPA_API_KEY)
    response = (
        supabase.table("exec_assitant_test")
        .select("*")
        .execute()
    )
    return response.data