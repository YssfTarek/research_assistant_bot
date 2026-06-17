from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
    response = model.invoke(messages)
    return response.content