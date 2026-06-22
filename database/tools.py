import os
from langchain_core.tools import tool
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPA_API_KEY = os.getenv("SUPA_API_KEY")
SUPA_URL = os.getenv("SUPA_URL")

supabase: Client = create_client(SUPA_URL, SUPA_API_KEY)

@tool
def fetch_department_metrics():
    """Queries Supabase to get the latest progress metrics for active projects."""
    response = (
        supabase.table("exec_assitant_test")
        .select("*")
        .execute()
    )
    return response.data