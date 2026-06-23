import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPA_API_KEY = os.getenv("SUPA_API_KEY")
SUPA_URL = os.getenv("SUPA_URL")

if not SUPA_URL or not SUPA_API_KEY:
    raise ValueError("Missing Supabase credentials in your .env file!")

# Initialize Supabase Client
supabase: Client = create_client(SUPA_URL, SUPA_API_KEY)

# High-quality synthetic corporate data
synthetic_departments = [
    {
        "department": "Engineering",
        "budget_allocated": 1500000,
        "budget_spent": 1420000,
        "team_velocity": 82, # lower velocity due to bottlenecks
        "bottleneck_metric": "Code review turnaround averages 4.2 days; tech debt handles 35% of sprint capacity."
    },
    {
        "department": "Marketing",
        "budget_allocated": 800000,
        "budget_spent": 850000, # Over budget
        "team_velocity": 95,
        "bottleneck_metric": "Ad acquisition costs rose 22% this quarter; creative asset approval pipeline is causing a 1-week campaign launch delay."
    },
    {
        "department": "Sales",
        "budget_allocated": 1200000,
        "budget_spent": 980000,
        "team_velocity": 91,
        "bottleneck_metric": "Inbound lead distribution takes 48 hours; enterprise onboarding bottlenecked by legal review cycles."
    },
    {
        "department": "Product",
        "budget_allocated": 500000,
        "budget_spent": 490000,
        "team_velocity": 60, # Severe bottleneck
        "bottleneck_metric": "Cross-dependency blocker: Engineering delays have pushed back 3 core feature launches by two quarters."
    },
    {
        "department": "Customer Success",
        "budget_allocated": 400000,
        "budget_spent": 395000,
        "team_velocity": 88,
        "bottleneck_metric": "Ticket spikes due to recent platform bugs increased average response times from 2 hours to 14 hours."
    }
]

def seed_database():
    print("🚀 Connecting to Supabase and clearing old test rows...")
    
    # Optional: Clear out previous rows to avoid messy duplicates during testing
    try:
        supabase.table("exec_assistant_test").delete().neq("department", "").execute()
    except Exception as e:
        print(f"Note on cleanup: {e} (Moving directly to insertion)")

    print("📥 Inserting fresh synthetic metrics...")
    
    # Insert rows
    response = supabase.table("exec_assistant_test").insert(synthetic_departments).execute()
    
    print("✅ Synthetic data successfully loaded into Supabase!")
    print(f"Rows created: {len(response.data)}")

if __name__ == "__main__":
    seed_database()