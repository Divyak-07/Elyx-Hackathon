# main.py
# Elyx Hackathon Project - Updated with Google Gemini Integration + Chat API

import json
import os
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# --- Pydantic Models for Data Validation ---
class Tag(BaseModel):
    type: Optional[str] = None
    linked_id: Optional[int] = None

class Message(BaseModel):
    id: int
    timestamp: datetime
    sender: str
    role: str
    content: str
    tags: Tag

class PersonaState(BaseModel):
    before: str
    after: str

class EpisodeAnalysis(BaseModel):
    month_name: str
    primary_goal_trigger: str
    friction_points: List[str]
    final_outcome: str
    persona_analysis: PersonaState

class ChatQuery(BaseModel):
    query: str

# --- FastAPI App ---
app = FastAPI(
    title="Elyx Member Journey API",
    description="API to serve Rohan Patel's 8-month health journey",
    version="1.2.0"
)

# --- CORS ---
# This is the section you need to update.
# We have added your Netlify URL to the list of allowed origins.
origins = [
    "https://elyx-hackathon.netlify.app", # Your deployed frontend
    "http://localhost:3000",             # For local development
    "http://127.0.0.1:5500",             # For local development
    "null"                               # For opening index.html directly
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)


# --- Google Gemini ---
try:
    # IMPORTANT: Set your GOOGLE_API_KEY as an environment variable in Render
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Could not configure Google AI: {e}")

def generate_with_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return "AI service is currently unavailable."

# --- Data Loading ---
def load_journey_data() -> List[Message]:
    try:
        with open("journey_data.json", "r") as f:
            data = json.load(f)
            return [Message(**item) for item in data]
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

MESSAGES = load_journey_data()

# --- AI Functions ---
def get_ai_analysis(month_name: str, messages: List[Message]) -> EpisodeAnalysis:
    month_msgs_str = []
    for m in messages:
        if m.timestamp.strftime("%B %Y") == month_name:
            month_msgs_str.append(json.dumps(m.dict(), indent=2, default=str))

    prompt = f"""
    Analyze the health journey of Rohan Patel for the month of {month_name}.
    Here are the messages for this month:
    {month_msgs_str}

    Based on these messages, please provide a concise analysis in the following JSON format:
    {{
      "month_name": "{month_name}",
      "primary_goal_trigger": "A short description of the main focus or event of the month.",
      "friction_points": ["A list of challenges or frustrations encountered.", "Another challenge."],
      "final_outcome": "A summary of the result or resolution by the end of the month.",
      "persona_analysis": {{
        "before": "Rohan's state of mind or engagement level at the start of the month.",
        "after": "Rohan's state of mind or engagement level at the end of the month."
      }}
    }}
    """
    ai_text = generate_with_gemini(prompt)

    try:
        cleaned_response = ai_text.strip().replace("```json", "").replace("```", "")
        analysis_data = json.loads(cleaned_response)
        return EpisodeAnalysis(**analysis_data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error decoding AI analysis response: {e}")
        return EpisodeAnalysis(
            month_name=month_name,
            primary_goal_trigger="Analysis is currently unavailable.",
            friction_points=[],
            final_outcome="Could not generate a summary.",
            persona_analysis=PersonaState(before="N/A", after="N/A")
        )

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to Elyx Member Journey API"}

@app.get("/messages", response_model=List[Message])
async def get_all_messages():
    if not MESSAGES:
        raise HTTPException(status_code=404, detail="Journey data not loaded")
    return MESSAGES

@app.get("/messages/timeline", response_model=List[Message])
async def get_timeline_events():
    milestones = [msg for msg in MESSAGES if msg.tags.type == "milestone"]
    if not milestones:
        raise HTTPException(status_code=404, detail="No milestones found")
    return milestones

@app.get("/messages/decision/{message_id}", response_model=Dict)
async def get_decision_and_reasons(message_id: int):
    decision_message = next((m for m in MESSAGES if m.id == message_id and m.tags.type == "decision"), None)
    if not decision_message:
        raise HTTPException(status_code=404, detail=f"No decision with ID {message_id}")
    reasons = [m for m in MESSAGES if m.tags.type == "reason" and m.tags.linked_id == message_id]
    return {"decision": decision_message, "reasons": reasons}

@app.get("/metrics/internal")
async def get_internal_metrics():
    role_counts = {}
    for msg in MESSAGES:
        if msg.role not in ["Member", "Personal Assistant"]:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
    return {"total_elyx_team_interactions": sum(role_counts.values()), "interactions_by_role": role_counts}

@app.get("/episodes/{month_name}", response_model=EpisodeAnalysis)
async def analyze_month_endpoint(month_name: str):
    decoded_month_name = month_name.replace("%20", " ")
    return get_ai_analysis(decoded_month_name, MESSAGES)

@app.post("/chat")
async def chat_with_ai(query: ChatQuery):
    context = "\n".join([f"- {m.sender} ({m.role}) on {m.timestamp.strftime('%B %d, %Y')}: {m.content}" for m in MESSAGES])
    
    prompt = f"""
    You are an AI assistant for Elyx, a personalized health service. Your name is Elyx AI.
    Your task is to answer questions about the health journey of a member named Rohan Patel.
    You must answer based *only* on the conversation history provided below.
    Be friendly, concise, and helpful.
    If the answer cannot be found in the conversation history, state that you do not have that information in the provided context.

    Here is the full conversation history for Rohan Patel's journey:
    ---
    {context}
    ---

    Based on the provided history, please answer the following question:
    Question: "{query.query}"
    """
    ai_response = generate_with_gemini(prompt)
    return {"response": ai_response}