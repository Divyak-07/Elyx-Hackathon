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
origins = [
    "https://elyx-hackathon.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google Gemini ---
try:
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
    month_msgs = [m for m in messages if m.timestamp.strftime("%B %Y") == month_name]

    prompt = f"""
    Analyze Rohan Patel's health journey in {month_name}.
    Messages:
    {json.dumps([m.dict() for m in month_msgs], indent=2, default=str)}

    Provide:
    - Primary goal/trigger
    - Friction points (list)
    - Final outcome
    - Persona analysis: before and after
    """
    ai_text = generate_with_gemini(prompt)

    if not ai_text or "AI service is currently unavailable" in ai_text:
        return EpisodeAnalysis(
            month_name=month_name,
            primary_goal_trigger="Ongoing health optimization.",
            friction_points=["Coordination challenges."],
            final_outcome="Steady progress.",
            persona_analysis=PersonaState(before="Following plan", after="More consistent")
        )

    return EpisodeAnalysis(
        month_name=month_name,
        primary_goal_trigger=ai_text,
        friction_points=[],
        final_outcome="See AI summary above",
        persona_analysis=PersonaState(before="", after="")
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

@app.get("/metrics/sentiment")
async def get_sentiment_metrics():
    return [
        {"month": "January 2025", "score": -0.2},
        {"month": "February 2025", "score": 0.1},
        {"month": "March 2025", "score": -0.5},
        {"month": "April 2025", "score": 0.4},
        {"month": "May 2025", "score": -0.3},
        {"month": "June 2025", "score": 0.7},
        {"month": "July 2025", "score": 0.8},
        {"month": "August 2025", "score": 0.9}
    ]

@app.get("/reports/weekly")
async def get_weekly_report():
    prompt = "Summarize key events in Rohan Patel's health journey this week. Output in HTML bullet points."
    ai_summary = generate_with_gemini(prompt)
    return {"week_of": "August 11, 2025", "summary": ai_summary}

@app.get("/analysis/{month_name}", response_model=EpisodeAnalysis)
async def analyze_month(month_name: str):
    return get_ai_analysis(month_name, MESSAGES)

@app.post("/chat")
async def chat_with_ai(query: ChatQuery):
    prompt = f"Answer the question: {query.query}\nUse Rohan Patel's journey as context."
    ai_response = generate_with_gemini(prompt)
    return {"response": ai_response}
