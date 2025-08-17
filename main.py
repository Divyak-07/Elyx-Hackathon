# main.py
# Elyx Hackathon Project - Updated with Google Gemini Integration

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

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Elyx Member Journey API",
    description="An API to serve the 8-month communication log for Rohan Patel's health journey.",
    version="1.1.0"
)

# --- CORS Middleware ---
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

# --- Configure Google AI Client ---
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Could not configure Google AI: {e}")

# --- Google Gemini Helper ---
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

# --- AI SIMULATION & GENERATION FUNCTIONS ---
def get_ai_analysis(month_name: str, messages: List[Message]) -> EpisodeAnalysis:
    # Extract messages for that month
    month_msgs = [
        m for m in messages
        if m.timestamp.strftime("%B %Y") == month_name
    ]

    prompt = f"""
    You are Elyx AI, analyzing Rohan Patel's health journey.
    Focus on month: {month_name}.
    
    Based on these messages:
    {json.dumps([m.dict() for m in month_msgs], indent=2, default=str)}

    Provide:
    - Primary goal/trigger
    - Friction points (list)
    - Final outcome
    - Persona analysis: before and after states (short sentences).
    """
    ai_text = generate_with_gemini(prompt)

    # If AI fails, return fallback
    if not ai_text or "AI service is currently unavailable" in ai_text:
        return EpisodeAnalysis(
            month_name=month_name,
            primary_goal_trigger="Ongoing health optimization.",
            friction_points=["Coordination challenges."],
            final_outcome="Steady progress made.",
            persona_analysis=PersonaState(before="Following the plan.", after="More consistent.")
        )

    # For now, just stuff AI text into primary_goal_trigger
    return EpisodeAnalysis(
        month_name=month_name,
        primary_goal_trigger=ai_text,
        friction_points=[],
        final_outcome="See analysis above.",
        persona_analysis=PersonaState(before="", after="")
    )

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Elyx Member Journey API"}

@app.get("/messages", response_model=List[Message], tags=["Messages"])
async def get_all_messages():
    if not MESSAGES:
        raise HTTPException(status_code=404, detail="Journey data not loaded.")
    return MESSAGES

@app.get("/messages/timeline", response_model=List[Message], tags=["Messages"])
async def get_timeline_events():
    milestones = [msg for msg in MESSAGES if msg.tags.type == 'milestone']
    if not milestones:
        raise HTTPException(status_code=404, detail="No milestone events found.")
    return milestones

@app.get("/messages/decision/{message_id}", response_model=Dict, tags=["Messages"])
async def get_decision_and_reasons(message_id: int):
    decision_message = next((msg for msg in MESSAGES if msg.id == message_id and msg.tags.type == 'decision'), None)
    if not decision_message:
        raise HTTPException(status_code=404, detail=f"Decision with ID {message_id} not found.")
    reason_messages = [msg for msg in MESSAGES if msg.tags.type == 'reason' and msg.tags.linked_id == message_id]
    return {"decision": decision_message, "reasons": reason_messages}

@app.get("/metrics/internal", response_model=Dict, tags=["Metrics"])
async def get_internal_metrics():
    if not MESSAGES:
        raise HTTPException(status_code=404, detail="Journey data not loaded.")
    role_counts = {}
    for msg in MESSAGES:
        if msg.role not in ["Member", "Personal Assistant"]:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
    return {"total_elyx_team_interactions": sum(role_counts.values()), "interactions_by_role": role_counts}

@app.get("/metrics/sentiment", tags=["Metrics"])
async def get_sentiment_metrics():
    sentiment_data = [
        {"month": "January 2025", "score": -0.2},
        {"month": "February 2025", "score": 0.1},
        {"month": "March 2025", "score": -0.5},
        {"month": "April 2025", "score": 0.4},
        {"month": "May 2025", "score": -0.3},
        {"month": "June 2025", "score": 0.7},
        {"month": "July 2025", "score": 0.8},
        {"month": "August 2025", "score": 0.9}
    ]
    return sentiment_data

@app.get("/reports/weekly", tags=["Reports"])
async def get_weekly_report():
    prompt = """
    Summarize the key events in Rohan Patel's health journey for the past week.
    Focus on structural health, recovery, cognitive health, and diagnostics.
    Format as HTML with bullet points.
    """
    ai_summary = generate_with_gemini(prompt)
    return {
        "week_of": "August 11, 2025",
        "summary": ai_summary
    }

@app.get("/analysis/{month_name}", response_model=EpisodeAnalysis, tags=["Analysis"])
async def analyze_month(month_name: str):
    return get_ai_analysis(month_name, MESSAGES)
