# main.py
# This file contains the FastAPI application updated to call the Google Gemini model.

import json
import os
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Body
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
    version="1.0.0"
)

# --- CORS Middleware ---
origins = [
    "https://elyx-hackathon.netlify.app",
    "http://localhost",
    "https://elyx-hackathon.onrender.com",
    "http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure Google AI Client ---
# The API key is read from an environment variable set in Render.
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Could not configure Google AI: {e}")


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
    # This remains a simulation for the persona analysis feature
    pre_written_analyses = {
        "February 2025": { "primary_goal_trigger": "Rohan expresses anxiety over an upcoming board presentation...", "friction_points": ["Garmin HR zones wrong...", "Plan is sparse..."], "final_outcome": "A foundational plan is created...", "persona_analysis": { "before": "Anxious and data-skeptical...", "after": "Becoming more engaged..." } },
        "May 2025": { "primary_goal_trigger": "Rohan wakes up with a sudden viral illness...", "friction_points": ["Frustration over setback..."], "final_outcome": "The team executes a 'Sick Day Protocol'...", "persona_analysis": { "before": "Feeling confident...", "after": "Frustrated but sees value..." } },
        "August 2025": { "primary_goal_trigger": "Rohan shifts focus to long-term goals...", "friction_points": ["Muscle soreness...", "Whoop strap rash..."], "final_outcome": "Long-term goals are formalized...", "persona_analysis": { "before": "Engaged member...", "after": "Proactive co-manager..." } }
    }
    if month_name in pre_written_analyses:
        return EpisodeAnalysis(month_name=month_name, **pre_written_analyses[month_name])
    else:
        return EpisodeAnalysis(month_name=month_name, primary_goal_trigger="Ongoing health optimization.", friction_points=["Logistical coordination."], final_outcome="Steady progress.", persona_analysis={ "before": "Following the plan.", "after": "More consistent." })

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def read_root(): return {"message": "Welcome to the Elyx Member Journey API"}

@app.get("/messages", response_model=List[Message], tags=["Messages"])
async def get_all_messages():
    if not MESSAGES: raise HTTPException(status_code=404, detail="Journey data not loaded.")
    return MESSAGES

# ... (other existing endpoints remain the same) ...
@app.get("/messages/timeline", response_model=List[Message], tags=["Messages"])
async def get_timeline_events():
    milestones = [msg for msg in MESSAGES if msg.tags.type == 'milestone']
    if not milestones: raise HTTPException(status_code=404, detail="No milestone events found.")
    return milestones

@app.get("/messages/decision/{message_id}", response_model=Dict, tags=["Messages"])
async def get_decision_and_reasons(message_id: int):
    decision_message = next((msg for msg in MESSAGES if msg.id == message_id and msg.tags.type == 'decision'), None)
    if not decision_message: raise HTTPException(status_code=404, detail=f"Decision with ID {message_id} not found.")
    reason_messages = [msg for msg in MESSAGES if msg.tags.type == 'reason' and msg.tags.linked_id == message_id]
    return {"decision": decision_message, "reasons": reason_messages}

@app.get("/metrics/internal", response_model=Dict, tags=["Metrics"])
async def get_internal_metrics():
    if not MESSAGES: raise HTTPException(status_code=404, detail="Journey data not loaded.")
    role_counts = {}
    for msg in MESSAGES:
        if msg.role not in ["Member", "Personal Assistant"]:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
    return {"total_elyx_team_interactions": sum(role_counts.values()), "interactions_by_role": role_counts}

@app.get("/metrics/sentiment", tags=["Metrics"])
async def get_sentiment_metrics():
    """
    Provides simulated monthly sentiment scores based on Rohan's journey.
    """
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

# FIX: Add the missing /reports/weekly endpoint
@app.get("/reports/weekly", tags=["Reports"])
async def get_weekly_report():
    """
    Generates a simulated AI-powered weekly summary.
    """
    return {
        "week_of": "August 11, 2025",
        "summary": """
        <p>This week saw a strong focus on long-term goal setting and proactive health measures.</p>
        <ul class='list-disc list-inside mt-2 space-y-1'>
            <li><b>Structural Health:</b> Long-term goals for strength (Deadlift 1.5x bodyweight), cardio (Top 10% VO2 Max), and stability were formalized. This provides clear, measurable targets for the next 12-24 months.</li>
            <li><b>Recovery Protocol:</b> A successful experiment was conducted using a post-workout protein/creatine shake, which subjectively reduced muscle soreness by 50% and objectively improved Whoop recovery scores. This protocol is now a permanent part of the plan.</li>
            <li><b>Cognitive Health:</b> A new goal to learn piano was endorsed by the medical team as an excellent intervention for building cognitive reserve and promoting neuroplasticity.</li>
            <li><b>Diagnostics:</b> Proactive diagnostics, including a DEXA scan, VO2 Max test, and a full-body MRI, have been scheduled for the upcoming weeks to establish a comprehensive health baseline.</li>
        </ul>
        <p class='mt-4'><b>Key takeaway:</b> The program is successfully transitioning from foundational stabilization to building a platform for high performance and longevity, with a focus on data-driven protocols and measurable, long-term goals.</p>
        """
    }

@app.get("/episodes/{month_name}", response_model=EpisodeAnalysis, tags=["Episodes"])
async def get_episode_analysis(month_name: str):
    try:
        month_messages = [msg for msg in MESSAGES if datetime.strptime(msg.timestamp.strftime('%B %Y'), '%B %Y') == datetime.strptime(month_name, '%B %Y')]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month format. Use 'Month YYYY'.")
    if not month_messages: raise HTTPException(status_code=404, detail=f"No data found for {month_name}.")
    return get_ai_analysis(month_name, month_messages)

@app.post("/chat", tags=["AI Agent"])
async def chat_with_agent(payload: ChatQuery):
    """
    NEW: Takes a user query, sends it to the Gemini model with conversation context,
    and returns the AI-generated answer.
    """
    if not genai.api_key:
        raise HTTPException(status_code=500, detail="Google API key is not configured on the server.")

    # Create a simplified string of the entire conversation for context
    conversation_context = "\n".join([f"[{msg.timestamp.strftime('%Y-%m-%d')}] {msg.sender}: {msg.content}" for msg in MESSAGES])

    prompt = f"""
    You are the Elyx AI assistant. Your role is to answer questions about a member's health journey based ONLY on the conversation history provided below.
    Be concise and helpful. Find the relevant decision and explain the reasons that led to it based on the conversation.

    --- CONVERSATION HISTORY ---
    {conversation_context}
    ----------------------------

    Based on the history, please answer the following question: "{payload.query}"
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Google AI: {str(e)}")