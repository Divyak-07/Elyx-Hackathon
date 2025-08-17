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

class SentimentPoint(BaseModel):
    month: str
    score: float

class WeeklyReport(BaseModel):
    week_of: str
    summary: str

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Elyx Member Journey API",
    description="An API to serve the 8-month communication log for Rohan Patel's health journey.",
    version="1.0.0"
)

# --- CORS Middleware ---
origins = [
    "null",
    "http://localhost",
    "http://localhost:8080",
    "https://elyx-hackathon.netlify.app"
]
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

def get_sentiment_scores() -> List[SentimentPoint]:
    """Simulates an AI sentiment analysis on Rohan's messages."""
    member_messages = [msg for msg in MESSAGES if msg.role == 'Member']
    monthly_scores = {}
    
    for msg in member_messages:
        month = msg.timestamp.strftime('%Y-%m')
        score = 0 # Neutral
        content = msg.content.lower()
        if any(word in content for word in ['good', 'excellent', 'better', 'powerful', 'great', 'successful']):
            score = 1 # Positive
        elif any(word in content for word in ['issue', 'problem', 'anxious', 'frustration', 'setback', 'wrong', 'not heard']):
            score = -1 # Negative
        
        if month not in monthly_scores:
            monthly_scores[month] = []
        monthly_scores[month].append(score)

    sentiment_trend = []
    for month, scores in sorted(monthly_scores.items()):
        avg_score = sum(scores) / len(scores)
        month_str = datetime.strptime(month, '%Y-%m').strftime('%b %Y')
        sentiment_trend.append(SentimentPoint(month=month_str, score=round(avg_score, 2)))
        
    return sentiment_trend

def get_weekly_report(end_date_str: str) -> WeeklyReport:
    """Simulates an AI generating a weekly report for a specific week."""
    summary = """
    <ul class='list-disc list-inside space-y-2'>
        <li><span class='font-semibold'>Key Achievement:</span> Successfully managed muscle soreness from the new strength program by implementing a post-workout nutrition protocol (protein/creatine shake). Recovery metrics improved significantly.</li>
        <li><span class='font-semibold'>New Goal:</span> Formalized long-term longevity goals with Rachel, including targets for strength (Deadlift 1.5x BW), cardio (VO2 Max), and stability.</li>
        <li><span class='font-semibold'>Logistics:</span> Resolved skin irritation from the Whoop strap by switching to a new band material.</li>
        <li><span class='font-semibold'>Focus for Next Week:</span> Continue adapting to the new strength program and schedule the baseline DEXA and VO2 Max tests.</li>
    </ul>
    """
    return WeeklyReport(week_of="August 11, 2025", summary=summary)


# --- API Endpoints ---
@app.get("/", tags=["General"])
async def read_root(): return {"message": "Welcome to the Elyx Member Journey API"}

@app.get("/messages", response_model=List[Message], tags=["Messages"])
async def get_all_messages():
    if not MESSAGES: raise HTTPException(status_code=404, detail="Journey data not loaded.")
    return MESSAGES

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
    role_counts = {msg.role: role_counts.get(msg.role, 0) + 1 for msg in MESSAGES if msg.role not in ["Member", "Personal Assistant"]}
    return {"total_elyx_team_interactions": sum(role_counts.values()), "interactions_by_role": role_counts}

@app.get("/episodes/{month_name}", response_model=EpisodeAnalysis, tags=["Episodes"])
async def get_episode_analysis(month_name: str):
    try:
        month_messages = [msg for msg in MESSAGES if datetime.strptime(msg.timestamp.strftime('%B %Y'), '%B %Y') == datetime.strptime(month_name, '%B %Y')]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month format. Use 'Month YYYY'.")
    if not month_messages: raise HTTPException(status_code=404, detail=f"No data found for {month_name}.")
    return get_ai_analysis(month_name, month_messages)

@app.get("/metrics/sentiment", response_model=List[SentimentPoint], tags=["AI Metrics"])
async def get_sentiment_trend():
    if not MESSAGES: raise HTTPException(status_code=404, detail="Journey data not loaded.")
    return get_sentiment_scores()

@app.get("/reports/weekly", response_model=WeeklyReport, tags=["AI Reports"])
async def generate_weekly_report(end_date: str = "2025-08-18"):
    if not MESSAGES: raise HTTPException(status_code=404, detail="Journey data not loaded.")
    return get_weekly_report(end_date)
    
@app.post("/chat", tags=["AI Agent"])
async def chat_with_agent(payload: ChatQuery):
    if not genai.api_key:
        raise HTTPException(status_code=500, detail="Google API key is not configured on the server.")

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
