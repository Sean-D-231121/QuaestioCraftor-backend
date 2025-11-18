import json
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    quiz_type: str
    difficulty: str
    question_count: int
    topic: str


@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/generate")
async def generate(req: QuizRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    # create client (reads api_key from env or pass api_key=api_key)
    client = OpenAI(api_key=api_key)

    prompt = f"""
    You are a quiz generation AI.
    
    Generate exactly {req.question_count} {req.difficulty}-level quiz questions on "{req.topic}".
    The quiz type is "{req.quiz_type}". Follow these strict rules:
    
    - If quiz_type = "MCQ":
        - Only multiple-choice questions.
        - Each must have exactly 4 options.
        - Include the correct answer in "answer".
        - Set "type": "MCQ".
    - If quiz_type = "True/False":
        - Only True or False questions.
        - No options field.
        - Set "type": "True/False".
        - Ensure an approximately equal number of questions have "answer": "True" and "answer": "False".
        - Randomly mix their order so there is no pattern or bias.
    - If quiz_type = "Mixed":
        - Exactly half MCQ and half True/False (if odd, add 1 extra MCQ).
        - Follow the same rules above for each type.
        
        Return strictly valid JSON:
        [
        {{
            "question": "...",
            "options": ["A", "B", "C", "D"],  # only for MCQ
            "answer": "...",
            "type": "MCQ" or "True/False"
            }}
            ]
            No text, no markdown, no commentary.
            """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        # extract text from response
        text = resp.choices[0].message.content.strip()
        # Try to parse the text as JSON
        try:
            quiz_json = json.loads(text)

            # Shuffle the entire question list
            random.shuffle(quiz_json)

            # Shuffle MCQ options and fix answers accordingly
            for q in quiz_json:
                if q.get("type") == "MCQ" and "options" in q and isinstance(q["options"], list):
                    correct = q["answer"]
                    opts = q["options"]

                    # Ensure answer exists in options before shuffling
                    if correct in opts:
                        random.shuffle(opts)
                        q["options"] = opts
                        # Update correct answer to reflect its new position
                        q["answer"] = correct
                    else:
                        # If GPT response was weird, leave options as-is
                        pass
            return {"quiz": quiz_json}  # Return the parsed JSON directly 
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Response from OpenAI is not valid JSON")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
