import json
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

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 2000

@app.post("/generate")
async def generate(req: PromptRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    # create client (reads api_key from env or pass api_key=api_key)
    client = OpenAI(api_key=api_key)

    
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=0.7,
        )
        # extract text from response
        text = resp.choices[0].message.content.strip()
        # Try to parse the text as JSON
        try:
            quiz_json = json.loads(text)
            return quiz_json  # Return the parsed JSON directly
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Response from OpenAI is not valid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
