"""Working News Analysis Module using Google Gemini + ADK (Escaped JSON + 6–7 Articles)"""

import os
import json
import re
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types as genai_types


# Load API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")


async def get_news_data_async(symbol: str):
    try:
        agent = LlmAgent(
            name="news_agent",
            model=Gemini(
                model="gemini-2.0-flash-001",
                api_key=API_KEY,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_output_tokens": 3000,
                },
            ),
            instruction=f"""
You are a financial news engine. You MUST return 6–7 articles.

Your tasks:

1. Use google_search tool MULTIPLE times with these queries:
   - "latest news {symbol} stock"
   - "{symbol} stock analysis"
   - "{symbol} financial news"
   - "{symbol} market outlook"
   - "{symbol} stock today"
   - "{symbol} investor news"
   - "{symbol} company updates"

2. Collect all articles found.
3. Remove duplicates.
4. Select the BEST 6–7 articles.
5. Determine sentiment → Bullish, Bearish, or Neutral.
6. Return STRICT JSON (NO markdown).

REQUIRED JSON FORMAT:
{{
  "overall_sentiment": "Bullish",
  "overall_sentiment_score": 0.42,
  "final_conclusion": "Short explanation of sentiment.",
  "articles": [
    {{"headline": "Title 1", "source": "Source"}},
    {{"headline": "Title 2", "source": "Source"}},
    {{"headline": "Title 3", "source": "Source"}},
    {{"headline": "Title 4", "source": "Source"}},
    {{"headline": "Title 5", "source": "Source"}},
    {{"headline": "Title 6", "source": "Source"}},
    {{"headline": "Title 7", "source": "Source"}}
  ]
}}

Rules:
- MUST return at least 6 articles (prefer 7).
- Articles must be from real news sources.
- NO markdown. Only JSON.
""",
            tools=[google_search],
        )

        runner = InMemoryRunner(agent=agent, app_name="news_app")
        session_service = runner.session_service

        await session_service.create_session(
            app_name="news_app",
            user_id="news_user",
            session_id=f"news_session_{symbol}",
        )

        user_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"Fetch latest news about {symbol} and return JSON only.")]
        )

        response_text = ""

        async for event in runner.run_async(
            user_id="news_user",
            session_id=f"news_session_{symbol}",
            new_message=user_message,
        ):
            if event.is_final_response():
                part = event.content.parts[0]
                response_text = getattr(part, "text", str(part))

        response_text = re.sub(r"```json|```", "", response_text).strip()

        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {
            "overall_sentiment": "Neutral",
            "overall_sentiment_score": 0,
            "final_conclusion": "Model did not return valid JSON.",
            "articles": [],
        }

    except Exception as e:
        print("NEWS ERROR:", e)
        return {
            "overall_sentiment": "Neutral",
            "overall_sentiment_score": 0,
            "final_conclusion": f"Internal error: {e}",
            "articles": [],
        }
