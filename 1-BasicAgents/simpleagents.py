from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.google import Gemini

import os 
from dotenv import load_dotenv

load_dotenv()

gemini = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    id="gemini-2.5-pro-exp-03-25"
)

agent = Agent(
    model=gemini,
    description="You are an assistant please reply based on the question",
    tools=[DuckDuckGoTools()],
    markdown=True
)

agent.print_response("What is the latest news in Malaysia?")