import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.crawl4ai import Crawl4aiTools

# Load environment variables
load_dotenv()

# Initialize the Gemini model
gemini = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    id="gemini-2.5-pro-exp-03-25"
)

# Create the agent with Crawl4aiTools
gold_agent = Agent(
    model=gemini,
    description=(
        "You are a helpful assistant that provides information about Maybank Gold Investment Account prices in Malaysia. "
        "When asked about gold prices, respond with current rates in Malaysian Ringgit (RM) per gram, showing both buying and selling prices. "
        "Provide the date of the rates."
    ),
    tools=[Crawl4aiTools(max_length=None)],
    show_tool_calls=False,
    markdown=True
)

# Example usage
if __name__ == "__main__":
    prompt = (
        "Go to https://www.maybank2u.com.my/maybank2u/malaysia/en/personal/rates/gold_and_silver.page "
        "and extract the latest Maybank Gold Investment Account rates. "
        "Return the date, selling price (RM/g), and buying price (RM/g) in a clear format."
    )
    gold_agent.print_response(prompt, stream=True)
