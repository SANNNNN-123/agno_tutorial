import os
import json
import requests
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

# Load environment variables
load_dotenv()

def get_gold_price(currency: str = "MYR") -> str:
    """Use this function to get the current gold price in the specified currency.
    
    Args:
        currency (str): The currency code (default is MYR for Malaysian Ringgit)
        
    Returns:
        str: JSON string with gold price information including price per gram
    """
    api_key = os.getenv("GOLD_API_KEY")
    
    if not api_key:
        return json.dumps({"error": "Gold API key not found in environment variables"})
    
    url = f"https://www.goldapi.io/api/XAU/{currency}"
    headers = {
        "x-access-token": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Get price per troy ounce
            price_per_ounce = data.get("price", 0)
            
            # Convert price from troy ounce to gram (1 troy ounce = 31.1034768 grams)
            price_per_gram = price_per_ounce / 31.1034768 if price_per_ounce else "N/A"
            
            result = {
                "price_per_ounce": price_per_ounce,
                "price_per_gram": price_per_gram,
                "currency": currency,
                "timestamp": data.get("timestamp", "N/A"),
                "high_price": data.get("high_price", "N/A"),
                "low_price": data.get("low_price", "N/A"),
                "open_price": data.get("open_price", "N/A"),
                "price_change_percentage": data.get("ch_percent", "N/A")
            }
            return json.dumps(result)
        else:
            return json.dumps({
                "error": f"Failed to fetch data: {response.status_code}",
                "message": response.text
            })
            
    except Exception as e:
        return json.dumps({"error": f"An error occurred: {str(e)}"})

# Initialize the Gemini model
gemini = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    id="gemini-2.5-pro-exp-03-25"
)

# Create the agent
gold_agent = Agent(
    model=gemini,
    description="You are a helpful assistant that provides information about gold prices in Malaysia. When asked about gold prices, respond with current rates in Malaysian Ringgit (MYR) per gram, highlight any trends, and provide helpful context about the gold market in Malaysia. Always emphasize the price per gram in your responses.",
    tools=[get_gold_price],
    show_tool_calls=True,
    markdown=True
)

# Example usage
if __name__ == "__main__":
    gold_agent.print_response("What is the current gold price in Malaysia?", stream=True)
