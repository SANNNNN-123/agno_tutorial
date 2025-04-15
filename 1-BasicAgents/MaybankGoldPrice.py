import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.crawl4ai import Crawl4aiTools
import requests
import pandas as pd
import json
from pydantic import BaseModel
from datetime import datetime


load_dotenv()

gemini = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    id="gemini-2.5-pro-exp-03-25"
)

class GoldPrice(BaseModel):
    date: str
    selling: float
    buying: float

gold_agent = Agent(
    model=gemini,
    description=(
        "You are a helpful assistant that provides information about gold investment account prices in Malaysia. "
        "When asked about gold prices, respond with current rates in Malaysian Ringgit (RM) per gram, showing both buying and selling prices. "
        "Provide the date of the rates. If asked for a comparison, show a markdown table comparing the banks."
    ),
    tools=[Crawl4aiTools(max_length=None)],
    show_tool_calls=False,
    markdown=False
)

def get_uob_gold_price():
    url = "https://www.uob.com.my/wsm/stayinformed.do?path=gia"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            content = res.text.replace("ITEM,", "")
            lines = content.strip().splitlines()
            columns = ["PRODUCT", "UNIT", "SELLING", "BUYING", "DATE", "TIME"]
            data = []
            for line in lines:
                if "GOLD SAVINGS ACCOUNT" in line:
                    values = line.split(",")
                    data.append(values)
            
            df = pd.DataFrame(data, columns=columns)
            if not df.empty:
                gsa = df.iloc[0]
                # Format date to match Maybank's YYYY-MM-DD format
                try:
                    # Extract just the date part (typically DD/MM/YYYY)
                    date_parts = gsa['DATE'].strip().split('/')
                    if len(date_parts) == 3:
                        day, month, year = date_parts
                        formatted_date = f"{year}-{month}-{day}"  # YYYY-MM-DD
                    else:
                        formatted_date = gsa['DATE'] 
                except Exception:
                    formatted_date = gsa['DATE']
                
                return {
                    "date": formatted_date,
                    "selling": float(gsa["SELLING"]),
                    "buying": float(gsa["BUYING"])
                }
    except Exception as e:
        print(f"Error fetching UOB data: {str(e)}")
    return None

def get_maybank_gold_price(agent):
    maybank_prompt = (
        "Go to https://www.maybank2u.com.my/maybank2u/malaysia/en/personal/rates/gold_and_silver.page "
        "and extract the latest Maybank Gold Investment Account rates. "
        "Respond ONLY with a valid JSON object with these exact keys: date, selling, buying. "
        "Do NOT include any explanation, markdown, or extra text. "
        "Example: {\"date\": \"2024-04-15\", \"selling\": 350.5, \"buying\": 345.0}"
    )
    try:
        response = agent.run(maybank_prompt, response_model=GoldPrice)
        if response and hasattr(response, "content"):
            content = response.content
            #print("Maybank agent raw content:", content)
            if isinstance(content, GoldPrice):
                return content.model_dump()
            elif isinstance(content, dict):
                return content
            elif isinstance(content, str):
                try:
                    if content.strip().startswith("```"):
                        content = content.strip().split("```")[-2]  
                    content = content.strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
                    data = json.loads(content)
                    return GoldPrice(**data).model_dump()
                except Exception as e:
                    print("DEBUG: Maybank JSON parse error:", e)
                    return {"date": "N/A", "selling": "N/A", "buying": "N/A"}
        return {"date": "N/A", "selling": "N/A", "buying": "N/A"}
    except Exception as e:
        print(f"Error fetching Maybank data: {str(e)}")
        return {"date": "N/A", "selling": "N/A", "buying": "N/A"}

if __name__ == "__main__":
    # 1. Get Maybank price
    maybank_data = get_maybank_gold_price(gold_agent)
    
    # 2. Get UOB price
    uob_data = get_uob_gold_price()

    # 3. Print markdown table
    print("\nGold Investment Account Prices Comparison:")
    print("=" * 50)
    print("""
| Bank    | Date           | Selling (RM/g) | Buying (RM/g) |
|---------|----------------|----------------|---------------|
| Maybank | {mb_date:<14} | {mb_sell:<14} | {mb_buy:<13} |
| UOB     | {uob_date:<14} | {uob_sell:<14} | {uob_buy:<13} |
""".format(
        mb_date=maybank_data.get('date', 'N/A'),
        mb_sell=str(maybank_data.get('selling', 'N/A')),
        mb_buy=str(maybank_data.get('buying', 'N/A')),
        uob_date=uob_data['date'] if uob_data else 'N/A',
        uob_sell=str(uob_data['selling']) if uob_data else 'N/A',
        uob_buy=str(uob_data['buying']) if uob_data else 'N/A'
    ))
