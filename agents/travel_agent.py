import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from serpapi import GoogleSearch
import sendgrid
from sendgrid.helpers.mail import Mail

load_dotenv()

# ─────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────

@tool
def search_flights(query: str) -> str:
    """Search for flight options based on the user's query."""
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": "google"
        })
        results = search.get_dict()
        organic = results.get("organic_results", [])[:3]

        if not organic:
            return "No flight results found."

        output = "✈️ Flight Results:\n\n"
        for r in organic:
            output += f"- {r.get('title')}\n"
            output += f"  {r.get('snippet')}\n\n"

        return output

    except Exception as e:
        return f"Flight error: {str(e)}"


@tool
def search_hotels(query: str) -> str:
    """Search for hotels based on the user's query."""
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": "google"
        })
        results = search.get_dict()
        organic = results.get("organic_results", [])[:3]

        if not organic:
            return "No hotel results found."

        output = "🏨 Hotel Results:\n\n"
        for r in organic:
            output += f"- {r.get('title')}\n"
            output += f"  {r.get('snippet')}\n\n"

        return output

    except Exception as e:
        return f"Hotel error: {str(e)}"


@tool
def search_attractions(query: str) -> str:
    """Search for tourist attractions based on the user's query."""
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": "google"
        })
        results = search.get_dict()
        organic = results.get("organic_results", [])[:3]

        if not organic:
            return "No attractions found."

        output = "🗺️ Attractions:\n\n"
        for r in organic:
            output += f"- {r.get('title')}\n"
            output += f"  {r.get('snippet')}\n\n"

        return output

    except Exception as e:
        return f"Attractions error: {str(e)}"


@tool
def send_travel_plan_email(recipient_email: str, travel_plan: str) -> str:
    """Send the generated travel plan to the user's email."""
    try:
        sg = sendgrid.SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"))

        message = Mail(
            from_email=os.getenv("SENDER_EMAIL"),
            to_emails=recipient_email,
            subject="Your Travel Plan ✈️",
            html_content=travel_plan.replace("\n", "<br>")
        )

        sg.send(message)
        return "Email sent successfully!"

    except Exception as e:
        return f"Email error: {str(e)}"


# ─────────────────────────────────────────
# LLM (GROQ FREE)
# ─────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ─────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────

def run_agent(user_input: str, chat_history: list, user_email: str = ""):
    try:
        # Step 1: Call tools
        flights = search_flights.invoke(user_input)
        hotels = search_hotels.invoke(user_input)
        attractions = search_attractions.invoke(user_input)

        # Step 2: Combine data
        combined_data = f"""
User Request:
{user_input}

{flights}

{hotels}

{attractions}
"""

        # Step 3: Generate final response
        response = llm.invoke([
            HumanMessage(content=f"Create a well-structured travel plan using this:\n{combined_data}")
        ])

        final_output = response.content

        # Step 4: Send email (optional)
        if user_email:
            send_travel_plan_email.invoke({
                "recipient_email": user_email,
                "travel_plan": final_output
            })

        return final_output, chat_history

    except Exception as e:
        return f"Error: {str(e)}", chat_history