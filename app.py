import streamlit as st
import requests
import os
import time
from datetime import datetime

st.set_page_config(
    page_title="Ai Travel Planner",
    layout="wide",
    page_icon="🚀"
)

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    DEEPSEEK_KEY = st.secrets["DEEPSEEK_KEY"]
except (KeyError, FileNotFoundError):
    HF_TOKEN = None
    DEEPSEEK_KEY = None

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

@st.cache_data(show_spinner=False, ttl=3600)
def generate_with_hf(prompt):
    """Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 800, "temperature": 0.5}
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
        return response.json()[0]['generated_text'] if response.status_code == 200 else None
    except Exception as e:
        st.error(f"HF API Error: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def generate_with_deepseek(prompt):
    """DeepSeek API"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=15)
        return response.json()['choices'][0]['message']['content'] if response.status_code == 200 else None
    except Exception as e:
        st.error(f"DeepSeek Error: {str(e)}")
        return None

def format_prompt(start, dest, days, budget, travel_date):
    return f"""Generate a detailed {days}-day travel itinerary from {start} to {dest} for {travel_date.strftime('%d %B %Y')} with total budget ₹{budget:,}.
Include realistic prices for:
- Transportation between locations
- Entry fees for attractions
- Food costs (breakfast/lunch/dinner)
- Hotel prices (budget/mid-range options)
- Emergency contacts for {dest}
start generating with table
make a table for price , activity etc,add famous must try foods , cuisines, activities and sum bonus tips also
"""

def show_fallback(start, dest, days, budget):
    """Fallback template"""
    daily_budget = budget // days
    itinerary = f"""
    ### ✨ {dest} Itinerary (Template)
    **Total Budget:** ₹{budget:,} | **Daily Budget:** ₹{daily_budget:,}
    """
    
    for day in range(1, days+1):
        itinerary += f"""
        **Day {day}:**
        - 08:00: Breakfast (₹{daily_budget//8:,})
        - 10:00: Morning Activity (₹{daily_budget//4:,})
        - 13:00: Lunch (₹{daily_budget//4:,})
        - 15:00: Afternoon Exploration (₹{daily_budget//4:,})
        - 19:00: Dinner (₹{daily_budget//4:,})
        - Stay: Hotel (₹{daily_budget//2:,})
        """
    
    st.markdown(itinerary)
    show_resources(dest)

def show_resources(dest):
    """Travel resources"""
    st.markdown(f"""
    ### 🧳 {dest} Resources
    - [Hotels](https://www.booking.com/city/in/{dest.lower()})
    - [Local Transport](https://www.redbus.in/)
    - [Emergency Contacts](tel:1363)
    """)

# ---- Main App ----
def main():
    # Sidebar with API status
    with st.sidebar:
        st.subheader("API Status")
        st.write(f"Hugging Face: {'✅ Active' if HF_TOKEN else '❌ Disabled'}")
        st.write(f"DeepSeek: {'✅ Active' if DEEPSEEK_KEY else '❌ Disabled'}")
    
    # Main interface
    st.title("AI Travel Planner")
    st.markdown("Plan budget trips across India with AI-powered itineraries")
    
    with st.form("travel_form"):
        col1, col2 = st.columns(2)
        with col1:
            start = st.text_input("From City", "Mumbai")
            dest = st.text_input("Destination City", "Goa")
            travel_date = st.date_input("Trip Date", datetime.now())
        with col2:
            days = st.slider("Duration (Days)", 1, 10, 3)
            budget = st.number_input("Total Budget (₹)", 5000, 100000, 15000, step=1000)
        
        if st.form_submit_button("Generate Itinerary"):
            with st.spinner("Creating your travel plan..."):
                start_time = time.time()
                prompt = format_prompt(start, dest, days, budget, travel_date)
                
                # Try APIs
                itinerary = None
                if DEEPSEEK_KEY:
                    itinerary = generate_with_deepseek(prompt)
                if not itinerary and HF_TOKEN:
                    itinerary = generate_with_hf(prompt)
                
                # Show results
                if itinerary:
                    st.success(f"Generated in {time.time()-start_time:.1f}s")
                    st.markdown(itinerary)
                    show_resources(dest)
                else:
                    st.warning("Using template itinerary")
                    show_fallback(start, dest, days, budget)

if __name__ == "__main__":
    main()