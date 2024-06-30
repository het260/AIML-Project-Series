import google.generativeai as genai
import streamlit as st

# Set up your Gemini API key
gemini_api_key = "GEMINI-API-KEY"    # Enter your Gemini API key here in inverted commas
genai.configure(api_key=gemini_api_key)

def fetch_recommendations(trip_preference, season):
    try:
        # Construct the prompt to query the Gemini API
        prompt = f"Recommend tourist places for a {trip_preference} trip during the {season} season."
        
        # Generate content using the Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching recommendations: {e}")
        return "Sorry, we couldn't fetch the recommendations at this time."
    
def fetch_tips(decided_trip, season):
    try:
        # Construct the prompt to query the Gemini API
        prompt = f"Give tips for the {decided_trip} trip in {season} season."
        
        # Generate content using the Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching tips: {e}")
        return "Sorry, we couldn't fetch the tips at this time."
    
def fetch_tickets(decided_trip, season):
    try:
        # Construct the prompt to query the Gemini API
        prompt = f"Recommend ticket booking platforms for {decided_trip} trip during the {season} season."
        
        # Generate content using the Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching tickets information: {e}")
        return "Sorry, we couldn't fetch the tickets information at this time."
    
def fetch_itinerary(decided_trip, trip_duration):
    try:
        # Construct the prompt to query the Gemini API
        prompt = f"Recommend trip itinerary for {decided_trip} trip for {trip_duration} days."
        
        # Generate content using the Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching the trip itinerary: {e}")
        return "Sorry, we couldn't fetch the trip itinerary at this time."
    
def fetch_localdining(decided_trip):
    try:
        # Construct the prompt to query the Gemini API
        prompt = f"Recommend local dinings and cuisines for {decided_trip} trip."
        
        # Generate content using the Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching local dining recommendations: {e}")
        return "Sorry, we couldn't fetch the dinings and cuisines at this time."

def main():
    st.title("TravelGuru - Trip Recommender Bot")
    st.write("Welcome to the TravelGuru! Let's plan your next adventure.")

    user_name = st.text_input("What is your name?")
    if user_name:
        option = st.selectbox("Choose an option:", ["Get Trip Recommendations", "Get Trip Details"])

        if option == "Get Trip Recommendations":
            trip_preference = st.selectbox("Do you want a Domestic (i.e. in India) or International (i.e. outside India) Trip?", ["Domestic", "International"])
            season = st.selectbox(f"{user_name}, which season is it right now?", ["Winter", "Summer", "Monsoon"])

            if st.button("Get Recommendations"):
                recommendations = fetch_recommendations(trip_preference, season)
                st.write("Here are some recommendations for you:")
                st.write(recommendations)

        elif option == "Get Trip Details":
            decided_trip = st.text_input("Enter the destination of the trip you have decided:")
            if decided_trip:
                season = st.selectbox(f"{user_name}, which season is it right now?", ["Winter", "Summer", "Monsoon"])
                
                if st.button("Get Tips"):
                    tips = fetch_tips(decided_trip, season)
                    st.write("Here are some tips for this trip:")
                    st.write(tips)

                if st.button("Get Ticket Booking Platforms"):
                    tickets = fetch_tickets(decided_trip, season)
                    st.write("Here are some ticket booking platforms for this trip:")
                    st.write(tickets)

                trip_duration = st.number_input("Enter the duration of the trip you have decided (in days):", min_value=1)
                if trip_duration:
                    if st.button("Get Itinerary"):
                        itinerary = fetch_itinerary(decided_trip, trip_duration)
                        st.write("Here is the itinerary (i.e. plan) of this trip:")
                        st.write(itinerary)

                if st.button("Get Local Dining and Cuisines"):
                    local_dining = fetch_localdining(decided_trip)
                    st.write(f"Here are some local dinings and cuisines at {decided_trip}:")
                    st.write(local_dining)

    st.write("Thank you for using TravelGuru. Have a safe and happy journey!")

if __name__ == "__main__":
    main()