import google.generativeai as genai
import streamlit as st

# Set up your Gemini API key
gemini_api_key = "GEMINI-API-KEY"       # Enter your Gemini API key here in inverted commas
genai.configure(api_key=gemini_api_key)

def fetch_demanding_degrees():
    try:
        prompt = "Recommend some of the degrees to pursue from college which are in demand nowadays."
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching recommendations: {e}")
        return "Sorry, we couldn't fetch the demanding degrees at this time."
    
def fetch_location(degree):
    try:
        prompt = f"Recommend places for pursuing {degree} degree."
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching recommendations: {e}")
        return "Sorry, we couldn't fetch the locations at this time."
    
def fetch_colleges(degree, location):
    try:
        prompt = f"Give the best colleges for {degree} degree at {location}."
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching recommendations: {e}")
        return "Sorry, we couldn't fetch the recommendations at this time."
    
def fetch_procedure(college, degree, location):
    try:
        prompt = f"Give procedure, requirements and deadlines for the admission in {college} college at {location} for {degree} degree."
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching recommendations: {e}")
        return "Sorry, we couldn't fetch the procedure, requirements and deadlines at this time."
    
def get_response(user_input):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while fetching recommendations: {e}")
        return "Sorry, we couldn't generate a response at this time."

def main():
    st.title("EduAdvisor - College Admission Q&A Bot")
    st.write("Welcome to EduAdvisor! Let's guide you through your college admission journey.")

    user_name = st.text_input("What is your name?")
    
    if user_name:
        st.write(f"Hello, {user_name}! Let's start with some recommendations.")

        if st.button("Fetch Demanding Degrees"):
            demanding_degrees = fetch_demanding_degrees()
            st.write("Some of the demanding degrees to pursue from colleges are:")
            st.write(demanding_degrees)

        degree = st.text_input("What is your desired degree to pursue?")
        
        if degree:
            if st.button("Fetch favourable locations for this degree"):
                favourable_locations = fetch_location(degree)
                st.write(f"Some of the favourable places to pursue {degree} degree are:")
                st.write(favourable_locations)

            location = st.text_input("What is your desired location of college?")
            
            if location:
                if st.button("Fetch recommended colleges at this location"):
                    recommended_colleges = fetch_colleges(degree, location)
                    st.write(f"Here are some recommended colleges for you in {location}:")
                    st.write(recommended_colleges)

                college = st.text_input("What is your desired college?")
                
                if college:
                    if st.button("Fetch admission procedure of this college"):
                        procedure = fetch_procedure(college, degree, location)
                        st.write(f"Here are procedure, requirements, and deadlines for the admission in {college} college:")
                        st.write(procedure)

                    st.write("Now you can ask questions related to admissions (write Done to end):")
                    
                    user_input = st.text_area("Enter your question:")
                    
                    if user_input.lower() != "done":
                        if st.button("Get Response"):
                            response = get_response(user_input)
                            st.write(response)
                    else:
                        st.write("Thank you for using EduAdvisor. Signing off with warm wishes for your bright future ahead!")

    # st.write("Thank you for using EduAdvisor. Signing off with warm wishes for your bright future ahead!")

if __name__ == "__main__":
    main()