import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the prompt
prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:  """

# Function to extract transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript_list])
        return transcript_text
    except Exception as e:
        st.error(f"Error extracting transcript: {e}")
        return None

# Function to generate summary using Google Gemini AI
def generate_gemini_content(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        
        # Check if response contains parts
        if hasattr(response, 'parts') and response.parts:
            return response.text
        else:
            st.error("No valid parts were returned in the response.")
            return "No valid content generated."
    except ValueError as ve:
        st.error(f"ValueError: {ve}")
        return "An error occurred while generating content."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred while generating content."

# Streamlit app title
st.title("YouTube Transcript to Detailed Notes Converter")

# Input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.write(f"Video ID: {video_id}")

# Button to get detailed notes
if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)
    
    if transcript_text:
        summary = generate_gemini_content(transcript_text, prompt)
        st.markdown("## Detailed Notes:")
        st.write(summary)
    else:
        st.error("Failed to extract transcript details. Please check the YouTube link.")
