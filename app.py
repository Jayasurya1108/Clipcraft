import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the prompt
default_prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:  """

# Function to extract the video ID from a YouTube URL
def extract_video_id(youtube_video_url):
    try:
        parsed_url = urlparse(youtube_video_url)
        video_id = parse_qs(parsed_url.query).get('v')
        if video_id:
            return video_id[0]
        else:
            raise ValueError("Invalid YouTube URL")
    except Exception as e:
        st.error(f"Error extracting video ID: {e}")
        return None

# Function to extract transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    video_id = extract_video_id(youtube_video_url)
    if not video_id:
        return None
    
    try:
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
        
        # Log the entire response for debugging
        st.write("Full Response:", response)

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

# Function to perform sentiment analysis
def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Function to extract keywords
def extract_keywords(text, num_keywords=10):
    blob = TextBlob(text)
    keywords = blob.noun_phrases
    return keywords[:num_keywords]

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to download summary
def download_summary(summary, filename="summary.txt"):
    b64 = base64.b64encode(summary.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Summary</a>'
    return href

# Streamlit app title
st.title("Clipcraft --> Your Video Summarizer")

# Input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

# Options for summary length and custom prompt
summary_length = st.selectbox("Select Summary Length:", ["Short", "Medium", "Detailed"])
custom_prompt = st.text_area("Custom Prompt (Optional):", default_prompt)

# Function to get appropriate summary length prompt
def get_prompt(length):
    if length == "Short":
        return "Summarize the following text in less than 100 words: "
    elif length == "Medium":
        return "Summarize the following text in 100-200 words: "
    else:
        return "Summarize the following text in 250-500 words: "

if youtube_link:
    video_id = extract_video_id(youtube_link)
    if video_id:
        st.write(f"Video ID: {video_id}")

# Button to get detailed notes
if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)
    
    if transcript_text:
        summary_prompt = custom_prompt if custom_prompt != default_prompt else get_prompt(summary_length)
        summary = generate_gemini_content(transcript_text, summary_prompt)
        
        st.markdown("## Detailed Notes:")
        st.write(summary)
        
        # Sentiment Analysis
        polarity, subjectivity = sentiment_analysis(transcript_text)
        st.write(f"Sentiment Polarity: {polarity}, Sentiment Subjectivity: {subjectivity}")
        
        # Keyword Extraction
        keywords = extract_keywords(transcript_text)
        st.write("Keywords:", keywords)
        
        # Generate Word Cloud
        st.markdown("## Word Cloud:")
        generate_wordcloud(transcript_text)
        
        # Download Summary
        st.markdown(download_summary(summary), unsafe_allow_html=True)
    else:
        st.error("Failed to extract transcript details. Please check the YouTube link.")
