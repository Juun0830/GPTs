import subprocess
from pydub import AudioSegment
import math
import glob
import os

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

llm = ChatOpenAI(
    temperature=0.1
)

import streamlit as st


# Function 

def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4","mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i", 
        video_path, 
        "-vn", 
        audio_path,
        ]
    subprocess.run(command)

def cut_audio_in_chunk(audio_path, chunk_size, chunk_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i+1) * chunk_len

        chunk = track[start_time:end_time]

        chunk.export(f"{chunk_folder}/chunk_{i}.mp3", format="mp3")

has_transcript = os.path.exists("./cache/example.txt")

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return

    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


# Streamlit

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="📀",
)

st.title("MeetingGPT")

with st.sidebar:
    video = st.file_uploader(
        "Video",
         type=["mp4","mp3","avi"],
    )

if video:
    chunks_folder = "./cache/meeting"
    with st.status("Loading Video...") as status:
        video_path = f"./cache/{video.name}"
        audio_path = video_path.replace("mp4","mp3")
        transcript_path = video_path.replace("mp4","txt")
        with open(video_path, "wb") as f:
            f.write(video.read())

        status.update(label="Extracting Audio...")
        extract_audio_from_video(video_path)

        status.update(label="Cutting Audio Segments...")
        cut_audio_in_chunk(audio_path, 8, chunks_folder)

        status.update(label="Transcribing Audio...")
        transcribe_chunks(chunks_folder, transcript_path)
    
transcript_tab, summary_tab, qa_tab = st.tabs(["Summary","QA"])

with transcript_tab:
    with open(transcript_path, "r") as file:
        st.write(file.read())

with summary_tab:
    start = st.button("Generate Summary")

    if start:
        loader = TextLoader(transcript_path)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800,
            chunk_overlap=100,
        )
        docs = loader.load_and_split(text_splitter=splitter)
        st.write(docs)

        first_summary_prompt = ChatPromptTemplate.from_template(
            """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:   
            """
            )
        
        first_summary_chain = first_summary_prompt | llm | StrOutputParser()

        summary = first_summary_chain.invoke(
            {"text": docs[0].page_content},
            )
        
        refine_prompt = ChatPromptTemplate.from_template(
            """
            Your job is to produce a final summary.
            We have provided an existing summary up to a certain point: {existing_summary}
            We have the opportunity to refine the existing summary (only if needed) with some more context below.
            ------------
            {context}
            ------------
            Given the new context, refine the original summary.
            If the context isn't useful, RETURN the original summary.
            """
        )

        refine_chain = refine_prompt | llm | StrOutputParser()

        with st.status("Summarizing...") as status:
            for i, doc in enumerate(docs[1:]):
                status.update(
                label=f"Processing document {i+1}/{len(docs)-1} "
                )
                summary = refine_chain.invoke(
                    {
                        "existing_summary": summary,
                        "context": doc.page_content,
                    }
                )
                
        st.write(summary)

with qa_tab:
    retriever = embed_file(transcript_path)







