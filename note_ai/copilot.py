import streamlit as st
from audiorecorder import audiorecorder

from pydub import AudioSegment

import os
import base64
import subprocess

from pathlib import Path
import pydub

import math
import glob

import time
from datetime import datetime

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Function 
# 1.Audio
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

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):

    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])
        
        os.remove(file)

def stream_data():
    for word in transcription.split(" "):
        yield word + " "
        time.sleep(0.02)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# 2.RAG

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"{file.name}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    cache_dir = LocalFileStore(f"./cache/embeddings/{file.name}")
    if not os.path.exists('cache_dir'):
        os.makedirs('cache_dir')

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


#LLM Prompt

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


#STREAMLIT

st.set_page_config(
    page_title="HARRY Copilot",
    page_icon="⚡️",
)

st.subheader("HARRY Copilot",help='help',divider="gray")

with st.sidebar:

    api_key_entered = st.session_state.get("api_key", "") != ""

    with st.expander("API KEY", expanded=not api_key_entered):
        st.markdown("Type your own AI API Key🔑")
        openai_api_key = st.text_input('',type="password", key="api_key")


    if openai_api_key:

        file_dir = "./file"
        recorded_dir = "./file/recorded"
        audio_dir = "./file/audiostream"
        note_dir = "./file/note"

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if not os.path.exists(note_dir):
            os.makedirs(note_dir)

        st.markdown("##### Task")
        selected_task = st.selectbox(
            'Choose a task',
            ('Recorded Content','Audio Stream'),
            index=None,
            placeholder="Tasks",
        )
        
        if selected_task == 'Recorded Content':

            if not os.path.exists(recorded_dir):
                os.makedirs(recorded_dir)
            
            uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv", "mp3"])
            if uploaded_video is not None:
                st.video(uploaded_video)

                st.session_state['video_uploading'] = True

        if selected_task == 'Audio Stream':

            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)

            recording = audiorecorder(start_prompt="Start recording", stop_prompt="Stop recording", pause_prompt="", key=None)

            if len(recording) > 0:
                st.session_state['recording'] = recording.export().read()
                st.audio(st.session_state['recording'])

                col1, col2, col3, col4 = st.columns([6.0, 3.8, 0.4, 0.1])

                with col1:
                    if st.button('Save&Transcription'):
                        record_name = st.text_input('Type the name of recording', key="record_name")
                        save_path = f"./file/audiostream/{record_name}.webm"
                        with open(save_path, "wb") as f:
                            f.write(st.session_state['recording'])                  
                        st.toast(f"Recording saved to {save_path}")
                        st.session_state['audio_recording'] = True

                with col2:
                    if st.button('Delete All'):
                        if 'recording' in st.session_state:
                            del st.session_state['recording']
                            st.toast("File does not exist")

        st.markdown("##### Archive")
            
        with st.expander("Recent Notes"):
            files = Path("./file/note").glob('*')
            sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
            
            for file in sorted_files[:5]:
                mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d')
                st.write(f"{file.name} ({mod_time})")


if 'video_uploading' in st.session_state and st.session_state['video_uploading']:

    st.markdown("##### Transcription")
    st.markdown("###### Raw Audio")

    chunks_folder = "./file/recorded/chunk"
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)

    with st.spinner("Loading Video..."):
        video_path = f"./file/recorded/{uploaded_video.name}"
        audio_path = video_path.replace("mp4","mp3")
        transcript_path = video_path.replace("mp4","txt")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

    with st.spinner("Extracting Audio..."):
        extract_audio_from_video(video_path)

    st.audio(audio_path)

    with st.spinner("Cutting Audio Segments..."):
        cut_audio_in_chunk(audio_path, 8, chunks_folder)

    with st.spinner("Transcribing Audio..."):
        transcribe_chunks(chunks_folder, transcript_path)

    st.markdown("###### Transcription")

    with open(transcript_path, "r") as file:
        transcription = file.read()

    with st.expander("Fold or Unfold", expanded=True):
        st.write(transcription)

    st.write("")
    st.markdown("##### The minutes of a meeting")

    summary_tab, qa_tab, note_tab = st.tabs(["Summary","QA", "Note"])

    with summary_tab:
        start = st.button("Generate Summary")

    with qa_tab:
        
        if not os.path.exists('./file/rag'):
            os.makedirs('./file/rag')

        file = open(transcript_path, "rb")

        if file:
            retriever = embed_file(file)
            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()
            message = st.chat_input("Ask anything about your file...")
            if message:
                send_message(message, "human")
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                )
                with st.chat_message("ai"):
                    chain.invoke(message)

        else:
            st.session_state["messages"] = []


elif 'audio_recording' in st.session_state and st.session_state['audio_recording']:

    st.markdown("##### Transcription")
    st.markdown("###### Raw Audio")

    save_path = f"./file/audiostream/{record_name}.webm"
    chunks_folder = "./file/recorded/chunk"
    transcript_path = save_path.replace("webm","txt")
    
    with st.spinner("Cutting Audio Segments..."):
        cut_audio_in_chunk(save_path, 0.1, chunks_folder)

    st.audio(st.session_state['recording'])

    with st.spinner("Transcribing Audio..."):
        transcribe_chunks(chunks_folder, transcript_path)

    st.markdown("###### Transcription")

    with open(transcript_path, "r") as file:
        transcription = file.read()

    with st.expander("Fold or Unfold", expanded=True):
        st.write_stream(stream_data)

    st.write("")
    st.markdown("##### The minutes of a meeting")

    summary_tab, qa_tab, note_tab = st.tabs(["Summary","QA", "Note"])

else:
    st.write("")




