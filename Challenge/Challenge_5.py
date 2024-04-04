"""
이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
파일 업로드 및 채팅 기록을 구현합니다.
사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
"""

# Retreiver
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

# Memory
from langchain.memory import ConversationBufferMemory

# Chat
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda

#Interface
import streamlit as st


# Function
@st.cache_data(show_spinner="Embedding file..")
def embed_file(file):
	openai.api_key=f'{api_key}'
	file_content = file.read() 
	file_path = f"./Challenge/Challenge_files/{file.name}"
	with open(file_path, "wb") as f: 
		f.write(file_content) 
	
	cache_dir = LocalFileStore(f"./Challenge/Challenge_cache/embeddings/{file.name}") 
	splitter = CharacterTextSplitter.from_tiktoken_encoder(
		separator="\n", chunk_size=600, chunk_overlap=100,
	)
	
	loader = UnstructuredFileLoader(file_path) 
	docs = loader.load_and_split(text_splitter=splitter) 
	
	embeddings = OpenAIEmbeddings() 
	cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
	 
	vectorstore = FAISS.from_documents(docs, cached_embeddings)
	retriever = vectorstore.as_retriever()
	
	return retriever


def send_message(message, role, save=True): 
	with st.chat_message(role): 
		st.markdown(message) 
	if save: 
		st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
	for message in st.session_state["messages"]:
		send_message(message["message"],message["role"],save=False)


def format_docs(docs):
	return "\n\n".join(document.page_content for document in docs)


# Prompt Template
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


# Streamlit

st.title("⭐️RAG Challenge⭐️")

api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

if api_key:
 
    openai.api_key=f'{api_key}'

    llm = ChatOpenAI(
		temperature=0.1,
	)

    st.sidebar.markdown("### GitHub Repository")
    st.sidebar.markdown("[GitHub Repo Link](https://github.com/Juun0830/GPTs)")

    st.markdown("##### Documents")
    file = st.file_uploader(
        "Uploader",
        type=["pdf", "txt", "docx"],
    )

    st.markdown("##### AI Chat")
    if "messages" not in st.session_state:
	    st.session_state["messages"] = []

    if file:
        retriever = embed_file(file)
        send_message("I'm ready!!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs), #미니체인
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            response = chain.invoke(message)
            send_message(response.content, "ai")

    else:
        st.session_state["messages"] = []
else:
	st.sidebar.error("Please enter your OpenAI API Key to activate the document upload feature.")
