"""
함수 호출을 사용합니다.
유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
만점이면 st.ballons를 사용합니다.
유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
"""

import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)


def get_questions_prompt(difficulty):
    difficulty_text = {
        "Easy": "for elementary school students (simple and intuitive problems))",
        "Hard": "for high school students (difficult and complex problems)"
    }
    prompt_text = f"""
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context, make 10 (TEN) questions to test the user's knowledge about the text. The questions should be suitable {difficulty_text[difficulty]}.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Use (o) to signal the correct answer.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital of Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    Your turn!

    Context: {{context}}
    """
    return str(prompt_text)

if 'quiz_attempt' not in st.session_state:
    st.session_state.quiz_attempt = 0

def handle_quiz(docs):

    if st.session_state.quiz_attempt > 0:
        st.experimental_rerun()

    response = run_quiz_chain(docs, topic if topic else file.name)
    correct_answers = 0
    form_key = f"form_{st.session_state.quiz_attempt}"

    with st.form(key=form_key):
        for question in response["questions"]:
            st.write(question["question"])
            options = [answer["answer"] for answer in question["answers"]]
            value = st.radio(
                "Select an option.",
                options,
                index=None,
                key=f"{question['question']}_{st.session_state.quiz_attempt}",
            )
            if any(answer["correct"] and answer["answer"] == value for answer in question["answers"]):
                correct_answers += 1
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write(f"You got {correct_answers} out of {len(response['questions'])} correct.")
        if correct_answers == len(response["questions"]):
            st.balloons()
        else:
            retake_test = st.button("Retake Test")
            if retake_test:
                st.session_state.quiz_attempt += 1

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs




st.title("QuizGPT")

st.markdown(
    """
    Welcome to QuizGPT.
    """
    )

st.sidebar.markdown("### GitHub Repository")
st.sidebar.markdown("[GitHub Repo Link](https://github.com/Juun0830/GPTs)")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="api_key")

if openai_api_key:

    with st.sidebar:

        difficulty = st.sidebar.selectbox("Choose difficulty", ("Easy", "Hard"), key="difficulty")

        docs = None
        topic = None
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)

    llm = ChatOpenAI(
        temperature=0.1,
        api_key=openai_api_key,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        )

    questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
        Use (o) to signal the correct answer.
            
        Question examples:
            
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
            
        Your turn!
            
        Context: {context}
    """,
            )
        ]
    )

    questions_chain = {"context": format_docs} | questions_prompt | llm

    formatting_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a powerful formatting algorithm.
        
        You format exam questions into JSON format.
        Answers with (o) are the correct ones.
        
        Example Input:

        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
        
        
        Example Output:
        
        ```json
        {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                            {{
                                "answer": "Red",
                                "correct": false
                            }},
                            {{
                                "answer": "Yellow",
                                "correct": false
                            }},
                            {{
                                "answer": "Green",
                                "correct": false
                            }},
                            {{
                                "answer": "Blue",
                                "correct": true
                            }}
                    ]
                }},
                            {{
                    "question": "What is the capital or Georgia?",
                    "answers": [
                            {{
                                "answer": "Baku",
                                "correct": false
                            }},
                            {{
                                "answer": "Tbilisi",
                                "correct": true
                            }},
                            {{
                                "answer": "Manila",
                                "correct": false
                            }},
                            {{
                                "answer": "Beirut",
                                "correct": false
                            }}
                    ]
                }},
                            {{
                    "question": "When was Avatar released?",
                    "answers": [
                            {{
                                "answer": "2007",
                                "correct": false
                            }},
                            {{
                                "answer": "2001",
                                "correct": false
                            }},
                            {{
                                "answer": "2009",
                                "correct": true
                            }},
                            {{
                                "answer": "1998",
                                "correct": false
                            }}
                    ]
                }},
                {{
                    "question": "Who was Julius Caesar?",
                    "answers": [
                            {{
                                "answer": "A Roman Emperor",
                                "correct": true
                            }},
                            {{
                                "answer": "Painter",
                                "correct": false
                            }},
                            {{
                                "answer": "Actor",
                                "correct": false
                            }},
                            {{
                                "answer": "Model",
                                "correct": false
                            }}
                    ]
                }}
            ]
        }}
        ```
        Your turn!

        Questions: {context}

    """,
            )
        ]
    )

    formatting_chain = formatting_prompt | llm

    if not docs:
        st.markdown(
            """
        you are authorized.
                        
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.     
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
        )
    else:
        handle_quiz(docs)