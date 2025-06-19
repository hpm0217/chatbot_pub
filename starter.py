import streamlit as st
import base64

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from dotenv import load_dotenv
import os

#load_dotenv()

# --- 함수/클래스 정의 ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

from pdfminer.high_level import extract_text

def get_pdf_text(filename):
    raw_text = extract_text(filename)
    return raw_text

def process_uploaded_file(uploaded_file):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # loader
        raw_text = get_pdf_text(uploaded_file)
        if not raw_text or len(raw_text.strip()) == 0:
            st.error("PDF에서 텍스트를 추출할 수 없습니다! 🐾")
            return None, None

        # splitter
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
        )
        all_splits = text_splitter.create_documents([raw_text])
        if not all_splits or len(all_splits) == 0:
            st.error("PDF에서 분할된 문서가 없습니다! 🐾")
            return None, None

        print("총 " + str(len(all_splits)) + "개의 passage")

        # storage
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        return vectorstore, raw_text
    return None, None

def generate_response(query_text, vectorstore, callback):
    # retriever 
    docs_list = vectorstore.similarity_search(query_text, k=3)
    docs = ""
    for i, doc in enumerate(docs_list):
        docs += f"'문서{i+1}':{doc.page_content}\n"
    # generator
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True, callbacks=[callback])
    # chaining
    rag_prompt = [
        SystemMessage(
            content="너는 문서에 대해 질의응답을 하는 '문서봇'이야. 주어진 문서를 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 정확하게 나와있지 않으면 대답하지 마."
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n{docs}"
        ),
    ]
    response = llm(rag_prompt)
    return response.content

def generate_summarize(callback):
    PET_DOC_PATH = os.path.join(os.path.dirname(__file__), "pet_doc.pdf")
    if not os.path.exists(PET_DOC_PATH):
        return "pet_doc.pdf 파일이 현재 폴더에 없습니다! 🐾"
    raw_text = get_pdf_text(PET_DOC_PATH)
    if not raw_text or len(raw_text.strip()) == 0:
        return "pet_doc.pdf에서 텍스트를 추출할 수 없습니다."
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True, callbacks=[callback])
    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서를 'Notion style'로 요약해줘. 중요한 내용만."
        ),
        HumanMessage(
            content=raw_text
        ),
    ]
    response = llm(rag_prompt)
    return response.content

# --- UI 및 상태 관리 코드 ---
# 귀여운 동물 이모지와 색상 테마
ANIMAL_EMOJI = "🐾🐶🐱🐾"
ANIMAL_BG_COLOR = "#FFF8E7"
ANIMAL_SIDEBAR_COLOR = "#FFE4E1"
ANIMAL_HEADER = f"<h1 style='text-align:center; color:#FF69B4;'>{ANIMAL_EMOJI} PET-LAW 챗봇 {ANIMAL_EMOJI}</h1>"
ANIMAL_SUBTITLE = "<p style='text-align:center; color:#A0522D;'>안녕하세요! 귀여운 동물 친구들의 법률정보를 알려주는 PET-LAW 챗봇이에요!<br>궁금한 점을 물어보세요! 🐾</p>"

# 배경색 스타일 적용
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {ANIMAL_BG_COLOR};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {ANIMAL_SIDEBAR_COLOR};
        }}
    </style>
""", unsafe_allow_html=True)

# 상단에 동물 이모지와 타이틀
st.markdown(ANIMAL_HEADER, unsafe_allow_html=True)
st.markdown(ANIMAL_SUBTITLE, unsafe_allow_html=True)

# OpenAI API Key 입력
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
save_button = st.sidebar.button("Save Key")
api_key_valid = False
if save_button and len(api_key)>10:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key saved successfully! 🐾")
    st.session_state["api_key_saved"] = True
if "api_key_saved" in st.session_state and st.session_state["api_key_saved"]:
    api_key_valid = True

# pet_doc.pdf 자동 로드 (API 키가 있을 때만)
PET_DOC_PATH = os.path.join(os.path.dirname(__file__), "pet_doc.pdf")
vectorstore, raw_text = None, None
if api_key_valid:
    if not os.path.exists(PET_DOC_PATH):
        st.error("pet_doc.pdf 파일이 현재 폴더에 없습니다! 🐾")
        st.stop()
    vectorstore, raw_text = process_uploaded_file(PET_DOC_PATH)
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.session_state['raw_text'] = raw_text

# chatbot greatings
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content="안녕하세요! 저는 귀여운 동물 친구들의 정보를 알려주는 펫닥 챗봇이에요! 궁금한 점을 물어보세요! 🐶🐱"
        )
    ]

# conversation history print 
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)
    
# API 키 안내 및 입력창 활성화 제어
if not api_key_valid:
    st.info("챗봇을 사용하려면 먼저 OpenAI API 키를 입력하고 저장 버튼을 눌러주세요! 🐾")
    st.chat_input("API 키를 입력해야 챗봇을 사용할 수 있습니다!", disabled=True)
else:
    prompt = st.chat_input("'요약' 또는 동물에 대해 궁금한 점을 입력해보세요! 🐾")
    if prompt:
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            try:
                if prompt == "요약":
                    response = generate_summarize(stream_handler)
                else:
                    response = generate_response(prompt, st.session_state['vectorstore'], stream_handler)
            except Exception as e:
                response = f"⚠️ 답변 생성 중 오류가 발생했어요: {e}"
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
