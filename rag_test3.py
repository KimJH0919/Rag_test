import streamlit as st
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
import os


def main():
    st.set_page_config(
        page_title="Streamlit_Rag",
        page_icon=":books:"
    )

    st.title("_Private Data :red[Q/A Chat]_ :books:")

    # 세션 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 사이드바: 파일 업로드 + API 키
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload your file", type=['pdf', 'docx'], accept_multiple_files=True
        )
        google_api_key = st.text_input(
            "Google API Key", key="chatbot_api_key", type="password"
        )
        os.environ["GOOGLE_API_KEY"] = google_api_key
        process = st.button("Process")

    # 처리 버튼 누르면 인덱스 생성
    if process:
        if not google_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        documents = get_text(uploaded_files)
        text_chunks = get_text_chunks(documents)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True

    # 채팅 내역 초기 메시지
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
        }]

    # 대화 내역 렌더링
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # StreamlitChatMessageHistory
    StreamlitChatMessageHistory(key="chat_messages")

    # 사용자 입력 처리
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.conversation({"question": query})
                st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_docs = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_docs[:3]:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        # 어시스턴트 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text: str) -> int:
    """tiktoken 대신 Rust 컴파일 없이 fallback: 문자 수로 토큰 길이 대체"""
    return len(text)


def get_text(uploaded_files):
    docs = []
    for file in uploaded_files:
        fn = file.name
        with open(fn, "wb") as f:
            f.write(file.getvalue())
            logger.info(f"Uploaded {fn}")
        if fn.lower().endswith(".pdf"):
            loader = PyPDFLoader(fn)
        elif fn.lower().endswith(".docx"):
            loader = Docx2txtLoader(fn)
        elif fn.lower().endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(fn)
        else:
            continue
        docs.extend(loader.load_and_split())
    return docs


def get_text_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return splitter.split_documents(docs)


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(chunks, embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
