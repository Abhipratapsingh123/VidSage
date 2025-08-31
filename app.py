import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Streamlit Config

st.set_page_config(page_title="YouTube RAG", page_icon="🎥", layout="wide")
st.title("YouTube Video Assistant")


# Session State (store vectorstore in session)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None


# function to extract video id from rhe url

def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from URL or return the same if already an ID.
    """
    # Common YouTube URL patterns
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/)([A-Za-z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return url  # If user already entered just the video ID


with st.container():
    url_input = st.text_input("Enter YouTube URL or Video ID:", "")
    video_id = extract_video_id(url_input,)

    if st.button("Process Video"):
        if video_id:
            with st.spinner("Fetching Script of the video..."):
                try:
                    ytt_api = YouTubeTranscriptApi()
                    transcript_list = ytt_api.fetch( video_id, languages=['hi', 'en'])
                    transcript = " ".join(chunk.text for chunk in transcript_list)

                    # Step 1b: Text splitting
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=190)
                    chunks = splitter.create_documents([transcript])

                    # Step 1c: Embeddings + Vector Store
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001"
                    )
                    vector_store = FAISS.from_documents(chunks, embeddings)

                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = vector_store.as_retriever(
                        search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.6}
                    )
                    st.session_state.video_id = video_id

                    st.success(
                        "Video processed successfully! Now ask questions below.")
                except TranscriptsDisabled:
                    st.error(" No captions available for this video")
        else:
            st.warning("Please enter a YouTube video ID first.")


# Chat / Ask Questions

if st.session_state.retriever:
    st.subheader("Ask Questions about the Video")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Chat input 
    if user_input := st.chat_input("Ask something about the video..."):
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):

            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

            prompt = PromptTemplate(
                template="""You are a helpful assistant.
                Always answer strictly based on the provided transcript context.  

                - Detect the language of the user’s question (Hindi, English or mixed) and reply in the same language and tone.  
                - Do not translate unless the user specifically asks.  
                - If the transcript does not have enough information, then make the output better by adding proper and updated information from your side"  
                - If the question asked from the user's side is completely out of the context then say I don't know.  
                - If the user asks something you cannot do, respond with: "I can't help you with this."  

                Context:
                {context}

                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                "context": st.session_state.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            parser = StrOutputParser()
            # main chain
            main_chain = parallel_chain | prompt | model | parser

            answer = main_chain.invoke(user_input)

        st.session_state.chat_history.append(AIMessage(content=answer))

        with st.chat_message("assistant"):
            st.markdown(answer)
