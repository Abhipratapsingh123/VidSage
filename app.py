import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import re
from dotenv import load_dotenv

# Load environment variables
api_key = st.secrets["GOOGLE_API_KEY"]

# Streamlit Config
st.set_page_config(page_title="YouTube RAG", page_icon="🎥", layout="wide")
st.title("YouTube Video Assistant")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None

# Function to extract video ID
def extract_video_id(url: str) -> str:
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/)([A-Za-z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return url  # already an ID

# Input container
with st.container():
    url_input = st.text_input("Enter YouTube URL or Video ID:", "")
    video_id = extract_video_id(url_input)

    if st.button("Process Video"):
        if video_id:
            with st.spinner("Fetching transcript..."):
                try:
                    transcript_list = YouTubeTranscriptApi().get_transcript(video_id, languages=['hi', 'en'])
                    transcript = " ".join(chunk["text"] for chunk in transcript_list)

                    # Step 1b: Split text into chunks
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=190)
                    chunks = splitter.create_documents([transcript])

                    # Step 1c: Create embeddings + vector store
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vector_store = FAISS.from_documents(chunks, embeddings)

                    # Cache in session
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = vector_store.as_retriever(
                        search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.6}
                    )
                    st.session_state.video_id = video_id

                    st.success("✅ Video processed successfully! You can now ask questions below.")
                except TranscriptsDisabled:
                    st.error("❌ No captions available for this video.")
        else:
            st.warning("⚠️ Please enter a valid YouTube video URL or ID.")

# Chat / Ask Questions
if st.session_state.retriever:
    st.subheader("💬 Ask Questions about the Video")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Chat input
    if user_input := st.chat_input("Ask something about the video..."):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key
            )

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Always answer strictly based on the provided transcript context.  

                - Detect the language of the user’s question (Hindi, English or mixed) and reply in the same language and tone.    
                - If the transcript lacks enough details, supplement it with accurate, relevant, and up-to-date information.
                - If the question is out of context, politely mention it but still provide a useful, appropriate answer.
                - If you cannot help, respond: "I can't help you with this."

                Context:
                {context}

                Question:
                {question}
                """,
                input_variables=["context", "question"]
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            parallel_chain = RunnableParallel({
                "context": st.session_state.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            })

            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | model | parser

            answer = main_chain.invoke(user_input)

        st.session_state.chat_history.append(AIMessage(content=answer))
        with st.chat_message("assistant"):
            st.markdown(answer)
