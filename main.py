import os, dotenv, base64
import streamlit as st
import asyncio
import edge_tts
from streamlit_mic_recorder import speech_to_text
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Load environment variables
dotenv.load_dotenv(".env")

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Available voices for Text-to-Speech
voices = {
    "William":"en-AU-WilliamNeural",
    "James":"en-PH-JamesNeural",
    "Jenny":"en-US-JennyNeural",
    "US Guy":"en-US-GuyNeural",
    "Sawara":"hi-IN-SwaraNeural",
}

st.set_page_config(page_title="Medica ChatBot", layout="wide", page_icon="./assets/logo-.png")

# Title
st.markdown("""
    <h1 style='text-align: center;'>
        <span style='color: #fcfcfc;'>Medica</span>
    </h1>
""", unsafe_allow_html=True)

# Streamlit setup
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("## Medica ChatBot")
    st.write("This bot can answer questions related to Medical book for MBBS. It is very helpful for Medical Students")
    st.divider()

# Load vectorstore only once
if "vectorstore" not in st.session_state:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "medical-data"  # Same name used during indexing
    
    index = pc.Index(index_name)
    st.session_state["vectorstore"] = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Hey there! How can I assist you today?", "metadata": None}
    ]

def format_docs(docs):
    return "\n\n".join(
        [f'Document {i+1}:\n{doc.page_content}\n'
         f'Source: {doc.metadata.get("source", "Unknown")}\n'
         f'Category: {doc.metadata.get("category", "Unknown")}\n'
         f'Instructor: {doc.metadata.get("instructor", "N/A")}\n-------------'
         for i, doc in enumerate(docs)]
    )

def extract_metadata_from_docs(docs):
    """Extract metadata from retrieved documents for display"""
    metadata_list = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        # Extract just the filename from the full path
        if source != "Unknown" and "/" in source:
            source = source.split("/")[-1]
        if source != "Unknown" and "\\" in source:
            source = source.split("\\")[-1]
        
        metadata_list.append({
            "source": source,
            "page": page
        })
        
    return metadata_list

def is_medical_question(question):
    """Use LLM to intelligently determine if question requires medical knowledge from textbooks"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a classifier that determines whether a question requires medical textbook knowledge or can be answered with general conversation.

Respond with ONLY "MEDICAL" or "CASUAL" based on these criteria:

MEDICAL - Questions that require specific medical knowledge from textbooks:
- Medical conditions, diseases, syndromes
- Anatomy, physiology, pathology concepts
- Medical procedures, treatments, diagnostics
- Drug information, pharmacology
- Clinical symptoms and signs
- Medical terminology definitions
- MBBS study-related content
- Any question requiring factual medical information

CASUAL - Questions that are conversational or general:
- Greetings (hi, hello, hey)
- General conversation (how are you, what can you do)
- Thanks/appreciation
- Questions about the chatbot itself
- Non-medical general knowledge
- Casual chat

Analyze the intent and content, not just keywords."""),
        ("human", "{question}")
    ])
    
    try:
        response = (classifier_prompt | llm | StrOutputParser()).invoke({"question": question})
        return response.strip().upper() == "MEDICAL"
    except:
        # If classification fails, default to medical to be safe
        return True

def check_response_uses_context(response_text, retrieved_docs):
    """Check if the generated response actually uses information from retrieved documents"""
    if not retrieved_docs or not response_text:
        return False
    
    # If response contains the standard "I don't have this information" message,
    # it means the context wasn't useful
    if "I don't have this information in the available medical textbooks" in response_text:
        return False
    
    # Use LLM to determine if response uses the retrieved context
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    context_text = "\n".join([doc.page_content[:200] + "..." for doc in retrieved_docs[:2]])
    
    verification_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a verification system. Determine if a response uses information from provided context documents.

Respond with ONLY "YES" or "NO".

YES - If the response contains specific information, facts, definitions, or details that come from the context documents
NO - If the response is general, conversational, or doesn't reference specific information from the context

Be strict: only say YES if you can clearly see that specific information from the context was used in the response."""),
        ("human", """Context Documents:
{context}

Generated Response:
{response}

Does this response use specific information from the context documents?""")
    ])
    
    try:
        verification = (verification_prompt | llm | StrOutputParser()).invoke({
            "context": context_text,
            "response": response_text
        })
        return verification.strip().upper() == "YES"
    except:
        # If verification fails, assume it doesn't use context to be safe
        return False
def reset_conversation():
    st.session_state.pop('chat_history')
    
    st.session_state['chat_history'] = [
        
        {"role":"assistant", "content":"Hey there! How can I assist you today?", "metadata": None}
        
    ]

def simple_chat_response(question, chat_history):
    """Generate a simple chat response without RAG for casual questions"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    output_parser = StrOutputParser()
    
    casual_system_prompt = """You are a friendly medical assistant chatbot for MBBS students. 
    You're having a casual conversation with a student. Keep your responses warm, helpful, and concise.
    
    For greetings and casual questions:
    - Respond naturally and friendly
    - Mention that you're here to help with medical studies
    - Keep responses brief and conversational
    - Don't mention sources or retrieved documents
    
    If the student asks what you can do, explain that you can help with:
    - Medical textbook questions
    - MBBS study materials
    - Medical concepts and definitions
    - Explaining medical procedures and treatments
    
    Keep the tone conversational and supportive."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", casual_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm | output_parser
    return chain.stream({"question": question, "chat_history": chat_history})

def rag_qa_chain(question, retriever, chat_history):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    output_parser = StrOutputParser()

    # System prompt to contextualize the question
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. If the original question is in Roman Urdu or Hingish language,
    then translate it to accurate English. Do NOT answer the question, just reformulate and translate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    contextualize_q_chain = contextualize_q_prompt | llm | output_parser

    qa_system_prompt = """You are a specialized medical assistant designed to help MBBS students with their studies. 
    You MUST answer questions ONLY based on the retrieved medical textbook content provided in the context below. 
    DO NOT use your general knowledge or information outside of the provided context.

    IMPORTANT GUIDELINES:
    1. Answer ONLY from the retrieved documents/context provided
    2. If the information is not available in the context, respond with: "I don't have this information in the available medical textbooks. Please refer to your course materials or consult your instructor."
    3. Provide concise, accurate, and precise answers
    4. When giving definitions or explanations, include specific details like numbers, classifications, or lists when available in the context
    5. Use proper medical terminology as found in the source material
    6. If asked about symptoms, diseases, treatments, or medical procedures, stick strictly to what's written in the context
    7. For questions requiring step-by-step processes (like diagnostic procedures), list them exactly as mentioned in the source
    8. If the context contains contradictory information, mention this and present both viewpoints as they appear in the source
    9. If the question is irrelevant to MBBS topics, politely respond: "This question is outside the scope of MBBS content. Please ask a question related to your medical studies.

    RESPONSE FORMAT:
    - Start with direct, specific answers (like your tiger species example)
    - Follow with additional relevant details from the context
    - Use bullet points or numbered lists when the source material is structured that way
    - Keep responses focused and avoid unnecessary elaboration beyond what's in the context

    DO NOT:
    - Add information from your general medical knowledge
    - Speculate or infer beyond what's explicitly stated
    - Provide medical advice or diagnoses
    - Answer questions unrelated to the medical content in your knowledge base

    Retrieved Medical Textbook Content (Context):
    ------------
    {context}
    ------------
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    final_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever | format_docs
        )
        | prompt
        | final_llm
        | output_parser
    )
    
    return rag_chain.stream({"question": question, "chat_history": chat_history})

# Generate the speech from text
async def generate_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        await communicate.save(temp_file.name)
        temp_file_path = temp_file.name
    return temp_file_path

# Get audio player
def get_audio_player(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
        
# Text-to-Speech function which automatically plays the audio
def generate_voice(text, voice):
    text_to_speak = (text).translate(str.maketrans('', '', '#-*_😊👋😄😁🥳👍🤩😂😎')) # Removing special chars and emojis
    with st.spinner("Generating voice response..."):
        temp_file_path = asyncio.run(generate_speech(text_to_speak, voice)) 
        audio_player_html = get_audio_player(temp_file_path)  # Create an audio player
        st.markdown(audio_player_html, unsafe_allow_html=True)
        os.unlink(temp_file_path)

def display_metadata(metadata_list, message_index):
    """Display metadata in an expandable section"""
    if metadata_list:
        with st.expander(f"📚 View Sources ({len(metadata_list)} references)", expanded=False):
            for i, metadata in enumerate(metadata_list, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {metadata['source']}**")
                with col2:
                    if metadata['page'] != "Unknown":
                        st.write(f"Page {metadata['page']}")
                    else:
                        st.write("Page N/A")
                
                if i < len(metadata_list):
                    st.divider()

# Sidebar voice option selection
if st.sidebar.toggle("Enable Voice Response"):
    voice_option = st.sidebar.selectbox("Choose a voice for response:", options=list(voices.keys()), key="voice_response")

# Dividing the main interface into two parts
col1, col2 = st.columns([1, 5])

# Displaying chat history
for idx, message in enumerate(st.session_state.chat_history):
    avatar = "assets/user.png" if message["role"] == "user" else "assets/assistant.png"
    with col2:
        st.chat_message(message["role"], avatar=avatar).write(message["content"])
        
        # Display metadata for assistant messages (except the initial greeting)
        if message["role"] == "assistant" and message.get("metadata") is not None:
            display_metadata(message["metadata"], idx)

# Handle voice or text input
with col1:
    st.button("Reset", use_container_width=True, on_click=reset_conversation)

    with st.spinner("Converting speech to text..."):
        text = speech_to_text(language="en", just_once=True, key="STT", use_container_width=True)

query = st.chat_input("Type your question")

# Generate the response
if text or query:
    col2.chat_message("user", avatar="assets/user.png").write(text if text else query)
    
    st.session_state.chat_history.append({"role": "user", "content": text if text else query, "metadata": None})

    # Generate response
    with col2.chat_message("assistant", avatar="assets/assistant.png"):
        try:
            # Get the retriever and retrieve documents first
            retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.get_relevant_documents(text if text else query)
            
            # Extract metadata from retrieved documents
            metadata_list = extract_metadata_from_docs(retrieved_docs)
            
            # Generate and display the response
            response = st.write_stream(rag_qa_chain(question=text if text else query,
                                retriever=retriever,
                                chat_history=st.session_state.chat_history))
            
            # Add response to chat history with metadata
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response, 
                "metadata": metadata_list
            })
            
            # Display metadata immediately after the response
            display_metadata(metadata_list, len(st.session_state.chat_history) - 1)
            
        except Exception as e:
            st.error(f"An internal error occurred. Please check your internet connection")

    # Generate voice response if the user has enabled it
    if "voice_response" in st.session_state and st.session_state.voice_response:
        response_voice = st.session_state.voice_response
        generate_voice(response, voices[response_voice])
