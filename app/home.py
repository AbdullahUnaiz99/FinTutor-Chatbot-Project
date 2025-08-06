import streamlit as st
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings import get_embeddings
from src.vector_store import load_vector_store
from src.utils import load_environment
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="FinTutor Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .message-content {
        margin-left: 1rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .processing-indicator {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """Initialize all components with error handling"""
    try:
        # Load environment variables
        load_environment()
        
        # Initialize embeddings
        embeddings = get_embeddings()
        
        # Load vector store
        vector_store = load_vector_store("fintutor", embeddings)
        
        # Initialize LLM - SIMPLIFIED to prevent repetition
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model="mistralai/mistral-7b-instruct:free",
            temperature=0.3,
            max_tokens=300,
            streaming=False,
        )
        
        return embeddings, vector_store, llm
        
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None, None, None

def create_rag_chain(vector_store, llm):
    """Create RAG chain for question answering"""
    try:
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Simple prompt to prevent repetition
        system_prompt = (
            "You are a helpful finance tutor. "
            "Answer the question based on the context provided. "
            "Give a clear, concise answer in 2-3 sentences maximum. "
            "Stop after giving one complete answer."
            "\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Failed to create RAG chain: {str(e)}")
        return None

def process_user_question(question, rag_chain):
    """Process a single user question and return response"""
    try:
        with st.spinner("🤔 Thinking..."):
            response = rag_chain.invoke({"input": question})
            return response.get("answer", "I'm sorry, I couldn't generate a response.")
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 FinTutor Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Your Personal Finance Learning Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False
    
    # Initialize components
    if "components_initialized" not in st.session_state:
        with st.spinner("🔄 Initializing FinTutor... Please wait."):
            embeddings, vector_store, llm = initialize_components()
            
            if embeddings and vector_store and llm:
                rag_chain = create_rag_chain(vector_store, llm)
                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    st.session_state.components_initialized = True
                    st.success("✅ FinTutor initialized successfully!")
                else:
                    st.error("❌ Failed to create RAG chain")
                    st.stop()
            else:
                st.error("❌ Failed to initialize components")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### About FinTutor")
        st.markdown("""
        🎯 **Features:**
        - Finance concepts explanation
        - Bond and investment queries
        - Corporate finance guidance
        - Market analysis help
        
        📚 **Knowledge Base:**
        - Finance textbooks
        - Investment guides
        - Market analysis papers
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample questions
        st.markdown("### Try These Questions:")
        sample_questions = [
            "What is a bond?",
            "Explain compound interest",
            "What is portfolio diversification?",
            "How does the stock market work?",
            "What is financial risk?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}", disabled=st.session_state.processing):
                st.session_state.pending_question = question
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", disabled=st.session_state.processing):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Welcome to FinTutor! Ask me anything about finance, investments, bonds, or market analysis.")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-message user-message">
                    <div style="font-size: 1.5rem;">🧑‍💼</div>
                    <div class="message-content">
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message bot-message">
                    <div style="font-size: 1.5rem;">🤖</div>
                    <div class="message-content">
                        <strong>FinTutor:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)

    # Processing indicator
    if st.session_state.processing:
        st.markdown('''
        <div class="processing-indicator">
            <strong>🤔 Processing your question...</strong><br>
            Please wait while I generate a response.
        </div>
        ''', unsafe_allow_html=True)

    # Chat input section
    st.markdown("---")
    
    # Input and buttons
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        # Use a dynamic key to allow input clearing
        input_key = f"user_input_{len(st.session_state.messages)}"
        user_input = st.text_input(
            "Ask your finance question:",
            placeholder="e.g., What is the difference between stocks and bonds?",
            disabled=st.session_state.processing,
            key=input_key
        )
    
    with col2:
        ask_button = st.button(
            "Ask 💬", 
            type="primary",
            disabled=st.session_state.processing or not user_input.strip()
        )
    
    with col3:
        stop_button = st.button(
            "🛑 Stop",
            disabled=not st.session_state.processing,
            help="Stop current response generation"
        )

    # Handle stop button
    if stop_button:
        st.session_state.stop_processing = True
        st.session_state.processing = False
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "❌ Response generation stopped by user."
        })
        st.rerun()

    # Handle pending question from sidebar
    if hasattr(st.session_state, 'pending_question') and not st.session_state.processing:
        question = st.session_state.pending_question
        delattr(st.session_state, 'pending_question')
        
        # Add user message and start processing
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.processing = True
        st.session_state.stop_processing = False
        st.rerun()

    # Handle ask button
    if ask_button and user_input.strip() and not st.session_state.processing:
        # Add user message and start processing
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.processing = True
        st.session_state.stop_processing = False
        # Note: Input will clear automatically on rerun due to disabled state
        st.rerun()

    # Process question if needed
    if st.session_state.processing and not st.session_state.stop_processing:
        # Get the last user message
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            # Generate response
            response = process_user_question(last_message["content"], st.session_state.rag_chain)
            
            # Add response and stop processing
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.processing = False
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit, LangChain, and Pinecone | "
        "Powered by Mistral AI"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

