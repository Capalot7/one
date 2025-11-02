import os
import base64

import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
# import demjson
import ujson
import json
import re

load_dotenv()
groq_api_key = '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'
working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="InsightLens",
    page_icon="💬",
    layout="wide"  # Make layout wide for panel display
)

#### Preprocessing Functions

@st.cache_data
def load_document(file_path):
    """Reads and loads data from PDF, Word, PPT, PPTx, or TXT documents and returns LangChain documents."""
    file_extension = os.path.splitext(file_path)[1].lower()
    # try:
    if file_extension == '.pdf':
        loader = UnstructuredPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_extension in  '.pptx':
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    documents = loader.load()
    
    # Validate documents
    if not documents:
        raise ValueError("No content extracted from the document")
    
    # Adding Source Metadata
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    return documents
    # except Exception as e:
    #     st.error(f"Error loading document {os.path.basename(file_path)}: {str(e)}")
    #     return []

@st.cache_data
def setup_vectorstore(_documents, file_path):
    """Splits documents, embeds them, stores in FAISS, and returns vector database."""
    if not _documents:
        st.error("No documents provided to create vector store")
        return None
    
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        doc_chunks = splitter.split_documents(_documents)
        
        if not doc_chunks:
            st.error("No document chunks created after splitting")
            return None
        
        vector_store = FAISS.from_documents(doc_chunks, embedding)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def create_chain(vector_store):
    """Creates a conversational retrieval chain."""
    if vector_store is None:
        st.error("Cannot create conversation chain without a valid vector store")
        return None
    
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
    retriever = vector_store.as_retriever()

    memory = ConversationBufferMemory(
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    
    # Define the prompt template for the chain
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a PDF Chat Bot, named InsightLens.
            
            Please Provide a detailed yet concise response to user queries,
            based on available documents.
            
            if the query is not in the context provided, be creative with the resources or context we have or answer from your knowledge base.
            
            Always maintain a friendly and helpful tone.

            Use Markdown for formatting to enhance readability.

            If the user query is related to a specific section of the document, reference that section in your response.
            
            If a user asks something out of context answer it from your own knowledge base and mention that it is out of context, Remember always mention if a query is out of context
            and mention its source as your knowledge base.

            Dont mention "Query Response" before giving a response.

        Return your response in the following JSON format:
        {{
          "answer": "<your_answer_here>",
          "has_source": <true_or_false>
        }}

        Note that the JSON format is final. While the content may change according to user instructions, you must stick to the JSON format AT ALL COSTS.
        Return true for 'has_source' only if the answer is relevant to the source.
        Never mention has_source = true/false in the response generated to be displayed.

        **Examples:**

        1. **Query with context**:
        Question: What is the capital of France?
        Context: The capital of France is Paris.
        Response:
        {{
            "answer": "The capital of France is **Paris**.",
            "has_source": true
        }}

        2. **Query without context**:
        Question: What is the smell of rain like?
        Context: [No relevant information]
        Response:
        {{
            "answer": "The smell of rain is often described as **fresh** and **earthy**, caused by a bacteria called actinomycetes. This response is based on my knowledge base, as no relevant context was provided.",
            "has_source": false
        }}

        3. **Complex query with partial context**:
        Question: What are the causes of global warming and its impacts?
        Context: Global warming is caused by greenhouse gas emissions, such as carbon dioxide from burning fossil fuels.
        Response:
        {{
            "answer": "Global warming is caused by **greenhouse gas emissions**, such as carbon dioxide from burning fossil fuels, as stated in the document. Additionally, impacts include **rising sea levels**, **extreme weather events**, and **ecosystem disruption**, based on my knowledge base.",
            "has_source": true
        }}

        **Your Response:**
        Return your response for the following in the JSON format above.

        Question: {question}
        Context: {context}
        Response:
        """
    )

    # Create the chain with the custom prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    return chain

@st.cache_data
def reset_chat(all_file_path):
    st.session_state.chat_history = []
    st.session_state.show_source_dialog = False #Won't show source by default after typing message, after document change

def sanitize_json_string(raw_string):
    """Sanitize a JSON string to escape control characters and remove invalid characters."""
    # Strip leading/trailing whitespace and newlines
    sanitized = raw_string.strip()
    
    # Remove BOM and normalize encoding
    sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8-sig', errors='ignore')
    
    # Remove additional Unicode control characters
    sanitized = ''.join(c for c in sanitized if ord(c) >= 32 or c in '\n\r\t' or c == '\ufeff')
    sanitized = sanitized.replace('\ufeff', '')  # Explicitly remove BOM
    
    # Replace control characters with their escaped equivalents
    control_chars = {
        '\n': '\\n',
        '\r': '\\r',
        '\t': '\\t',
        '\b': '\\b',
        '\f': '\\f',
        '\v': '\\v',
        '\0': '\\0'
    }
    for char, escaped in control_chars.items():
        sanitized = sanitized.replace(char, escaped)
    
    # Ensure backslashes are properly escaped
    sanitized = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', sanitized)
    
    # Ensure the string starts with { and ends with }
    if not sanitized.startswith('{'):
        sanitized = sanitized.lstrip(' \n\r\t{')  # Remove any leading junk
        sanitized = '{' + sanitized
    if not sanitized.endswith('}'):
        sanitized = sanitized.rstrip(' \n\r\t}') + '}'
    
    return sanitized

### **Upload and Process Documents**
st.sidebar.title("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader("Upload Your PDF, Word, PPTx, or TXT File", type=['pdf', 'docx', 'pptx', 'txt'], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    all_file_path = []
    for uploaded_file in uploaded_files:
        file_path = f"{working_dir}/{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
   
        documents = load_document(file_path)
        if documents:  # Only add if documents were successfully loaded
            all_file_path.append(file_path)
            all_docs.extend(documents)

    # Store first uploaded file path for preview
    if all_docs and "source_path" not in st.session_state:
        st.session_state.source_path = all_file_path[0]

    if all_docs:
        st.session_state.vector_store = setup_vectorstore(all_docs, all_file_path)
        if st.session_state.vector_store:
            st.session_state.conversation_chain = create_chain(st.session_state.vector_store)
            file_names = ", ".join([file.name for file in uploaded_files])
            st.sidebar.success(f"Processed {len(uploaded_files)} file(s)")
            reset_chat(all_file_path)
        else:
            st.error("Failed to create vector store. Please check the documents and try again.")
    else:
        st.error("No valid documents were loaded. Please check the uploaded files.")

###  Main Chat Interface

# Add an image as a title
st.image("InsightLens_logo.png", width=400)
if not uploaded_files:
    st.write("Upload a file to start the chat")

# Chat History Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#initializing session state for dialog box
if "show_source_dialog" not in st.session_state:
    st.session_state.show_source_dialog = False

#To store current_source_path
if "current_source_path" not in st.session_state:
    st.session_state.current_source_path = None    

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User Input
user_input = st.chat_input("Ask InsightLens...")

# predefined_responses For Try-except block
predefined_responses = {
    "What is Global Warming": ["InsightLens is a document-based chatbot that allows you to interact with your PDFs.","global_warmimg.pdf"],
}

if user_input:
    # Store the original user query
    original_query = user_input
    
    # Add context to the query
    structured_query = user_input
    
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})

    with st.chat_message("user"):
        st.markdown(user_input)
    
    try:
        with st.chat_message("assistant"):
            if not uploaded_files:
                st.markdown("Upload A File First")
                st.session_state.chat_history.append({"role": "assistant", "content": "Upload A File First"})
            elif not hasattr(st.session_state, 'conversation_chain') or st.session_state.conversation_chain is None:
                st.markdown("No valid conversation chain available. Please upload valid documents.")
                st.session_state.chat_history.append({"role": "assistant", "content": "No valid conversation chain available. Please upload valid documents."})
            else:
                response = st.session_state.conversation_chain({"question": structured_query})
                
                # Parse the structured JSON response
                try:
                    raw_response = response['answer']
                    # sanitized_response = sanitize_json_string(raw_response)
                    structured_response = json.loads(raw_response)
                    # structured_response = ujson.loads(raw_response)
                    assistant_response = structured_response["answer"]
                    has_source = structured_response["has_source"]
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"JSON_ERROR: {e}")
                    print(f"Response received: {response['answer']}")
                    # Fallback in case JSON parsing fails
                    assistant_response = response['answer']
                    has_source = len(response.get("source_documents", [])) > 0
                
                st.markdown(assistant_response)
               
                # Show source only if has_source is True and source documents are available
                st.session_state.show_source_dialog = False               
                if has_source and response.get("source_documents"):
                    first_source = response["source_documents"][0].metadata.get("source", "Unknown")
                    source_path = os.path.join(working_dir, first_source)
                    st.markdown(f"**Response Source:** {first_source}")
                    st.session_state.current_source_path = source_path

                    def show_source():
                        st.session_state.show_source_dialog = True

                    st.button("View Source", on_click=show_source)
           
            # Append assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})  
            
            # Response Evaluation
            llm2 = ChatGroq(model="gemma2-9b-it")
            retriever = st.session_state.conversation_chain.retriever
            retrieved_docs = retriever.get_relevant_documents(user_input)

            evaluation_prompt = f"""
            You are an evaluator. Rate the chatbot's response on a scale of 1 to 10 for:
            1. **Answer Relevance** (Does the response correctly address the query?)
            2. **Context Relevance** (Is the response based on the retrieved documents?)
            3. **Hallucination** (Does the response contain incorrect/unverified information?)
            4. **Completeness** (Does the response fully answer the question?)
            
            Query: {structured_query}
            Chatbot Response: {assistant_response}
            Retrieved Context: {retrieved_docs}

            Provide ratings only in this format:
            - Answer Relevance: X/10
            - Context Relevance*: Y/10
            - Hallucination: Z/10
            - Completeness: B/10
            
            If the query is out of context give, mention that and give the metrics.
            apart from this dont add anything extra.
            """
            evaluation = llm2.invoke(evaluation_prompt)

            # **Display Metrics in the Right Sidebar**
            with st.sidebar:    
                st.sidebar.divider()    
                st.header("Model Evaluation Metrics")
                st.markdown(f"**Query:** {original_query}")
                st.markdown("**Evaluation Scores:**")
                st.markdown(evaluation.content)

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        
        if not uploaded_files:
            pass
        else:
            with st.chat_message("assistant"):
                if user_input in predefined_responses:
                    assistant_response = predefined_responses[user_input][0]
                    first_source = predefined_responses[user_input][1]
                else:
                    assistant_response = "I couldn't process your request. Please try again."
                    first_source = "Unknown"
                
                st.markdown(assistant_response)
                source_path = os.path.join(working_dir, first_source)
                st.markdown(f"**Source:** {first_source}") 
                st.session_state.current_source_path = source_path 
                    
                def show_source():
                    st.session_state.show_sourceuji_dialog = True

                st.button("View Source", on_click=show_source)    
            
            # Append assistant response to chat history    
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

# Track RPM limit
import time
from collections import deque

if "query_timestamps" not in st.session_state:
    st.session_state.query_timestamps = deque(maxlen=60)  # Store timestamps (last 60 seconds)

# When user submits a query
if user_input:
    current_time = time.time()
    st.session_state.query_timestamps.append(current_time)

# Remove timestamps older than 60 seconds
current_time = time.time()
st.session_state.query_timestamps = deque(
    [t for t in st.session_state.query_timestamps if current_time - t <= 60], 
    maxlen=60
)

# Display queries per minute in sidebar
with st.sidebar:
    st.sidebar.divider()
    st.sidebar.markdown(f""" **Rate Limits**       
                           
                        Queries per Minute:{len(st.session_state.query_timestamps)}
        Max RPM: 30""")            
    

# Dialog Box for Source Document Preview (Moved outside the response block)
if st.session_state.show_source_dialog and st.session_state.current_source_path:
    with st.container():
        st.markdown("### Source Document Preview")
        file_extension = os.path.splitext(st.session_state.current_source_path)[1].lower()
        if file_extension == '.pdf':
            with open(st.session_state.current_source_path, "rb") as file:
                pdf_data = base64.b64encode(file.read()).decode("utf-8")
                iframe_html = f"""
                <iframe src="data:application/pdf;base64,{pdf_data}" width="100%" height="500px" style="border:none;"></iframe>
                """
                st.markdown(iframe_html, unsafe_allow_html=True)
        else:
            st.markdown("Preview not available for this file type. Download the file to view its contents.")
            with open(st.session_state.current_source_path, "rb") as file:
                st.download_button(
                    label="Download Source File",
                    data=file,
                    file_name=os.path.basename(st.session_state.current_source_path),
                    mime="application/octet-stream"
                )
        
        def close_iframe():
            st.session_state.show_source_dialog = False
        
        st.button("Close", on_click=close_iframe)

