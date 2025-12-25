from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import uuid
import tempfile

app = FastAPI(title="FAQ")
# --- Configuration --
# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize PC
pc = Pinecone(api_key= os.getenv("PINECONE_API_KEY"))

INDEX_NAME= "FAQ"

# Initialize Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embedding': True}
)

DIMENSION = len(embeddings.embed_query("TEST")) # Embed 'TEST' and get the dimesion

# Create index if it doesn't exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name = INDEX_NAME,
        dimension= DIMENSION,
        metric= 'cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region= "us-east-1"
        )
    )

# Initialize vector store
vectorstore= PineconeVectorStore(
    index_name= INDEX_NAME,
    embedding= embeddings
)

# Define RAG Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 
    documents: list[Document]
    query: str
    answer: str

def create_rag_graph():
    """RAG with LangGraph multi-step reasoning"""
    
    llm = ChatOpenAI(
        model = MODEL_NAME,
        api_key= OPENROUTER_API_KEY,
        base_url= BASE_URL,
        temperature= 0.3 
    )

    # Node 1 : Query Rewriter
    async def rewrite_query(state: AgentState):
        """Rewrite query for better retrieval"""
        query = state["messages"][-1].content
        system_prompt = """User Queyr:{query}
                        Rewrite this query to be more specific and suitable for semantic search.
                        Return only the rewritten query nothing else."""
        response = await llm.ainvoke(HumanMessage(system_prompt))
        rewritten = response.content
        
        return {"query" : rewritten}
    
    # Node 2 : Retriever
    async def retrieve_document(state:AgentState):
        """Retrieve relevant document"""
        query = state.get("query", state["messages"][-1].content)
        
        docs = vectorstore.similarity_search(
            query= query,
            k = 3
        )
        
        return {"documents": docs}
    
    # Node 3 : Grader
    async def grade_document(state:AgentState):
        """Grade document relevancies"""
        query = state["query"]
        documents= state["documents"]
        
        relevant_doc = []
        for doc in documents:
            grade_prompt = f"""Query: {query}
                            Document: {doc.page_content[:500]}
                            Is this document relevant to the query? 
                            answer only 'yes' or 'no'."""
                            
            response = await llm.ainvoke(HumanMessage(grade_prompt))
            grade = response.content.strip().lower() # yes/ no 
            if grade== 'yes':
                relevant_doc.append(doc)
                
        return {"documents" : relevant_doc}
    
    
    # Node 4 : Generator 
    async def generate_answer(state:AgentState):
        """Generate final answer"""
        query = state["query"]
        documents = state["documents"]
        
        context = "\n\n".join([
            f"Document_{i+1}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Answer the question based on the next context
                Context: {context}
                Question: {query}
                answer:"""
                
        response = await llm.ainvoke(HumanMessage(prompt)) 
        answer = response.content
        
        return{
            "answer" : answer,
            "messages": [AIMessage(answer)]
        }
        
    
    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("rewrite" , rewrite_query)
    graph.add_node("retrieve" , retrieve_document)
    graph.add_node("grade" , grade_document )
    graph.add_node("generate" , generate_answer)
    
    # Add edges
    graph.add_edge(START, "rewrite")
    graph.add_edge('rewrite', 'retrieve')
    graph.add_edge('retrieve', 'grade')
    graph.add_edge('grade', 'generate')
    graph.add_edge('generate', END)
    
    return graph.compile()


# Initialize RAG graph
rag_graph = create_rag_graph()


 
# ---- Pydantic Models --------
class QueryRequest(BaseModel):
    query : str
    session_id : str = 'default'
    
class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    answer : str
    source : list[dict]
    num_source_used : int
    

@app.post("/document/upload")
async def upload_documents(file:UploadFile = File(...)):
    """Upload the process document"""
    
    # Save temporary
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load documents
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= 1000,
            chunk_overlap = 200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        doc_id = str(uuid.uuid4())
        for chunk in chunks :
            chunk.metadata.update({
                "doc_id" : doc_id,
                "filename" : file.filename
            })
            
            
        # Store in Pinecone
        vectorstore.add_documents(chunks)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return{
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_created": len(chunks),
            "status": "sucess"
        }
        
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed To load files {str(e)}")
        
        
@app.post("/query", response_model= QueryResponse)
async def query_rag(request: QueryResponse):
    """Query the RAG System"""
    
    # Run the graph
    result = rag_graph.ainvoke({
        "messages" : [HumanMessage(request.query)],
        "documents": [],
        "query": "",
        "answer": ""
    })
    
    return QueryResponse(
        query = request.query,
        rewritten_query= result["query"],
        answer = result["answer"],
        source = [
            {
                "content" : doc.page_content[:200],
                "filename": doc.metadata.get("filename", "unknown")
            }
            for doc in result["documents"]
        ],
        num_source_used= len(result["documents"])
    )
    
@app.get("/index/stats")
async def get_index_status():
    """Get pinecone index statistics"""
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    
    return{
        "total_vectorstore": stats.total_vector_count,
        "dimension" : stats.dimension,
        "Index_fullness" : stats.index_fullness,
        "namespace" : stats.namespace
    }
    
    
    
                
    
    