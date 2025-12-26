import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

# Create a TestClient instance
client = TestClient(app)

# --- Fixtures to Mock External Services ---

@pytest.fixture
def mock_graph_response():
    """
    This fixture simulates a successful response from our LangGraph brain.
    It returns what the graph.invoke() would return.
    """
    mock_doc = MagicMock()
    mock_doc.page_content = "To reset the device, hold the button for 5 seconds."
    mock_doc.metadata = {"filename": "manual.pdf"}

    return {
        "answer": "You need to hold the button for 5 seconds.",
        "documents": [mock_doc],
        "rewritten_query": "how to reset device",
        "generation_status": "go"
    }

@pytest.fixture
def mock_ingestion():
    """Mocks the PDF ingestion service so we don't need real PDFs or Pinecone."""
    return 15  # Simulates 15 chunks being indexed

# --- The Tests ---

@patch("app.api.v1.endpoints.rag_graph.ainvoke", new_callable=AsyncMock)
def test_query_endpoint_success(mock_ainvoke, mock_graph_response):
    """
    Test the /query endpoint.
    We mock the graph's 'ainvoke' method to return our fake response immediately.
    """
    # 1. Setup the mock behavior
    mock_ainvoke.return_value = mock_graph_response

    # 2. Make the request
    payload = {"query": "How do I reset?"}
    response = client.post("/api/v1/query", json=payload)

    # 3. Assertions (Check if the code worked)
    assert response.status_code == 200
    data = response.json()
    
    # Check structure matches our Pydantic model
    assert data["original_query"] == "How do I reset?"
    assert data["answer"] == "You need to hold the button for 5 seconds."
    assert data["status"] == "answered"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "manual.pdf"

@patch("app.api.v1.endpoints.rag_graph.ainvoke", new_callable=AsyncMock)
def test_query_endpoint_no_context(mock_ainvoke):
    """
    Test the case where the AI finds no relevant documents.
    """
    # 1. Setup mock for failure case
    mock_ainvoke.return_value = {
        "answer": "I don't know.",
        "documents": [],
        "rewritten_query": "unknown query",
        "generation_status": "stop"
    }

    # 2. Make request
    response = client.post("/query", json={"query": "What is the meaning of life?"})

    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_context"
    assert data["sources"] == []

@patch("app.api.v1.endpoints.ingest_pdf")
def test_upload_document(mock_ingest):
    """
    Test file upload. We mock 'ingest_pdf' to avoid parsing real PDFs.
    """
    # 1. Setup mock
    mock_ingest.return_value = 10 # Simulate 10 chunks indexed

    # 2. Create a fake file in memory
    files = {
        'file': ('test_manual.pdf', b'%PDF-1.4 content...', 'application/pdf')
    }

    # 3. Make request
    response = client.post("/api/v1/document/upload", files=files)

    # 4. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_manual.pdf"
    assert data["chunks_indexed"] == 10

def test_upload_invalid_file_type():
    """
    Test that uploading a non-PDF file returns a 400 error.
    """
    files = {
        'file': ('image.png', b'image data', 'image/png')
    }
    response = client.post("/document/upload", files=files)
    
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]