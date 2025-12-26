from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    session_id : str = "default"
    
class SourceInfo(BaseModel):
    content : str
    filename : str
    

class QueryRespnose(BaseModel):
    original_query: str
    rewritten_query: str
    answer : str
    sources: List[SourceInfo]
    status : str # 'answered' or 'no_context'