from pydantic import BaseModel

class Analyze(BaseModel):
    text : str

class Summary(BaseModel):
    summary : str

class SemanticSearchRequest(BaseModel):
    text: str