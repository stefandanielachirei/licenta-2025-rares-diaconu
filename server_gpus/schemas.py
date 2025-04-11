from pydantic import BaseModel
from typing import List
    
class ReviewInput(BaseModel):
    texts: List[str]
    max_tokens_threshold: int = 20
    
class SentimentRequest(BaseModel):
    texts: list[str]
    
class TextsRequest(BaseModel):
    texts: list[str]