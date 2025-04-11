from pydantic import BaseModel
from typing import List

class ReviewRequest(BaseModel):
    review: str
    
class ReviewInput(BaseModel):
    texts: List[str]
    max_tokens_threshold: int = 20