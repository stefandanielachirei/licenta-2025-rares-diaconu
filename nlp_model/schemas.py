from pydantic import BaseModel

class TextRequest(BaseModel):
    prompt: str