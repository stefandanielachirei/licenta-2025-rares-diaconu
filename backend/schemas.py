from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class UserCreateRequest(BaseModel):
    email: str
    role: str