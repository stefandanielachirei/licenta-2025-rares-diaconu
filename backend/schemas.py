from pydantic import BaseModel, ConfigDict
from typing import Optional

class PromptRequest(BaseModel):
    prompt: str

class UserCreateRequest(BaseModel):
    email: str
    role: str

class BookCreate(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    title: str
    author: str
    isbn: str
    goodreads_id: str
    image_url: str

class BookUpdate(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    title: Optional[str] = None
    author: Optional[str] = None
    isbn: Optional[str] = None
    goodreads_id: Optional[str] = None
    image_url: Optional[str] = None

class StatusUpdate(BaseModel):
    user_email: str
    book_id: int
    status: str