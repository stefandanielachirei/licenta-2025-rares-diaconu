from pydantic import BaseModel

class LoginRequestModel(BaseModel):
    username: str
    password: str

class ValidateTokenRequestModel(BaseModel):
    token: str

class LogoutRequestModel(BaseModel):
    token: str

class RegisterRequestModel(BaseModel):
    username: str
    password: str

class ChangePasswordRequestModel(BaseModel):
    username: str
    current_password: str
    new_password: str

class DeleteUserRequestModel(BaseModel):
    username: str

class DeleteAccountRequestModel(BaseModel):
    username: str
