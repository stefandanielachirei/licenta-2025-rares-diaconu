from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import grpc
import idm_service_pb2
import idm_service_pb2_grpc
from schemas import LoginRequestModel, ValidateTokenRequestModel, LogoutRequestModel, ChangePasswordRequestModel, DeleteUserRequestModel, RegisterRequestModel, DeleteAccountRequestModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
security = HTTPBearer()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

grpc_server_address = "idm_service:50051"

def get_grpc_stub():
    try:
        channel = grpc.insecure_channel(grpc_server_address)
        stub = idm_service_pb2_grpc.IDMServiceStub(channel)
        return stub
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not connect to gRPC server: {str(e)}")

async def validate_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    stub = get_grpc_stub()

    try:
        grpc_request = idm_service_pb2.ValidateTokenRequest(token=token)
        grpc_response = stub.ValidateToken(grpc_request)

        if not grpc_response.valid:
            raise HTTPException(status_code=403, detail=grpc_response.message)

        return {
            "username" : grpc_response.username,
            "role" : grpc_response.role
        }

    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token validation failed: {str(e)}")


@app.post("/login")
async def login(request: LoginRequestModel):
    try:
        grpc_request = idm_service_pb2.LoginRequest(
            username=request.username,
            password=request.password
        )

        stub = get_grpc_stub()
        grpc_response = stub.Login(grpc_request)

        if not grpc_response.token:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {"token" : grpc_response.token}

    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.get("/validate")
async def validate(user_info: dict = Depends(validate_bearer_token)):
    return {
        "username" : user_info["username"],
        "role" : user_info["role"],
        "message" : "Token is valid",
    }

@app.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    stub = get_grpc_stub()
    try:
        grpc_request = idm_service_pb2.LogoutRequest(
            token=token
        )
        grpc_response = stub.Logout(grpc_request)

        if not grpc_response.success:
            raise HTTPException(status_code=400, detail=grpc_response.message)

        return {"message" : grpc_response.message}

    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")


@app.post("/changePassword")
async def change_password(request: ChangePasswordRequestModel, user_info: dict = Depends(validate_bearer_token)):
    
    if user_info["username"] != request.username:
        raise HTTPException(status_code=403, detail="You can only change your own password.")

    stub = get_grpc_stub()

    grpc_request_change_password = idm_service_pb2.ChangePasswordRequest(
        username=request.username,
        current_password=request.current_password,
        new_password=request.new_password
    )

    try:
        grpc_response_change_password = stub.ChangePassword(grpc_request_change_password)

        if not grpc_response_change_password.success:
            raise HTTPException(status_code=403, detail=grpc_response_change_password.message)

        return {"message" : grpc_response_change_password.message}

    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Password change failed: {str(e)}")

@app.post("/deleteUser")
async def delete_user(request: DeleteUserRequestModel, user_info: dict = Depends(validate_bearer_token)):

    if user_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")

    stub = get_grpc_stub()

    grpc_request_delete = idm_service_pb2.DeleteUserRequest(
        username = request.username
    )

    try:
        grpc_response_delete = stub.DeleteUser(grpc_request_delete)
        if not grpc_response_delete.success:
            raise HTTPException(status_code=403, detail=grpc_response_delete.message)

        return {"message": grpc_response_delete.message}

    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User deletion failed: {str(e)}")

@app.post("/register")
async def register(request: RegisterRequestModel):

    stub = get_grpc_stub()

    grpc_request = idm_service_pb2.RegisterRequest(
        username = request.username,
        password = request.password
    )

    try:
        grpc_response = stub.Register(grpc_request)
        if not grpc_response.success:
            raise HTTPException(status_code=400, detail=grpc_response.message)
        
        return {"message" : grpc_response.message}

    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User registration failed: {str(e)}")

@app.post("/deleteAccount")
async def delete_account(request: DeleteAccountRequestModel, user_info: dict = Depends(validate_bearer_token)):
    
    if user_info["role"] == "admin":
        raise HTTPException(status_code=403, detail="Permission denied")

    stub = get_grpc_stub()

    grpc_request = idm_service_pb2.DeleteAccountRequest(
        username = request.username
    )

    try:
        grpc_response = stub.DeleteAccount(grpc_request)
        if not grpc_response.success:
            raise HTTPException(status_code=400, detail=grpc_response.message)

        return {"message": grpc_response.message}

    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Account deletion failed: {str(e)}")