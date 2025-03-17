import grpc
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import idm_service_pb2
import idm_service_pb2_grpc

class TokenValidationMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        if scope["path"] == "/save-user" and request.method == "POST":
            await self.app(scope, receive, send)
            return

        if request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        authorization = request.headers.get("Authorization")

        if not authorization or not authorization.startswith("Bearer "):
            response = JSONResponse(
                status_code=401, content={"detail": "Authorization header missing or invalid"}
            )
            await response(scope, receive, send)
            return
        
        token = authorization.split(" ")[1]

        try:
            with grpc.insecure_channel("idm_service:50051") as channel:
                stub = idm_service_pb2_grpc.IDMServiceStub(channel)
                response = stub.ValidateToken(idm_service_pb2.ValidateTokenRequest(token=token))

                if not response.valid:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
                
                scope["user"] = {"username" : response.username, "role" : response.role}

        except grpc.RpcError as e:
            response = JSONResponse(
                status_code=500, content={"detail": f"Token validation failed: {str(e)}"}
            )
            await response(scope, receive, send)
            return
        
        await self.app(scope, receive, send)