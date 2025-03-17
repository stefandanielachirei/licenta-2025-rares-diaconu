import grpc
from concurrent import futures
import hashlib
import jwt
import os
import uuid
from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import User, BlacklistedToken, UserRole, Base
import idm_service_pb2
import idm_service_pb2_grpc
from datetime import datetime, timedelta

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@idm_db:5432/idm_db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
secret_key = "e180f017c88101758c609be62fef16adacae5c655d51dac5dd9adfc073ce901bdd50caa734114e002b262c8e33b8d9714b25b09309b87e199e2a8b234ff9cd44ef5e05b1fb1f2e26f56801e2955ee872108eb4d3a12c74b6f17a9b6a3db555a66364a1d4fb84057c8fb44116b4b2b0991805262c3813b8e91cbb409498e0feb5a9dcaedf751c48c8ee6d764fb86ec25a27c472219283e7e0ebac7dcb26ed2cec3f7b18795ba820cf8c00e287eb97f78944d34e53ad898a1e50a2c6004376e947f57f85a82755f97b83467e15f85da7b27abbdeb6638d1ed7b38b627b1fa425fe8c7696abf34d02ce595465d8908883c2f3a61e2d9253a284af6dc8950926c9c3"

def hashpassword(password: str, salt: str = "default_salt") -> str :
    salted_password = f"{salt}{password}".encode('utf-8')
    return hashlib.sha256(salted_password).hexdigest()

def verify_password(plain_password: str, hashed_password: str, salt: str = "default_salt") -> bool:
    return hashpassword(plain_password, salt) == hashed_password


class IDMServiceServices(idm_service_pb2_grpc.IDMServiceServicer) :
    def __init__(self) :
        self.db = SessionLocal()

    def Login(self, request, context) :
        try:
            user = self.db.query(User).filter(User.email == request.username).first()
            if user is None or user.password is None:
                print("Login attempt failed: user does not exist or has no password set.")
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Invalid credentials.")
                return idm_service_pb2.LoginResponse()

            hashed_password = hashpassword(request.password)
            if hashed_password != user.password :
                print("Authentification failed")
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Invalid credentials.")
                return idm_service_pb2.LoginResponse()
            

            expiration_time = datetime.utcnow() + timedelta(hours=2)
            token_payload = {
                'iss' : '[::]:50051',
                'sub' : user.email,
                'exp' : expiration_time,
                'jti' : str(uuid.uuid4()),
                'role' : user.role.name
            }

            token = jwt.encode(token_payload, secret_key, algorithm='HS256')
            print(f"It's authentificating {request.username} token:{token}")
            return idm_service_pb2.LoginResponse(token=token)

        except Exception as e:
            print(f"Error during authentification: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return idm_service_pb2.LoginResponse()

    def ValidateToken(self, request, context) :
        try:
            token = request.token
            if self.db.query(BlacklistedToken).filter(BlacklistedToken.token == token).first():
                print(f"Token is blacklisted: {token}")
                return idm_service_pb2.ValidateTokenResponse(valid=False, username="", role="")
            try:
                payload = jwt.decode(token, secret_key, algorithms=['HS256'])
                username = payload.get("sub")
                role = payload.get("role")
                print(f"Decoded token: {payload}")
                if datetime.utcnow() > datetime.utcfromtimestamp(payload["exp"]):
                    self.blacklist_token(token)
                    return idm_service_pb2.ValidateTokenResponse(valid = False, username="", role="")
                
                return idm_service_pb2.ValidateTokenResponse(valid = True, username=username, role=role)

            except jwt.ExpiredSignatureError:
                print(f"Token has expired: {token}")
                self.blacklist_token(token)
                return idm_service_pb2.ValidateTokenResponse(valid = False, username="", role="")

            except jwt.InvalidTokenError:
                print(f"Invalid token: {token}")
                self.blacklist_token(token)
                return idm_service_pb2.ValidateTokenResponse(valid = False, username="", role="")

        except Exception as e:
            print(f"Error during token validation: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return idm_service_pb2.ValidateTokenResponse(valid = False, username="", role="")
                    
    def Logout(self, request, context) :
        try:
            token = request.token
            self.blacklist_token(token)
            print(f"Token successfully blacklisted")
            return idm_service_pb2.LogoutResponse(success=True, message="Token successfully blacklisted")

        except Exception as e:
            print(f"Error during logout: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return idm_service_pb2.LogoutResponse(success=False, message="Failed to blacklist token")


    def DeleteUser(self, request, context):
        try:
            user = self.db.query(User).filter(User.email == request.username).first()
            if not user:
                print(f"User with email {request.username} does not exist.")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("User not found")
                return idm_service_pb2.DeleteUserResponse(success=False, message="User not found")
            
            self.db.delete(user)
            self.db.commit()

            print(f"User with email {request.username} successfully deleted.")
            return idm_service_pb2.DeleteUserResponse(success=True, message="User deleted successfully.")
        except Exception as e:
            print(f"Error during user deletion: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return idm_service_pb2.DeleteUserResponse(success=False, message="Failed to delete user")
            
    
    def blacklist_token(self, token):
        try:
            blacklistedToken = BlacklistedToken(token=token)
            self.db.add(blacklistedToken)
            self.db.commit()
            print("Token added to the blacklist")
        except IntegrityError:
            self.db.rollback()
            print("Token already in the blacklist")
        except Exception as e:
            self.db.rollback()
            print(f"Error during token blacklisting: {str(e)}")

    def ChangePassword(self, request, context):
        try:
            user = self.db.query(User).filter(User.email == request.username).first()
            if not user:
                print(f"User with email {request.username} does not exist.")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("User not found")
                return idm_service_pb2.ChangePasswordResponse(success=False, message="User not found")
            
            if not verify_password(request.current_password, user.password):
                print(f"Wrong current password for user {request.username}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Wrong current password")
                return idm_service_pb2.ChangePasswordResponse(success=False, message="Wrong current password")

            user.password = hashpassword(request.new_password, salt="default_salt")
            self.db.commit()

            print(f"Password changed successfully for user: {request.username}")
            return idm_service_pb2.ChangePasswordResponse(success=True, message="Password changed successfully")

        except Exception as e:
            print(f"Error during password change: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return idm_service_pb2.ChangePasswordResponse(success=False, message="Failed to change password")

    def Register(self, request, context):
        try:
            existing_user = self.db.query(User).filter(User.email == request.username).first()
            if existing_user:
                print(f"User with email {request.username} already exists.")
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details("User already exists")
                return idm_service_pb2.RegisterResponse(success=False, message="User already exists")
            
            user = User(email = request.username, password = hashpassword(request.password), role = UserRole.user)
            self.db.add(user)
            self.db.commit()

            print(f"User registered: {request.username}")
            return idm_service_pb2.RegisterResponse(success=True, message="User registered successfully")
        except Exception as e:
            print(f"Error during user registration: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return idm_service_pb2.RegisterResponse(success=False, message="Failed to register user")
     
def initialize_admin():
    try:
        db = SessionLocal()
        existing_admin = db.query(User).filter(User.role == UserRole.admin).first()
        if existing_admin:
            print("Admin user already exists. Skipping initialization.")
            return
        
        admin_email = os.getenv("ADMIN_EMAIL")
        admin_password = os.getenv("ADMIN_PASSWORD")

        hashed_password = hashpassword(admin_password)

        admin_user = User(email = admin_email, password = hashed_password, role = UserRole.admin)

        db.add(admin_user)
        db.commit()

        print(f"Admin initialized successfully with email: {admin_email}")
        
    except Exception as e:
        print(f"Error during admin initialization: {str(e)}")
    finally:
        db.close()
            
            
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    idm_service_pb2_grpc.add_IDMServiceServicer_to_server(IDMServiceServices(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    print("Initializing admin user...")
    initialize_admin()
    print("Starting gRPC server...")
    serve()