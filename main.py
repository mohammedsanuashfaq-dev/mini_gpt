from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Fake database
users = []

# Data model
class User(BaseModel):
    name: str
    age: int


# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to FastAPI!"}


# Create User
@app.post("/users")
def create_user(user: User):
    users.append(user)
    return {
        "message": "User created successfully",
        "user": user
    }


# Get All Users
@app.get("/users")
def get_users():
    return users


# Get Single User
@app.get("/users/{user_id}")
def get_user(user_id: int):
    if user_id < 0 or user_id >= len(users):
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return users[user_id]


# Update User
@app.put("/users/{user_id}")
def update_user(user_id: int, user: User):
    if user_id < 0 or user_id >= len(users):
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    users[user_id] = user

    return {
        "message": "User updated successfully",
        "user": user
    }
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    if user_id < 0 or user_id >= len(users):
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    deleted_user = users.pop(user_id)

    return {
        "message": "User deleted successfully",
        "deleted_user": deleted_user
    }

@app.get("/users/search")
def search_users(name:str, age: int):
    result = []

    for user in users:
        if user["name"] == name and user["age"] == age:
            result.append(user)

    return result