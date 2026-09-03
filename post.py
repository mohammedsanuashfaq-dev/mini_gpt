from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel

class User(BaseModel):
    name: str

@app.post("/welcome")
def welcome(user: User):
    return {
        "message": f"Welcome {user.name}!"
    }