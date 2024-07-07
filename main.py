from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
classifier = pipeline("zero-shot-classification")

labels = ["play music","joke","clean the room","dance","hug"]

class RequestModel(BaseModel):
    input: str
    actions: list

@app.post("/decide")
def get_response(request: RequestModel):
    prompt = request.input
    actions = request.actions
    result = classifier(prompt, actions)
    return result

    # print(f"Text: {prompt}")
    # print("Predicted Labels:")
    # for label, score in zip(result["labels"], result["scores"]):
    #     print(f"- {label}: {score}")
    # print(type(result))
    # return result
