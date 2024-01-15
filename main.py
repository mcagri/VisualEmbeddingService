import json

from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
from Model import ModelHandler
from typing import Annotated
from pydantic import BaseModel
import uvicorn

app = FastAPI()


@app.post("/embedimage/")
async def get_image_embedding(file: Annotated[bytes, File()]):
    try:
        output = {"embedding": ModelHandler.get_embedding(file)}
        return output
    except Exception as e:
        return {"message": "invalid request}"}


if __name__ == '__main__':
    uvicorn.run(app, port=15002, host="0.0.0.0")
