from fastapi import FastAPI
from app.api.v1.simplifier_endpoint import router


app = FastAPI()
app.include_router(router, prefix="/api/v1")