from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import predict, train

app = FastAPI(
    title="Finance Transaction Categorization ML Service",
    description="ML service for categorizing financial transactions",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1/predict", tags=["predict"])
app.include_router(train.router, prefix="/api/v1/train", tags=["train"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"} 