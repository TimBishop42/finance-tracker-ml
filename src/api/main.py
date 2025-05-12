from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, generate_latest
from src.api.routes import router as api_router
from src.config import settings

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create metrics
PREDICTION_COUNTER = Counter(
    'prediction_total',
    'Total number of predictions made',
    ['endpoint']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing predictions',
    ['endpoint']
)
TRAINING_COUNTER = Counter(
    'training_total',
    'Total number of training operations',
    ['status']
)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    """Health check endpoint with rate limiting."""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain") 