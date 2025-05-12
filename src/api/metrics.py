from prometheus_client import Counter, Histogram

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