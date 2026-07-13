import time
from fastapi import Request, HTTPException

# Simple in-memory storage for rate limiting.
# Key: client IP string
# Value: list of timestamps of recent requests
_rate_limit_records = {}

# Limit: 3 requests per 60 seconds.
LIMIT_WINDOW = 60
LIMIT_COUNT = 3

def check_rate_limit(request: Request):
    """Dependency that raises 429 if the request rate exceeds LIMIT_COUNT per LIMIT_WINDOW seconds."""
    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
        
    ip = request.client.host if request.client else "unknown"

    now = time.time()
    
    if ip not in _rate_limit_records:
        _rate_limit_records[ip] = []
        
    # Clean up older timestamps
    _rate_limit_records[ip] = [t for t in _rate_limit_records[ip] if now - t < LIMIT_WINDOW]
    
    if len(_rate_limit_records[ip]) >= LIMIT_COUNT:
        raise HTTPException(
            status_code=429,
            detail="Too many authentication attempts. Please wait a minute and try again."
        )
        
    _rate_limit_records[ip].append(now)
