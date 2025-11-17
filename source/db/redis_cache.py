# backend/db/redis_cache.py

import os
import json
from settings import settings
from typing import Optional

try:
    import redis
except Exception as e:
    redis = None

REDIS_URL = settings.REDIS_URL or (
    f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
)

_redis_client = None


def get_redis_client():
    global _redis_client, redis
    if _redis_client is None:
        if redis is None:
            raise RuntimeError("redis-py not installed. Install: pip install redis")
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# status helpers
def set_job_status(job_id: str, status_payload: dict, expire_seconds: Optional[int] = 3600):
    """
    Store job status JSON under key: job:{job_id}
    """
    r = get_redis_client()
    key = f"job:{job_id}"
    r.set(key, json.dumps(status_payload))
    if expire_seconds:
        r.expire(key, expire_seconds)


def get_job_status(job_id: str) -> Optional[dict]:
    r = get_redis_client()
    key = f"job:{job_id}"
    val = r.get(key)
    if not val:
        return None
    return json.loads(val)


# chat cache helpers (store chat session results or recent messages)
def cache_chat(session_id: str, payload: dict, expire_seconds: Optional[int] = 3600):
    r = get_redis_client()
    key = f"chat:{session_id}"
    r.set(key, json.dumps(payload))
    if expire_seconds:
        r.expire(key, expire_seconds)


def get_chat(session_id: str) -> Optional[dict]:
    r = get_redis_client()
    key = f"chat:{session_id}"
    val = r.get(key)
    if not val:
        return None
    return json.loads(val)
