import os
import redis

ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"

redis_client = None

if ENABLE_CACHE:
    redis_url = os.getenv("REDIS_URL")

    if redis_url:
        redis_client = redis.from_url(redis_url, decode_responses=True)
    else:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )