import time
# A decorator to log the time taken by a function to execute
def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__} to execute: {end-start} seconds")
        return result
    return wrapper

# Logging input and output
# from traceloop.sdk import Traceloop

# Traceloop.init(disable_batch=True, api_key="216b6cac8cb6b080938422349f1712888bb60b74fc04d0bf92264b3946a744cddda664dd2907c726c1393e6905348eb9")