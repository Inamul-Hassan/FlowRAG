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