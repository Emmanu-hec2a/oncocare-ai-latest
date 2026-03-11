import uvicorn
import os

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
