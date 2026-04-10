from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).parent

app.mount(
    "/outputs",
    StaticFiles(directory=BASE_DIR / "data" / "outputs"),
    name="outputs"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)