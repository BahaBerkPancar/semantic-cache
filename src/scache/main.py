from fastapi import FastAPI

app = FastAPI(title="semantic-cache")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
