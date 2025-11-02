import os, json, logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("roots-api")

client = OpenAI()  # needs OPENAI_API_KEY in env

class GenerateRequest(BaseModel):
    data_json: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    save_files: bool = True
    output_dir: str = "posts"

class GenerateResponse(BaseModel):
    updated_items: List[Dict[str, Any]]
    saved_files: List[str] = []

app = FastAPI(title="Roots Content Pipeline API", version="1.0.0")

def deep_research(row: Dict[str, Any]) -> str:
    title = row.get("title") or "TODO"
    prompt = f"""You are a Deep Research Agent for Roots. Collect neutral, verifiable facts.
If unknown, write TODO. Topic: {title}
Return a thorough brief with sections and citations (names + URLs)."""
    resp = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        tools=[{"type": "web_search"}],
        input=prompt,
    )
    return resp.output_text

def write_blog(title: str, research: str) -> str:
    prompt = f"""You are a Professional Blog Writing Agent for Roots...
Title: {title}

Use this research:
{research}
"""
    resp = client.responses.create(model="gpt-5-mini-2025-08-07", input=prompt)
    return resp.output_text

def editing(draft: str) -> str:
    prompt = f"""You are a Professional Editor for Roots. Return only edited markdown.

{draft}"""
    resp = client.responses.create(model="gpt-5-mini-2025-08-07", input=prompt)
    return resp.output_text

def save_post(title: str, content: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
    path = os.path.join(out_dir, f"{safe or 'post'}.mdx")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def load_items(req: GenerateRequest) -> List[Dict[str, Any]]:
    if req.data is not None:
        if not isinstance(req.data, list):
            raise HTTPException(400, "`data` must be a list of objects.")
        return req.data
    if req.data_json is not None:
        try:
            parsed = json.loads(req.data_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON in `data_json`: {e}")
        if not isinstance(parsed, list):
            raise HTTPException(400, "`data_json` must decode to a list of objects.")
        return parsed
    raise HTTPException(400, "Provide `data` (parsed) or `data_json` (string).")

@app.post("/generate-blogs")
def generate_blogs(req: GenerateRequest):
    items = load_items(req)
    saved_files: List[str] = []
    updated: List[Dict[str, Any]] = []

    for row in items:
        if str(row.get("status", "")).lower() != "idea":
            updated.append(row)
            continue

        title = row.get("title") or f"Post {row.get('row_number', '')}"
        try:
            research = deep_research(row)
            draft = write_blog(title, research)
            post = editing(draft)

            row.update({"research": research, "draft": draft, "post": post, "status": "completed"})
            updated.append(row)
        except Exception as e:
            logger.exception("Failed: %s", title)
            row.update({"status": "error", "error": str(e)})
            updated.append(row)

    # Return plain JSON mapping title -> blog content
    blog_map = {row["title"]: row["post"] for row in updated if row.get("status") == "completed"}
    return blog_map

@app.get("/healthz")
def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("main:app", host="0.0.0.0", port=int("10000"), reload=True)
