from fastapi import FastAPI, HTTPException, Request
import logging, os, uvicorn, time
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from Bots.Data_injestion import Agent_Result
from typing import Optional, List, Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("Agentic AI pipeline")

#Middleware
class SuccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        process_time = round(time.time() - start_time, 4)

        if response.status_code == 200:
            response.headers["X-Status"] = "Success"
            response.headers["X-Process-Time"] = str(process_time)

            logger.info(
                f"SUCCESS | {request.method} {request.url.path} | {process_time}s"
            )

        return response

#Fast api
app = FastAPI(title="Agentic AI")
logger.info("Agentic AI Initialised")

#Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


app.add_middleware(SuccessMiddleware)


@app.get("/Get Info")
def root_user():
    return {"messages": "Agentic AI application is healthy"}

#Request/Response model
class Queryrequest(BaseModel):
    question: str = Field(description="Get the user Question")
    session_id: Optional[str] = Field(None, descripyion="Conversation/Session identifier")
    trace_id: bool = Field(default=False, description="Whether to return trust and governance metadata")


class Evidence(BaseModel):
    Type: Literal['Internal', 'Web Source'] = Field(description="Its a Information source")
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    heading: Optional[str] = None
    sub_heading: Optional[str] = None
    snippet: Optional[str] = None


class TrustMetrics(BaseModel):
    confidence: Literal["High", "Medium", "Low"]
    evidence_count: int
    web_fall_back: bool
    memory_used: bool


class response(BaseModel):
    answer: str
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidences")
    trust: TrustMetrics


# Main Chat Endpoint

@app.post("/Chat Discussion", response_model=response)
async def Response(request: Queryrequest):
    try:
        # Call your backend
        answer_text = Agent_Result(request.question)
        if not answer_text:
            answer_text = "No relevant answer found."

        # Suppose backend returns evidence list; if not, use empty
        backend_evidence_list = []  # Replace with actual evidence from backend if available

        # Safely convert to Pydantic Evidence objects
        evidence = [Evidence(**e) for e in backend_evidence_list if e]

        return response(
            answer=answer_text,
            evidence=evidence,  # use the safe list here
            trust=TrustMetrics(
                confidence="Medium",  # must match Literal
                evidence_count=len(evidence),
                web_fall_back=False,
                memory_used=True
            )
        )

    except Exception as e:
        # Always return valid Pydantic model even if error occurs
        return response(
            answer=f"Error: {str(e)}",
            evidence=[],
            trust=TrustMetrics(
                confidence="Medium",
                evidence_count=0,
                web_fall_back=False,
                memory_used=True
            )
        )



if __name__ == "__main__":
    uvicorn.run("Fastapi:app", host="0.0.0.0", port=8000, reload=True)

#python -m uvicorn Fastapi:app --reload
