# rag_chatbot.py
# [일반] Springboot <-> chatbot 용 (속도 9초)
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime
import requests
from pydantic import BaseModel

# 1. 환경변수 불러오기
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-guideline")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SPRINGBOOT_LOG_URL = os.getenv("SPRINGBOOT_LOG_URL")  # 예: https://your-backend.com/api/chat-log

# 2. FastAPI 앱 생성
app = FastAPI()

# 3. 요청/응답 스키마 정의
class Query(BaseModel):
    session_id: int 
    question: str

class Answer(BaseModel):
    answer: str
    source_pages: list[str]

# 4. 유사 문단 검색
def retrieve_context(query, top_k=3):
    embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    contexts = [match["metadata"]["text"] for match in results["matches"]]
    pages = [str(match["metadata"].get("page", "unknown")) for match in results["matches"]]
    return "\n".join(contexts), pages

# 5. GPT 응답 생성
def generate_answer(question, context):
    prompt = f"""너는 악성민원 대응 가이드 문서를 기반으로 법률 상담을 도와주는 역할이야.

다음은 문서에서 발췌한 참고 내용이야:

{context}

위 내용을 참고해서 사용자의 질문에 정확하고 간결하게 답변해줘:
"{question}"
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini", # 더 빠름
        messages=[
            {"role": "system", "content": "너는 법률 문서를 기반으로 상담하는 도우미야."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# 6. Spring Boot 로그 전송
def log_to_springboot(session_id: int, question: str, answer: str, pages: list[str]):
    payload = {
        "sessionId": session_id,
        "question": question,
        "answer": answer,
        "sourcePages": pages,
        # "timestamp": datetime.utcnow().isoformat() <- springboot 의 created_at이 있어서 불필요
    }
    try:
        res = requests.post(SPRINGBOOT_LOG_URL, json=payload, timeout=5)
        res.raise_for_status()
    except Exception as e:
        print("⚠️ Spring Boot 로그 전송 실패:", e)

# 7. API 엔드포인트 정의
@app.post("/ask", response_model=Answer)
async def ask(query: Query):
    context, pages = retrieve_context(query.question)
    answer = generate_answer(query.question, context)
    return {"answer": answer, "source_pages": pages}
