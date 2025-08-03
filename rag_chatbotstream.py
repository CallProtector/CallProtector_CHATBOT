# [SSE] 스트리밍 방식 Springboot <-> chatbot 용 (속도 4.5초)
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone

# ✅ 1. 환경 변수 로드
load_dotenv()

# ✅ 2. FastAPI 앱 초기화
app = FastAPI()

# ✅ 3. OpenAI & Pinecone 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-guideline")  # Pinecone 인덱스 이름

# ✅ 4. 요청 모델 정의
class Query(BaseModel):
    session_id: int
    question: str

# ✅ 5. 유사 문단 검색 함수
def retrieve_context(query: str, top_k: int = 3):
    embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    contexts = [match["metadata"]["text"] for match in results["matches"]]
    return "\n".join(contexts)

# ✅ 6. GPT 스트리밍 + SSE
@app.post("/stream")
async def stream_chat(query: Query):
    # 🔍 Pinecone 검색
    context = retrieve_context(query.question)

    # 💬 프롬프트 생성
    prompt = f"""
    너는 악성민원 대응 가이드 문서를 기반으로 법률 상담을 도와주는 역할이야.
    다음은 문서에서 발췌한 참고 내용이야:
    {context}

    위 내용을 참고해서 사용자의 질문에 정확하고 간결하게 답변해줘:
    "{query.question}"
    """

    # GPT 스트리밍 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 법률 문서를 기반으로 상담하는 도우미야."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    # SSE 이벤트 생성기
    async def event_generator():
        buffer = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                buffer += delta
                if buffer.endswith((" ", "\n")):  # 공백 포함 시 즉시 flush
                    yield f"data: {buffer}\n\n"
                    buffer = ""
        if buffer:  # 남은 텍스트 flush
            yield f"data: {buffer}\n\n"



    return EventSourceResponse(event_generator())
