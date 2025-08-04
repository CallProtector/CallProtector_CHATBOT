import os
import json
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
index = pc.Index("legal-guideline")

# ✅ 4. 요청 모델 정의
class Query(BaseModel):
    session_id: int
    question: str

# ✅ 5. 유사 문단 검색 (본문+메타데이터 포함)
def retrieve_context(query: str, top_k: int = 5):
    embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    context_blocks = []
    source_pages = []
    for match in results["matches"]:
        meta = match["metadata"]
        context_blocks.append(
            f"📌 유형: {meta.get('유형', '없음')}\n"
            f"📖 본문: {meta.get('본문', '')}\n"
            f"⚖ 관련 법률: {meta.get('관련 법률', '없음')}\n"
            f"📝 요약: {meta.get('요약', '')}\n"
        )
        source_pages.append({
            "유형": meta.get("유형", "없음"),
            "관련법률": meta.get("관련 법률", "없음")
        })

    return "\n---\n".join(context_blocks), source_pages

# ✅ 6. GPT 스트리밍 + JSON 응답
@app.post("/stream")
async def stream_chat(query: Query):
    context, source_pages = retrieve_context(query.question)
    source_hint = json.dumps(source_pages, ensure_ascii=False, indent=2)

    prompt = f"""
너는 악성민원 대응 및 관련 법률 상담을 도와주는 AI야.
아래 참고 자료를 바탕으로 사용자의 질문에 자연스러운 문장으로 답변하고, 반드시 코드 블록 없이 JSON만 출력해.
문장에는 자연스러운 한글 띄어쓰기를 유지해.

출력 형식 예시:
{{
  "answer": "자연스러운 한글 답변 문장",
  "sourcePages": [
    {{
      "유형": "유형",
      "관련법률": "관련 법률"
    }}
  ]
}}

### 참고 자료:
{context}

### 질문:
{query.question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 악성민원 대응 가이드 및 관련 법률 문서를 기반으로 상담하는 전문가 AI야. 반드시 JSON 형식으로만 출력하고 추가 설명은 하지 마."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    async def event_generator():
        full_response = ""
        # 🔹 스트리밍: 실시간 UI 표시
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield f"data: {delta}\n\n"  # UI 표시용 (실시간)
        
        # 🔹 마지막에만 완성된 JSON 별도 전송 (파싱 용)
        yield f"data: [JSON]{full_response}\n\n"
        yield "data: [END]\n\n"

    return EventSourceResponse(event_generator())