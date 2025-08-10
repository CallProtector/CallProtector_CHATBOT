# 일반 채팅, 상담별 채팅 AI 서버를 하나로 통합하기 위한 파일 


from fastapi import FastAPI
from rag_chatbotstream import router as rag_chatbotstream
from rag_analyzeCallsession import router as analyze_router  # 경로에 맞게 수정
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🚀 라우터 등록
app.include_router(rag_chatbotstream)
app.include_router(analyze_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Spring에서 서빙되는 HTML 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)