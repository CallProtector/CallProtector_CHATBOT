from fastapi import FastAPI
from rag_chatbotstream import router as chat_router
from rag_analyzeCallsession import router as analyze_router
from rag_callchatbotstream import router as callchat_router

app = FastAPI()

app.include_router(chat_router,   prefix="/ai/chat")
app.include_router(analyze_router, prefix="/ai/callsession")
app.include_router(callchat_router, prefix="/ai/callchat")

