import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ✅ 1. 환경변수 로드
load_dotenv()

# ✅ 2. OpenAI & Pinecone 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ✅ 3. Pinecone 인덱스 설정
index_name = "legal-guideline"
region = os.getenv("PINECONE_ENV")  # 예: "us-east-1"

if index_name not in pc.list_indexes().names():
    print(f"📌 인덱스 '{index_name}' 없음 → 새로 생성합니다.")
    pc.create_index(
        name=index_name,
        dimension=1536,  # text-embedding-ada-002 차원
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region)
    )

index = pc.Index(index_name)

# ✅ 4. JSON 파일 로드
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"📂 총 {len(documents)}개의 문서를 업로드합니다...")

# ✅ 5. 업로드 전 기존 인덱스 초기화 여부 (선택)
clear_index = input("기존 인덱스를 초기화할까요? (y/n): ").strip().lower()
if clear_index == "y":
    index.delete(delete_all=True)
    print("🗑️ 기존 인덱스를 초기화했습니다.")

# ✅ 6. 문서별 임베딩 생성 및 업로드
for i, doc in enumerate(documents, start=1):
    embedding = client.embeddings.create(
        input=doc["text"],
        model="text-embedding-ada-002"
    ).data[0].embedding

    index.upsert([
        {
            "id": doc["id"],
            "values": embedding,
            "metadata": {
                "본문": doc["text"],
                "유형": doc["metadata"].get("유형", ""),
                "관련 법률": doc["metadata"].get("관련 법률", ""),
                "주요 키워드": doc["metadata"].get("주요 키워드", []),
                "요약": doc["metadata"].get("요약", "")
            }
        }
    ])

    print(f"✅ [{i}/{len(documents)}] {doc['id']} 업로드 완료")

print("🎉 모든 문서 임베딩 및 업로드가 완료되었습니다.")
