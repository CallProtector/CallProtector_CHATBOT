import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ★ 오레곤(us-west-2)
index_name = os.getenv("PINECONE_INDEX", "legal-guideline-usw2")
cloud = "aws"
region = "us-west-2"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# 인덱스 생성 여부 판단
existing = pc.list_indexes().names()
created_now = False
if index_name not in existing:
    print(f"📌 인덱스 '{index_name}' 없음 → {cloud}/{region} 에 새로 생성합니다.")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )
    created_now = True

index = pc.Index(index_name)

# documents.json 로드
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
print(f"📂 총 {len(documents)}개의 문서를 업로드합니다...")

# ★  초기화 로직: 새로 만든 인덱스면 스킵, 기존이면 존재하는 네임스페이스만 삭제
if not created_now:
    try:
        stats = index.describe_index_stats()
        namespaces = list((stats or {}).get("namespaces", {}).keys())
    except Exception:
        namespaces = []

    if namespaces:
        clear_index = input(
            f"인덱스 '{index_name}'의 {len(namespaces)}개 네임스페이스를 초기화할까요? (y/n): "
        ).strip().lower()
        if clear_index == "y":
            for ns in namespaces:
                try:
                    index.delete(delete_all=True, namespace=ns)
                    print(f"🗑️ 네임스페이스 '{ns}' 삭제 완료")
                except NotFoundException:
                    pass
    else:
        print("ℹ️ 삭제할 네임스페이스가 없습니다(빈 인덱스). 초기화 스킵.")

# 업로드
for i, doc in enumerate(documents, start=1):
    emb = client.embeddings.create(model=EMBED_MODEL, input=doc["text"]).data[0].embedding
    index.upsert([{
        "id": doc["id"],
        "values": emb,
        "metadata": {
            "본문": doc["text"],
            "유형": doc["metadata"].get("유형", ""),
            "관련 법률": doc["metadata"].get("관련 법률", ""),
            "주요 키워드": doc["metadata"].get("주요 키워드", []),
            "요약": doc["metadata"].get("요약", "")
        }
    }])
    print(f"✅ [{i}/{len(documents)}] {doc['id']} 업로드 완료")

print("🎉 오레곤(us-west-2) 인덱스 업로드 완료.")
