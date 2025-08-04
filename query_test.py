import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# 1. 환경 변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-guideline")

# 2. 테스트 쿼리
query = "악성민원은 무엇인가요?"
query_embedding = client.embeddings.create(
    input=query,
    model="text-embedding-ada-002"
).data[0].embedding

# 3. Pinecone 검색
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

# 4. 결과 출력
for match in results["matches"]:
    print(f"📌 점수: {match['score']:.4f}")
    print(f"본문: {match['metadata']['본문']}")
    print(f"요약: {match['metadata']['요약']}\n")
