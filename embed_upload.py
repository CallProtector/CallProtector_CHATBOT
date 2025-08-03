import os
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from uuid import uuid4
from openai import OpenAI

# 1. 환경변수 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Pinecone 인스턴스 생성
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 3. 인덱스 정보 설정
index_name = "legal-guideline"
cloud = "aws"
region = os.getenv("PINECONE_ENV")  # ex: us-east-1

# 4. 인덱스 없으면 생성
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

# 5. 인덱스 연결
index = pc.Index(index_name)

# 6. 업로드할 문단들 (page_id와 함께)
documents = [
    {"id": "page_1", "text": "악성민원은 고객응대근로자에게 피해를 초래하는 부적절한 언행 또는 민원내용을 말한다."},
    {"id": "page_2", "text": "성희롱, 욕설, 협박 등은 선 종료 기준에 따라 ARS 경고멘트를 송출하고 통화를 종료한다."},
    {"id": "page_3", "text": "과도한 보상요구는 심의 절차를 거쳐 처리하며 상담사는 즉답하지 않는다."},
    {"id": "page_4", "text": "악·강성민원은 민원인의 부적절한 행동(예: 성희롱, 욕설, 협박, 고성 등)과 과도한 민원 내용 및 요청 방식(예: 무리한 요구, 민원 반복 등)으로 인해 고객응대근로자와 다른 민원인에게 피해와 불편을 초래하는 민원을 말한다."},
    {"id": "page_5", "text": "악·강성민원은 불쾌·불안한 감정을 유발하거나 민원처리를 방해하고, 민원서비스를 오·남용하는 행태를 포함한다."},
    {"id": "page_6", "text": "악·강성민원은 17가지 유형으로 분류되며, 대표적으로 성희롱, 욕설, 협박, 모욕, 고성, 억지주장, 장난전화, 반복민원 등이 있다."},
    {"id": "page_7", "text": "성희롱, 욕설, 협박 등의 위법 행위는 ARS 경고멘트를 송출한 후 즉시 통화를 종료하고, 관리 담당자가 전담 관리 여부를 판단한다."},
    {"id": "page_8", "text": "과도한 보상요구의 경우, 상담사는 즉답하지 않고 소속 부서장에게 보고하여 보상심의 절차에 따라 처리한다."},
    {"id": "page_9", "text": "억지주장, 하소연, 무리한 요구 등은 2회 상담 곤란 안내 후 부서장 보고 및 3회차 발생 시 통화 종료 후 전담관리 검토에 들어간다."},
    {"id": "page_10", "text": "상습 강요(콜백 요청, 특정상담사 요구 등)는 사유 확인에 불응할 경우 ARS 종료멘트를 통해 통화를 종료하며, 반복 시 차단 대상이 될 수 있다."},
    {"id": "page_11", "text": "장난 또는 거짓민원은 허위신고로 판단될 경우 통화를 종료하고, 관련 법률(경범죄처벌법 등)에 따라 법적 책임이 따를 수 있음을 안내한다."},
    {"id": "page_12", "text": "반복 민원은 동일 민원을 정당한 사유 없이 3회 이상 반복 시 ‘민원처리법 제23조’에 따라 종결 처리할 수 있다."},
    {"id": "page_13", "text": "장시간 통화는 다른 시민의 상담 기회를 침해하므로 1회 통화시간 제한을 안내하고, 제한 시간 경과 시 통화를 종료한다."},
    {"id": "page_14", "text": "민원인의 성희롱 발언은 성적 수치심을 유발할 수 있으며, 이는 악성민원으로 분류된다."},
    {"id": "page_15", "text": "욕설, 폭언, 협박은 반복 시 선 종료 대상이며, 경찰 민원 콜센터는 3개월 차단 조치를 적용할 수 있다."},
    {"id": "page_16", "text": "고객의 고성, 반말, 짜증은 상담사의 업무에 지장을 주며 강성민원으로 대응된다."},
    {"id": "page_17", "text": "말꼬리 잡기나 트집은 정상적인 상담 진행을 방해하는 행위로 간주된다."},
    {"id": "page_18", "text": "개인정보 제공 거부는 민원 접수에 지장을 주며, 최소한의 정보 제공이 필요함을 안내한다."},
    {"id": "page_19", "text": "민원요지가 불명확할 경우, 주취 또는 소통불가 여부를 판단하여 종료할 수 있다."},
    {"id": "page_20", "text": "무리한 보상요구는 접수 단계에서 사실 관계 확인 후 내부 절차에 따라 판단한다."},
    {"id": "page_21", "text": "민원인이 타 기관 상담사 연결을 요구하는 경우, 정해진 업무 범위 외 사안으로 종결 가능하다."},
    {"id": "page_22", "text": "장난전화는 사전 고지 후 종료되며, 반복 시 법적 대응이 가능하다."},
    {"id": "page_23", "text": "동일 민원의 반복 제기는 민원처리법에 따라 내부 종결이 가능하다."},
    {"id": "page_24", "text": "민원인의 반복된 콜백 요구는 '상습 강요'에 해당하며, 상담 업무 범위를 벗어난 요구로 차단 조치될 수 있다."},
    {"id": "page_25", "text": "성희롱은 '성폭력범죄의 처벌 등에 관한 특례법 제13조'에 따라 2년 이하 징역 또는 2천만원 이하 벌금에 처해질 수 있다."},
    {"id": "page_26", "text": "위협, 협박은 '정보통신망법 제44조의7 제1항 제3호'에 따라 1년 이하 징역 또는 1천만원 이하 벌금 대상이다."},
    {"id": "page_27", "text": "모욕, 고성 등의 발언이 지속되면 상담사는 ARS 경고 후 통화를 종료할 수 있다."},
    {"id": "page_28", "text": "상담사의 안내를 지속적으로 방해하거나 트집 잡을 경우 '업무방해'로 간주된다."},
    {"id": "page_29", "text": "보상 심의는 소속 부서장이 주관하며, 결과에 따라 ARS 안내 후 종결된다."},
    {"id": "page_30", "text": "상담사는 반복되는 억지주장에 대해 통화 종료 후 악성민원으로 등록할 수 있다."},
    {"id": "page_31", "text": "정당한 사유 없는 개인정보 제공 거부는 민원 접수를 제한하는 사유가 된다."},
    {"id": "page_32", "text": "업무 범위를 초과하는 요구는 민원 처리 규정을 벗어나며, 안내 후 종료될 수 있다."},
    {"id": "page_33", "text": "상담 업무를 방해하는 고압적 태도는 강성민원으로 분류된다."}
]


# 7. OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 8. 임베딩 및 업로드
batch = []
for doc in documents:
    # OpenAI Embedding API 사용
    vector = client.embeddings.create(
        input=[doc["text"]],
        model="text-embedding-ada-002"
    ).data[0].embedding

    metadata = {
        "text": doc["text"],
        "source": "dasan-guide",
        "page": str(doc["id"])  # 문자열로 저장
    }

    batch.append((str(uuid4()), vector, metadata))

index.upsert(vectors=batch)
print(f"✅ 업로드 완료! 총 문단 수: {len(batch)}")
