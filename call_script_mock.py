# call_script_mock.py

def get_script_by_session_id(session_id: int):
    if session_id == 21:
        return [
            {"speaker": "INBOUND", "text": "이거 반품 안된 거 같은데요."},
            {"speaker": "OUTBOUND", "text": "확인해드리겠습니다. 주문번호 알려주시겠어요?"},
            {"speaker": "INBOUND", "text": "123456입니다. 환불 빨리 좀 처리해주세요."}
        ]
    else:
        return []
