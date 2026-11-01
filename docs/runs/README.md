# 저장된 모델 실행 기록

평가에 사용한 실제 모델 응답의 공개 사본입니다. 재채점은 이 저장된 응답을 다시 평가할 뿐 현재 모델을 새로 검증하지 않습니다.

| 폴더 | 내용 | `max_tokens` |
|---|---|---|
| `step5_original/` | 샘플 5개의 최초 실행. `top_03`은 `finish_reason: length`로 실패했습니다. | 6000 |
| `step5_top_03_followup/` | `top_03`만 다시 실행한 후속 기록 | 12000 |

두 기록은 설정이 달라 하나의 점수로 합치지 않습니다.

## 가린 값

- `exchanges[].response.id`의 OpenRouter 생성 ID 12개를 `[GENERATION_ID_REDACTED]`로 바꿨습니다.
- API 키, 인증 헤더, 실제 `user_id`는 기록에 없었습니다.

그 밖의 모델 응답, 오류, 설정, 토큰 사용량, 도구 호출은 그대로입니다.
