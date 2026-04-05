# Self-Healing Text-to-SQL Pipeline

실행 피드백 기반 Text-to-SQL 오류 자율 수정(Self-healing) 파이프라인 설계 및 평가

## 개요

Text-to-SQL 모델(gpt-4o-mini)이 생성한 SQL이 실행 오류나 오답을 반환할 때, 오류 유형을 감지하고 맞춤 피드백으로 최대 3회까지 자율 수정하는 파이프라인입니다.

세 가지 방법을 동일 평가셋에서 비교합니다:
- **No-Repair Baseline**: 초기 SQL 생성 후 수정 없이 채점
- **Naive Self-Correction**: 에러 메시지만 피드백으로 제공
- **Targeted Self-Healing**: Syntax/Semantic 분류 기반 동적 피드백 (스키마 링킹 + 데이터 샘플 주입 + CoT)

## 프로젝트 구조

```
├── data/
│   ├── mini_dev_sqlite.json        # BIRD Mini-Dev 500문항 평가셋
│   └── dev_databases/              # SQLite DB 파일 (별도 다운로드)
├── results/                        # 실험 결과 JSON
├── baseline.py                     # 베이스라인 SQL 생성 + 채점
├── naive_self_healing.py           # naive 단순 피드백 자율 수정
├── self_healing.py                 # 동적 멀티턴 자율 수정 (최대 3회)
└── scorer.py                       # 범용 채점 스크립트
```

## 환경 설정

```bash
pip install openai python-dotenv pandas tqdm tabulate

echo "OPENAI_API_KEY=sk-..." > .env
```

`dev_databases/`는 [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev)에서 다운로드하여 `data/` 아래에 배치합니다.

## Dataset

본 프로젝트는 [BIRD-SQL Mini-Dev](https://github.com/bird-bench/mini_dev) 데이터셋을 사용합니다.

- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- 500문항 (simple 148 / moderate 250 / challenging 102)
- 11개 DB