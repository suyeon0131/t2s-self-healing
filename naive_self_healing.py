import json
import sqlite3
import pandas as pd
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ 에러: OPENAI_API_KEY를 찾을 수 없습니다.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# 경로 설정
DB_BASE_PATH = "./data/dev_databases"
INPUT_PATH = "./results/failure_corpus_official.json"
OUTPUT_PATH = "./results/naive_fixed_corpus.json"


def call_llm_to_fix_query_naive(question, pred_sql, feedback_prompt):
    """GPT-4o-mini를 호출하여 '단순 피드백'만으로 쿼리를 수정하게 합니다.
    스키마, 데이터 샘플, CoT 등 추가 정보 일절 제공하지 않음."""

    system_message = "You are a world-class SQLite database expert. Your task is to perfectly fix the incorrect SQL query based on the provided feedback."

    user_message = f"""
[User Question]
{question}

[Your Previous Incorrect SQL]
{pred_sql}

[Feedback & Clues]
{feedback_prompt}

Carefully review the feedback above. Return ONLY the corrected SQLite query within a markdown block (```sql ... ```). Do not include any other explanations.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0
    )

    fixed_sql = response.choices[0].message.content
    fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
    return fixed_sql


def execute_sql(db_id, sql_query):
    """SQL을 실제 DB에서 실행합니다."""
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return True, "Success", results
    except Exception as e:
        return False, str(e), None


def compare_results(gold_res, pred_res):
    """BIRD 공식 EX: set 비교"""
    if gold_res is None or pred_res is None:
        return False
    return set(pred_res) == set(gold_res)


def main():
    print("🚀 Naive 자율 수정(대조군) 파이프라인 가동을 시작합니다... (3턴)")

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        failed_samples = json.load(f)

    # ── 이어서 실행 (resume) 지원 ──
    fixed_results = []
    start_idx = 0

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            fixed_results = json.load(f)
        start_idx = len(fixed_results)
        if start_idx >= len(failed_samples):
            print(f"⏭️  이미 완료됨 ({start_idx}건). 건너뜁니다.")
            return
        print(f"📂 기존 결과 {start_idx}건 로드. {start_idx}번부터 이어서 실행합니다.")

    print(f"📋 실패 코퍼스 {len(failed_samples)}건 중 {start_idx}번부터 교정 시작\n")

    for idx in tqdm(range(start_idx, len(failed_samples))):
        sample = failed_samples[idx]
        db_id = sample['db_id']
        question = sample['question']
        gold_sql = sample['gold_sql']

        current_sql = sample['pred_sql']
        current_exec_ok = sample['error_type'] != "Syntax Error"
        current_exec_msg = sample['error_message']

        # 정답 데이터
        gold_ok, _, gold_res = execute_sql(db_id, gold_sql)

        turn = 0
        max_turns = 3
        is_healed = False
        turn_log = []

        while turn < max_turns and not is_healed:

            # 🎯 Naive 피드백: 추가 정보 일절 없음
            if not current_exec_ok:
                feedback_prompt = f"A Syntax Error occurred. Execution failed with the following error:\n{current_exec_msg}"
            else:
                feedback_prompt = "The query executed without syntax errors, but the returned results were incorrect (Result Mismatch). Please rewrite the query to fix the logic."

            try:
                fixed_sql = call_llm_to_fix_query_naive(question, current_sql, feedback_prompt)
            except Exception as e:
                fixed_sql = current_sql

            # 실행 및 채점
            pred_ok, pred_msg, pred_res = execute_sql(db_id, fixed_sql)

            turn_entry = {
                "turn": turn + 1,
                "error_type_at_entry": "Syntax Error" if not current_exec_ok else "Semantic Error",
                "input_sql": current_sql,
                "output_sql": fixed_sql,
                "exec_success": pred_ok,
                "exec_message": pred_msg
            }

            if not pred_ok:
                current_exec_ok = False
                current_exec_msg = pred_msg
                current_sql = fixed_sql
                turn_entry["result"] = "syntax_error"
            else:
                if compare_results(gold_res, pred_res):
                    is_healed = True
                    current_sql = fixed_sql
                    turn_entry["result"] = "healed"
                else:
                    current_exec_ok = True
                    current_exec_msg = "Result Mismatch"
                    current_sql = fixed_sql
                    turn_entry["result"] = "semantic_error"

            turn_log.append(turn_entry)
            turn += 1

        result = {**sample}
        result['fixed_sql'] = current_sql
        result['turn_used'] = turn
        result['is_healed'] = is_healed
        result['turn_log'] = turn_log
        fixed_results.append(result)

        # 중간 저장 (50건마다)
        if (idx + 1) % 50 == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(fixed_results, f, ensure_ascii=False, indent=4)
            healed_so_far = sum(1 for r in fixed_results if r.get('is_healed'))
            print(f"\n  💾 중간 저장 ({len(fixed_results)}건, 치유율: {healed_so_far}/{len(fixed_results)})")

    # 최종 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(fixed_results, f, ensure_ascii=False, indent=4)

    # ── 최종 통계 ──
    healed_count = sum(1 for r in fixed_results if r.get('is_healed'))
    total = len(fixed_results)

    healed_by_turn = {}
    for r in fixed_results:
        if r.get('is_healed'):
            t = r.get('turn_used', 0)
            healed_by_turn[t] = healed_by_turn.get(t, 0) + 1

    # 악화 분석
    worsened = sum(1 for r in fixed_results
                   if r.get('error_type') == 'Semantic Error'
                   and r.get('turn_log')
                   and not r['turn_log'][-1].get('exec_success'))

    print("\n" + "=" * 50)
    print("📊 [Naive Self-correction 최종 리포트 (3턴)]")
    print(f"실패 코퍼스: {total}건")
    print(f"자율 교정 성공: {healed_count}건 (교정률: {healed_count/total*100:.1f}%)")
    print(f"교정 실패: {total - healed_count}건")
    print(f"턴별 치유 분포: {dict(sorted(healed_by_turn.items()))}")
    print(f"Semantic→Syntax 악화: {worsened}건")

    for diff in ['simple', 'moderate', 'challenging']:
        diff_total = sum(1 for r in fixed_results if r.get('difficulty') == diff)
        diff_healed = sum(1 for r in fixed_results if r.get('difficulty') == diff and r.get('is_healed'))
        if diff_total > 0:
            print(f"  {diff}: {diff_healed}/{diff_total} ({diff_healed/diff_total*100:.1f}%)")

    print("=" * 50)
    print(f"\n✅ 결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()