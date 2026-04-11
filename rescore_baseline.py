"""
rescore_baseline.py - 공식 EX 기준(set 비교)으로 baseline 재채점 + 실패 코퍼스 재추출

기존 baseline_results.json의 pred_sql을 그대로 사용하되,
채점 로직만 BIRD 공식 기준(set(gold) == set(pred))으로 변경

사용법:
  python rescore_baseline.py
"""

import json
import sqlite3
import os
from tqdm import tqdm

DB_DIR = "./data/dev_databases/"
INPUT_PATH = "./results/baseline_results.json"
OUTPUT_BASELINE = "./results/baseline_results_official.json"
OUTPUT_FAILURES = "./results/failure_corpus_official.json"


def execute_sql_raw(db_id, sql_query):
    """SQL을 실행하고 cursor.fetchall() 결과를 반환 (공식 스크립트와 동일한 방식)"""
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return True, "Success", results
    except Exception as e:
        return False, str(e), None


def compare_results_official(gold_res, pred_res):
    """BIRD 공식 EX: set(predicted_res) == set(ground_truth_res)"""
    if gold_res is None or pred_res is None:
        return False
    return set(pred_res) == set(gold_res)


def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    print(f"📂 기존 baseline 로드: {len(baseline)}건")

    rescored = []
    success_count = 0

    for item in tqdm(baseline, desc="재채점 중"):
        db_id = item['db_id']
        gold_sql = item['gold_sql']
        pred_sql = item['pred_sql']

        gold_ok, _, gold_res = execute_sql_raw(db_id, gold_sql)
        pred_ok, pred_msg, pred_res = execute_sql_raw(db_id, pred_sql)

        if not pred_ok:
            ex_pass = False
            error_type = "Syntax Error"
        elif compare_results_official(gold_res, pred_res):
            ex_pass = True
            error_type = None
            success_count += 1
        else:
            ex_pass = False
            error_type = "Semantic Error"

        result = {
            "question_id": item.get("question_id"),
            "db_id": db_id,
            "question": item["question"],
            "evidence": item.get("evidence", ""),
            "difficulty": item.get("difficulty", ""),
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "ex_pass": ex_pass,
            "error_type": error_type,
            "error_message": pred_msg if error_type == "Syntax Error" else (
                "Result Mismatch (EX Fail)" if error_type else None
            )
        }
        rescored.append(result)

    # 저장
    with open(OUTPUT_BASELINE, 'w', encoding='utf-8') as f:
        json.dump(rescored, f, indent=4, ensure_ascii=False)

    failures = [r for r in rescored if not r['ex_pass']]
    with open(OUTPUT_FAILURES, 'w', encoding='utf-8') as f:
        json.dump(failures, f, indent=4, ensure_ascii=False)

    # 통계
    total = len(rescored)
    syntax_err = sum(1 for r in rescored if r.get('error_type') == 'Syntax Error')
    semantic_err = sum(1 for r in rescored if r.get('error_type') == 'Semantic Error')

    print("\n" + "=" * 50)
    print("📊 [공식 EX 기준 재채점 결과]")
    print(f"전체: {total}건")
    print(f"EX Pass: {success_count}건 ({success_count/total*100:.1f}%)")
    print(f"EX Fail: {len(failures)}건")
    print(f"  - Syntax Error: {syntax_err}건")
    print(f"  - Semantic Error: {semantic_err}건")

    for diff in ['simple', 'moderate', 'challenging']:
        dt = sum(1 for r in rescored if r.get('difficulty') == diff)
        dp = sum(1 for r in rescored if r.get('difficulty') == diff and r.get('ex_pass'))
        if dt > 0:
            print(f"  {diff}: {dp}/{dt} ({dp/dt*100:.1f}%)")

    print("=" * 50)
    print(f"\n✅ 재채점 결과: {OUTPUT_BASELINE}")
    print(f"✅ 실패 코퍼스 ({len(failures)}건): {OUTPUT_FAILURES}")

    # 기존 대비 변화
    old_pass = sum(1 for r in baseline if r.get('ex_pass'))
    print(f"\n📌 기존 scorer 기준: {old_pass}/500 ({old_pass/500*100:.1f}%)")
    print(f"📌 공식 EX 기준:    {success_count}/500 ({success_count/500*100:.1f}%)")
    print(f"📌 차이: +{success_count - old_pass}건 (행 순서 무시로 추가 통과)")


if __name__ == "__main__":
    main()