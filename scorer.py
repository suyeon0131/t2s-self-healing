"""
scorer.py - 범용 채점기

수정된 SQL이 담긴 JSON 파일을 받아서 gold SQL과 비교 채점합니다.
naive_fixed_corpus.json, multi_turn_healing.json 등 어떤 결과든 채점 가능.

사용법:
  python scorer.py --input ./results/naive_fixed_corpus.json --output ./results/naive_scored.json
  python scorer.py --input ./results/multi_turn_healing.json --output ./results/multi_turn_scored.json
"""

import json
import sqlite3
import pandas as pd
import os
import argparse
from tqdm import tqdm

DB_DIR = "./data/dev_databases/"

def execute_sql(db_id, sql_query):
    """SQL을 실제 DB에서 실행합니다."""
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return True, "Success", df
    except Exception as e:
        return False, str(e), None

def compare_results(gold_df, pred_df):
    """정답 표와 예측 표의 내용이 완전히 일치하는지(EX) 검사합니다."""
    if gold_df is None or pred_df is None:
        return False
    try:
        gold_values = set(tuple(row) for row in gold_df.values.tolist())
        pred_values = set(tuple(row) for row in pred_df.values.tolist())
        return gold_values == pred_values
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="범용 SQL 채점기")
    parser.add_argument("--input", required=True, help="채점할 JSON 파일 경로")
    parser.add_argument("--output", default=None, help="채점 결과 저장 경로 (미지정 시 자동 생성)")
    parser.add_argument("--sql_key", default="fixed_sql", help="채점 대상 SQL 키 (기본: fixed_sql)")
    parser.add_argument("--gold_key", default="gold_sql", help="정답 SQL 키 (기본: gold_sql)")
    args = parser.parse_args()

    # 출력 경로 자동 생성
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_scored.json"

    print(f"📂 입력: {args.input}")
    print(f"📝 채점 대상 키: {args.sql_key}")
    print(f"🎯 정답 키: {args.gold_key}")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🚀 {len(data)}건 채점 시작\n")

    success_count = 0
    failure_corpus = []
    scored_results = []

    for item in tqdm(data):
        db_id = item['db_id']
        question = item['question']
        gold_sql = item[args.gold_key]
        pred_sql = item[args.sql_key]

        # 정답 실행
        gold_success, _, gold_df = execute_sql(db_id, gold_sql)

        # 예측 실행
        pred_success, exec_msg, pred_df = execute_sql(db_id, pred_sql)

        # 채점
        if not pred_success:
            error_type = "Syntax Error"
            ex_pass = False
        else:
            if compare_results(gold_df, pred_df):
                error_type = None
                ex_pass = True
                success_count += 1
            else:
                error_type = "Semantic Error"
                ex_pass = False

        scored = {**item}
        scored['scored_ex_pass'] = ex_pass
        scored['scored_error_type'] = error_type
        scored['scored_exec_message'] = exec_msg if error_type == "Syntax Error" else None
        scored_results.append(scored)

    # 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(scored_results, f, indent=4, ensure_ascii=False)

    # ── 통계 출력 ──
    total = len(scored_results)
    syntax_errors = sum(1 for r in scored_results if r.get('scored_error_type') == 'Syntax Error')
    semantic_errors = sum(1 for r in scored_results if r.get('scored_error_type') == 'Semantic Error')

    # 원래 에러 유형 → 수정 후 결과 분석
    orig_syntax = sum(1 for r in scored_results if r.get('error_type') == 'Syntax Error')
    orig_semantic = sum(1 for r in scored_results if r.get('error_type') == 'Semantic Error')
    syntax_to_fix = sum(1 for r in scored_results if r.get('error_type') == 'Syntax Error' and r.get('scored_ex_pass'))
    semantic_to_fix = sum(1 for r in scored_results if r.get('error_type') == 'Semantic Error' and r.get('scored_ex_pass'))
    semantic_to_syntax = sum(1 for r in scored_results 
                            if r.get('error_type') == 'Semantic Error' 
                            and r.get('scored_error_type') == 'Syntax Error')

    print("\n" + "=" * 50)
    print("📊 [채점 결과 리포트]")
    print(f"채점 대상: {total}건")
    print(f"교정 성공 (EX Pass): {success_count}건 ({success_count/total*100:.1f}%)")
    print(f"교정 실패: {total - success_count}건")
    print(f"  - 여전히 Syntax Error: {syntax_errors}건")
    print(f"  - 여전히 Semantic Error: {semantic_errors}건")
    print(f"\n전환 분석:")
    print(f"  Syntax Error → 교정 성공: {syntax_to_fix}/{orig_syntax}")
    print(f"  Semantic Error → 교정 성공: {semantic_to_fix}/{orig_semantic}")
    print(f"  Semantic Error → 악화(Syntax): {semantic_to_syntax}건")

    # Difficulty별
    print(f"\nDifficulty별 교정률:")
    for diff in ['simple', 'moderate', 'challenging']:
        dt = sum(1 for r in scored_results if r.get('difficulty') == diff)
        dp = sum(1 for r in scored_results if r.get('difficulty') == diff and r.get('scored_ex_pass'))
        if dt > 0:
            print(f"  {diff}: {dp}/{dt} ({dp/dt*100:.1f}%)")

    print("=" * 50)
    print(f"\n✅ 채점 결과 저장: {args.output}")

if __name__ == "__main__":
    main()