"""
select_clause_repair.py - SELECT Clause Candidate Repair

목적:
  OVER_SELECT(pred 컬럼 수 > gold 컬럼 수) 케이스에서
  시스템이 직접 SELECT 절의 컬럼을 하나씩 제거한 후보 SQL을 생성하고
  gold와 실행 비교로 repair 성공 여부를 판단한다.

방식:
  1. sqlglot AST로 SELECT 절 컬럼 목록 추출
  2. 컬럼을 하나씩 제거한 후보 SQL 생성
  3. gold와 set 비교 (공식 EX)
  4. 성공하면 repair 완료

주의:
  gold 기반 oracle evaluation — 실제 운영에서는 gold를 모르므로 PoC 용도

사용법:
  python select_clause_repair.py

출력:
  results/select_clause_repair_log.json
"""

import json
import sqlite3
import re
import os
import openpyxl
from tqdm import tqdm

try:
    import sqlglot
    from sqlglot import exp as sqlglot_exp
except ImportError:
    print("❌ sqlglot이 필요합니다: pip install sqlglot")
    exit(1)

DB_BASE_PATH = "./data/dev_databases"
SHAPE_LOG_PATH = "./results/result_shape_analysis_log.json"
FAILURE_CORPUS_PATH = "./results/failure_corpus_official.json"
OUTPUT_LOG_PATH = "./results/select_clause_repair_log.json"
FAIL_LABEL_PATH = "./results/error_labeling_233.xlsx"


# ============================================================
# 수작업 라벨 로드
# ============================================================

def load_manual_labels():
    labels = {}
    if not os.path.exists(FAIL_LABEL_PATH):
        return labels
    wb = openpyxl.load_workbook(FAIL_LABEL_PATH)
    ws = wb['오류 라벨링']
    for row in range(2, 235):
        qid = ws.cell(row=row, column=2).value
        label = ws.cell(row=row, column=12).value
        if qid and label:
            labels[qid] = label
    return labels


# ============================================================
# SQL 실행 및 비교
# ============================================================

def execute_sql_rows(db_id, sql):
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)


def compare_results(gold_rows, pred_rows):
    """공식 EX: set 비교"""
    if gold_rows is None or pred_rows is None:
        return False
    return set(gold_rows) == set(pred_rows)


# ============================================================
# SELECT 절 컬럼 추출 및 후보 생성
# ============================================================

def extract_select_expressions(pred_sql):
    """sqlglot AST로 SELECT 절 expressions 추출
    
    Returns:
        list of (expression_sql, expression_alias)
        예: [("COUNT(*)", "cnt"), ("T1.name", "name"), ...]
    """
    try:
        parsed = sqlglot.parse_one(pred_sql, dialect="sqlite",
                                   error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return []

    select = parsed.find(sqlglot_exp.Select)
    if not select:
        return []

    expressions = []
    for expr in select.expressions:
        expr_sql = expr.sql(dialect="sqlite")
        alias = expr.alias if hasattr(expr, 'alias') and expr.alias else None
        expressions.append((expr_sql, alias))

    return expressions


def generate_select_candidates(pred_sql, pred_cols, gold_cols):
    """SELECT 절에서 컬럼을 제거한 후보 SQL 생성
    
    전략:
    - diff = pred_cols - gold_cols 만큼 컬럼을 제거
    - 단순 제거(가장 마지막 컬럼부터)와 조합 제거 시도
    - sqlglot으로 SELECT 절만 교체
    
    Returns:
        list of (candidate_sql, description, removed_cols)
    """
    candidates = []
    diff = pred_cols - gold_cols

    if diff <= 0:
        return []

    expressions = extract_select_expressions(pred_sql)
    if not expressions or len(expressions) <= gold_cols:
        return []

    n = len(expressions)

    # 전략 1: 단순 제거 — 마지막 diff개 컬럼 제거
    kept = expressions[:gold_cols]
    removed = expressions[gold_cols:]
    candidate_sql = rebuild_select(pred_sql, kept)
    if candidate_sql and candidate_sql != pred_sql:
        candidates.append((
            candidate_sql,
            f"remove_last_{diff}: removed {[e[0][:30] for e in removed]}",
            [e[0] for e in removed]
        ))

    # 전략 2: 처음 diff개 컬럼 제거 (앞쪽이 불필요 컬럼인 경우)
    if diff < n:
        kept2 = expressions[diff:]
        removed2 = expressions[:diff]
        candidate_sql2 = rebuild_select(pred_sql, kept2)
        if candidate_sql2 and candidate_sql2 != pred_sql and candidate_sql2 != candidates[0][0] if candidates else True:
            candidates.append((
                candidate_sql2,
                f"remove_first_{diff}: removed {[e[0][:30] for e in removed2]}",
                [e[0] for e in removed2]
            ))

    # 전략 3: diff=1인 경우 각 컬럼을 하나씩 제거한 n개 후보
    if diff == 1 and n <= 8:  # 너무 많은 후보 방지
        for i in range(n):
            kept3 = expressions[:i] + expressions[i+1:]
            removed3 = [expressions[i]]
            candidate_sql3 = rebuild_select(pred_sql, kept3)
            if candidate_sql3 and candidate_sql3 != pred_sql:
                # 중복 체크
                if not any(c[0] == candidate_sql3 for c in candidates):
                    candidates.append((
                        candidate_sql3,
                        f"remove_idx_{i}: removed [{expressions[i][0][:30]}]",
                        [expressions[i][0]]
                    ))

    return candidates


def rebuild_select(pred_sql, kept_expressions):
    """SELECT 절을 kept_expressions로 재구성
    
    sqlglot으로 파싱 후 SELECT 절만 교체
    """
    try:
        parsed = sqlglot.parse_one(pred_sql, dialect="sqlite",
                                   error_level=sqlglot.ErrorLevel.IGNORE)
        select = parsed.find(sqlglot_exp.Select)
        if not select:
            return None

        # 새 SELECT 표현식 생성
        new_exprs_sql = ", ".join(expr_sql for expr_sql, _ in kept_expressions)

        # SELECT ~ FROM 사이를 교체
        sql_upper = pred_sql.upper()
        select_pos = sql_upper.find("SELECT")
        from_pos = sql_upper.find("\nFROM")
        if from_pos == -1:
            from_pos = sql_upper.find(" FROM")
            if from_pos == -1:
                return None

        # SELECT 다음부터 FROM 전까지 교체
        before = pred_sql[:select_pos + len("SELECT")]
        after = pred_sql[from_pos:]
        new_sql = before + " " + new_exprs_sql + after

        return new_sql
    except Exception:
        return None


# ============================================================
# 메인
# ============================================================

def main():
    print("🔧 SELECT Clause Candidate Repair 시작")

    # shape 분석 결과에서 OVER_SELECT 케이스 추출
    with open(SHAPE_LOG_PATH, 'r', encoding='utf-8') as f:
        shape_logs = json.load(f)

    over_select_cases = [
        x for x in shape_logs
        if x.get('shape_class') == 'OVER_SELECT' and not x.get('is_healed')
    ]
    print(f"📂 OVER_SELECT 대상: {len(over_select_cases)}건")

    # failure corpus에서 gold SQL 로드
    with open(FAILURE_CORPUS_PATH, 'r', encoding='utf-8') as f:
        failures = json.load(f)
    failure_map = {r.get('question_id'): r for r in failures}

    manual_labels = load_manual_labels()

    logs = []
    success_count = 0

    for case in tqdm(over_select_cases):
        qid = case['question_id']
        db_id = case['db_id']
        pred_cols = case.get('pred_cols', 0)
        gold_cols = case.get('gold_cols', 0)

        failure = failure_map.get(qid)
        if not failure:
            continue

        gold_sql = failure.get('gold_sql', '')
        fixed_sql = failure.get('fixed_sql', case.get('pred_sql', ''))

        # v2의 최종 fixed_sql 사용 (self-healing 후 결과)
        # shape analysis는 fixed_sql 기준이었으므로 동일하게 사용
        pred_sql = fixed_sql

        # gold 실행
        gold_ok, gold_rows, gold_err = execute_sql_rows(db_id, gold_sql)
        if not gold_ok:
            logs.append({
                "question_id": qid,
                "db_id": db_id,
                "status": "gold_exec_error",
                "repaired": False,
            })
            continue

        # 후보 생성
        candidates = generate_select_candidates(pred_sql, pred_cols, gold_cols)

        repaired = False
        repair_sql = None
        repair_desc = None
        tried = []

        for cand_sql, cand_desc, removed_cols in candidates:
            pred_ok, pred_rows, pred_err = execute_sql_rows(db_id, cand_sql)

            trial = {
                "description": cand_desc,
                "candidate_sql": cand_sql[:300],
                "exec_success": pred_ok,
                "exec_error": pred_err,
                "row_count": len(pred_rows) if pred_rows else 0,
                "ex_pass": False,
            }

            if pred_ok and compare_results(gold_rows, pred_rows):
                trial["ex_pass"] = True
                repaired = True
                repair_sql = cand_sql
                repair_desc = cand_desc

            tried.append(trial)

            if repaired:
                break

        if repaired:
            success_count += 1

        logs.append({
            "question_id": qid,
            "db_id": db_id,
            "difficulty": case.get('difficulty'),
            "manual_label": manual_labels.get(qid),
            "detected_type": case.get('detected_type'),
            "pred_cols": pred_cols,
            "gold_cols": gold_cols,
            "diff": pred_cols - gold_cols,
            "candidates_tried": len(tried),
            "repaired": repaired,
            "repair_description": repair_desc,
            "trials": tried,
        })

    with open(OUTPUT_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # 통계
    total = len(logs)
    repaired_count = sum(1 for x in logs if x.get('repaired'))
    no_candidate = sum(1 for x in logs if x.get('candidates_tried', 0) == 0)

    print(f"\n{'='*50}")
    print(f"📊 SELECT Clause Candidate Repair 결과")
    print(f"{'='*50}")
    print(f"대상: {total}건")
    print(f"  후보 없음: {no_candidate}건")
    print(f"  시도: {total - no_candidate}건")
    tried = total - no_candidate
    if tried > 0:
        print(f"  repair 성공: {repaired_count}건 ({repaired_count/tried*100:.1f}% of tried)")

    # diff별 성공률
    print(f"\ndiff별 성공률:")
    diff_stats = {}
    for x in logs:
        d = x.get('diff', 0)
        if d not in diff_stats:
            diff_stats[d] = {'total': 0, 'repaired': 0}
        diff_stats[d]['total'] += 1
        if x.get('repaired'):
            diff_stats[d]['repaired'] += 1
    for d in sorted(diff_stats.keys()):
        s = diff_stats[d]
        rate = s['repaired'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  diff={d}: {s['repaired']}/{s['total']} ({rate:.1f}%)")

    # 성공 케이스 상세
    print(f"\n성공 케이스:")
    for x in logs:
        status = "✅" if x.get('repaired') else "❌"
        print(f"  {status} qid={x['question_id']} diff={x.get('diff')} tried={x.get('candidates_tried',0)} label={x.get('manual_label','?')}")

    print(f"\n✅ 로그: {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()