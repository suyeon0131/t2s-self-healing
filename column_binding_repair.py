"""
column_binding_repair.py - Column Binding Candidate Repair

목적:
  Prompt 주입 없이, 시스템이 직접 column binding을 수정한 후보 SQL을 생성하고
  실행 결과로 repair 성공 여부를 판단한다.

대상:
  column_reference_probe에서 true_wrong_binding으로 확인된 케이스
  (candidate_tables가 있는 invalid_ref)

방식:
  1. invalid_ref에서 qualifier → resolved_table 매핑 확인
  2. candidate_table로 교체한 후보 SQL 생성
     - FROM/JOIN 절에서 resolved_table → candidate_table 교체
     - alias 재매핑
  3. 후보 SQL 실행
  4. 공식 EX(set 비교)로 gold와 비교
  5. 성공하면 repair 완료

사용법:
  python column_binding_repair.py

출력:
  results/column_binding_repair_log.json  (전체 로그)
"""

import json
import sqlite3
import re
import os
import openpyxl
from tqdm import tqdm

DB_BASE_PATH = "./data/dev_databases"
PROBE_LOG_PATH = "./results/column_reference_probe_log.json"
FAILURE_CORPUS_PATH = "./results/failure_corpus_official.json"
OUTPUT_LOG_PATH = "./results/column_binding_repair_log.json"

# true_wrong_binding으로 수작업 확인된 케이스 (question_id 기준)
TRUE_WRONG_BINDING_QIDS = {
    1251, 1254, 1270, 1302,  # thrombosis_prediction
    1037,                     # european_football_2
    962,                      # formula_1
    682,                      # codebase_community
    45, 62,                   # california_schools (candidate 있는 것만)
    94, 149, 168, 119,        # financial
}


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
# 후보 SQL 생성: column rebinding
# ============================================================

SQL_KEYWORDS = {
    "where", "join", "on", "group", "order", "having", "limit",
    "inner", "left", "right", "full", "cross", "union", "select",
    "from", "and", "or", "not", "in", "is", "as", "by", "set"
}


def find_join_insert_pos(sql):
    """JOIN 절을 삽입할 올바른 위치 찾기
    WHERE, GROUP BY, HAVING, ORDER BY, LIMIT 중 가장 앞에 오는 것 앞에 삽입
    """
    s = sql.rstrip().rstrip(";")
    patterns = [
        r"\bWHERE\b",
        r"\bGROUP\s+BY\b",
        r"\bHAVING\b",
        r"\bORDER\s+BY\b",
        r"\bLIMIT\b",
    ]
    positions = []
    for p in patterns:
        m = re.search(p, s, re.IGNORECASE)
        if m:
            positions.append(m.start())
    return min(positions) if positions else len(s)


def get_fk_join_condition(db_id, table_a, table_b):
    """두 테이블 간 FK join 조건 찾기
    
    Returns:
        (join_col_a, join_col_b) or None
    """
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # table_a → table_b FK
        cursor.execute(f"PRAGMA foreign_key_list([{table_a}])")
        for row in cursor.fetchall():
            if row[2].lower() == table_b.lower():
                conn.close()
                return row[3], row[4]  # from_col, to_col

        # table_b → table_a FK (역방향)
        cursor.execute(f"PRAGMA foreign_key_list([{table_b}])")
        for row in cursor.fetchall():
            if row[2].lower() == table_a.lower():
                conn.close()
                return row[4], row[3]  # to_col, from_col (역방향이라 스왑)

        conn.close()
    except Exception:
        pass
    return None


def generate_join_insertion_sqls(db_id, pred_sql, needs_join, alias_map):
    """candidate_table이 SQL에 없는 경우 JOIN 절 삽입

    전략:
    1. 현재 SQL에 사용된 테이블 중 candidate_table과 FK가 있는 테이블 찾기
    2. FK join 조건 생성
    3. JOIN 절 삽입 + column reference 교체

    Returns:
        list of (candidate_sql, description, strategy)
    """
    candidates = []

    # 현재 SQL에서 사용된 실제 테이블 목록
    used_tables = {v: None for v in alias_map.values()}  # table_lower: alias

    # alias → table 역매핑
    table_to_alias = {}
    for alias, table in alias_map.items():
        if alias != table:  # 별도 alias가 있는 경우
            table_to_alias[table] = alias

    for nj in needs_join:
        qualifier = nj['qualifier']
        column = nj['column']
        cand_table = nj['candidate_table']
        cand_table_lower = cand_table.lower()

        # candidate_table과 FK 연결되는 현재 사용 테이블 찾기
        join_anchor = None
        join_col_anchor = None
        join_col_cand = None

        for used_table in used_tables:
            result = get_fk_join_condition(db_id, used_table, cand_table_lower)
            if result:
                join_col_anchor, join_col_cand = result
                join_anchor = used_table
                break

        if not join_anchor:
            continue  # FK 연결 못 찾음

        # anchor 테이블의 alias 찾기
        anchor_alias = table_to_alias.get(join_anchor, join_anchor)

        # candidate_table의 새 alias 생성
        # 이미 사용 중인 단일 문자 alias 피하기
        used_aliases = set(alias_map.keys())
        new_alias = None
        for char in 'labcdefghijkmnopqrstuvwxyz':
            if char not in used_aliases:
                new_alias = char
                break
        if not new_alias:
            new_alias = cand_table_lower[:3]  # 테이블명 앞 3글자

        # JOIN 절 생성
        join_clause = (
            f"JOIN [{cand_table}] {new_alias} "
            f"ON {anchor_alias}.{join_col_anchor} = {new_alias}.{join_col_cand}"
        )

        # pred SQL에 JOIN 삽입 — WHERE/GROUP BY/ORDER BY 등 앞에 삽입
        insert_pos = find_join_insert_pos(pred_sql)
        base = pred_sql.rstrip().rstrip(";")
        new_sql = base[:insert_pos].rstrip() + "\n" + join_clause + "\n" + base[insert_pos:].lstrip()

        # column reference 교체: qualifier.column → new_alias.column
        new_sql = re.sub(
            rf'\b{re.escape(qualifier)}\.{re.escape(column)}\b',
            f'{new_alias}.{column}',
            new_sql,
            flags=re.IGNORECASE
        )

        if new_sql != pred_sql:
            candidates.append((
                new_sql,
                f"join_insert:{cand_table} via {join_anchor}.{join_col_anchor}={cand_table}.{join_col_cand}, {qualifier}.{column}→{new_alias}.{column}",
                "join_insertion"
            ))

    # 중복 제거
    seen = set()
    unique = []
    for sql, desc, strategy in candidates:
        if sql not in seen:
            seen.add(sql)
            unique.append((sql, desc, strategy))

    return unique


def generate_candidate_sqls(pred_sql, invalid_refs):
    """invalid_ref의 candidate_tables로 교체한 후보 SQL 생성

    전략 A (우선): candidate_table이 이미 SQL에 있으면
                   해당 alias로 column reference만 교체 (e.IGG to l.IGG)
    전략 B (차순): candidate_table이 SQL에 없으면
                   needs_join_insertion으로 분류 (이번엔 생성 안 함)
    table replacement는 사용하지 않음 (너무 공격적)

    Returns:
        list of (candidate_sql, description, strategy)
    """
    candidates = []
    needs_join = []

    for ref in invalid_refs:
        qualifier = ref.get('qualifier', '')
        resolved_table = ref.get('resolved_table', '')
        column = ref.get('column', '')
        candidate_tables = ref.get('candidate_tables', [])

        if not candidate_tables:
            continue

        for cand_table in candidate_tables[:2]:
            # 전략 A: candidate_table이 이미 SQL에 있는지 확인
            # FROM/JOIN 절에서 alias 찾기
            alias_pattern = re.compile(
                rf'\b{re.escape(cand_table)}\b\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)',
                re.IGNORECASE
            )
            alias_match = alias_pattern.search(pred_sql)

            if alias_match:
                cand_alias = alias_match.group(1).lower()
                # keyword면 alias 없는 경우
                if cand_alias in SQL_KEYWORDS:
                    cand_alias = None

                if cand_alias:
                    # column reference만 교체: qualifier.column → cand_alias.column
                    swapped = re.sub(
                        rf'\b{re.escape(qualifier)}\.{re.escape(column)}\b',
                        f'{cand_alias}.{column}',
                        pred_sql,
                        flags=re.IGNORECASE
                    )
                    if swapped != pred_sql:
                        candidates.append((
                            swapped,
                            f"alias_swap:{qualifier}.{column}→{cand_alias}.{column}",
                            "alias_swap"
                        ))
                else:
                    # cand_table이 있는데 alias가 없으면 테이블명 직접 사용
                    swapped = re.sub(
                        rf'\b{re.escape(qualifier)}\.{re.escape(column)}\b',
                        f'{cand_table}.{column}',
                        pred_sql,
                        flags=re.IGNORECASE
                    )
                    if swapped != pred_sql:
                        candidates.append((
                            swapped,
                            f"table_ref_swap:{qualifier}.{column}→{cand_table}.{column}",
                            "table_ref_swap"
                        ))
            else:
                # 전략 B: candidate_table이 SQL에 없음 → JOIN 추가 필요
                needs_join.append({
                    "qualifier": qualifier,
                    "column": column,
                    "resolved_table": resolved_table,
                    "candidate_table": cand_table,
                    "reason": "candidate_table_not_in_sql"
                })

    # 중복 제거
    seen = set()
    unique = []
    for sql, desc, strategy in candidates:
        if sql not in seen:
            seen.add(sql)
            unique.append((sql, desc, strategy))

    return unique, needs_join


# ============================================================
# 메인
# ============================================================

def main():
    print("🔧 Column Binding Candidate Repair 시작")

    # probe log 로드
    with open(PROBE_LOG_PATH, 'r', encoding='utf-8') as f:
        probe_logs = json.load(f)

    # failure corpus 로드 (gold SQL 확인용)
    with open(FAILURE_CORPUS_PATH, 'r', encoding='utf-8') as f:
        failures = json.load(f)

    failure_map = {r.get('question_id'): r for r in failures}

    # true_wrong_binding 케이스만 필터
    target_cases = [
        log for log in probe_logs
        if log.get('question_id') in TRUE_WRONG_BINDING_QIDS
        and log.get('column_probe', {}).get('invalid_refs')
    ]

    print(f"📂 대상: {len(target_cases)}건\n")

    logs = []
    success_count = 0

    for case in tqdm(target_cases):
        qid = case['question_id']
        db_id = case['db_id']
        pred_sql = case['pred_sql']
        invalid_refs = case['column_probe'].get('invalid_refs', [])

        # candidate_tables 있는 것만
        actionable_refs = [r for r in invalid_refs if r.get('candidate_tables')]
        if not actionable_refs:
            logs.append({
                "question_id": qid,
                "db_id": db_id,
                "manual_label": case.get('manual_label'),
                "status": "no_candidate",
                "candidates_tried": 0,
                "repaired": False,
            })
            continue

        # gold SQL
        failure = failure_map.get(qid)
        if not failure:
            continue
        gold_sql = failure.get('gold_sql', '')

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

        # 후보 SQL 생성 (alias swap)
        candidates, needs_join = generate_candidate_sqls(pred_sql, actionable_refs)

        # JOIN insertion 후보도 생성
        alias_map = {}
        alias_pattern = re.compile(
            r'\b(?:FROM|JOIN)\s+["`\[]?([A-Za-z_][A-Za-z0-9_]*)["`\]]?'
            r'(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?',
            re.IGNORECASE
        )
        for table, alias in alias_pattern.findall(pred_sql):
            t_l = table.lower()
            alias_map[t_l] = t_l
            if alias and alias.lower() not in SQL_KEYWORDS:
                alias_map[alias.lower()] = t_l

        if needs_join:
            join_candidates = generate_join_insertion_sqls(db_id, pred_sql, needs_join, alias_map)
            candidates.extend(join_candidates)

        repaired = False
        repair_sql = None
        repair_desc = None
        tried = []

        for cand_sql, cand_desc, cand_strategy in candidates:
            pred_ok, pred_rows, pred_err = execute_sql_rows(db_id, cand_sql)

            trial = {
                "description": cand_desc,
                "strategy": cand_strategy,
                "candidate_sql": cand_sql,
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
                break

            tried.append(trial)

        if repaired:
            success_count += 1
            print(f"  ✅ qid={qid} ({db_id}): {repair_desc}")

        logs.append({
            "question_id": qid,
            "db_id": db_id,
            "manual_label": case.get('manual_label'),
            "difficulty": case.get('difficulty'),
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "invalid_refs": actionable_refs,
            "candidates_tried": len(tried),
            "needs_join_insertion": needs_join,
            "trials": tried,
            "repaired": repaired,
            "repair_sql": repair_sql,
            "repair_description": repair_desc,
        })

    # 저장
    with open(OUTPUT_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # 통계
    total = len(logs)
    repaired = sum(1 for x in logs if x.get('repaired'))
    no_candidate = sum(1 for x in logs if x.get('status') == 'no_candidate')

    print(f"\n{'='*50}")
    print(f"📊 Column Binding Candidate Repair 결과")
    print(f"{'='*50}")
    print(f"대상: {total}건")
    print(f"  candidate 없음: {no_candidate}건")
    print(f"  시도: {total - no_candidate}건")
    print(f"  repair 성공: {repaired}건 ({repaired/(total-no_candidate)*100:.1f}% of tried)" if total - no_candidate > 0 else "")
    print(f"\n상세:")
    for x in logs:
        status = "✅ 성공" if x.get('repaired') else "❌ 실패"
        tried_n = x.get('candidates_tried', 0)
        print(f"  qid={x['question_id']} {status} (시도={tried_n}건) label={x.get('manual_label')}")

    print(f"\n✅ 로그: {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()