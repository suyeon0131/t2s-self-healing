"""
shape_to_operator_repair.py - Shape-to-Operator Mapping Pipeline

목적:
  v2 self-healing 실패 케이스에 대해 실행 결과 shape를 분류하고,
  shape에 맞는 repair operator를 자동으로 선택하여 적용한다.

Shape → Operator 매핑:
  OVER_SELECT    → SELECT Clause Repair (컬럼 subset 조합)
  column binding → Column Binding Repair (FK JOIN 삽입 + alias swap)
  기타           → 스킵 (현재 operator 없음)

주의:
  gold 기반 oracle evaluation — 실제 운영에서는 gold를 모르므로 PoC 용도

사용법:
  python shape_to_operator_repair.py

출력:
  results/shape_to_operator_repair_log.json
"""

import json
import sqlite3
import re
import os
import openpyxl
from tqdm import tqdm
from itertools import combinations

try:
    import sqlglot
    from sqlglot import exp as sqlglot_exp
except ImportError:
    print("❌ sqlglot이 필요합니다: pip install sqlglot")
    exit(1)

DB_BASE_PATH = "./data/dev_databases"
V2_HEALING_PATH = "./results/multi_turn_healing_v2.json"
FAILURE_CORPUS_PATH = "./results/failure_corpus_official.json"
AST_PROBE_PATH = "./results/column_reference_probe_ast_log.json"
OUTPUT_LOG_PATH = "./results/shape_to_operator_repair_log.json"
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
            labels[str(qid)] = label
    return labels


# ============================================================
# 공통 유틸
# ============================================================

def execute_sql_rows(db_id, sql):
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_count = len(cursor.description) if cursor.description else 0
        conn.close()
        return True, rows, col_count, None
    except Exception as e:
        return False, None, 0, str(e)


def compare_results(gold_rows, pred_rows):
    if gold_rows is None or pred_rows is None:
        return False
    return set(gold_rows) == set(pred_rows)


def classify_shape(pred_rows, pred_cols, gold_rows, gold_cols):
    pred_n = len(pred_rows) if pred_rows is not None else 0
    gold_n = len(gold_rows) if gold_rows is not None else 0
    if pred_n == 0 and gold_n > 0:
        return "EMPTY_RESULT"
    if pred_cols > gold_cols:
        return "OVER_SELECT"
    if pred_cols < gold_cols:
        return "UNDER_SELECT"
    if pred_n < gold_n:
        return "OVER_AGG"
    if pred_n > gold_n:
        return "MISSING_AGG"
    if pred_n == gold_n and pred_cols == gold_cols:
        return "SAME_SHAPE_VALUE_DIFF"
    return "STRUCTURAL_MISMATCH"


# ============================================================
# Operator 1: SELECT Clause Repair
# ============================================================

def extract_select_expressions(pred_sql):
    try:
        parsed = sqlglot.parse_one(pred_sql, dialect="sqlite",
                                   error_level=sqlglot.ErrorLevel.IGNORE)
        select = parsed.find(sqlglot_exp.Select)
        if not select:
            return []
        return [(expr.sql(dialect="sqlite"), expr.alias if hasattr(expr, 'alias') else None)
                for expr in select.expressions]
    except Exception:
        return []


def rebuild_select(pred_sql, kept_expressions):
    """SELECT 절을 kept_expressions로 재구성 — CTE/서브쿼리 포함 처리"""
    try:
        # sqlglot AST 기반 재구성 시도
        parsed = sqlglot.parse_one(pred_sql, dialect="sqlite",
                                   error_level=sqlglot.ErrorLevel.IGNORE)

        # 가장 바깥쪽 SELECT 찾기
        select_node = parsed.find(sqlglot_exp.Select)
        if select_node:
            new_exprs = []
            for expr_sql, _ in kept_expressions:
                try:
                    expr_node = sqlglot.parse_one(expr_sql, dialect="sqlite",
                                                   error_level=sqlglot.ErrorLevel.IGNORE)
                    new_exprs.append(expr_node)
                except Exception:
                    pass

            if new_exprs and len(new_exprs) == len(kept_expressions):
                select_node.set("expressions", new_exprs)
                return parsed.sql(dialect="sqlite")

        # fallback: 문자열 기반 (CTE 없는 단순 SQL)
        sql_upper = pred_sql.upper()

        # CTE 있으면 CTE 다음의 SELECT 위치 찾기
        with_pos = sql_upper.find("WITH ")
        if with_pos != -1:
            # WITH 절 이후 최상위 SELECT 찾기 — 괄호 깊이 추적
            depth = 0
            i = with_pos
            while i < len(sql_upper):
                if sql_upper[i] == '(':
                    depth += 1
                elif sql_upper[i] == ')':
                    depth -= 1
                elif depth == 0 and sql_upper[i:i+6] == 'SELECT':
                    select_pos = i
                    break
                i += 1
            else:
                return None
        else:
            select_pos = sql_upper.find("SELECT")

        if select_pos == -1:
            return None

        # SELECT 이후 FROM 위치 찾기 (괄호 깊이 0인 FROM)
        depth = 0
        i = select_pos + len("SELECT")
        from_pos = -1
        while i < len(sql_upper):
            if sql_upper[i] == '(':
                depth += 1
            elif sql_upper[i] == ')':
                depth -= 1
            elif depth == 0 and sql_upper[i:i+5] in (' FROM', '\nFROM'):
                from_pos = i
                break
            i += 1

        if from_pos == -1:
            return None

        new_exprs_sql = ", ".join(expr_sql for expr_sql, _ in kept_expressions)
        before = pred_sql[:select_pos + len("SELECT")]
        after = pred_sql[from_pos:]
        return before + " " + new_exprs_sql + after

    except Exception:
        return None


def select_clause_repair(db_id, pred_sql, pred_cols, gold_cols, gold_rows):
    """OVER_SELECT: combinations 기반 SELECT subset 후보 생성"""
    if pred_cols <= gold_cols:
        return False, None, []

    expressions = extract_select_expressions(pred_sql)
    if not expressions or len(expressions) <= gold_cols:
        return False, None, []

    n = len(expressions)
    if n > 10:
        return False, None, [{"status": "skipped_too_many_cols", "n": n}]

    tried = []
    for keep_indices in combinations(range(n), gold_cols):
        kept = [expressions[i] for i in keep_indices]
        removed = [expressions[i] for i in range(n) if i not in keep_indices]
        cand_sql = rebuild_select(pred_sql, kept)
        if not cand_sql or cand_sql == pred_sql:
            continue

        ok, pred_rows, _, err = execute_sql_rows(db_id, cand_sql)
        trial = {
            "strategy": "select_subset",
            "keep_indices": list(keep_indices),
            "candidate_sql": cand_sql,
            "exec_success": ok,
            "exec_error": err,
            "ex_pass": False,
        }
        if ok and compare_results(gold_rows, pred_rows):
            trial["ex_pass"] = True
            tried.append(trial)
            return True, cand_sql, tried
        tried.append(trial)

    return False, None, tried


# ============================================================
# Operator 2: Column Binding Repair
# ============================================================

SQL_KEYWORDS = {
    "where", "join", "on", "group", "order", "having", "limit",
    "inner", "left", "right", "full", "cross", "union", "select",
    "from", "and", "or", "not", "in", "is", "as", "by", "set"
}


def quote_ident(name):
    if name is None:
        return ""
    return '"' + str(name).replace('"', '""') + '"'


def qcol(alias, column):
    return f"{quote_ident(alias)}.{quote_ident(column)}"


def replace_qualified_column(sql, old_qualifier, old_column, new_qualifier, new_column):
    replacement = qcol(new_qualifier, new_column)
    patterns = [
        rf'{re.escape(old_qualifier)}\s*\.\s*"{re.escape(old_column)}"',
        rf'"{re.escape(old_qualifier)}"\s*\.\s*"{re.escape(old_column)}"',
        rf'{re.escape(old_qualifier)}\s*\.\s*{re.escape(old_column)}',
    ]
    new_sql = sql
    for p in patterns:
        new_sql = re.sub(p, replacement, new_sql, flags=re.IGNORECASE)
    return new_sql


def find_join_insert_pos(sql):
    s = sql.rstrip().rstrip(";")
    patterns = [r"\bWHERE\b", r"\bGROUP\s+BY\b", r"\bHAVING\b", r"\bORDER\s+BY\b", r"\bLIMIT\b"]
    positions = []
    for p in patterns:
        m = re.search(p, s, re.IGNORECASE)
        if m:
            positions.append(m.start())
    return min(positions) if positions else len(s)


def get_fk_join_condition(db_id, table_a, table_b):
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA foreign_key_list([{table_a}])")
        for row in cursor.fetchall():
            if row[2].lower() == table_b.lower():
                conn.close()
                return row[3], row[4]
        cursor.execute(f"PRAGMA foreign_key_list([{table_b}])")
        for row in cursor.fetchall():
            if row[2].lower() == table_a.lower():
                conn.close()
                return row[4], row[3]
        conn.close()
    except Exception:
        pass
    return None


def column_binding_repair(db_id, pred_sql, invalid_refs, gold_rows):
    """Column Binding: alias swap + JOIN insertion"""
    if not invalid_refs:
        return False, None, []

    # alias 맵
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

    tried = []
    actionable = [r for r in invalid_refs if r.get('candidate_tables') and
                  r.get('issue') == 'column_not_in_resolved_table']

    if not actionable:
        return False, None, []

    # 전략 A: alias swap
    for ref in actionable:
        qualifier = ref['qualifier']
        column = ref['column']
        for cand_table in ref['candidate_tables'][:2]:
            alias_pat = re.compile(
                rf'\b{re.escape(cand_table)}\b\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)',
                re.IGNORECASE
            )
            m = alias_pat.search(pred_sql)
            if m and m.group(1).lower() not in SQL_KEYWORDS:
                cand_alias = m.group(1)
                swapped = replace_qualified_column(pred_sql, qualifier, column, cand_alias, column)
                if swapped != pred_sql:
                    ok, pred_rows, _, err = execute_sql_rows(db_id, swapped)
                    trial = {"strategy": "alias_swap", "candidate_sql": swapped,
                             "exec_success": ok, "exec_error": err, "ex_pass": False}
                    if ok and compare_results(gold_rows, pred_rows):
                        trial["ex_pass"] = True
                        tried.append(trial)
                        return True, swapped, tried
                    tried.append(trial)

    # 전략 B: JOIN insertion
    for ref in actionable:
        qualifier = ref['qualifier']
        column = ref['column']
        resolved_table = ref['resolved_table'].lower()

        for cand_table in ref['candidate_tables'][:2]:
            cand_table_lower = cand_table.lower()

            # FK anchor 탐색
            join_anchor = None
            join_col_anchor = None
            join_col_cand = None
            for used_table in alias_map.values():
                result = get_fk_join_condition(db_id, used_table, cand_table_lower)
                if result:
                    join_col_anchor, join_col_cand = result
                    join_anchor = used_table
                    break

            if not join_anchor:
                continue

            anchor_alias = next((a for a, t in alias_map.items() if t == join_anchor and a != t), join_anchor)

            # 새 alias 생성
            used_aliases = set(alias_map.keys())
            new_alias = next((c for c in 'labcdefghijkmnopqrstuvwxyz' if c not in used_aliases), cand_table_lower[:3])

            join_clause = (
                f"JOIN {quote_ident(cand_table)} {quote_ident(new_alias)} "
                f"ON {qcol(anchor_alias, join_col_anchor)} = {qcol(new_alias, join_col_cand)}"
            )

            insert_pos = find_join_insert_pos(pred_sql)
            base = pred_sql.rstrip().rstrip(";")
            new_sql = base[:insert_pos].rstrip() + "\n" + join_clause + "\n" + base[insert_pos:].lstrip()
            new_sql = replace_qualified_column(new_sql, qualifier, column, new_alias, column)

            ok, pred_rows, _, err = execute_sql_rows(db_id, new_sql)
            trial = {"strategy": "join_insertion", "candidate_sql": new_sql,
                     "exec_success": ok, "exec_error": err, "ex_pass": False}
            if ok and compare_results(gold_rows, pred_rows):
                trial["ex_pass"] = True
                tried.append(trial)
                return True, new_sql, tried
            tried.append(trial)

    return False, None, tried


# ============================================================
# 메인
# ============================================================

def main():
    print("🔧 Shape-to-Operator Mapping Pipeline 시작")

    with open(V2_HEALING_PATH, 'r', encoding='utf-8') as f:
        v2_results = json.load(f)

    # v2 healing 결과 로드 (gold_sql, fixed_sql 참조용)
    with open(V2_HEALING_PATH, 'r', encoding='utf-8') as f:
        v2_results = json.load(f)
    v2_map = {r.get('question_id'): r for r in v2_results}

    # 전체 baseline 실패 262건 대상 (v2 성공 여부 무관)
    # → column_binding은 v2 성공 케이스도 포함해야 하고
    #   select_clause는 v2 실패 케이스에서만 의미 있음
    with open(FAILURE_CORPUS_PATH, 'r', encoding='utf-8') as f:
        failures = json.load(f)
    failure_map = {r.get('question_id'): r for r in failures}

    # shape 분류 및 operator 적용을 위해
    # pred_sql: v2 fixed_sql (self-healing 후 최종 SQL)
    # gold_sql: failure_corpus의 gold_sql
    all_cases = []
    for r in v2_results:
        qid = r.get('question_id')
        failure = failure_map.get(qid)
        if not failure:
            continue
        all_cases.append({
            'question_id': qid,
            'db_id': r.get('db_id'),
            'difficulty': r.get('difficulty'),
            'is_healed': r.get('is_healed', False),
            'fixed_sql': r.get('fixed_sql', ''),
            'gold_sql': failure.get('gold_sql', ''),
            'turn_log': r.get('turn_log', []),
        })

    print(f"📂 전체 대상: {len(all_cases)}건 (v2 성공 포함)")
    print(f"   - v2 성공: {sum(1 for x in all_cases if x['is_healed'])}건")
    print(f"   - v2 실패: {sum(1 for x in all_cases if not x['is_healed'])}건\n")

    # AST probe 결과 로드 (column binding용)
    ast_probe_map = {}
    if os.path.exists(AST_PROBE_PATH):
        with open(AST_PROBE_PATH, 'r', encoding='utf-8') as f:
            ast_logs = json.load(f)
        ast_probe_map = {x['question_id']: x for x in ast_logs}

    manual_labels = load_manual_labels()

    logs = []
    repaired_count = 0
    shape_stats = {}
    operator_stats = {"select_clause": {"tried": 0, "success": 0},
                      "column_binding": {"tried": 0, "success": 0},
                      "skipped": 0}

    for case in tqdm(all_cases):
        qid = case['question_id']
        db_id = case['db_id']
        pred_sql = case['fixed_sql']
        gold_sql = case['gold_sql']
        is_healed = case['is_healed']

        if not pred_sql or not gold_sql:
            continue

        # gold 실행
        gold_ok, gold_rows, gold_cols, gold_err = execute_sql_rows(db_id, gold_sql)
        if not gold_ok:
            logs.append({"question_id": qid, "status": "gold_exec_error"})
            continue

        # pred 실행 → shape 분류
        pred_ok, pred_rows, pred_cols, pred_err = execute_sql_rows(db_id, pred_sql)
        if not pred_ok:
            shape_class = "EXEC_ERROR"
        else:
            shape_class = classify_shape(pred_rows, pred_cols, gold_rows, gold_cols)

        # v2 이미 성공한 케이스는 column_binding만 시도 (select_clause는 스킵)
        if is_healed:
            shape_stats[shape_class] = shape_stats.get(shape_class, 0) + 1

            # column binding 오류가 있으면 시도 (v2 성공 여부 무관)
            ast_probe = ast_probe_map.get(qid, {})
            invalid_refs = ast_probe.get('column_probe', {}).get('invalid_refs', []) if ast_probe else []
            has_column_binding = any(
                r.get('issue') == 'column_not_in_resolved_table' and r.get('candidate_tables')
                for r in invalid_refs
            )

            if has_column_binding:
                operator_stats["column_binding"]["tried"] += 1
                repaired, repair_sql, trials = column_binding_repair(db_id, pred_sql, invalid_refs, gold_rows)
                if repaired:
                    operator_stats["column_binding"]["success"] += 1
                    repaired_count += 1
                logs.append({
                    "question_id": qid,
                    "db_id": db_id,
                    "is_healed": True,
                    "shape_class": shape_class,
                    "has_column_binding": has_column_binding,
                    "operator_used": "column_binding",
                    "repaired": repaired,
                    "repair_sql": repair_sql,
                    "candidates_tried": len(trials),
                })
            else:
                logs.append({
                    "question_id": qid,
                    "db_id": db_id,
                    "is_healed": True,
                    "shape_class": shape_class,
                    "operator_used": None,
                    "repaired": False,
                })
            continue

        shape_stats[shape_class] = shape_stats.get(shape_class, 0) + 1

        # Shape → Operator 선택
        operator_used = None
        repaired = False
        repair_sql = None
        trials = []

        # Column Binding 감지 (AST probe 결과 활용)
        ast_probe = ast_probe_map.get(qid, {})
        invalid_refs = ast_probe.get('column_probe', {}).get('invalid_refs', []) if ast_probe else []
        has_column_binding = any(
            r.get('issue') == 'column_not_in_resolved_table' and r.get('candidate_tables')
            for r in invalid_refs
        )

        if has_column_binding:
            operator_used = "column_binding"
            operator_stats["column_binding"]["tried"] += 1
            repaired, repair_sql, trials = column_binding_repair(db_id, pred_sql, invalid_refs, gold_rows)
            if repaired:
                operator_stats["column_binding"]["success"] += 1

        elif shape_class == "OVER_SELECT":
            operator_used = "select_clause"
            operator_stats["select_clause"]["tried"] += 1
            repaired, repair_sql, trials = select_clause_repair(
                db_id, pred_sql, pred_cols, gold_cols, gold_rows
            )
            if repaired:
                operator_stats["select_clause"]["success"] += 1

        else:
            operator_used = None
            operator_stats["skipped"] += 1

        if repaired:
            repaired_count += 1

        logs.append({
            "question_id": qid,
            "db_id": db_id,
            "difficulty": case.get('difficulty'),
            "is_healed": False,
            "manual_label": manual_labels.get(str(qid)),
            "detected_type": case.get('turn_log', [{}])[0].get('detected_type') if case.get('turn_log') else None,
            "shape_class": shape_class,
            "pred_cols": pred_cols,
            "gold_cols": gold_cols,
            "has_column_binding": has_column_binding,
            "operator_used": operator_used,
            "repaired": repaired,
            "repair_sql": repair_sql,
            "candidates_tried": len(trials),
        })

    with open(OUTPUT_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # 통계
    total = len(logs)
    print(f"\n{'='*60}")
    print(f"📊 Shape-to-Operator Mapping 결과")
    print(f"{'='*60}")
    print(f"전체 실패 케이스: {total}건")
    print(f"추가 교정 성공: {repaired_count}건")

    print(f"\nShape 분포:")
    for sc, cnt in sorted(shape_stats.items(), key=lambda x: -x[1]):
        print(f"  {sc:<30} {cnt}건")

    print(f"\nOperator 적용 결과:")
    for op, stat in operator_stats.items():
        if op == "skipped":
            print(f"  skipped: {stat}건")
        else:
            t = stat['tried']
            s = stat['success']
            rate = s/t*100 if t > 0 else 0
            print(f"  {op}: {s}/{t}건 ({rate:.1f}%)")

    print(f"\n성공 케이스:")
    for x in logs:
        if x.get('repaired'):
            print(f"  ✅ qid={x['question_id']} shape={x['shape_class']} op={x['operator_used']} label={x.get('manual_label')}")

    print(f"\n✅ 로그: {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()