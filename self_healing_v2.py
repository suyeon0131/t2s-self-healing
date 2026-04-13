import json
import sqlite3
import re
import pandas as pd
from openai import OpenAI
import os
import sys
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
OUTPUT_PATH = "./results/multi_turn_healing_v2.json"


# ============================================================
# 기존 유틸 함수 (self_healing.py와 동일)
# ============================================================

def extract_tables_from_sql(pred_sql):
    """예측된 SQL 쿼리에서 FROM과 JOIN 뒤에 오는 테이블 이름만 추출합니다."""
    sql_lower = pred_sql.lower().replace('\n', ' ')
    pattern = r'(?:from|join)\s+([a-zA-Z0-9_]+)'
    tables = re.findall(pattern, sql_lower)
    return list(set(tables))

def get_db_schema(db_id):
    """해당 DB의 모든 테이블 생성문(스키마)을 가져옵니다."""
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return "-- 스키마를 찾을 수 없습니다."
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schemas = [row[0] for row in cursor.fetchall() if row[0] is not None]
    conn.close()
    
    return "\n".join(schemas)

def get_sample_rows(db_id, table_names):
    """DB에 접속하여 특정 테이블들의 실제 데이터 3줄을 가져옵니다."""
    db_path = f"{DB_BASE_PATH}/{db_id}/{db_id}.sqlite"
    samples_text = ""
    
    if not os.path.exists(db_path):
        return "\n[DB 파일을 찾을 수 없습니다.]\n"

    try:
        conn = sqlite3.connect(db_path)
        for table in table_names:
            try:
                query = f"SELECT * FROM [{table}] LIMIT 3"
                df = pd.read_sql_query(query, conn)
                samples_text += f"\n[Table: {table} Sample Data]\n"
                samples_text += df.to_markdown(index=False) + "\n"
            except Exception:
                samples_text += f"\n[Table: {table} — Could not read]\n"
        conn.close()
    except Exception as e:
        samples_text += f"\n데이터 샘플을 가져오는 중 에러 발생: {e}\n"
        
    return samples_text

def execute_sql(db_id, sql_query):
    """에이전트가 생성한 SQL을 실제 DB에서 실행해봅니다."""
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return True, "Success", df
    except Exception as e:
        return False, str(e), None

def compare_results(gold_df, pred_df):
    """에이전트의 결과가 정답과 일치하는지 확인합니다. (공식 EX: set 비교)"""
    if gold_df is None or pred_df is None:
        return False
    try:
        gold_values = set(tuple(row) for row in gold_df.values.tolist())
        pred_values = set(tuple(row) for row in pred_df.values.tolist())
        return gold_values == pred_values
    except:
        return False


# ============================================================
# 3-Type Gold-Free Detector
# ============================================================

def detect_value_format(db_id, pred_sql):
    """VALUE_FORMAT 감지: SQL의 WHERE 리터럴이 DB에 실제 존재하는지 확인"""
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return False, []
    
    # SQL에서 column = 'value' 패턴 추출
    where_patterns = re.findall(r"(\w+)\s*=\s*'([^']+)'", pred_sql, re.IGNORECASE)
    if not where_patterns:
        return False, []
    
    used_tables = extract_tables_from_sql(pred_sql)
    if not used_tables:
        return False, []
    
    mismatches = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        for column, value in where_patterns:
            for table in used_tables:
                try:
                    # 해당 컬럼이 이 테이블에 존재하는지 먼저 확인
                    cursor.execute(f"PRAGMA table_info([{table}])")
                    columns = [row[1].lower() for row in cursor.fetchall()]
                    if column.lower() not in columns:
                        continue
                    
                    # 정확한 값 존재 여부 확인
                    cursor.execute(
                        f"SELECT COUNT(*) FROM [{table}] WHERE [{column}] = ?",
                        (value,)
                    )
                    count = cursor.fetchone()[0]
                    
                    if count == 0:
                        # 대소문자 불일치 확인
                        cursor.execute(
                            f"SELECT DISTINCT [{column}] FROM [{table}] "
                            f"WHERE LOWER([{column}]) = LOWER(?) LIMIT 5",
                            (value,)
                        )
                        case_matches = [row[0] for row in cursor.fetchall()]
                        
                        if case_matches:
                            mismatches.append({
                                "table": table,
                                "column": column,
                                "used_value": value,
                                "actual_values": case_matches,
                                "issue": "case_mismatch"
                            })
                        else:
                            # 아예 없는 값 — DISTINCT 상위 값 조회
                            cursor.execute(
                                f"SELECT DISTINCT [{column}] FROM [{table}] LIMIT 10"
                            )
                            actual_vals = [row[0] for row in cursor.fetchall()]
                            mismatches.append({
                                "table": table,
                                "column": column,
                                "used_value": value,
                                "actual_values": actual_vals,
                                "issue": "value_not_found"
                            })
                    break  # 첫 번째 매칭 테이블에서 확인 완료
                except Exception:
                    continue
        
        conn.close()
    except Exception:
        return False, []
    
    return len(mismatches) > 0, mismatches


def detect_join_path(db_id, pred_sql, schema_text):
    """JOIN_PATH 감지: FK 그래프 기반으로 테이블 연결 누락 확인"""
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return False, {}
    
    # 스키마에서 모든 테이블 이름 추출
    all_tables = re.findall(
        r'CREATE\s+TABLE\s+["\[\`]?(\w+)["\]\`]?',
        schema_text, re.IGNORECASE
    )
    all_tables_lower = [t.lower() for t in all_tables]
    
    # SQL에서 사용된 테이블
    used_tables = [t.lower() for t in extract_tables_from_sql(pred_sql)]
    
    # FK 관계 추출
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        fk_list = []
        for table in all_tables:
            cursor.execute(f"PRAGMA foreign_key_list([{table}])")
            for row in cursor.fetchall():
                fk_list.append({
                    "from_table": table.lower(),
                    "from_column": row[3],
                    "to_table": row[2].lower(),
                    "to_column": row[4]
                })
        conn.close()
    except Exception:
        return False, {}
    
    if not fk_list:
        return False, {}
    
    # 인접 리스트 구축
    adj = {}
    for fk in fk_list:
        ft, tt = fk['from_table'], fk['to_table']
        adj.setdefault(ft, set()).add(tt)
        adj.setdefault(tt, set()).add(ft)
    
    # 사용된 테이블 중 FK 그래프에 있는 것만 필터
    used_in_graph = [t for t in used_tables if t in adj]
    
    if len(used_in_graph) < 2:
        return False, {}
    
    # BFS로 연결성 확인
    visited = set()
    queue = [used_in_graph[0]]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in used_in_graph and neighbor not in visited:
                queue.append(neighbor)
    
    unreachable = [t for t in used_in_graph if t not in visited]
    
    info = {
        "fk_relations": fk_list,
        "used_tables": used_tables,
        "unreachable": unreachable
    }
    
    return len(unreachable) > 0, info


def detect_error_type(db_id, pred_sql, exec_success, exec_message, schema_text):
    """3-type 오류 감지. 우선순위: SYNTAX → VALUE_FORMAT → JOIN_PATH → GENERAL
    
    Returns:
        (error_type: str, detection_info: dict)
    """
    # Type 1: SYNTAX — 실행 실패
    if not exec_success:
        return "SYNTAX", {"message": exec_message}
    
    # Type 2: VALUE_FORMAT — 리터럴 불일치
    is_value_error, mismatches = detect_value_format(db_id, pred_sql)
    if is_value_error:
        return "VALUE_FORMAT", {"mismatches": mismatches}
    
    # Type 3: JOIN_PATH — 테이블 연결 누락
    is_join_error, join_info = detect_join_path(db_id, pred_sql, schema_text)
    if is_join_error:
        return "JOIN_PATH", join_info
    
    # 폴백: 일반 Semantic Error (기존 방식)
    return "GENERAL", {}


# ============================================================
# 유형별 맞춤 Repair Prompt Template
# ============================================================

def build_syntax_feedback(exec_message):
    """SYNTAX: 에러 로그 주입 (기존과 동일)"""
    return f"A Syntax Error occurred. Please fix the syntax based on the following system error log:\n{exec_message}"


def build_value_format_feedback(db_id, pred_sql, mismatches):
    """VALUE_FORMAT: 리터럴 불일치를 명시적으로 지적 + 실제 DB 값 제시"""
    feedback = "ERROR TYPE: Value / Format Mismatch\n"
    feedback += "The query executed but returned WRONG results. "
    feedback += "The following string literals in your WHERE clause do NOT match the actual values in the database:\n\n"
    
    for m in mismatches:
        feedback += f"  Column [{m['table']}.{m['column']}]:\n"
        feedback += f"    Your query uses: '{m['used_value']}'\n"
        if m['issue'] == 'case_mismatch':
            feedback += f"    Correct value(s) in DB: {', '.join(repr(v) for v in m['actual_values'])}\n"
            feedback += f"    → Case mismatch! Use the exact casing from the database.\n\n"
        else:
            feedback += f"    This value does NOT exist in the database.\n"
            feedback += f"    Available values: {', '.join(repr(v) for v in m['actual_values'][:10])}\n"
            feedback += f"    → Use one of the actual values listed above.\n\n"
    
    feedback += "REPAIR STRATEGY:\n"
    feedback += "1. Replace ALL mismatched string literals with the exact values shown above.\n"
    feedback += "2. Check date formats — verify if dates use 'YYYY-MM-DD', 'YYYYMMDD', or 'YYYYMM' by looking at the sample data.\n"
    feedback += "3. Do NOT change the query logic — only fix the value mismatches.\n"
    
    # 데이터 샘플도 추가 (기존 방식 유지)
    used_tables = extract_tables_from_sql(pred_sql)
    sample_data = get_sample_rows(db_id, used_tables)
    feedback += f"\n{sample_data}"
    
    return feedback


def build_join_path_feedback(db_id, join_info):
    """JOIN_PATH: FK 관계를 명시적으로 나열하고 올바른 경로 제시"""
    feedback = "ERROR TYPE: JOIN Path Error\n"
    feedback += "The query executed but returned WRONG results. "
    feedback += "The table join structure is likely incorrect — missing a bridge table, wrong join column, or unnecessary table.\n\n"
    
    feedback += f"Tables currently used in your SQL: {', '.join(join_info.get('used_tables', []))}\n"
    
    if join_info.get('unreachable'):
        feedback += f"Unreachable tables (not connected via FK): {', '.join(join_info['unreachable'])}\n\n"
    
    # FK 관계 나열
    fk_list = join_info.get('fk_relations', [])
    if fk_list:
        feedback += "Foreign Key Relationships in this database:\n"
        for fk in fk_list:
            feedback += f"  {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
    
    feedback += "\nREPAIR STRATEGY:\n"
    feedback += "1. LIST all entities mentioned in the question.\n"
    feedback += "2. For EACH entity, identify the correct table from the schema.\n"
    feedback += "3. TRACE the shortest FK path connecting all required tables using the relationships above.\n"
    feedback += "4. If two tables are not directly connected, find the BRIDGE TABLE that links them.\n"
    feedback += "5. Use explicit JOIN ... ON syntax with correct FK columns.\n"
    
    # 데이터 샘플도 추가
    used_tables = join_info.get('used_tables', [])
    sample_data = get_sample_rows(db_id, used_tables)
    feedback += f"\n{sample_data}"
    
    return feedback


def build_general_feedback(db_id, pred_sql):
    """GENERAL: 기존 방식 — 데이터 샘플 + 스키마 (폴백)"""
    used_tables = extract_tables_from_sql(pred_sql)
    sample_data_text = get_sample_rows(db_id, used_tables)
    return f"The query executed without errors but returned incorrect data. Please check the actual database sample rows below and rewrite the query:\n{sample_data_text}"


def build_feedback(detected_type, detection_info, db_id, pred_sql, exec_message):
    """감지된 유형에 따라 적절한 피드백을 생성합니다."""
    if detected_type == "SYNTAX":
        return build_syntax_feedback(exec_message)
    elif detected_type == "VALUE_FORMAT":
        return build_value_format_feedback(db_id, pred_sql, detection_info.get("mismatches", []))
    elif detected_type == "JOIN_PATH":
        return build_join_path_feedback(db_id, detection_info)
    else:  # GENERAL
        return build_general_feedback(db_id, pred_sql)


# ============================================================
# LLM 호출 (기존과 동일)
# ============================================================

def call_llm_to_fix_query(question, pred_sql, feedback_prompt, schema_text):
    """GPT-4o-mini를 호출하여 Schema Linking + CoT 방식으로 쿼리를 수정하게 합니다."""
    
    system_message = "You are a world-class SQLite database expert. Your task is to perfectly fix the incorrect SQL query based on the provided feedback."
    
    user_message = f"""
[User Question]
{question}

[Your Previous Incorrect SQL]
{pred_sql}

[Database Schema]
{schema_text}

[Feedback & Clues]
{feedback_prompt}

Carefully review the schema and feedback above. To solve complex logical or mathematical errors, you MUST follow these thinking steps before writing the final SQL:
1. Schema Linking: Identify all necessary tables, columns, and how they should be joined (e.g., bridge tables) using the [Database Schema].
2. Logical & Mathematical Operations: Identify if aggregations (SUM, AVG), groupings, or specific mathematical formulas are required.
3. Value Mapping: Ensure string values match the exact formats shown in the [Feedback & Clues] sample data.

Output your response in the following format:
[Thinking Process]
(Write your step-by-step reasoning here)

[Final SQL]
```sql
(Write your final corrected SQLite query here)
```
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0 
    )

    full_response = response.choices[0].message.content

    match = re.search(r'```sql\s*(.*?)\s*```', full_response, re.DOTALL | re.IGNORECASE)
    if match:
        fixed_sql = match.group(1).strip()
    else:
        fixed_sql = full_response.replace("```sql", "").replace("```", "").strip()
        
    return fixed_sql


# ============================================================
# 메인 루프
# ============================================================

def main():
    print("🚀 자율 수정(Self-healing v2) 에이전트 가동을 시작합니다...")
    print("📌 3-Type Detector: SYNTAX / VALUE_FORMAT / JOIN_PATH + GENERAL 폴백")
    
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
    
    print(f"📋 실패 코퍼스 {len(failed_samples)}건 중 {start_idx}번부터 치유 시작\n")
    
    for idx in tqdm(range(start_idx, len(failed_samples))):
        sample = failed_samples[idx]
        db_id = sample['db_id']
        question = sample['question']
        gold_sql = sample['gold_sql']
        
        current_sql = sample['pred_sql']
        current_exec_ok = sample['error_type'] != "Syntax Error"
        current_exec_msg = sample['error_message']
        
        schema_text = get_db_schema(db_id)
        gold_success, _, gold_df = execute_sql(db_id, gold_sql)
        
        turn = 0
        max_turns = 3
        is_success = False
        turn_log = []
        
        while turn < max_turns and not is_success:
            
            # 1. 3-Type Detector로 오류 유형 감지
            detected_type, detection_info = detect_error_type(
                db_id, current_sql, current_exec_ok, current_exec_msg, schema_text
            )
            
            # 2. 유형별 맞춤 피드백 생성
            feedback_prompt = build_feedback(
                detected_type, detection_info, db_id, current_sql, current_exec_msg
            )
            
            # 3. LLM에게 수정 요청
            try:
                fixed_sql = call_llm_to_fix_query(question, current_sql, feedback_prompt, schema_text)
            except Exception as e:
                fixed_sql = current_sql
            
            # 4. 자체 검증
            pred_success, exec_msg, pred_df = execute_sql(db_id, fixed_sql)
            
            turn_entry = {
                "turn": turn + 1,
                "detected_type": detected_type,
                "input_sql": current_sql,
                "output_sql": fixed_sql,
                "exec_success": pred_success,
                "exec_message": exec_msg
            }
            
            if not pred_success:
                current_exec_ok = False
                current_exec_msg = exec_msg
                current_sql = fixed_sql
                turn_entry["result"] = "syntax_error"
            else:
                if compare_results(gold_df, pred_df):
                    is_success = True
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
        result['is_healed'] = is_success
        result['turn_log'] = turn_log
        fixed_results.append(result)
        
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
    
    # 감지 유형별 통계
    type_stats = {}
    for r in fixed_results:
        if r.get('turn_log'):
            first_type = r['turn_log'][0].get('detected_type', 'unknown')
            if first_type not in type_stats:
                type_stats[first_type] = {"total": 0, "healed": 0}
            type_stats[first_type]["total"] += 1
            if r.get('is_healed'):
                type_stats[first_type]["healed"] += 1
    
    print("\n" + "=" * 50)
    print("📊 [Self-healing v2 최종 리포트]")
    print(f"실패 코퍼스: {total}건")
    print(f"자율 교정 성공: {healed_count}건 (교정률: {healed_count/total*100:.1f}%)")
    print(f"교정 실패: {total - healed_count}건")
    print(f"턴별 치유 분포: {dict(sorted(healed_by_turn.items()))}")
    
    print(f"\n감지 유형별 교정률 (첫 턴 기준):")
    for t, s in sorted(type_stats.items(), key=lambda x: -x[1]['total']):
        rate = s['healed'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {t}: {s['healed']}/{s['total']} ({rate:.1f}%)")
    
    for diff in ['simple', 'moderate', 'challenging']:
        diff_total = sum(1 for r in fixed_results if r.get('difficulty') == diff)
        diff_healed = sum(1 for r in fixed_results if r.get('difficulty') == diff and r.get('is_healed'))
        if diff_total > 0:
            print(f"  {diff}: {diff_healed}/{diff_total} ({diff_healed/diff_total*100:.1f}%)")
    
    print("=" * 50)
    print(f"\n✅ 결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()