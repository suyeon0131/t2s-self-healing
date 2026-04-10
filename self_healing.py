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
OUTPUT_PATH = "./results/multi_turn_healing.json"

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
    """에이전트의 결과가 정답과 일치하는지 확인합니다."""
    if gold_df is None or pred_df is None:
        return False
    try:
        return gold_df.values.tolist() == pred_df.values.tolist()
    except:
        return False

def main():
    print("🚀 자율 수정(Self-healing) 에이전트 가동을 시작합니다...")
    
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
        
        # 턴을 진행하며 계속 업데이트될 변수들
        current_sql = sample['pred_sql']
        current_error_type = sample['error_type']
        current_error_msg = sample['error_message']
        
        # 채점을 위해 정답(Gold) 데이터를 미리 뽑아둠
        gold_success, _, gold_df = execute_sql(db_id, gold_sql)
        
        turn = 0
        max_turns = 3
        is_success = False
        turn_log = []
        
        # 최대 3번의 재시도 루프 시작
        while turn < max_turns and not is_success:
            
            # 1. 현재 에러 상태에 따른 동적 템플릿 교체
            if current_error_type == "Syntax Error":
                feedback_prompt = f"A Syntax Error occurred. Please fix the syntax based on the following system error log:\n{current_error_msg}"
            else: 
                used_tables = extract_tables_from_sql(current_sql)
                sample_data_text = get_sample_rows(db_id, used_tables)
                feedback_prompt = f"The query executed without errors but returned incorrect data. Please check the actual database sample rows below and rewrite the query:\n{sample_data_text}"
            
            schema_text = get_db_schema(db_id)
            
            # 2. LLM에게 수정 요청
            try:
                fixed_sql = call_llm_to_fix_query(question, current_sql, feedback_prompt, schema_text)
            except Exception as e:
                fixed_sql = current_sql  # API 에러 시 원본 유지
            
            # 3. 에이전트 내부 자체 검증 (실시간 채점)
            pred_success, exec_msg, pred_df = execute_sql(db_id, fixed_sql)
            
            # 턴 로그 기록
            turn_entry = {
                "turn": turn + 1,
                "error_type_at_entry": current_error_type,
                "input_sql": current_sql,
                "output_sql": fixed_sql,
                "exec_success": pred_success,
                "exec_message": exec_msg
            }
            
            if not pred_success:
                # 고치다가 문법 에러를 낸 경우 → Syntax로 전환
                current_error_type = "Syntax Error"
                current_error_msg = exec_msg
                current_sql = fixed_sql
                turn_entry["result"] = "syntax_error"
            else:
                if compare_results(gold_df, pred_df):
                    is_success = True
                    current_sql = fixed_sql
                    turn_entry["result"] = "healed"
                else:
                    # 결과가 틀린 경우 → Semantic 유지
                    current_error_type = "Semantic Error"
                    current_error_msg = "Result Mismatch"
                    current_sql = fixed_sql
                    turn_entry["result"] = "semantic_error"
            
            turn_log.append(turn_entry)
            turn += 1
            
        # 루프 종료 후 최종 결과 저장
        result = {**sample}
        result['fixed_sql'] = current_sql
        result['turn_used'] = turn
        result['is_healed'] = is_success
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
    
    # 턴별 치유 분포
    healed_by_turn = {}
    for r in fixed_results:
        if r.get('is_healed'):
            t = r.get('turn_used', 0)
            healed_by_turn[t] = healed_by_turn.get(t, 0) + 1
    
    print("\n" + "="*50)
    print("📊 [Self-healing 최종 리포트]")
    print(f"실패 코퍼스: {total}건")
    print(f"자율 교정 성공: {healed_count}건 (교정률: {healed_count/total*100:.1f}%)")
    print(f"교정 실패: {total - healed_count}건")
    print(f"턴별 치유 분포: {dict(sorted(healed_by_turn.items()))}")
    
    # Difficulty별 분석
    for diff in ['simple', 'moderate', 'challenging']:
        diff_total = sum(1 for r in fixed_results if r.get('difficulty') == diff)
        diff_healed = sum(1 for r in fixed_results if r.get('difficulty') == diff and r.get('is_healed'))
        if diff_total > 0:
            print(f"  {diff}: {diff_healed}/{diff_total} ({diff_healed/diff_total*100:.1f}%)")
    
    print("="*50)
    print(f"\n✅ 결과 저장: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
