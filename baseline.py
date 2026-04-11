import json
import sqlite3
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 환경변수 로드 및 LLM 클라이언트 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 경로 설정
DATA_PATH = "./data/mini_dev_sqlite.json"
DB_DIR = "./data/dev_databases/"
OUTPUT_DIR = "./results/"

def get_db_schema(db_id):
    """해당 DB의 모든 테이블 생성문(스키마)을 가져옵니다."""
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return ""
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schemas = [row[0] for row in cursor.fetchall() if row[0] is not None]
    conn.close()
    
    return "\n".join(schemas)

def generate_sql_with_llm(question, schema, evidence=""):
    """GPT-4o-mini로 초기 SQL을 생성합니다."""
    
    evidence_section = ""
    if evidence and evidence.strip():
        evidence_section = f"\n[Evidence / Hints]\n{evidence}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a world-class SQLite database expert. Generate a correct SQL query for the given question."
            },
            {
                "role": "user",
                "content": f"""[User Question]
{question}

[Database Schema]
{schema}
{evidence_section}
Generate the correct SQLite query to answer the question.
Return ONLY the SQL query within a markdown block (```sql ... ```). Do not include any explanations."""
            }
        ],
        temperature=0.0
    )
    
    raw = response.choices[0].message.content
    # ```sql ... ``` 블록 추출
    import re
    match = re.search(r'```sql\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.replace("```sql", "").replace("```", "").strip()

def execute_sql(db_id, sql_query):
    """생성된 SQL을 실제 DB에서 실행해보고 에러가 나는지 확인합니다."""
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return True, "Success", df
    except Exception as e:
        return False, str(e), None
    
def compare_results(gold_df, pred_df):
    if gold_df is None or pred_df is None:
        return False
    try:
        gold_values = set(tuple(row) for row in gold_df.values.tolist())
        pred_values = set(tuple(row) for row in pred_df.values.tolist())
        return gold_values == pred_values
    except:
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("⏳ 데이터셋을 로드합니다...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_samples = data  # 500문항 전체
    
    # ── 이어서 실행 (resume) 지원 ──
    baseline_path = os.path.join(OUTPUT_DIR, "baseline_results.json")
    all_results = []
    start_idx = 0
    
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        start_idx = len(all_results)
        if start_idx >= len(test_samples):
            print(f"⏭️  이미 완료됨 ({start_idx}건). 건너뜁니다.")
            return
        print(f"📂 기존 결과 {start_idx}건 로드. {start_idx}번부터 이어서 실행합니다.")
    
    failure_corpus = []
    success_count = sum(1 for r in all_results if r.get('ex_pass'))

    print(f"🚀 총 {len(test_samples)}개의 샘플 중 {start_idx}번부터 베이스라인 추론을 시작합니다.\n")
    
    for i in tqdm(range(start_idx, len(test_samples))):
        item = test_samples[i]
        db_id = item['db_id']
        question = item['question']
        gold_sql = item['SQL']  # mini_dev_sqlite.json은 'SQL' 키 사용
        evidence = item.get('evidence', '')
        difficulty = item.get('difficulty', '')
        question_id = item.get('question_id', i)
        
        # 스키마 로드 + LLM으로 SQL 생성
        schema = get_db_schema(db_id)
        try:
            pred_sql = generate_sql_with_llm(question, schema, evidence)
        except Exception as e:
            pred_sql = f"-- LLM ERROR: {e}"
        
        # 1. 정답 SQL 실행
        gold_success, _, gold_df = execute_sql(db_id, gold_sql)
        
        # 2. 예측 SQL 실행
        pred_success, exec_msg, pred_df = execute_sql(db_id, pred_sql)
        
        # 3. 결과 분류
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
        
        result = {
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "ex_pass": ex_pass,
            "error_type": error_type,
            "error_message": exec_msg if error_type == "Syntax Error" else (
                "Result Mismatch (EX Fail)" if error_type else None
            )
        }
        all_results.append(result)
        
        # 중간 저장 (50건마다)
        if (i + 1) % 50 == 0:
            with open(baseline_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            current_success = sum(1 for r in all_results if r.get('ex_pass'))
            print(f"\n  💾 중간 저장 ({len(all_results)}건, 현재 정확도: {current_success/len(all_results)*100:.1f}%)")
            
    # ── 최종 저장 ──
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # 실패 코퍼스 별도 저장
    failure_corpus = [r for r in all_results if not r.get('ex_pass')]
    failure_path = os.path.join(OUTPUT_DIR, "failure_corpus.json")
    with open(failure_path, 'w', encoding='utf-8') as f:
        json.dump(failure_corpus, f, indent=4, ensure_ascii=False)
    
    # 최종 통계
    total = len(all_results)
    success_count = sum(1 for r in all_results if r.get('ex_pass'))
    syntax_errors = sum(1 for r in all_results if r.get('error_type') == 'Syntax Error')
    semantic_errors = sum(1 for r in all_results if r.get('error_type') == 'Semantic Error')
    
    print("\n" + "="*50)
    print("📊 [최종 결과 리포트]")
    print(f"전체 테스트: {total}건")
    print(f"정답 (EX Pass): {success_count}건")
    print(f"실패 (EX Fail): {total - success_count}건")
    print(f"  - Syntax Error: {syntax_errors}건")
    print(f"  - Semantic Error: {semantic_errors}건")
    print(f"현재 모델 정확도(EX): {(success_count/total)*100:.1f}%")
    
    # Difficulty별 분석
    for diff in ['simple', 'moderate', 'challenging']:
        diff_total = sum(1 for r in all_results if r.get('difficulty') == diff)
        diff_pass = sum(1 for r in all_results if r.get('difficulty') == diff and r.get('ex_pass'))
        if diff_total > 0:
            print(f"  {diff}: {diff_pass}/{diff_total} ({diff_pass/diff_total*100:.1f}%)")
    
    print("="*50)
    print(f"\n✅ 전체 결과: {baseline_path}")
    print(f"✅ 실패 코퍼스 ({len(failure_corpus)}건): {failure_path}")

if __name__ == "__main__":
    main()
