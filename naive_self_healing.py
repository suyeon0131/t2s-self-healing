import json
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
INPUT_PATH = "./results/failure_corpus.json"
OUTPUT_PATH = "./results/naive_fixed_corpus.json"

def call_llm_to_fix_query_naive(question, pred_sql, feedback_prompt):
    """GPT-4o-mini를 호출하여 '단순 피드백'만으로 쿼리를 수정하게 합니다."""
    
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

def main():
    print("🚀 Naive 자율 수정(대조군) 파이프라인 가동을 시작합니다...")
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        failed_samples = json.load(f)
    
    # ── 이어서 실행 (resume) 지원 ──
    naive_results = []
    start_idx = 0
    
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            naive_results = json.load(f)
        start_idx = len(naive_results)
        if start_idx >= len(failed_samples):
            print(f"⏭️  이미 완료됨 ({start_idx}건). 건너뜁니다.")
            return
        print(f"📂 기존 결과 {start_idx}건 로드. {start_idx}번부터 이어서 실행합니다.")
    
    print(f"📋 실패 코퍼스 {len(failed_samples)}건 중 {start_idx}번부터 교정 시작\n")
    
    for idx in tqdm(range(start_idx, len(failed_samples))):
        sample = failed_samples[idx]
        question = sample['question']
        pred_sql = sample['pred_sql']
        error_type = sample['error_type']
        error_msg = sample['error_message']
        
        # 🎯 Naïve 피드백 로직: 데이터 단서 일체 제공 안 함
        if error_type == "Syntax Error":
            feedback_prompt = f"A Syntax Error occurred. Execution failed with the following error:\n{error_msg}"
        else: 
            feedback_prompt = "The query executed without syntax errors, but the returned results were incorrect (Result Mismatch). Please rewrite the query to fix the logic."
            
        try:
            fixed_sql = call_llm_to_fix_query_naive(question, pred_sql, feedback_prompt)
        except Exception as e:
            print(f"\n  ⚠️ LLM 호출 실패: {e}")
            fixed_sql = pred_sql  # 실패 시 원본 유지
        
        result = {**sample}
        result['fixed_sql'] = fixed_sql
        result['feedback_used'] = feedback_prompt
        naive_results.append(result)
        
        # 중간 저장 (50건마다)
        if (idx + 1) % 50 == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(naive_results, f, ensure_ascii=False, indent=4)
            print(f"\n  💾 중간 저장 ({len(naive_results)}건)")
        
    # 최종 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(naive_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 대조군 쿼리 교정 완료! '{OUTPUT_PATH}' ({len(naive_results)}건)")
    print("📌 다음 단계: baseline.py로 이 파일을 채점하세요.")

if __name__ == "__main__":
    main()
