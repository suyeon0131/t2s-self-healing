"""
convert_to_bird_format.py - 우리 결과 JSON을 BIRD 공식 평가 포맷으로 변환

BIRD 공식 포맷: 각 줄에 "SQL\t----- bird -----\tdb_id"
gold SQL 파일도 동일한 포맷

사용법:
  # baseline 결과 변환
  python convert_to_bird_format.py --input ./results/baseline_results.json --sql_key pred_sql

  # naive 결과 변환 (실패 코퍼스 → fixed_sql 기준)
  python convert_to_bird_format.py --input ./results/naive_fixed_corpus.json --sql_key fixed_sql

  # multi-turn 결과 변환
  python convert_to_bird_format.py --input ./results/multi_turn_healing.json --sql_key fixed_sql

  # gold SQL 파일 생성 (평가셋 기준)
  python convert_to_bird_format.py --input ./data/mini_dev_sqlite.json --sql_key SQL --output ./results/gold.sql

  # 전체 결과 변환 (baseline pass + healing 결과 병합)
  python convert_to_bird_format.py --merge --baseline ./results/baseline_results.json --healing ./results/multi_turn_healing.json --output ./results/predict_targeted.sql
"""

import json
import argparse
import os

SEPARATOR = "\t----- bird -----\t"

def convert_single(input_path, sql_key, output_path):
    """단일 JSON 파일을 BIRD 포맷으로 변환"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = []
    for item in data:
        sql = item[sql_key].replace('\n', ' ').replace('\t', ' ').strip()
        db_id = item['db_id']
        lines.append(f"{sql}{SEPARATOR}{db_id}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ 변환 완료: {output_path} ({len(lines)}건)")


def convert_merged(baseline_path, healing_path, healing_sql_key, output_path):
    """baseline pass + healing 결과를 병합하여 전체 500문항 예측 파일 생성
    
    baseline에서 pass한 건 → pred_sql 그대로
    baseline에서 fail한 건 → healing의 fixed_sql 사용
    """
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    with open(healing_path, 'r', encoding='utf-8') as f:
        healing = json.load(f)
    
    # healing 결과를 question_id로 인덱싱
    healing_map = {}
    for item in healing:
        qid = item.get('question_id')
        healing_map[qid] = item
    
    lines = []
    for item in baseline:
        qid = item.get('question_id')
        
        if item.get('ex_pass'):
            # baseline에서 이미 맞힌 건 → 원래 pred_sql 사용
            sql = item['pred_sql']
        elif qid in healing_map:
            # healing 결과가 있으면 → fixed_sql 사용
            sql = healing_map[qid][healing_sql_key]
        else:
            # healing 대상이 아닌 경우 → 원래 pred_sql 유지
            sql = item['pred_sql']
        
        sql = sql.replace('\n', ' ').replace('\t', ' ').strip()
        db_id = item['db_id']
        lines.append(f"{sql}{SEPARATOR}{db_id}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ 병합 변환 완료: {output_path} ({len(lines)}건)")


def main():
    parser = argparse.ArgumentParser(description="BIRD 공식 평가 포맷 변환기")
    parser.add_argument("--input", help="변환할 JSON 파일 경로")
    parser.add_argument("--sql_key", default="pred_sql", help="SQL이 담긴 키 (기본: pred_sql)")
    parser.add_argument("--output", default=None, help="출력 파일 경로 (미지정 시 자동 생성)")
    
    # 병합 모드
    parser.add_argument("--merge", action="store_true", help="baseline + healing 병합 모드")
    parser.add_argument("--baseline", help="병합 모드: baseline 결과 JSON")
    parser.add_argument("--healing", help="병합 모드: healing 결과 JSON")
    parser.add_argument("--healing_key", default="fixed_sql", help="병합 모드: healing SQL 키 (기본: fixed_sql)")
    
    args = parser.parse_args()
    
    if args.merge:
        if not args.baseline or not args.healing:
            print("❌ --merge 모드에서는 --baseline과 --healing을 모두 지정해야 합니다.")
            return
        output = args.output or args.healing.replace('.json', '_bird.sql')
        convert_merged(args.baseline, args.healing, args.healing_key, output)
    else:
        if not args.input:
            print("❌ --input을 지정해주세요.")
            return
        output = args.output or args.input.replace('.json', '_bird.sql')
        convert_single(args.input, args.sql_key, output)


if __name__ == "__main__":
    main()
