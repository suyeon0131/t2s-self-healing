"""
convert_to_bird_format.py - 우리 결과 JSON을 BIRD 공식 평가 포맷으로 변환

BIRD 공식 스크립트가 기대하는 포맷:
  - predicted: JSON dict {"0": "SQL\t----- bird -----\tdb_id", "1": ...}
  - gold:      텍스트 파일, 한 줄에 "SQL\tdb_id"
  - diff:      JSONL 파일, 한 줄에 {"difficulty": "simple", ...}

사용법:
  # 1. gold 파일 + diff(jsonl) 파일 생성 (처음 한 번만)
  python convert_to_bird_format.py --gold --input ./data/mini_dev_sqlite.json

  # 2. baseline 예측 변환
  python convert_to_bird_format.py --input ./results/baseline_results.json --sql_key pred_sql

  # 3. naive 전체 (baseline pass + naive fix 병합)
  python convert_to_bird_format.py --merge --baseline ./results/baseline_results.json --healing ./results/naive_fixed_corpus.json --output ./results/predict_naive.json

  # 4. targeted 전체 (baseline pass + multi-turn fix 병합)
  python convert_to_bird_format.py --merge --baseline ./results/baseline_results.json --healing ./results/multi_turn_healing.json --output ./results/predict_targeted.json
"""

import json
import argparse
import os

SEPARATOR = "\t----- bird -----\t"


def clean_sql(sql):
    """SQL에서 줄바꿈, 탭을 공백으로 치환"""
    return sql.replace('\n', ' ').replace('\t', ' ').strip()


def convert_pred(input_path, sql_key, output_path):
    """단일 JSON → BIRD pred 포맷 (JSON dict)"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for i, item in enumerate(data):
        sql = clean_sql(item[sql_key])
        db_id = item['db_id']
        result[str(i)] = f"{sql}{SEPARATOR}{db_id}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ pred 변환 완료: {output_path} ({len(result)}건)")


def convert_merged(baseline_path, healing_path, healing_sql_key, output_path):
    """baseline pass + healing 결과 병합 → BIRD pred 포맷 (JSON dict)"""
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    with open(healing_path, 'r', encoding='utf-8') as f:
        healing = json.load(f)

    healing_map = {}
    for item in healing:
        qid = item.get('question_id')
        healing_map[qid] = item

    result = {}
    for i, item in enumerate(baseline):
        qid = item.get('question_id')

        if item.get('ex_pass'):
            sql = item['pred_sql']
        elif qid in healing_map:
            sql = healing_map[qid][healing_sql_key]
        else:
            sql = item['pred_sql']

        sql = clean_sql(sql)
        db_id = item['db_id']
        result[str(i)] = f"{sql}{SEPARATOR}{db_id}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 병합 변환 완료: {output_path} ({len(result)}건)")


def convert_gold(input_path, output_dir):
    """mini_dev_sqlite.json → gold.sql (텍스트) + diff.jsonl (JSONL)"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # gold.sql: "SQL\tdb_id" 한 줄씩
    gold_path = os.path.join(output_dir, "gold.sql")
    with open(gold_path, 'w', encoding='utf-8') as f:
        for item in data:
            sql = clean_sql(item['SQL'])
            db_id = item['db_id']
            f.write(f"{sql}\t{db_id}\n")

    # diff.jsonl: 한 줄에 하나의 JSON
    diff_path = os.path.join(output_dir, "diff.jsonl")
    with open(diff_path, 'w', encoding='utf-8') as f:
        for item in data:
            line = json.dumps({
                "question_id": item.get("question_id"),
                "db_id": item["db_id"],
                "difficulty": item.get("difficulty", "simple")
            }, ensure_ascii=False)
            f.write(line + "\n")

    print(f"✅ gold 변환 완료: {gold_path} ({len(data)}건)")
    print(f"✅ diff 변환 완료: {diff_path} ({len(data)}건)")


def main():
    parser = argparse.ArgumentParser(description="BIRD 공식 평가 포맷 변환기")
    parser.add_argument("--input", help="변환할 JSON 파일 경로")
    parser.add_argument("--sql_key", default="pred_sql", help="SQL이 담긴 키 (기본: pred_sql)")
    parser.add_argument("--output", default=None, help="출력 파일 경로")
    parser.add_argument("--output_dir", default="./results", help="gold/diff 출력 디렉토리")

    # gold 모드
    parser.add_argument("--gold", action="store_true", help="gold.sql + diff.jsonl 생성")

    # 병합 모드
    parser.add_argument("--merge", action="store_true", help="baseline + healing 병합 모드")
    parser.add_argument("--baseline", help="병합 모드: baseline 결과 JSON")
    parser.add_argument("--healing", help="병합 모드: healing 결과 JSON")
    parser.add_argument("--healing_key", default="fixed_sql", help="병합 모드: healing SQL 키")

    args = parser.parse_args()

    if args.gold:
        if not args.input:
            print("❌ --gold 모드에서는 --input을 지정해주세요.")
            return
        os.makedirs(args.output_dir, exist_ok=True)
        convert_gold(args.input, args.output_dir)

    elif args.merge:
        if not args.baseline or not args.healing:
            print("❌ --merge 모드에서는 --baseline과 --healing을 모두 지정해야 합니다.")
            return
        output = args.output or args.healing.replace('.json', '_bird.json')
        convert_merged(args.baseline, args.healing, args.healing_key, output)

    else:
        if not args.input:
            print("❌ --input을 지정해주세요.")
            return
        output = args.output or args.input.replace('.json', '_bird.json')
        convert_pred(args.input, args.sql_key, output)


if __name__ == "__main__":
    main()