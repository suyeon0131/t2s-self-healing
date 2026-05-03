"""
result_shape_analysis.py - 실행 결과 Shape 기반 오류 분해

목적:
  v2 self-healing 실패 케이스(특히 GENERAL 폴백)에서
  gold_df vs pred_df 실행 결과 형태를 비교하여 오류 패턴을 분류한다.

분류 기준:
  EXEC_ERROR       : pred SQL 실행 자체 실패
  EMPTY_RESULT     : pred 0행, gold > 0행
  OVER_SELECT      : pred 컬럼 수 > gold 컬럼 수
  UNDER_SELECT     : pred 컬럼 수 < gold 컬럼 수
  OVER_AGG         : pred 행 수 < gold 행 수 (과집계)
  MISSING_AGG      : pred 행 수 > gold 행 수 (집계 누락)
  SAME_SHAPE_VALUE_DIFF : shape 같은데 값 다름
  STRUCTURAL_MISMATCH   : shape가 완전히 다름

사용법:
  python result_shape_analysis.py

출력:
  results/result_shape_analysis_log.json
  results/result_shape_analysis_stats.txt
"""

import json
import sqlite3
import re
import os
import openpyxl
from tqdm import tqdm

DB_BASE_PATH = "./data/dev_databases"
INPUT_PATH = "./results/multi_turn_healing_v2.json"  # v2 최종 결과
OUTPUT_LOG_PATH = "./results/result_shape_analysis_log.json"
OUTPUT_STATS_PATH = "./results/result_shape_analysis_stats.txt"
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


def extract_main_type(label):
    if not label:
        return 'unknown'
    label = str(label).strip()
    main = label.split('(')[0].strip() if '(' in label else label
    if main == 'COMPLEX':
        match = re.search(r'\(([^)]+)\)', label)
        if match:
            return match.group(1).split(',')[0].strip()
        return 'COMPLEX'
    return main


# ============================================================
# SQL 실행
# ============================================================

def execute_sql(db_id, sql):
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


# ============================================================
# Shape 비교
# ============================================================

def classify_shape(pred_rows, pred_cols, gold_rows, gold_cols):
    """pred vs gold 실행 결과 shape 비교 → 오류 유형 분류"""

    pred_n = len(pred_rows) if pred_rows is not None else 0
    gold_n = len(gold_rows) if gold_rows is not None else 0

    # EMPTY_RESULT
    if pred_n == 0 and gold_n > 0:
        return "EMPTY_RESULT"

    # 컬럼 수 차이
    if pred_cols > gold_cols:
        return "OVER_SELECT"
    if pred_cols < gold_cols:
        return "UNDER_SELECT"

    # 같은 컬럼 수
    if pred_n < gold_n:
        return "OVER_AGG"
    if pred_n > gold_n:
        return "MISSING_AGG"

    # 행/컬럼 수 동일
    if pred_n == gold_n and pred_cols == gold_cols:
        return "SAME_SHAPE_VALUE_DIFF"

    return "STRUCTURAL_MISMATCH"


# ============================================================
# 메인
# ============================================================

def main():
    print("🔍 Result Shape Analysis 시작 (LLM 호출 없음)")

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        v2_results = json.load(f)

    manual_labels = load_manual_labels()
    print(f"📂 v2 결과: {len(v2_results)}건")
    print(f"📂 수작업 라벨: {len(manual_labels)}건\n")

    logs = []

    for r in tqdm(v2_results):
        db_id = r['db_id']
        qid = r.get('question_id')
        is_healed = r.get('is_healed', False)
        gold_sql = r.get('gold_sql', '')
        fixed_sql = r.get('fixed_sql', '')  # v2 최종 출력 SQL

        # detector 감지 유형 (1턴 기준)
        tl = r.get('turn_log', [])
        detected_type = tl[0].get('detected_type', 'unknown') if tl else 'unknown'

        manual_label = manual_labels.get(qid)

        # 교정 성공 케이스는 스킵 (EX pass이므로)
        if is_healed:
            logs.append({
                "question_id": qid,
                "db_id": db_id,
                "difficulty": r.get('difficulty'),
                "is_healed": True,
                "detected_type": detected_type,
                "manual_label": manual_label,
                "shape_class": "HEALED",
            })
            continue

        # gold 실행
        gold_ok, gold_rows, gold_cols, gold_err = execute_sql(db_id, gold_sql)

        # pred (fixed_sql) 실행
        pred_ok, pred_rows, pred_cols, pred_err = execute_sql(db_id, fixed_sql)

        if not pred_ok:
            shape_class = "EXEC_ERROR"
        elif not gold_ok:
            shape_class = "GOLD_EXEC_ERROR"
        else:
            shape_class = classify_shape(pred_rows, pred_cols, gold_rows, gold_cols)

        logs.append({
            "question_id": qid,
            "db_id": db_id,
            "difficulty": r.get('difficulty'),
            "is_healed": False,
            "detected_type": detected_type,
            "manual_label": manual_label,
            "shape_class": shape_class,
            "pred_rows": len(pred_rows) if pred_rows is not None else None,
            "pred_cols": pred_cols,
            "gold_rows": len(gold_rows) if gold_rows is not None else None,
            "gold_cols": gold_cols,
            "pred_error": pred_err,
        })

    # ============================================================
    # 통계
    # ============================================================

    failed = [x for x in logs if not x.get('is_healed')]
    total_failed = len(failed)

    # shape 분포
    shape_counts = {}
    for x in failed:
        sc = x['shape_class']
        shape_counts[sc] = shape_counts.get(sc, 0) + 1

    # detected_type별 shape 분포
    det_shape = {}
    for x in failed:
        dt = x['detected_type']
        sc = x['shape_class']
        if dt not in det_shape:
            det_shape[dt] = {}
        det_shape[dt][sc] = det_shape[dt].get(sc, 0) + 1

    # 수작업 라벨별 shape 분포
    label_shape = {}
    for x in failed:
        label = extract_main_type(x.get('manual_label'))
        sc = x['shape_class']
        if label not in label_shape:
            label_shape[label] = {}
        label_shape[label][sc] = label_shape[label].get(sc, 0) + 1

    shape_classes = [
        "EXEC_ERROR", "EMPTY_RESULT", "OVER_SELECT", "UNDER_SELECT",
        "OVER_AGG", "MISSING_AGG", "SAME_SHAPE_VALUE_DIFF", "STRUCTURAL_MISMATCH",
        "GOLD_EXEC_ERROR"
    ]

    stats_lines = []
    stats_lines.append("=" * 70)
    stats_lines.append("📊 Result Shape Analysis 통계")
    stats_lines.append("=" * 70)
    stats_lines.append(f"전체 v2 결과: {len(logs)}건")
    stats_lines.append(f"교정 성공: {len(logs) - total_failed}건")
    stats_lines.append(f"교정 실패: {total_failed}건\n")

    stats_lines.append("=== Shape 분류 분포 ===")
    for sc in shape_classes:
        cnt = shape_counts.get(sc, 0)
        if cnt > 0:
            stats_lines.append(f"  {sc:<30} {cnt:>5}건 ({cnt/total_failed*100:.1f}%)")

    stats_lines.append("\n=== Detector 유형별 Shape 분포 ===")
    for dt in sorted(det_shape.keys(), key=lambda x: -sum(det_shape[x].values())):
        dt_total = sum(det_shape[dt].values())
        stats_lines.append(f"\n  [{dt}] 총 {dt_total}건")
        for sc in shape_classes:
            cnt = det_shape[dt].get(sc, 0)
            if cnt > 0:
                stats_lines.append(f"    {sc:<30} {cnt:>4}건 ({cnt/dt_total*100:.1f}%)")

    stats_lines.append("\n=== 수작업 라벨별 Shape 분포 ===")
    for label in sorted(label_shape.keys(), key=lambda x: -sum(label_shape[x].values())):
        l_total = sum(label_shape[label].values())
        stats_lines.append(f"\n  [{label}] 총 {l_total}건")
        for sc in shape_classes:
            cnt = label_shape[label].get(sc, 0)
            if cnt > 0:
                stats_lines.append(f"    {sc:<30} {cnt:>4}건 ({cnt/l_total*100:.1f}%)")

    stats_lines.append("\n" + "=" * 70)

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    with open(OUTPUT_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_STATS_PATH, 'w', encoding='utf-8') as f:
        f.write(stats_text)

    print(f"\n✅ 로그: {OUTPUT_LOG_PATH}")
    print(f"✅ 통계: {OUTPUT_STATS_PATH}")


if __name__ == "__main__":
    main()