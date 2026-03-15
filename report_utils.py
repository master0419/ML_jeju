"""
=================================================================
실험 보고서 자동 생성 유틸리티 (report_utils.py)
=================================================================
- 테스트 실행 시 CSV 결과 파일 + MD 분석보고서 자동 생성
- 이전 실행 결과와 비교 분석
- model/history/ 에 보고서 저장, result/ 에 CSV 저장
=================================================================

[사용법]
  from report_utils import generate_experiment_report, find_latest_csv

  report_path = generate_experiment_report(
      results_df=results_df,
      config=CONFIG,
      experiment_name="로그 변환 적용",
      changes_summary=["타겟에 np.log1p() 변환 적용"],
      csv_paths={"모델 비교 CSV": "result/xxx.csv"},
      feature_cols=feature_cols,
  )
=================================================================
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime


# ============================================================
# 유틸리티 함수
# ============================================================

def find_latest_csv(results_dir="./result", prefix="model_comparison"):
    """가장 최근 결과 CSV를 찾습니다.

    Args:
        results_dir: 결과 폴더 경로
        prefix: CSV 파일명 접두사

    Returns:
        가장 최근 CSV 파일 경로 (없으면 None)
    """
    pattern = os.path.join(results_dir, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def find_latest_report(history_dir="./model/history"):
    """가장 최근 분석보고서(MD)를 찾습니다.

    Returns:
        가장 최근 MD 파일 경로 (없으면 None)
    """
    pattern = os.path.join(history_dir, "report_*.md")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _fmt(val, force_int=False):
    """숫자를 읽기 좋게 포맷합니다."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if force_int or abs(val) >= 1e6:
        return f"{val:,.0f}"
    if abs(val) >= 100:
        return f"{val:,.1f}"
    return f"{val:,.2f}"


def _change_badge(pct):
    """변화율에 따른 판정 배지를 반환합니다."""
    if pct < -5:
        return "🟢 대폭 개선"
    elif pct < 0:
        return "✅ 소폭 개선"
    elif pct == 0:
        return "➡️ 동일"
    elif pct < 5:
        return "⚠️ 소폭 악화"
    else:
        return "❌ 대폭 악화"


# ============================================================
# 메인: 분석보고서 생성
# ============================================================

def generate_experiment_report(
    results_df,
    config,
    experiment_name="",
    changes_summary=None,
    timestamp=None,
    csv_paths=None,
    previous_results_df=None,
    previous_csv_path=None,
    feature_cols=None,
):
    """실험 분석보고서(MD)를 생성하고 경로를 반환합니다.

    Args:
        results_df: 현재 모델 비교 결과 DataFrame
                    (columns: Model, MAE, MAPE(%), MSE, RMSE, Time(s))
        config: CONFIG dict
        experiment_name: 실험명 (비어있으면 기본명 사용)
        changes_summary: 변경 사항 리스트
        timestamp: 실행 ID 타임스탬프 (None이면 현재 시각)
        csv_paths: 생성된 CSV 파일 경로 dict
                   예: {"모델 비교 CSV": "result/model_comparison_xxx.csv"}
        previous_results_df: 비교할 이전 결과 DataFrame (None이면 자동 탐색)
        previous_csv_path: 이전 CSV 파일 경로
        feature_cols: 사용된 피처 컬럼 리스트

    Returns:
        report_path: 생성된 보고서 파일 절대 경로
    """
    if changes_summary is None:
        changes_summary = []
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not experiment_name:
        experiment_name = f"실험 실행 ({timestamp})"

    history_dir = config.get("history_dir", "./model/history")
    results_dir = config.get("results_dir", "./result")
    os.makedirs(history_dir, exist_ok=True)

    # --- 이전 결과 자동 탐색 ---
    if previous_results_df is None:
        prev_csv = find_latest_csv(results_dir, "model_comparison")
        if prev_csv:
            try:
                previous_results_df = pd.read_csv(prev_csv)
                previous_csv_path = prev_csv
            except Exception:
                pass

    # --- 정렬 ---
    sorted_results = results_df.sort_values(
        "MAE", ascending=True, na_position="last"
    ).reset_index(drop=True)

    best_row = sorted_results.iloc[0]

    # --- MD 보고서 조립 ---
    L = []  # lines

    # ── 헤더 ──
    L.append(f"# 📊 분석보고서: {experiment_name}")
    L.append("")
    L.append(
        f"> **Run ID**: `{timestamp}` &nbsp;|&nbsp; "
        f"**생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    L.append("")
    L.append("---")
    L.append("")

    # ── 1. 실험 개요 ──
    L.append("## 1. 실험 개요")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|:-----|:---|")
    L.append(f"| 실험명 | {experiment_name} |")
    L.append(f"| 타겟 변수 | `{config.get('target_col', 'N/A')}` |")
    L.append(
        f"| 예측 구간 | {config.get('forecast_start', '?')} ~ "
        f"(+{config.get('forecast_months', '?')}개월) |"
    )
    L.append(f"| 검증셋 | {config.get('valid_months', '?')}개월 |")
    L.append(f"| 슬라이딩 윈도우 | {config.get('window_size', '?')}개월 |")
    n_features = len(feature_cols) if feature_cols else "?"
    L.append(f"| 피처 수 | {n_features}개 |")
    active = [k for k, v in config.get("models", {}).items() if v]
    L.append(f"| 활성 모델 수 | {len(active)}개 |")
    L.append("")

    # ── 2. 변경 사항 ──
    L.append("## 2. 변경 사항")
    L.append("")
    if changes_summary:
        for i, ch in enumerate(changes_summary, 1):
            L.append(f"{i}. {ch}")
    else:
        L.append("- _(기본 실행 — 변경 사항 없음)_")
    L.append("")

    # ── 3. 사용 피처 ──
    L.append("## 3. 사용 피처")
    L.append("")
    if feature_cols:
        chunks = [
            feature_cols[i:i + 5] for i in range(0, len(feature_cols), 5)
        ]
        for chunk in chunks:
            L.append(
                ", ".join(f"`{c}`" for c in chunk)
            )
    else:
        L.append("_(피처 정보 없음)_")
    L.append("")

    # ── 4. 하이퍼파라미터 ──
    L.append("## 4. 모델 하이퍼파라미터")
    L.append("")
    L.append("<details>")
    L.append("<summary>클릭하여 펼치기</summary>")
    L.append("")
    for model_name in active:
        if model_name.startswith("Ensemble"):
            continue
        hp = config.get("model_params", {}).get(model_name, {})
        if hp:
            L.append(f"**{model_name}**")
            L.append("```")
            for k, v in hp.items():
                L.append(f"  {k}: {v}")
            L.append("```")
            L.append("")
    ens_cfg = config.get("ensemble", {})
    if ens_cfg:
        L.append("**Ensemble 설정**")
        L.append("```")
        for k, v in ens_cfg.items():
            L.append(f"  {k}: {v}")
        L.append("```")
        L.append("")
    L.append("</details>")
    L.append("")

    # ── 5. 실험 결과 ──
    L.append("## 5. 실험 결과")
    L.append("")
    L.append("| 순위 | 모델 | MAE | MAPE(%) | MSE | RMSE | Time(s) |")
    L.append("|:----:|------|----:|--------:|----:|-----:|--------:|")

    for rank, (_, row) in enumerate(sorted_results.iterrows(), 1):
        model = row.get("Model", "")
        mae = _fmt(row.get("MAE"), force_int=True)
        mape = (
            f"{row['MAPE(%)']:.2f}"
            if pd.notna(row.get("MAPE(%)"))
            else "N/A"
        )
        mse_val = row.get("MSE")
        mse_str = f"{mse_val:.2e}" if pd.notna(mse_val) else "N/A"
        rmse = _fmt(row.get("RMSE"), force_int=True)
        t = (
            f"{row['Time(s)']:.3f}"
            if pd.notna(row.get("Time(s)"))
            else "N/A"
        )
        trophy = " 🏆" if rank == 1 else ""
        L.append(
            f"| {rank} | **{model}**{trophy} | {mae} | {mape} | "
            f"{mse_str} | {rmse} | {t} |"
        )
    L.append("")
    L.append(
        f"> **최적 모델**: {best_row['Model']}  \n"
        f"> MAE = **{_fmt(best_row['MAE'], True)}** &nbsp;|&nbsp; "
        f"MAPE = **{best_row.get('MAPE(%)', 0):.2f}%**"
    )
    L.append("")

    # ── 6. 이전 대비 비교 ──
    L.append("## 6. 이전 결과 대비 비교")
    L.append("")

    if previous_results_df is not None and len(previous_results_df) > 0:
        if previous_csv_path:
            prev_name = os.path.basename(previous_csv_path)
            L.append(f"**비교 대상**: `{prev_name}`")
            L.append("")

        L.append(
            "| 모델 | 이전 MAE | 현재 MAE | 변화량 | 변화율 | 판정 |"
        )
        L.append(
            "|------|--------:|--------:|-------:|-------:|------|"
        )

        improved = 0
        worsened = 0
        unchanged = 0

        for _, row in sorted_results.iterrows():
            model = row.get("Model", "")
            cur_mae = row.get("MAE")

            prev_row = previous_results_df[
                previous_results_df["Model"] == model
            ]
            if len(prev_row) > 0 and pd.notna(cur_mae):
                prev_mae = prev_row.iloc[0].get("MAE")
                if pd.notna(prev_mae) and prev_mae != 0:
                    diff = cur_mae - prev_mae
                    pct = (diff / prev_mae) * 100
                    badge = _change_badge(pct)
                    sign = "+" if diff > 0 else ""
                    L.append(
                        f"| {model} | {_fmt(prev_mae, True)} | "
                        f"{_fmt(cur_mae, True)} | "
                        f"{sign}{_fmt(diff, True)} | "
                        f"{sign}{pct:.1f}% | {badge} |"
                    )
                    if diff < 0:
                        improved += 1
                    elif diff > 0:
                        worsened += 1
                    else:
                        unchanged += 1
                else:
                    L.append(
                        f"| {model} | N/A | {_fmt(cur_mae, True)} "
                        f"| - | - | 🆕 신규 |"
                    )
            else:
                L.append(
                    f"| {model} | - | {_fmt(cur_mae, True)} "
                    f"| - | - | 🆕 신규 |"
                )

        L.append("")

        # 종합 분석
        L.append("### 종합 분석")
        L.append("")
        L.append(f"- ✅ 개선된 모델: **{improved}개**")
        L.append(f"- ❌ 악화된 모델: **{worsened}개**")
        if unchanged > 0:
            L.append(f"- ➡️ 동일: **{unchanged}개**")

        # 최적 모델 비교
        prev_best = previous_results_df.sort_values("MAE").iloc[0]
        prev_best_mae = prev_best.get("MAE")
        cur_best_mae = best_row.get("MAE")

        if (
            pd.notna(prev_best_mae)
            and pd.notna(cur_best_mae)
            and prev_best_mae != 0
        ):
            best_diff = cur_best_mae - prev_best_mae
            best_pct = (best_diff / prev_best_mae) * 100
            sign = "+" if best_diff > 0 else ""
            L.append("")
            L.append("**최적 모델 변화:**")
            L.append(
                f"- 이전 최적: **{prev_best.get('Model', '?')}** "
                f"(MAE: {_fmt(prev_best_mae, True)})"
            )
            L.append(
                f"- 현재 최적: **{best_row['Model']}** "
                f"(MAE: {_fmt(cur_best_mae, True)})"
            )
            L.append(
                f"- 변화: **{sign}{best_pct:.1f}%** "
                f"({sign}{_fmt(best_diff, True)})"
            )
        L.append("")
    else:
        L.append(
            "> ℹ️ 이전 실행 결과가 없어 비교할 수 없습니다. (첫 실행)"
        )
        L.append("")

    # ── 7. 관련 파일 ──
    L.append("## 7. 관련 파일")
    L.append("")
    if csv_paths:
        for label, fpath in csv_paths.items():
            rel = os.path.relpath(fpath, history_dir).replace("\\", "/")
            L.append(f"- [{label}]({rel})")
    else:
        L.append("_(관련 파일 없음)_")
    L.append("")

    # ── 푸터 ──
    L.append("---")
    L.append(
        f"*자동 생성된 분석보고서입니다. "
        f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})*"
    )

    # --- 저장 ---
    report_filename = f"report_{timestamp}.md"
    report_path = os.path.join(history_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"\n  📝 분석보고서: {os.path.abspath(report_path)}")

    return report_path
