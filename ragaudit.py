"""ragaudit —— 给检索增强链路做一次端到端体检。

设计原则：
1. **先定位错在哪一段**。检索、拼接、生成三段各自有独立的失效模式，
   归因错了就白调参数。
2. **归因要有证据**。每一段给出它凭什么被判定为有问题。
3. **不做自动调参**。只给那一段的处置建议，改不改是人的决定。

仅依赖 Python 标准库。

用法:
    python ragaudit.py --help
    python ragaudit.py --report rag_report.json
    python ragaudit.py --report rag_report.json --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------
# 三段的判定阈值
# ---------------------------------------------------------------------------

THRESHOLDS = {
    # 检索段
    "retrieval_no_hit": 0.15,        # 召回为空的比例上限
    "retrieval_low_coverage": 0.4,    # 关键片段命中率下限
    "retrieval_too_many": 8,          # 召回数超过这个算噪声过多
    # 拼接段
    "context_truncated": 0.2,         # 上下文被截断的比例上限
    "context_too_long": 0.8,          # 上下文用满预算的比例上限
    "context_dup_ratio": 0.3,         # 上下文重复内容占比上限
    # 生成段
    "answer_no_support": 0.25,        # 答案无依据比例上限
    "answer_too_short": 20,           # 答案短于这个字符数算异常
    "answer_too_long": 4000,
    "refusal_rate": 0.3,              # 拒答率上限
}

STAGES = ("retrieval", "context", "generation")


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_report(path: str) -> list[dict]:
    """读入一次 RAG 链路的逐条记录。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError as e:
        raise ValueError(f"读不了 {path}：{e}") from e

    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败：{e}") from e
        return [x for x in data if isinstance(x, dict)]

    rows: list[dict] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"第 {lineno} 行 JSON 解析失败：{e}") from e
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _f(rec: dict, *names: str, default: float = 0.0) -> float:
    for n in names:
        v = rec.get(n)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    return default


def _b(rec: dict, *names: str, default: bool = False) -> bool:
    for n in names:
        if n in rec:
            return bool(rec[n])
    return default


# ---------------------------------------------------------------------------
# 逐段体检
# ---------------------------------------------------------------------------

def audit_retrieval(rows: list[dict]) -> dict:
    """检索段体检。"""
    total = len(rows)
    no_hit = 0
    low_cov = 0
    too_many = 0
    coverage_sum = 0.0
    count_sum = 0.0
    evidence: list[str] = []

    for r in rows:
        hits = r.get("retrieved") or r.get("hits") or []
        n = len(hits) if isinstance(hits, list) else 0
        count_sum += n
        if n == 0:
            no_hit += 1
        elif n > THRESHOLDS["retrieval_too_many"]:
            too_many += 1

        cov = _f(r, "hit_coverage", "coverage", "gold_coverage")
        coverage_sum += cov
        if n > 0 and cov < THRESHOLDS["retrieval_low_coverage"]:
            low_cov += 1

    no_hit_rate = no_hit / total if total else 0.0
    low_cov_rate = low_cov / total if total else 0.0
    too_many_rate = too_many / total if total else 0.0
    avg_count = count_sum / total if total else 0.0
    avg_cov = coverage_sum / total if total else 0.0

    findings: list[dict] = []
    if no_hit_rate > THRESHOLDS["retrieval_no_hit"]:
        findings.append({
            "issue": "大量查询召回为空",
            "rate": round(no_hit_rate, 4),
            "threshold": THRESHOLDS["retrieval_no_hit"],
            "evidence": f"{no_hit}/{total} 条没有召回到任何片段",
        })
    if low_cov_rate > THRESHOLDS["retrieval_no_hit"]:
        findings.append({
            "issue": "召回到了但没覆盖到关键内容",
            "rate": round(low_cov_rate, 4),
            "threshold": THRESHOLDS["retrieval_low_coverage"],
            "evidence": f"{low_cov} 条的命中覆盖率低于 "
                        f"{THRESHOLDS['retrieval_low_coverage']}",
        })
    if too_many_rate > 0.1:
        findings.append({
            "issue": "召回片段过多，可能是噪声",
            "rate": round(too_many_rate, 4),
            "threshold": 0.1,
            "evidence": f"{too_many} 条召回了超过 "
                        f"{THRESHOLDS['retrieval_too_many']} 个片段",
        })

    return {
        "stage": "retrieval",
        "label": "检索段",
        "findings": findings,
        "metrics": {
            "no_hit_rate": round(no_hit_rate, 4),
            "low_coverage_rate": round(low_cov_rate, 4),
            "avg_hit_count": round(avg_count, 2),
            "avg_coverage": round(avg_cov, 4),
        },
        "advice": _advice_retrieval(findings),
    }


def audit_context(rows: list[dict]) -> dict:
    """拼接段体检。"""
    total = len(rows)
    truncated = 0
    too_long = 0
    dup_rows = 0
    dup_sum = 0.0

    for r in rows:
        if _b(r, "truncated", "is_truncated"):
            truncated += 1
        if _b(r, "context_full", "near_limit"):
            too_long += 1
        d = _f(r, "duplicate_ratio", "dup_ratio")
        dup_sum += d
        if d > THRESHOLDS["context_dup_ratio"]:
            dup_rows += 1

    trunc_rate = truncated / total if total else 0.0
    long_rate = too_long / total if total else 0.0
    dup_rate = dup_rows / total if total else 0.0
    avg_dup = dup_sum / total if total else 0.0

    findings: list[dict] = []
    if trunc_rate > THRESHOLDS["context_truncated"]:
        findings.append({
            "issue": "上下文被截断",
            "rate": round(trunc_rate, 4),
            "threshold": THRESHOLDS["context_truncated"],
            "evidence": f"{truncated}/{total} 条的上下文超出了预算",
        })
    if long_rate > THRESHOLDS["context_too_long"]:
        findings.append({
            "issue": "上下文长期顶满预算",
            "rate": round(long_rate, 4),
            "threshold": THRESHOLDS["context_too_long"],
            "evidence": f"{too_long} 条的上下文用量接近上限",
        })
    if dup_rate > THRESHOLDS["context_dup_ratio"]:
        findings.append({
            "issue": "上下文里重复内容过多",
            "rate": round(dup_rate, 4),
            "threshold": THRESHOLDS["context_dup_ratio"],
            "evidence": f"{dup_rows} 条的重复内容占比超过 "
                        f"{THRESHOLDS['context_dup_ratio']}，平均 "
                        f"{avg_dup:.2f}",
        })

    return {
        "stage": "context",
        "label": "拼接段",
        "findings": findings,
        "metrics": {
            "truncated_rate": round(trunc_rate, 4),
            "near_limit_rate": round(long_rate, 4),
            "duplicate_row_rate": round(dup_rate, 4),
            "avg_duplicate_ratio": round(avg_dup, 4),
        },
        "advice": _advice_context(findings),
    }


def audit_generation(rows: list[dict]) -> dict:
    """生成段体检。"""
    total = len(rows)
    no_support = 0
    too_short = 0
    too_long = 0
    refused = 0
    lens: list[int] = []

    for r in rows:
        answer = str(r.get("answer", r.get("output", "")))
        lens.append(len(answer))
        if not _b(r, "has_support", "grounded", default=True):
            no_support += 1
        if len(answer) < THRESHOLDS["answer_too_short"]:
            too_short += 1
        if len(answer) > THRESHOLDS["answer_too_long"]:
            too_long += 1
        if _b(r, "refused", "refusal"):
            refused += 1

    no_support_rate = no_support / total if total else 0.0
    short_rate = too_short / total if total else 0.0
    long_rate = too_long / total if total else 0.0
    refusal_rate = refused / total if total else 0.0
    avg_len = sum(lens) / len(lens) if lens else 0.0

    findings: list[dict] = []
    if no_support_rate > THRESHOLDS["answer_no_support"]:
        findings.append({
            "issue": "答案找不到依据支撑",
            "rate": round(no_support_rate, 4),
            "threshold": THRESHOLDS["answer_no_support"],
            "evidence": f"{no_support}/{total} 条答案无法对应到任何召回到的片段",
        })
    if refusal_rate > THRESHOLDS["refusal_rate"]:
        findings.append({
            "issue": "拒答率偏高",
            "rate": round(refusal_rate, 4),
            "threshold": THRESHOLDS["refusal_rate"],
            "evidence": f"{refused}/{total} 条直接拒答",
        })
    if short_rate > 0.15:
        findings.append({
            "issue": "异常短的答案偏多",
            "rate": round(short_rate, 4),
            "threshold": 0.15,
            "evidence": f"{too_short} 条答案短于 "
                        f"{THRESHOLDS['answer_too_short']} 字符",
        })
    if long_rate > 0.1:
        findings.append({
            "issue": "异常长的答案偏多",
            "rate": round(long_rate, 4),
            "threshold": 0.1,
            "evidence": f"{too_long} 条答案长于 "
                        f"{THRESHOLDS['answer_too_long']} 字符",
        })

    return {
        "stage": "generation",
        "label": "生成段",
        "findings": findings,
        "metrics": {
            "no_support_rate": round(no_support_rate, 4),
            "refusal_rate": round(refusal_rate, 4),
            "too_short_rate": round(short_rate, 4),
            "avg_answer_len": round(avg_len, 1),
        },
        "advice": _advice_generation(findings),
    }


def _advice_retrieval(findings: list[dict]) -> str:
    if not findings:
        return "检索段没有明显问题，先看别的段。"
    keys = " ".join(f["issue"] for f in findings)
    if "召回为空" in keys:
        return ("先查查询改写与索引覆盖：召回为空时调分块或换查询几乎无效。"
                "确认查询里的关键词确实出现在被索引的内容里。")
    if "覆盖" in keys:
        return ("召回到了但没覆盖到关键内容：优先调分块粒度，"
                "再考虑加一路向量召回作为补充。")
    return "召回片段过多：先降 top-k，或加一层相关性过滤。"


def _advice_context(findings: list[dict]) -> str:
    if not findings:
        return "拼接段没有明显问题，先看别的段。"
    keys = " ".join(f["issue"] for f in findings)
    if "截断" in keys:
        return ("上下文被截断：先查拼接顺序——被截掉的往往是最该被看到的。"
                "把最相关的片段往前排，比加大预算更有效。")
    if "重复" in keys:
        return "重复内容过多：在拼接前去重，同一来源的多次召回合并成一段。"
    return "上下文顶满预算：检查是不是把所有召回都塞进去了，先筛再拼。"


def _advice_generation(findings: list[dict]) -> str:
    if not findings:
        return "生成段没有明显问题，先看别的段。"
    keys = " ".join(f["issue"] for f in findings)
    if "依据" in keys:
        return ("答案没有依据支撑：如果检索段是好的，问题多半出在"
                "「模型没被允许用」或「要求引用」这一步，而不是检索本身。")
    if "拒答" in keys:
        return "拒答率偏高：检查系统提示里是否把「不确定」写成了「必须回答」。"
    return "答案长度异常：检查生成参数与输出格式要求是否匹配。"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def audit(rows: list[dict]) -> dict:
    """整条链路的体检。"""
    stages = [
        audit_retrieval(rows),
        audit_context(rows),
        audit_generation(rows),
    ]

    # 归因：问题最集中的那一段排在最前
    ranked = sorted(
        stages,
        key=lambda s: (-len(s["findings"]),
                       -max([f["rate"] for f in s["findings"]], default=0.0)))
    for i, s in enumerate(ranked, 1):
        s["priority"] = i

    return {
        "sample_count": len(rows),
        "stages": stages,
        "primary_stage": ranked[0]["stage"] if ranked else "",
        "primary_label": ranked[0]["label"] if ranked else "",
        "has_issue": any(s["findings"] for s in stages),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render(result: dict) -> str:
    out: list[str] = []
    out.append("RAG 链路体检")
    out.append("=" * 60)
    out.append(f"样本数：{result['sample_count']}")
    if not result["has_issue"]:
        out.append("")
        out.append("三段都没有发现明显问题。")
        return "\n".join(out)

    out.append(f"主要嫌疑段：{result['primary_label']}")
    out.append("")

    for s in sorted(result["stages"], key=lambda x: x["priority"]):
        out.append(f"【第 {s['priority']} 优先】{s['label']}")
        if not s["findings"]:
            out.append("  未发现问题")
        else:
            for f in s["findings"]:
                out.append(f"  ! {f['issue']}（占比 {f['rate']:.1%}）")
                out.append(f"    证据：{f['evidence']}")
        out.append(f"  建议：{s['advice']}")
        out.append("")

    out.append("注：归因基于可观测指标，不等于因果结论。")
    return "\n".join(out).rstrip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ragaudit",
        description="给检索增强链路做一次端到端体检（仅依赖标准库）",
        epilog=(
            "记录格式（JSONL 或 JSON 数组），字段都可省略：\n"
            "  {\"query\":\"...\",\"retrieved\":[\"a\",\"b\"],\"hit_coverage\":0.8,\n"
            "   \"truncated\":false,\"duplicate_ratio\":0.1,\"answer\":\"...\",\n"
            "   \"has_support\":true,\"refused\":false}\n"
            "  python ragaudit.py --report rag_report.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--report", required=True, help="逐条记录文件")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    try:
        rows = load_report(args.report)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not rows:
        print("没有记录。", file=sys.stderr)
        return 2

    result = audit(rows)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(render(result))
    return 1 if result["has_issue"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
