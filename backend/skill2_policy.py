"""Skill2 policy helpers: safety mode, duration budget, and 3D validation."""
from __future__ import annotations

import re
from typing import Any, Dict, List

FIVE_STEP_HEADERS = ["【入境】", "【觉察】", "【接纳】", "【充能】", "【收束】"]

BLOCK_RULES: List[tuple[str, List[str]]] = [
    ("driving_or_machinery", ["开车", "驾驶", "骑车", "骑行", "操作机械", "机器运转"]),
    ("moving_or_walking", ["走路", "行走中", "跑步中", "移动中", "过马路"]),
    ("dangerous_environment", ["高空", "危险环境", "施工现场", "攀爬中"]),
    ("self_harm", ["自杀", "轻生", "不想活", "结束生命", "伤害自己", "伤害他人"]),
    ("psychosis", ["幻觉", "幻听", "妄想", "被害妄想"]),
    ("doctor_forbidden", ["医生不建议冥想", "医生禁止冥想", "不能做冥想"]),
    ("ptsd", ["ptsd", "创伤后应激", "创伤复现"]),
    ("severe_depression", ["重度抑郁", "急性抑郁", "抑郁发作"]),
]

SIMPLIFY_SCENE_KEYWORDS = [
    "地铁",
    "公交",
    "公共场所",
    "办公室",
    "工位",
    "商场",
    "排队",
]

SIMPLIFY_EMOTION_KEYWORDS = [
    "情绪崩溃",
    "快崩溃",
    "恐慌",
    "失控",
]

FORBIDDEN_PHRASES = [
    "清空头脑",
    "丢掉痛苦",
    "忘掉一切",
    "深入感受",
    "面对痛苦",
    "释放创伤",
    "长时间放空",
    "完全失去意识",
]

FORBIDDEN_MEDICAL_TERMS = ["治愈", "治疗", "治好", "缓解症状", "诊断"]
FORBIDDEN_COMMAND_TERMS = ["必须", "一定要", "不许动", "不准睁眼"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def classify_script_mode(
    question: str,
    where: str,
    what: str,
    core_goal: str,
    chat_history_text: str = "",
) -> Dict[str, str]:
    combined = _normalize(" ".join([question, where, what, core_goal, chat_history_text]))

    for reason, keywords in BLOCK_RULES:
        if any(_normalize(keyword) in combined for keyword in keywords):
            return {"mode": "block", "reason": reason}

    if any(_normalize(keyword) in combined for keyword in SIMPLIFY_SCENE_KEYWORDS):
        return {"mode": "simplify", "reason": "public_space"}

    if any(_normalize(keyword) in combined for keyword in SIMPLIFY_EMOTION_KEYWORDS):
        return {"mode": "simplify", "reason": "emotional_overload"}

    return {"mode": "full", "reason": "safe"}


def build_block_response(reason: str) -> str:
    generic = (
        "现在不适合进行这类正念引导。你的安全最重要，"
        "请先离开危险情境并确保身边有现实支持。"
        "如果你愿意，我可以先用一句话陪你稳定呼吸：吸气4秒，呼气6秒，重复3轮。"
    )
    reason_map = {
        "driving_or_machinery": "你当前可能在驾驶/操作设备中，不适合进行闭眼或深度放松练习。",
        "moving_or_walking": "你当前处于移动中，不适合进行该练习。",
        "dangerous_environment": "你当前环境存在明显风险，请先确保人身安全。",
        "self_harm": "你提到了可能的自伤/他伤风险，建议立刻联系当地紧急支持资源。",
        "psychosis": "你提到了可能的幻觉或妄想体验，建议优先寻求专业医疗支持。",
        "doctor_forbidden": "你提到医生明确不建议冥想，请优先遵循医嘱。",
        "ptsd": "你提到了明确创伤/PTSD线索，当前不适合进行通用引导练习。",
        "severe_depression": "你提到了重度抑郁急性线索，建议优先连接专业支持。",
    }
    lead = reason_map.get(reason, "当前场景不满足安全练习条件。")
    return f"{lead}{generic}"


def duration_targets(duration_min: int, mode: str) -> Dict[str, int]:
    safe_duration = max(1, min(30, int(duration_min or 8)))
    # More conservative speaking speed for Chinese guided narration.
    chars_per_min = 145 if mode == "full" else 125
    target = safe_duration * chars_per_min
    return {
        "target": target,
        "min": int(target * 0.88),
        "max": int(target * 1.12),
        "chars_per_min": chars_per_min,
    }


def step_char_budgets(duration_min: int, mode: str) -> Dict[str, int]:
    targets = duration_targets(duration_min, mode)
    ratios = [0.22, 0.22, 0.22, 0.18, 0.16] if mode == "full" else [0.26, 0.20, 0.22, 0.16, 0.16]
    keys = ["入境", "觉察", "接纳", "充能", "收束"]
    budgets: Dict[str, int] = {}
    for key, ratio in zip(keys, ratios):
        budgets[key] = int(targets["target"] * ratio)
    return budgets


def _extract_section(text: str, header: str, next_header: str | None) -> str:
    if header not in text:
        return ""
    start = text.find(header) + len(header)
    end = len(text) if next_header is None else text.find(next_header, start)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def _contains_any(text: str, candidates: List[str]) -> List[str]:
    normalized = _normalize(text)
    return [item for item in candidates if _normalize(item) in normalized]


def _estimate_minutes_by_chars(text: str, chars_per_min: int) -> float:
    chars = len(re.sub(r"\s+", "", text or ""))
    if chars_per_min <= 0:
        return 0.0
    pause_seconds = 0
    for sec in re.findall(r"停顿\s*(\d{1,2})\s*秒", text or ""):
        try:
            pause_seconds += int(sec)
        except Exception:
            pass
    return (chars / chars_per_min) + (pause_seconds / 60.0)


def _personalization_hits(script: str, where: str, what: str, core_goal: str) -> int:
    normalized = _normalize(script)
    hits = 0
    for slot in [where, what, core_goal]:
        value = (slot or "").strip()
        if not value:
            continue
        token = _normalize(value)
        if len(token) >= 2 and token in normalized:
            hits += 1
    return hits


def evaluate_script_quality(
    script: str,
    duration_min: int,
    mode: str,
    where: str,
    what: str,
    core_goal: str,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "pass": True,
        "safety_score": 100,
        "professional_score": 100,
        "personalization_score": 100,
        "issues": [],
    }

    targets = duration_targets(duration_min, mode)
    char_count = len(re.sub(r"\s+", "", script or ""))
    est_minutes = _estimate_minutes_by_chars(script, targets["chars_per_min"])
    report["char_count"] = char_count
    report["estimated_minutes"] = round(est_minutes, 2)
    report["duration_target"] = targets

    if char_count < targets["min"] or char_count > targets["max"]:
        report["professional_score"] -= 35
        report["issues"].append(
            f"时长不匹配：当前约{char_count}字，目标区间{targets['min']}-{targets['max']}字。"
        )

    hit_forbidden = _contains_any(script, FORBIDDEN_PHRASES + FORBIDDEN_MEDICAL_TERMS + FORBIDDEN_COMMAND_TERMS)
    if hit_forbidden:
        report["safety_score"] -= 55
        report["issues"].append(f"触发禁用表达：{', '.join(hit_forbidden[:6])}")

    positions = []
    for header in FIVE_STEP_HEADERS:
        idx = script.find(header)
        positions.append(idx)
    if any(idx < 0 for idx in positions):
        report["professional_score"] -= 30
        report["issues"].append("缺少五步标题（入境/觉察/接纳/充能/收束）之一。")
    else:
        if positions != sorted(positions):
            report["professional_score"] -= 25
            report["issues"].append("五步顺序错误，必须按入境→觉察→接纳→充能→收束。")
        for i, header in enumerate(FIVE_STEP_HEADERS):
            next_header = FIVE_STEP_HEADERS[i + 1] if i + 1 < len(FIVE_STEP_HEADERS) else None
            section = _extract_section(script, header, next_header)
            if len(re.sub(r"\s+", "", section)) < 24:
                report["professional_score"] -= 10
                report["issues"].append(f"{header}内容过短，建议补充可朗读引导句。")

    if mode == "simplify":
        if "闭上眼" in script or "深度放松" in script:
            report["safety_score"] -= 30
            report["issues"].append("简化场景不得包含闭眼或深度放松指令。")
        if not any(key in script for key in ["睁眼", "轻微", "短呼吸", "保持警觉"]):
            report["professional_score"] -= 15
            report["issues"].append("简化场景应强调睁眼、短呼吸、保持环境感知。")

    p_hits = _personalization_hits(script, where=where, what=what, core_goal=core_goal)
    report["personalization_hits"] = p_hits
    if p_hits == 0:
        report["personalization_score"] -= 40
        report["issues"].append("个性化不足：未体现用户场景/状态/核心目的。")
    elif p_hits == 1:
        report["personalization_score"] -= 20
        report["issues"].append("个性化一般：建议至少体现2个用户要素。")

    if report["safety_score"] < 70 or report["professional_score"] < 70 or report["personalization_score"] < 60:
        report["pass"] = False

    return report


def render_validation_feedback(report: Dict[str, Any]) -> str:
    if report.get("pass"):
        return "校验通过。"
    issues = report.get("issues", [])
    if not issues:
        return "未通过校验，请提升安全性、结构完整性和个性化。"
    return "；".join(issues[:6])
