"""Skill1: demand confirmation and routing for mindfulness assistant."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

Route = Literal["intervene", "simple_qa", "retrieval_qa", "script_gen"]
RiskLevel = Literal["low", "medium", "high"]
SafetyReason = Literal["none", "crisis", "dangerous_context"]


@dataclass
class ScriptRequirements:
    duration_min: Optional[int] = None
    where: Optional[str] = None
    what: Optional[str] = None
    core_goal: Optional[str] = None


@dataclass
class Skill1Decision:
    route: Route
    risk_level: RiskLevel = "low"
    safety_reason: SafetyReason = "none"
    negative_emotion: bool = False
    state_tags: List[str] = field(default_factory=list)
    energy_tags: List[str] = field(default_factory=list)
    script_requirements: ScriptRequirements = field(default_factory=ScriptRequirements)
    missing_slots: List[str] = field(default_factory=list)
    next_question: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


HIGH_RISK_KEYWORDS = [
    "自杀",
    "轻生",
    "不想活",
    "结束生命",
    "活不下去",
    "想死",
    "伤害自己",
    "伤害他人",
    "杀了自己",
    "杀了他",
]

MEDIUM_RISK_KEYWORDS = [
    "崩溃",
    "绝望",
    "撑不住",
    "没有意义",
]

DANGEROUS_CONTEXT_KEYWORDS = [
    "正在开车",
    "我在开车",
    "开车中",
    "驾驶中",
    "高速上",
    "骑车中",
    "过马路",
    "操作机器",
    "高空作业",
]

SCRIPT_INTENT_KEYWORDS = [
    "正念引导语",
    "引导语脚本",
    "冥想脚本",
    "正念脚本",
    "冥想引导",
    "引导我冥想",
    "生成脚本",
    "做一段冥想",
    "带我做冥想",
    "带我练",
    "来一段正念",
    "来一段冥想",
    "做个睡前冥想",
    "放松练习",
]

SCRIPT_ACCEPTANCE_KEYWORDS = [
    "好",
    "好呀",
    "好的",
    "好啊",
    "可以",
    "可以的",
    "行",
    "行的",
    "来吧",
    "开始",
    "开始吧",
    "试试",
    "那就来",
    "再试一次",
    "重试",
]

SCRIPT_CANCEL_KEYWORDS = [
    "取消",
    "不用了",
    "先不",
    "等会",
    "换个问题",
]

PRACTICE_INVITE_MARKERS = [
    "我可以陪你做一段很短的正念练习",
    "我可以陪你做一段正念练习",
    "如果你愿意，我们可以做一段",
    "如果你愿意，我可以带你做",
]

SIMPLE_QA_KEYWORDS = [
    "你好",
    "hi",
    "hello",
    "谢谢",
    "多谢",
    "再见",
    "你是谁",
    "你能做什么",
    "今天天气",
    "天气不错",
    "天气真好",
    "早上好",
    "晚上好",
    "在吗",
]

EMOTIONAL_SUPPORT_KEYWORDS = [
    "不开心",
    "难受",
    "焦虑",
    "紧张",
    "烦",
    "烦躁",
    "压力大",
    "压力很大",
    "低落",
    "郁闷",
    "心累",
    "沮丧",
    "委屈",
    "失落",
    "情绪很差",
    "很糟",
    "崩溃",
]

CHITCHAT_KEYWORDS = [
    "今天天气",
    "天气真好",
    "天气不错",
    "下雨了",
    "吃了吗",
    "早上好",
    "晚上好",
    "午安",
    "晚安",
    "在吗",
    "哈哈",
]

KNOWLEDGE_QUERY_HINTS = [
    "什么是",
    "是什么",
    "定义",
    "原理",
    "机制",
    "区别",
    "文档",
    "资料",
    "出处",
    "引用",
    "根据",
    "论文",
    "科学依据",
    "研究",
    "为什么",
    "怎么理解",
    "介绍一下",
    "有何区别",
    "有什么区别",
    "mbsr",
    "mbct",
]

CORE_GOAL_KEYWORDS = {
    "身体扫描": ["身体扫描", "body scan", "扫身体"],
    "呼吸觉察": ["呼吸觉察", "观呼吸", "breath awareness"],
    "念头觉察": ["念头觉察", "观察念头", "观察想法", "思绪很多"],
    "内心接纳": ["内心接纳", "接纳情绪", "不评判", "自我接纳"],
    "放松入睡": ["放松入睡", "睡前放松", "入睡", "助眠"],
    "缓解焦虑": ["缓解焦虑", "减轻焦虑", "降低焦虑", "平复焦虑"],
    "减压放松": ["减压", "释放压力", "缓解压力", "放松身心"],
    "提升专注": ["提升专注", "专注学习", "专注工作", "进入心流"],
}

CORE_GOAL_SIGNAL_KEYWORDS = [
    "缓解",
    "减轻",
    "降低",
    "释放",
    "减压",
    "放松",
    "平静",
    "稳定",
    "专注",
    "入睡",
    "接纳",
    "觉察",
    "恢复",
    "提神",
    "安心",
]

CORE_GOAL_IGNORE_PHRASES = [
    "生成脚本",
    "做脚本",
    "引导语脚本",
    "冥想脚本",
    "正念脚本",
]

SLOT_QUESTIONS = {
    "duration_min": "你希望这段引导语大概多长时间？（例如 5 分钟 / 10 分钟）",
    "where": "你现在是在什么场景里使用？（例如 通勤路上 / 办公室 / 睡前床上）",
    "what": "你当下正在做什么或刚做完什么？（例如 开会前、加班后、准备睡觉）",
    "core_goal": "这段冥想最想帮你达成什么？（可选：身体扫描 / 呼吸觉察 / 念头觉察 / 内心接纳；也可自由描述，如“缓解焦虑”）",
}


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _safe_strip(text: Optional[str]) -> str:
    return (text or "").strip()


def detect_risk_level(text: str) -> RiskLevel:
    if _contains_any(text, HIGH_RISK_KEYWORDS):
        return "high"
    if _contains_any(text, MEDIUM_RISK_KEYWORDS):
        return "medium"
    return "low"


def detect_safety_reason(text: str) -> SafetyReason:
    if _contains_any(text, HIGH_RISK_KEYWORDS):
        return "crisis"
    if _contains_any(text, DANGEROUS_CONTEXT_KEYWORDS):
        return "dangerous_context"
    return "none"


def detect_script_intent(text: str) -> bool:
    clean = _safe_strip(text)
    if not clean:
        return False
    if _contains_any(clean, SCRIPT_INTENT_KEYWORDS):
        return True
    # Semantic practice request patterns, e.g. "我想做一段睡前冥想"
    pattern = re.compile(
        r"(我想|想|希望|请|帮我|带我|可以).{0,10}(做|来|进行|开始|练|试试).{0,12}(正念|冥想|练习)",
        flags=re.IGNORECASE,
    )
    if pattern.search(clean):
        return True
    return False


def detect_state_tags(text: str) -> List[str]:
    tags: List[str] = []
    mappings = [
        ("mind:greed", ["贪心", "执着", "放不下"]),
        ("mind:anger", ["愤怒", "生气", "烦躁", "暴躁", "怨"]),
        ("mind:delusion", ["迷茫", "混乱", "无明", "看不清"]),
        ("mind:arrogance", ["自负", "傲慢", "看不起"]),
        ("mind:doubt", ["怀疑自己", "犹豫", "反复想", "拿不准"]),
        ("mind:cognitive_distortion", ["灾难化", "非黑即白", "过度概括", "应该化"]),
        ("body:sleep", ["失眠", "睡不着", "睡眠差", "早醒"]),
        ("body:diet", ["暴食", "食欲", "不想吃", "饮食紊乱"]),
        ("body:discomfort", ["头痛", "胸闷", "心慌", "不舒服", "胃痛"]),
        ("body:fatigue", ["疲劳", "很累", "没力气", "精力不足"]),
        ("env:clutter", ["杂乱", "凌乱", "收拾不完"]),
        ("env:noise", ["噪音", "很吵", "嘈杂", "干扰多"]),
        ("env:relationship", ["关系紧张", "吵架", "冷战", "被误解"]),
        ("task:overload", ["待办很多", "做不完", "任务堆积", "压力山大"]),
        ("task:hidden_goal", ["目标不清", "不知道重点", "隐形目标"]),
        ("task:goal_conflict", ["目标冲突", "两难", "进退两难"]),
        ("task:waiting_no_feedback", ["等反馈", "没回复", "石沉大海", "没有进展"]),
    ]
    for tag, keywords in mappings:
        if _contains_any(text, keywords):
            tags.append(tag)
    return tags


def detect_energy_tags(text: str) -> List[str]:
    tags: List[str] = []
    if _contains_any(text, ["没兴趣", "提不起劲", "无聊", "缺乏动力", "没有掌控感"]):
        tags.append("dopamine_low")
    if _contains_any(
        text,
        ["没精神", "没生命力", "麻木", "压抑", "感受不到美", "没有活力"],
    ):
        tags.append("vitality_low")
    if _contains_any(text, ["不自信", "自我否定", "我不行", "没价值", "不配"]):
        tags.append("belief_low")
    return tags


def extract_duration_minutes(text: str) -> Optional[int]:
    match = re.search(r"(\d{1,3})\s*(分钟|分|mins|min)", text, flags=re.IGNORECASE)
    if match:
        value = int(match.group(1))
        if 1 <= value <= 180:
            return value
    if "一刻钟" in text:
        return 15
    if "半小时" in text:
        return 30
    return None


def extract_where(text: str) -> Optional[str]:
    scene_keywords = [
        "通勤路上",
        "地铁上",
        "车上",
        "办公室",
        "工位",
        "图书馆",
        "家里",
        "卧室",
        "床上",
        "睡前",
        "会议室",
        "公园",
    ]
    for item in scene_keywords:
        if item in text:
            return item

    scene_hint_match = re.search(r"(场景|地点)\s*(是|为)?\s*([^，。！？\n]{1,16})", text)
    if scene_hint_match:
        raw = scene_hint_match.group(3).strip()
        cleaned = re.sub(r"(里|上|中|旁|附近|内|外)$", "", raw)
        cleaned = re.sub(r"(自习|学习|工作|开会|休息|冥想|使用)$", "", cleaned)
        if cleaned:
            return cleaned

    direct_at_match = re.search(r"(?:我)?在([^，。！？\n]{1,16})", text)
    if direct_at_match:
        raw = direct_at_match.group(1).strip()
        cleaned = re.sub(r"(里|上|中|旁|附近|内|外)$", "", raw)
        cleaned = re.sub(r"(自习|学习|工作|开会|休息|冥想|使用)$", "", cleaned)
        if cleaned and not cleaned.startswith(
            ("做", "写", "学", "看", "赶", "复习", "处理", "准备", "刷", "聊")
        ):
            return cleaned

    match = re.search(r"在([^，。！？\n]{1,12})(里|上|中|旁|附近)", text)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return None


def extract_what(text: str) -> Optional[str]:
    activity_keywords = [
        "自习",
        "学习",
        "工作",
        "开会前",
        "开会后",
        "加班后",
        "学习前",
        "学习后",
        "准备睡觉",
        "刚吵完架",
        "写报告",
        "刷手机",
        "做家务",
    ]
    for item in activity_keywords:
        if item in text:
            return item

    status_label_match = re.search(
        r"(?:当下状态|状态|现在状态|我现在状态)\s*[:：]\s*([^，。！？\n]{1,24})",
        text,
    )
    if status_label_match:
        return status_label_match.group(1).strip()

    recent_done_match = re.search(
        r"(刚做完|刚完成|刚结束|刚处理完)\s*([^，。！？\n]{1,20})",
        text,
    )
    if recent_done_match:
        return f"{recent_done_match.group(1)}{recent_done_match.group(2)}".strip()

    doing_match = re.search(
        r"(?:正在|在)\s*(做|写|学|看|赶|准备|复习|处理|开会|上课|通勤|运动|休息)\s*([^，。！？\n]{0,16})",
        text,
    )
    if doing_match:
        return f"{doing_match.group(1)}{doing_match.group(2)}".strip()

    scene_activity_match = re.search(
        r"在[^，。！？\n]{1,16}(自习|学习|工作|开会|赶项目|复习|写作业|刷手机|休息)",
        text,
    )
    if scene_activity_match:
        return scene_activity_match.group(1)

    match = re.search(r"(正在|在)([^，。！？\n]{2,18})(时|的时候|中)", text)
    if match:
        return match.group(2)
    return None


def extract_core_goal(text: str) -> Optional[str]:
    def clean_candidate(raw: str) -> str:
        candidate = raw.strip(" ，。！？；：\n\t")
        candidate = re.sub(
            r"^(我想要|我想|想要|希望|想|目标是|目的是|主要是|核心目的是|想让我|为了)",
            "",
            candidate,
        ).strip()
        return candidate

    def is_ignored_candidate(candidate: str) -> bool:
        return any(ignore in candidate for ignore in CORE_GOAL_IGNORE_PHRASES)

    for goal, keywords in CORE_GOAL_KEYWORDS.items():
        if _contains_any(text, keywords):
            return goal

    labeled_goal_match = re.search(
        r"(?:核心目的|目的|目标|想达成)\s*(?:是|为)?\s*[:：]?\s*([^，。！？\n]{2,28})",
        text,
    )
    if labeled_goal_match:
        candidate = clean_candidate(labeled_goal_match.group(1))
        if candidate and not is_ignored_candidate(candidate):
            return candidate

    intent_goal_match = re.search(
        r"(?:我想要|我想|想要|希望|想|为了)\s*([^，。！？\n]{2,28})",
        text,
    )
    if intent_goal_match:
        candidate = clean_candidate(intent_goal_match.group(1))
        if (
            candidate
            and not is_ignored_candidate(candidate)
            and (
                _contains_any(candidate, CORE_GOAL_SIGNAL_KEYWORDS)
                or _contains_any(candidate, ["焦虑", "压力", "紧张", "烦躁", "低落", "情绪"])
            )
        ):
            return candidate

    direct_goal_match = re.search(
        r"(缓解|减轻|降低|释放|稳定|平复)\s*([^，。！？\n]{0,12}(?:焦虑|压力|紧张|烦躁|情绪))",
        text,
    )
    if direct_goal_match:
        return f"{direct_goal_match.group(1)}{direct_goal_match.group(2)}".strip()

    return None


def extract_script_requirements(text: str) -> ScriptRequirements:
    return ScriptRequirements(
        duration_min=extract_duration_minutes(text),
        where=extract_where(text),
        what=extract_what(text),
        core_goal=extract_core_goal(text),
    )


def merge_script_requirements(*requirements: ScriptRequirements) -> ScriptRequirements:
    merged = ScriptRequirements()
    for req in requirements:
        if req.duration_min is not None:
            merged.duration_min = req.duration_min
        if req.where:
            merged.where = req.where
        if req.what:
            merged.what = req.what
        if req.core_goal:
            merged.core_goal = req.core_goal
    return merged


def extract_requirements_from_ai_history(chat_history: Optional[List[Dict[str, str]]]) -> ScriptRequirements:
    req = ScriptRequirements()
    if not chat_history:
        return req

    duration_pattern = re.compile(r"时长[:：]\s*(\d{1,3})\s*分钟")
    where_pattern = re.compile(r"场景[:：]\s*([^；;。]+)")
    what_pattern = re.compile(r"当下状态[:：]\s*([^；;。]+)")
    goal_pattern = re.compile(r"核心目的[:：]\s*([^；;。]+)")

    for message in chat_history:
        ai_text = _safe_strip(message.get("ai"))
        if not ai_text:
            continue

        duration_match = duration_pattern.search(ai_text)
        if duration_match:
            value = int(duration_match.group(1))
            if 1 <= value <= 180:
                req.duration_min = value

        where_match = where_pattern.search(ai_text)
        if where_match:
            value = where_match.group(1).strip()
            if value and value not in ("未确认", "未提供"):
                req.where = value

        what_match = what_pattern.search(ai_text)
        if what_match:
            value = what_match.group(1).strip()
            if value and value not in ("未确认", "未提供"):
                req.what = value

        goal_match = goal_pattern.search(ai_text)
        if goal_match:
            value = goal_match.group(1).strip()
            if value and value not in ("未确认", "未提供"):
                req.core_goal = value

    return req


def get_missing_slots(req: ScriptRequirements) -> List[str]:
    missing: List[str] = []
    if req.duration_min is None:
        missing.append("duration_min")
    if req.where is None:
        missing.append("where")
    if req.what is None:
        missing.append("what")
    if req.core_goal is None:
        missing.append("core_goal")
    return missing


def next_question_for_missing_slots(missing_slots: List[str]) -> str:
    if not missing_slots:
        return ""
    first_missing = missing_slots[0]
    return SLOT_QUESTIONS.get(first_missing, "可以再多说一点你的需求吗？")


def is_simple_qa(text: str) -> bool:
    clean = _safe_strip(text)
    if len(clean) <= 8 and _contains_any(clean, SIMPLE_QA_KEYWORDS):
        return True
    return _contains_any(clean, SIMPLE_QA_KEYWORDS) and "?" not in clean and "？" not in clean


def is_emotional_support_turn(text: str) -> bool:
    clean = _safe_strip(text)
    if not clean:
        return False
    has_emotion = _contains_any(clean, EMOTIONAL_SUPPORT_KEYWORDS)
    asks_knowledge = _contains_any(clean, KNOWLEDGE_QUERY_HINTS)
    return has_emotion and not asks_knowledge


def is_chitchat_turn(text: str) -> bool:
    clean = _safe_strip(text)
    if not clean:
        return False
    if _contains_any(clean, CHITCHAT_KEYWORDS):
        return True
    if len(clean) <= 12 and _contains_any(clean, ["你好", "hi", "hello", "谢谢", "再见"]):
        return True
    return False


def has_negative_emotion(text: str) -> bool:
    return _contains_any(_safe_strip(text), EMOTIONAL_SUPPORT_KEYWORDS)


def is_knowledge_query(text: str) -> bool:
    clean = _safe_strip(text)
    if not clean:
        return False
    if _contains_any(clean, KNOWLEDGE_QUERY_HINTS):
        return True
    if ("?" in clean or "？" in clean) and _contains_any(clean, ["正念", "冥想", "mbsr", "mbct", "身体扫描"]):
        return True
    return False


def has_any_script_slot(req: ScriptRequirements) -> bool:
    return any(
        value is not None
        for value in [req.duration_min, req.where, req.what, req.core_goal]
    )


def is_script_collection_active(chat_history: Optional[List[Dict[str, str]]]) -> bool:
    if not chat_history:
        return False
    last_ai = ""
    for message in reversed(chat_history):
        ai_text = _safe_strip(message.get("ai"))
        if ai_text:
            last_ai = ai_text
            break
    if not last_ai:
        return False
    markers = [
        "我收到你想生成正念引导语脚本",
        "需求确认完成，可以进入 Skill2",
        "回复“开始生成脚本”即可",
        "我先确认一个最关键问题",
        "你希望这段引导语大概多长时间",
        "你现在是在什么场景里使用",
        "你当下正在做什么或刚做完什么",
        "这段冥想最想帮你达成什么",
        "我暂时没能生成脚本",
        "正在生成脚本",
        "收到。",
    ]
    return _contains_any(last_ai, markers)


def is_short_acceptance(text: str) -> bool:
    clean = re.sub(r"\s+", "", _safe_strip(text))
    if not clean:
        return False
    if clean in SCRIPT_ACCEPTANCE_KEYWORDS:
        return True
    if len(clean) <= 6 and any(clean.startswith(k) for k in ["好", "可", "行", "来", "开"]):
        return True
    return False


def did_ai_invite_practice(chat_history: Optional[List[Dict[str, str]]]) -> bool:
    if not chat_history:
        return False
    last_ai = ""
    for message in reversed(chat_history):
        ai_text = _safe_strip(message.get("ai"))
        if ai_text:
            last_ai = ai_text
            break
    if not last_ai:
        return False
    return _contains_any(last_ai, PRACTICE_INVITE_MARKERS)


def summarize_requirements(req: ScriptRequirements) -> str:
    return (
        f"时长：{req.duration_min or '未确认'} 分钟；"
        f"场景：{req.where or '未确认'}；"
        f"当下状态：{req.what or '未确认'}；"
        f"核心目的：{req.core_goal or '未确认'}。"
    )


def format_intervention_message(reason: SafetyReason = "none") -> str:
    if reason == "dangerous_context":
        return (
            "你现在的安全最重要。你提到自己可能处在不适合做正念练习的场景（例如驾驶或操作设备）。"
            "请先专注当前环境，等到停稳并确保安全后，我们再开始。"
            "如果你愿意，等你安全下来后我可以用 1 分钟带你做一个超短放松练习。"
        )
    return (
        "你现在的感受很重要，你不需要一个人扛着。"
        "如果你有立即伤害自己或他人的风险，请立刻联系当地紧急电话。"
        "如果你在美国，可拨打或短信 988（24小时免费）；"
        "如果你在中国，可拨打 120/110 或联系当地心理援助热线。"
        "如果你愿意，我也可以先陪你做 30 秒呼吸：吸气 4 秒，停 2 秒，呼气 6 秒，重复 3 轮。"
    )


def format_script_followup_message(decision: Skill1Decision) -> str:
    if decision.next_question:
        return f"收到。{decision.next_question}"
    return "收到。你可以继续补充场景、状态和核心目的。"


def format_script_ready_message(decision: Skill1Decision) -> str:
    summary = summarize_requirements(decision.script_requirements)
    return f"需求确认完成，正在生成脚本。{summary}"


def decision_from_dict(data: Dict) -> Skill1Decision:
    req_data = data.get("script_requirements", {}) if isinstance(data, dict) else {}
    req = ScriptRequirements(
        duration_min=req_data.get("duration_min"),
        where=req_data.get("where"),
        what=req_data.get("what"),
        core_goal=req_data.get("core_goal"),
    )
    return Skill1Decision(
        route=data.get("route", "retrieval_qa"),
        risk_level=data.get("risk_level", "low"),
        safety_reason=data.get("safety_reason", "none"),
        negative_emotion=bool(data.get("negative_emotion", False)),
        state_tags=data.get("state_tags", []),
        energy_tags=data.get("energy_tags", []),
        script_requirements=req,
        missing_slots=data.get("missing_slots", []),
        next_question=data.get("next_question", ""),
    )


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"[，,、;；\n]+", value)
        return [part.strip() for part in parts if part.strip()]
    return []


def _normalize_route(route: Any) -> Route:
    value = str(route or "").strip().lower()
    route_map = {
        "intervene": "intervene",
        "crisis": "intervene",
        "safety": "intervene",
        "simple_qa": "simple_qa",
        "simple": "simple_qa",
        "chat": "simple_qa",
        "retrieval_qa": "retrieval_qa",
        "retrieval": "retrieval_qa",
        "rag": "retrieval_qa",
        "script_gen": "script_gen",
        "script": "script_gen",
        "generate_script": "script_gen",
    }
    return route_map.get(value, "retrieval_qa")


def _normalize_risk_level(risk_level: Any) -> RiskLevel:
    value = str(risk_level or "").strip().lower()
    risk_map = {
        "high": "high",
        "critical": "high",
        "medium": "medium",
        "mid": "medium",
        "moderate": "medium",
        "low": "low",
        "none": "low",
    }
    return risk_map.get(value, "low")


def _normalize_safety_reason(safety_reason: Any) -> SafetyReason:
    value = str(safety_reason or "").strip().lower()
    reason_map = {
        "none": "none",
        "crisis": "crisis",
        "self_harm": "crisis",
        "dangerous_context": "dangerous_context",
        "driving": "dangerous_context",
    }
    return reason_map.get(value, "none")


def _coerce_duration_minutes(value: Any) -> Optional[int]:
    if isinstance(value, int):
        if 1 <= value <= 180:
            return value
        return None
    if isinstance(value, float):
        int_value = int(value)
        if 1 <= int_value <= 180:
            return int_value
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            int_value = int(stripped)
            if 1 <= int_value <= 180:
                return int_value
        return extract_duration_minutes(stripped)
    return None


def _clean_slot_text(value: Any, max_len: int = 32) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text[:max_len]
    return text


def coerce_skill1_decision(
    raw: Dict[str, Any],
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Skill1Decision:
    history = chat_history or []
    human_history = [msg.get("human", "") for msg in history if msg.get("human")]
    merged_text = "\n".join(human_history + [question]).lower()
    current_text = _safe_strip(question).lower()

    fallback = analyze_skill1(question, chat_history)
    if not isinstance(raw, dict):
        return fallback

    route = _normalize_route(raw.get("route"))
    risk_level = _normalize_risk_level(raw.get("risk_level"))
    safety_reason = _normalize_safety_reason(raw.get("safety_reason"))
    negative_emotion = bool(raw.get("negative_emotion", False)) or fallback.negative_emotion

    # Rule-first safety guardrail: if text has explicit safety triggers, always intervene.
    detected_safety_reason = detect_safety_reason(merged_text)
    if detected_safety_reason != "none":
        route = "intervene"
        safety_reason = detected_safety_reason
        risk_level = "high" if detected_safety_reason == "crisis" else "medium"

    if route in ("retrieval_qa", "simple_qa") and fallback.route == "script_gen":
        route = "script_gen"
    if route == "retrieval_qa" and is_emotional_support_turn(current_text):
        route = "simple_qa"

    state_tags = _coerce_str_list(raw.get("state_tags")) or fallback.state_tags
    energy_tags = _coerce_str_list(raw.get("energy_tags")) or fallback.energy_tags

    raw_req = raw.get("script_requirements", {})
    if not isinstance(raw_req, dict):
        raw_req = {}

    history_req = extract_script_requirements("\n".join(human_history).lower())
    ai_history_req = extract_requirements_from_ai_history(history)
    fallback_req = merge_script_requirements(ai_history_req, history_req, extract_script_requirements(merged_text))
    duration_min = _coerce_duration_minutes(raw_req.get("duration_min"))
    where = _clean_slot_text(raw_req.get("where"))
    what = _clean_slot_text(raw_req.get("what"))
    core_goal = _clean_slot_text(raw_req.get("core_goal"))

    req = merge_script_requirements(
        fallback_req,
        ScriptRequirements(
            duration_min=duration_min,
            where=where,
            what=what,
            core_goal=core_goal,
        ),
    )

    if route == "script_gen":
        missing_slots = get_missing_slots(req)
        next_question = next_question_for_missing_slots(missing_slots)
    else:
        missing_slots = []
        next_question = ""

    return Skill1Decision(
        route=route,
        risk_level=risk_level,
        safety_reason=safety_reason,
        negative_emotion=negative_emotion,
        state_tags=state_tags,
        energy_tags=energy_tags,
        script_requirements=req,
        missing_slots=missing_slots,
        next_question=next_question,
    )


def analyze_skill1(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Skill1Decision:
    history = chat_history or []
    human_history = [msg.get("human", "") for msg in history if msg.get("human")]
    merged_text = "\n".join(human_history + [question]).lower()
    current_text = _safe_strip(question).lower()

    risk_level = detect_risk_level(merged_text)
    safety_reason = detect_safety_reason(merged_text)
    if safety_reason == "crisis":
        risk_level = "high"
    elif safety_reason == "dangerous_context" and risk_level == "low":
        risk_level = "medium"

    state_tags = detect_state_tags(merged_text)
    energy_tags = detect_energy_tags(merged_text)
    negative_emotion = has_negative_emotion(current_text)

    if safety_reason != "none" or risk_level == "high":
        return Skill1Decision(
            route="intervene",
            risk_level=risk_level,
            safety_reason=safety_reason,
            negative_emotion=negative_emotion,
            state_tags=state_tags,
            energy_tags=energy_tags,
        )

    script_collection_active = is_script_collection_active(history)
    invited_to_practice = did_ai_invite_practice(history)
    current_req = extract_script_requirements(current_text)
    cancelled = _contains_any(current_text, SCRIPT_CANCEL_KEYWORDS)
    should_continue_slot_collection = script_collection_active and not cancelled
    accepted_after_invite = invited_to_practice and is_short_acceptance(current_text)
    script_intent = (
        detect_script_intent(current_text)
        or should_continue_slot_collection
        or accepted_after_invite
    )

    if script_intent:
        history_req = extract_script_requirements("\n".join(human_history).lower())
        ai_history_req = extract_requirements_from_ai_history(history)
        current_req_full = extract_script_requirements(merged_text)
        script_requirements = merge_script_requirements(ai_history_req, history_req, current_req_full)
        missing_slots = get_missing_slots(script_requirements)
        return Skill1Decision(
            route="script_gen",
            risk_level=risk_level,
            safety_reason=safety_reason,
            negative_emotion=negative_emotion,
            state_tags=state_tags,
            energy_tags=energy_tags,
            script_requirements=script_requirements,
            missing_slots=missing_slots,
            next_question=next_question_for_missing_slots(missing_slots),
        )

    if is_emotional_support_turn(current_text):
        route: Route = "simple_qa"
    elif is_chitchat_turn(current_text):
        route = "simple_qa"
    elif is_simple_qa(current_text):
        route: Route = "simple_qa"
    elif is_knowledge_query(current_text):
        route = "retrieval_qa"
    else:
        route = "retrieval_qa"

    return Skill1Decision(
        route=route,
        risk_level=risk_level,
        safety_reason=safety_reason,
        negative_emotion=negative_emotion,
        state_tags=state_tags,
        energy_tags=energy_tags,
    )
