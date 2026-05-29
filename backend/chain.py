import os
import logging
import json
import re
from operator import itemgetter
from typing import Any, Dict, List, Optional, Sequence

import weaviate
from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.retrieval import HybridWeaviateRetriever, infer_retrieval_filters
from backend.skill2_policy import (
    build_block_response,
    classify_script_mode,
    evaluate_script_quality,
    render_validation_feedback,
    step_char_budgets,
)
from backend.skill1 import (
    coerce_skill1_decision,
    decision_from_dict,
    format_intervention_message,
    format_script_followup_message,
    format_script_ready_message,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.ingest import get_embeddings_model
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatCohere
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    chain,
)
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client

logger = logging.getLogger(__name__)

RESPONSE_TEMPLATE = """\
你是一个谨慎、温和的正念助手。请仅基于检索到的资料回答问题，不要编造。

回答要求：
1) 优先中文，简洁清晰，控制在 6 句以内。
2) 只使用 context 中的信息；如信息不足，请直接说“我暂时无法从资料中确认”。
3) 如果引用到不同资料，可在对应句末标注 [1] [2]。
4) 语气保持支持性，不做医疗诊断。

<context>
{context}
</context>
"""

COHERE_RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

SIMPLE_QA_TEMPLATE = """\
你是正念陪伴助手，正在处理不需要检索的简单问题。
用户负面情绪标记：{negative_emotion}

回答要求：
1) 用中文，简明、温和、可执行。
2) 允许给出 1-3 条短建议。
3) 不做医疗诊断，不夸大承诺。
4) 如用户显著痛苦，先共情并建议寻求现实支持系统。
5) 当负面情绪标记为 true 且用户未明确要求练习时，结尾可温柔补一句邀请：
“如果你愿意，我可以陪你做一段很短的正念练习。”\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

SKILL1_DECISION_TEMPLATE = """\
你是“Skill1 决策器”，负责根据聊天上下文识别路由和槽位。
你必须输出 JSON 对象，禁止输出 markdown、解释、代码块。

路由定义：
- intervene: 有明显自伤/他伤风险，需要安全干预
- simple_qa: 简单支持性对话或无需检索的问题
- retrieval_qa: 需要知识库检索的问题
- script_gen: 用户希望生成正念引导语脚本，或正在补齐脚本需求槽位

请结合“历史对话 + 最新用户输入”，输出：
{{
  "route": "intervene|simple_qa|retrieval_qa|script_gen",
  "risk_level": "low|medium|high",
  "safety_reason": "none|crisis|dangerous_context",
  "negative_emotion": true,
  "state_tags": ["mind:*","body:*","env:*","task:*"],
  "energy_tags": ["dopamine_low","vitality_low","belief_low"],
  "script_requirements": {{
    "duration_min": null,
    "where": null,
    "what": null,
    "core_goal": null
  }}
}}

要求：
1) 如用户出现高危信号或危险场景，必须 route=intervene。
2) 用户仅在倾诉情绪且未明确要练习时，route=simple_qa，不要提前切到 script_gen。
3) core_goal 可以自由文本，如“缓解焦虑”“快速平静下来”。
4) 对不确定字段返回 null，不要编造。
5) 只返回 JSON 对象本身。\
"""

SKILL1_FEWSHOT_EXAMPLES = """\
Few-shot 参考（仅用于学习路由，不要复制文本）：
示例1
输入：我现在真的不想活了，感觉没有意义
输出：{"route":"intervene","risk_level":"high","safety_reason":"crisis","negative_emotion":true,"state_tags":[],"energy_tags":[],"script_requirements":{"duration_min":null,"where":null,"what":null,"core_goal":null}}

示例2
输入：我在开车，有点慌，能带我做冥想吗
输出：{"route":"intervene","risk_level":"medium","safety_reason":"dangerous_context","negative_emotion":true,"state_tags":[],"energy_tags":[],"script_requirements":{"duration_min":null,"where":null,"what":null,"core_goal":null}}

示例3
输入：今天天气不错，但工作有点烦
输出：{"route":"simple_qa","risk_level":"low","safety_reason":"none","negative_emotion":true,"state_tags":["task:overload"],"energy_tags":[],"script_requirements":{"duration_min":null,"where":null,"what":null,"core_goal":null}}

示例4
输入：什么是身体扫描，它和观呼吸有什么区别？
输出：{"route":"retrieval_qa","risk_level":"low","safety_reason":"none","negative_emotion":false,"state_tags":[],"energy_tags":[],"script_requirements":{"duration_min":null,"where":null,"what":null,"core_goal":null}}

示例5
输入：我想做一段睡前冥想，帮我放松入睡
输出：{"route":"script_gen","risk_level":"low","safety_reason":"none","negative_emotion":false,"state_tags":["body:sleep"],"energy_tags":[],"script_requirements":{"duration_min":null,"where":"睡前","what":null,"core_goal":"放松入睡"}}
"""

# Keep this template for future expansion, but current follow-up uses deterministic short prompts.
SCRIPT_REQUIREMENT_FOLLOWUP_TEMPLATE = """\
deterministic_followup_reserved\
"""

SKILL2_SCRIPT_TEMPLATE = """\
你是资深正念引导语脚本作者。请基于已确认槽位生成一段可直接朗读的中文正念引导语。

要求：
1) 只输出脚本文本正文，不要解释、不要标题、不要 JSON。
2) 语气温和、慢节奏、第二人称（“你”）。
3) 安全边界：不做医疗诊断，不承诺治愈，不使用绝对化表达。
4) 必须严格使用以下五个标题并保持顺序：
   【入境】【觉察】【接纳】【充能】【收束】
5) 每个标题下都要有可朗读内容，不能空段落。
6) 可使用“（停顿3秒）”等停顿提示，便于朗读。
7) 禁止表达：
   - 清空头脑、丢掉痛苦、忘掉一切
   - 深入感受、面对痛苦、释放创伤
   - 治愈/治疗/治好/缓解症状/诊断
   - 必须/一定要/不许动/不准睁眼
8) 必须遵守字数预算（来自输入）：总字数范围与各步骤建议字数。
9) 若提供了 RAG 参考片段，可借鉴其表达方式，但不要逐字复制。\
"""

SKILL2_REWRITE_TEMPLATE = """\
你是脚本质量修正器。请根据“原稿+校验反馈”输出一版修正稿。

硬性要求：
1) 只输出修正后的脚本正文，不要解释。
2) 必须保留并按顺序使用五个标题：【入境】【觉察】【接纳】【充能】【收束】。
3) 严格修复校验反馈指出的问题（时长、安全、个性化）。
4) 禁止使用禁用表达和医疗化词汇。\
"""

SCRIPT2_START_KEYWORDS = [
    "开始生成脚本",
    "开始生成",
    "直接生成",
    "现在生成",
    "生成吧",
    "开始吧",
    "继续生成",
    "开始写脚本",
]


client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


class EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return []

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return []


def get_retriever() -> BaseRetriever:
    try:
        weaviate_client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
            startup_period=None,
        )
        return HybridWeaviateRetriever(
            client=weaviate_client,
            embedding_model=get_embeddings_model(),
            class_name=WEAVIATE_DOCS_INDEX_NAME,
            k=6,
            fetch_k=20,
            alpha=0.58,
        )
    except Exception as exc:
        logger.exception("Failed to initialize Weaviate retriever; falling back to empty retriever: %s", exc)
        return EmptyRetriever()


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def stringify_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
    lines: List[str] = []
    for message in chat_history or []:
        human = message.get("human")
        ai = message.get("ai")
        if human:
            lines.append(f"用户: {human}")
        if ai:
            lines.append(f"助手: {ai}")
    return "\n".join(lines) if lines else "(empty)"


def parse_skill1_json(raw_content: Any) -> Dict[str, Any]:
    if isinstance(raw_content, list):
        raw_text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    else:
        raw_text = str(raw_content or "")
    text = raw_text.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    fence_cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if fence_cleaned:
        try:
            parsed = json.loads(fence_cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def is_route(payload: Dict, route: str) -> bool:
    return payload.get("skill1", {}).get("route") == route


def has_missing_script_slots(payload: Dict) -> bool:
    if not is_route(payload, "script_gen"):
        return False
    return bool(payload.get("skill1", {}).get("missing_slots"))


def format_skill1_intervene_response(payload: Dict) -> str:
    skill1 = payload.get("skill1", {}) if isinstance(payload, dict) else {}
    safety_reason = str(skill1.get("safety_reason", "none") or "none")
    return format_intervention_message(reason=safety_reason)


def format_skill1_script_followup_response(payload: Dict) -> str:
    decision = decision_from_dict(payload.get("skill1", {}))
    return format_script_followup_message(decision)


def format_skill1_script_ready_response(payload: Dict) -> str:
    decision = decision_from_dict(payload.get("skill1", {}))
    return format_script_ready_message(decision)


def should_generate_skill2(payload: Dict) -> bool:
    return is_route(payload, "script_gen") and not has_missing_script_slots(payload)


def apply_retrieval_filters(payload: Dict) -> Dict:
    query = str(payload.get("question", "") or "")
    skill1 = payload.get("skill1", {})
    filters = infer_retrieval_filters(query=query, skill1=skill1)
    return {
        **payload,
        "retrieval_filters": filters,
    }


def fetch_script_docs_with_fallback(
    retriever_obj: BaseRetriever,
    query: str,
    filters: Dict[str, str],
    limit: int = 4,
) -> List[Document]:
    search_fn = getattr(retriever_obj, "search", None)
    if callable(search_fn):
        try:
            docs = search_fn(query=query, filters=filters, limit=limit)
            return docs if isinstance(docs, list) else []
        except Exception as exc:
            logger.warning("Skill2 hybrid search failed, fallback to invoke: %s", exc)
    try:
        docs = retriever_obj.invoke(query) or []
        return docs if isinstance(docs, list) else []
    except Exception as exc:
        logger.warning("Skill2 retriever invoke failed: %s", exc)
        return []


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    if isinstance(retriever, HybridWeaviateRetriever):
        def retrieve_with_dynamic_filters(payload: Dict) -> List[Document]:
            query = str(payload.get("question", "") or "")
            filters = payload.get("retrieval_filters", {}) or {}
            return retriever.search(query=query, filters=filters, limit=6)

        retriever_chain = RunnableLambda(retrieve_with_dynamic_filters).with_config(run_name="FindDocs")
    else:
        retriever_chain = create_retriever_chain(
            llm,
            retriever,
        ).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    retrieval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    default_response_synthesizer = retrieval_prompt | llm

    simple_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SIMPLE_QA_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    simple_response_synthesizer = (simple_prompt | llm | StrOutputParser()).with_config(
        run_name="SimpleQAResponse"
    )

    skill1_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SKILL1_DECISION_TEMPLATE),
            (
                "human",
                "{fewshot_examples}\n\n历史对话：\n{chat_history_text}\n\n最新用户输入：\n{question}",
            ),
        ]
    )

    def attach_skill1_decision(request: Dict, config: Optional[Dict] = None) -> Dict:
        question = request.get("question", "")
        chat_history = request.get("chat_history")
        raw_decision: Dict[str, Any] = {}
        try:
            prompt_messages = skill1_prompt.format_messages(
                fewshot_examples=SKILL1_FEWSHOT_EXAMPLES,
                chat_history_text=stringify_chat_history(chat_history),
                question=question,
            )
            llm_response = llm.invoke(prompt_messages, config=config)
            raw_decision = parse_skill1_json(getattr(llm_response, "content", ""))
        except Exception as exc:
            logger.warning("Skill1 LLM decision failed, falling back to heuristic parser: %s", exc)
            raw_decision = {}

        decision = coerce_skill1_decision(
            raw=raw_decision,
            question=question,
            chat_history=chat_history,
        )
        return {
            **request,
            "skill1": decision.to_dict(),
        }

    skill2_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SKILL2_SCRIPT_TEMPLATE),
            (
                "human",
                "请根据以下信息生成脚本：\n"
                "- 模式：{script_mode}（full=完整五步；simplify=简化睁眼短呼吸）\n"
                "- 时长（分钟）：{duration_min}\n"
                "- 场景：{where}\n"
                "- 当下状态：{what}\n"
                "- 核心目的：{core_goal}\n"
                "- 状态标签：{state_tags}\n"
                "- 能量标签：{energy_tags}\n\n"
                "- 总字数目标：约{target_chars_total}字\n"
                "- 五步字数预算：{step_char_budgets}\n\n"
                "用户刚刚的确认语：{latest_user_message}\n\n"
                "RAG可参考片段（可选）：\n{script_context}",
            ),
        ]
    )

    def build_skill2_prompt_inputs(payload: Dict) -> Dict[str, Any]:
        decision = decision_from_dict(payload.get("skill1", {}))
        req = decision.script_requirements
        duration_min = req.duration_min or 8
        where = req.where or "未提供"
        what = req.what or "未提供"
        core_goal = req.core_goal or "放松并回到当下"
        state_tags = "、".join(decision.state_tags) if decision.state_tags else "无"
        energy_tags = "、".join(decision.energy_tags) if decision.energy_tags else "无"
        latest_user_message = str(payload.get("question", "") or "")
        history_text = stringify_chat_history(payload.get("chat_history"))

        mode_result = classify_script_mode(
            question=latest_user_message,
            where=where,
            what=what,
            core_goal=core_goal,
            chat_history_text=history_text,
        )
        script_mode = mode_result.get("mode", "full")
        mode_reason = mode_result.get("reason", "safe")

        if script_mode == "block":
            return {
                "script_mode": "block",
                "mode_reason": mode_reason,
                "block_response": build_block_response(mode_reason),
            }

        budgets = step_char_budgets(duration_min=duration_min, mode=script_mode)

        script_query = (
            f"正念引导语 脚本 {duration_min}分钟 "
            f"场景:{where} 状态:{what} 目标:{core_goal} "
            "入境 觉察 接纳 充能 收束"
        )
        script_docs = []
        try:
            # RAG is optional for Skill2: retrieve script chunks when available.
            script_filters = {"track": "skill_script"}
            if where in ["睡眠", "焦虑", "压力", "感恩"]:
                script_filters["scenario"] = where
            script_docs = fetch_script_docs_with_fallback(
                retriever_obj=retriever,
                query=script_query,
                filters=script_filters,
                limit=4,
            )
        except Exception as exc:
            logger.warning("Skill2 RAG retrieval failed, continue without context: %s", exc)
            script_docs = []

        if not isinstance(script_docs, list):
            script_docs = []
        script_context = format_docs(script_docs[:4]) if script_docs else "（无可用参考片段）"

        return {
            "script_mode": script_mode,
            "mode_reason": mode_reason,
            "duration_min": duration_min,
            "where": where,
            "what": what,
            "core_goal": core_goal,
            "state_tags": state_tags,
            "energy_tags": energy_tags,
            "latest_user_message": latest_user_message,
            "script_context": script_context,
            "target_chars_total": sum(budgets.values()),
            "step_char_budgets": (
                f"入境≈{budgets['入境']}字；觉察≈{budgets['觉察']}字；接纳≈{budgets['接纳']}字；"
                f"充能≈{budgets['充能']}字；收束≈{budgets['收束']}字"
            ),
        }

    def generate_skill2_with_validation(payload: Dict, config: Optional[Dict] = None) -> str:
        inputs = build_skill2_prompt_inputs(payload)
        if inputs.get("script_mode") == "block":
            return str(inputs.get("block_response", "当前场景不适合进行该练习。"))

        base_messages = skill2_prompt.format_messages(**inputs)
        base_response = llm.invoke(base_messages, config=config)
        script = str(getattr(base_response, "content", "") or "").strip()
        if not script:
            return "我暂时没能生成脚本。你愿意我再试一次吗？"

        report = evaluate_script_quality(
            script=script,
            duration_min=int(inputs.get("duration_min", 8)),
            mode=str(inputs.get("script_mode", "full")),
            where=str(inputs.get("where", "")),
            what=str(inputs.get("what", "")),
            core_goal=str(inputs.get("core_goal", "")),
        )
        if report.get("pass"):
            return script

        feedback = render_validation_feedback(report)
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SKILL2_REWRITE_TEMPLATE),
                (
                    "human",
                    "原稿：\n{draft}\n\n校验反馈：\n{feedback}\n\n"
                    "修订约束：\n"
                    "- 时长：{duration_min}分钟\n"
                    "- 模式：{script_mode}\n"
                    "- 场景：{where}\n"
                    "- 状态：{what}\n"
                    "- 核心目的：{core_goal}\n"
                    "- 总字数目标：约{target_chars_total}字\n"
                    "- 各步骤字数建议：{step_char_budgets}",
                ),
            ]
        )
        rewrite_messages = rewrite_prompt.format_messages(
            draft=script,
            feedback=feedback,
            duration_min=inputs.get("duration_min", 8),
            script_mode=inputs.get("script_mode", "full"),
            where=inputs.get("where", "未提供"),
            what=inputs.get("what", "未提供"),
            core_goal=inputs.get("core_goal", "未提供"),
            target_chars_total=inputs.get("target_chars_total", 0),
            step_char_budgets=inputs.get("step_char_budgets", ""),
        )
        rewrite_response = llm.invoke(rewrite_messages, config=config)
        rewritten = str(getattr(rewrite_response, "content", "") or "").strip()
        return rewritten or script

    skill2_script_chain = RunnableLambda(generate_skill2_with_validation).with_config(
        run_name="Skill2GenerateWithValidation"
    )

    def build_script_followup_response(payload: Dict) -> str:
        decision = decision_from_dict(payload.get("skill1", {}))
        return format_script_followup_message(decision)

    script_followup_chain = RunnableLambda(build_script_followup_response).with_config(
        run_name="Skill1FollowupDeterministic"
    )

    cohere_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COHERE_RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSequence:
        return cohere_prompt | llm.bind(source_documents=input["docs"])

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="xiaomi_mimo_v2_flash",
            deepseek_r1=default_response_synthesizer,
            mistralai_devstral=default_response_synthesizer,
            deepseek_r1t2_chimera=default_response_synthesizer,
            glm_4_5_air=default_response_synthesizer,
        )
        | StrOutputParser()
    ).with_config(run_name="GenerateResponse")

    retrieval_qa_chain = (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | RunnableLambda(apply_retrieval_filters).with_config(run_name="ApplyRetrievalFilters")
        | context
        | response_synthesizer
    ).with_config(run_name="RetrievalQAResponse")

    def build_simple_qa_inputs(payload: Dict) -> Dict[str, Any]:
        skill1 = payload.get("skill1", {}) if isinstance(payload, dict) else {}
        return {
            "question": payload.get("question", ""),
            "chat_history": serialize_history(payload),
            "negative_emotion": "true" if bool(skill1.get("negative_emotion", False)) else "false",
        }

    simple_qa_chain = (
        RunnableLambda(build_simple_qa_inputs).with_config(run_name="SimpleQABuildInput")
        | simple_response_synthesizer
    ).with_config(run_name="SimpleQABranch")

    return (
        RunnableLambda(attach_skill1_decision).with_config(run_name="Skill1Decision")
        | RunnableBranch(
            (
                RunnableLambda(lambda x: is_route(x, "intervene")).with_config(
                    run_name="RouteInterveneCheck"
                ),
                RunnableLambda(format_skill1_intervene_response).with_config(
                    run_name="InterventionResponse"
                ),
            ),
            (
                RunnableLambda(has_missing_script_slots).with_config(
                    run_name="RouteScriptFollowupCheck"
                ),
                script_followup_chain,
            ),
            (
                RunnableLambda(
                    lambda x: is_route(x, "script_gen")
                    and not has_missing_script_slots(x)
                    and should_generate_skill2(x)
                ).with_config(run_name="RouteSkill2GenerateCheck"),
                skill2_script_chain,
            ),
            (
                RunnableLambda(
                    lambda x: is_route(x, "script_gen")
                    and not has_missing_script_slots(x)
                    and not should_generate_skill2(x)
                ).with_config(run_name="RouteScriptReadyCheck"),
                RunnableLambda(format_skill1_script_ready_response).with_config(
                    run_name="ScriptReady"
                ),
            ),
            (
                RunnableLambda(lambda x: is_route(x, "simple_qa")).with_config(
                    run_name="RouteSimpleQACheck"
                ),
                simple_qa_chain,
            ),
            retrieval_qa_chain,
        ).with_config(run_name="Skill1Router")
    )


gpt_3_5 = ChatOpenAI(
    model="xiaomi/mimo-v2-flash:free",
    temperature=0,
    streaming=True,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
deepseek_r1 = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free",
    temperature=0,
    streaming=True,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
devstral = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    temperature=0,
    streaming=True,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
chimera = ChatOpenAI(
    model="tngtech/deepseek-r1t2-chimera:free",
    temperature=0,
    streaming=True,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
glm_4 = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    temperature=0,
    streaming=True,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
llm = gpt_3_5.configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    default_key="xiaomi_mimo_v2_flash",
    deepseek_r1=deepseek_r1,
    mistralai_devstral=devstral,
    deepseek_r1t2_chimera=chimera,
    glm_4_5_air=glm_4,
).with_fallbacks(
    [deepseek_r1, devstral, chimera, glm_4]
)

retriever = get_retriever()
answer_chain = create_chain(llm, retriever)
