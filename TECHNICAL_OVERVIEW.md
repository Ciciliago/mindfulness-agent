# Mindfulness Agent 技术文档

## 1. 项目目标
本项目是一个基于 `LangChain + FastAPI + Next.js + Weaviate` 的正念陪伴 Agent，核心能力包括：

- 基于知识库的问答（RAG）
- Skill1：意图识别与需求确认（多轮槽位收集）
- Skill2：根据已确认需求生成正念引导语脚本
- 安全兜底与可观测（LangSmith trace、反馈）

---

## 2. 当前系统架构

### 2.1 后端
- 入口：`backend/main.py`
- 主链路：`backend/chain.py`
- Skill1 规则与规范化：`backend/skill1.py`
- 向量化与入库：`backend/ingest.py`、`_scripts/import_semantic_chunks_weaviate.py`

### 2.2 前端
- 聊天界面：`frontend/app/components/ChatWindow.tsx`
- 消息气泡/trace：`frontend/app/components/ChatMessageBubble.tsx`
- 调用后端：`/chat/stream_log`（LangServe）

### 2.3 存储
- 向量库：Weaviate（当前为语义 chunk 入库）
- 会话状态：前端内存态 `chat_history`（随请求传后端）
- 运行日志：`.run/backend.log`、`.run/frontend.log`

---

## 3. 核心流程

### 3.1 主路由（Skill1）
每轮请求先经过 Skill1 决策，输出结构化信息：

- `route`: `intervene | simple_qa | retrieval_qa | script_gen`
- `risk_level`
- `state_tags` / `energy_tags`
- `script_requirements`: `duration_min / where / what / core_goal`
- `missing_slots`

当前是 **LLM 结构化识别优先 + 规则兜底**：

1. LLM 依据历史对话和当前输入输出 JSON
2. `coerce_skill1_decision` 做规范化与安全修正
3. 路由到对应分支

### 3.2 四类分支
- `intervene`：危机场景干预回复
- `simple_qa`：无需检索的支持性回应
- `retrieval_qa`：RAG 问答（向量检索）
- `script_gen`：脚本需求收集与脚本生成

### 3.3 Skill2 触发与生成
当 `script_gen` 且槽位齐全后：

- 若用户输入包含“开始生成脚本/直接生成”等触发词 -> 进入 Skill2 生成
- 否则先返回“需求确认完成，可进入 Skill2”

Skill2 会：

1. 读取 Skill1 槽位
2. 可选从 RAG 检索脚本参考片段（`retriever.invoke`）
3. 按模板生成可朗读脚本正文

---

## 4. RAG 数据链路

### 4.1 数据来源（当前仓库内）
- `documents/`（理论资料）
- `data/mindfulness script_raw data/`（脚本原始资料）
- `data/structured data and embeddings  with meta data/`（结构化产物、chunk、embedding）

### 4.2 入库方式（当前）
- 主要使用“语义 chunk”写入 Weaviate
- 检索默认是向量检索（`by_text=False`）
- embedding 模型与向量维度已对齐（1536 维）

---

## 5. 运行与调试

### 5.1 启动/停止
- 启动：`_scripts/dev_up.sh`
- 停止：`_scripts/dev_down.sh`
- 前端：`http://127.0.0.1:3000`
- 后端：`http://127.0.0.1:8080/docs`

### 5.2 常见问题
- `ERR_CONNECTION_REFUSED`：通常是进程退出，重启服务并查看 `.run/*.log`
- 回答 `Hmm, I'm not sure.`：多为检索无相关内容或路由到检索分支但语料不足
- View trace 无响应：前端 `window.open` 异步行为被拦截时需按当前实现先开空白页再跳转

---

## 6. 你给出的设计 vs 当前实现状态

下表按“已实现 / 部分实现 / 未实现”评估。

### 6.1 RAG知识库构建&检索
- 已实现：
  - 多来源资料清洗、chunk、向量化、Weaviate 入库
  - 元数据字段基础透传（如 `source/title`，以及你的 enriched 文件）
  - Skill2 可选使用 RAG 脚本片段
- 部分实现：
  - 元数据过滤检索（已具备数据基础，但主检索链未系统化启用按字段过滤）
- 未实现：
  - **BM25 + 向量混合检索**
  - **Reranker 重排**（如 bge-reranker/cohere rerank）

### 6.2 意图识别&安全防护
- 已实现：
  - 分层思路：LLM 结构化路由 + 规则兜底
  - 四意图路由（危机/闲聊/问答/练习）
  - 闲聊情绪识别并可引导到练习确认流程
- 部分实现：
  - 高危拦截规则存在，但“100% 正则拦截”需补更严格词表、上下文判别和误报控制
- 未实现：
  - 完整 Few-shot 评测集与离线精度报表

### 6.3 引导语生成
- 已实现：
  - 按槽位（时长/场景/状态/目的）生成脚本
  - 可结合 RAG 参考片段
- 部分实现：
  - 五步模板思想已有，但当前 prompt 未强制显式输出“入境-觉察-接纳-充能-收束”标记段
- 未实现：
  - 历史偏好驱动的个性化风格持久化（跨会话）

### 6.4 多维校验与自愈闭环
- 已实现：
  - 基础安全语气约束、LLM fallback 链（模型层）
- 未实现（核心缺口）：
  - 三维评分器（安全/专业/个性化）自动判分
  - Reflexion 反馈回写与自动重试闭环
  - 多级策略化 fallback（按失败类型分流）

### 6.5 三层记忆架构与认知进化
- 已实现：
  - 短期记忆（当前会话 `chat_history`）
- 未实现（核心缺口）：
  - 中期记忆（N次会话）
  - 长期偏好记忆（稳定特征）
  - SQLite + FTS5 记忆存储与检索
  - 反馈驱动的主动反思与偏好固化

---

## 7. 建议的下一阶段落地顺序

建议按“先稳住效果，再做复杂能力”推进：

1. `RAG 检索升级`：混合检索（BM25+向量）+ Reranker  
2. `Skill2 质量闸门`：输出后自动评分，不达标触发重写  
3. `记忆层`：先做 SQLite 中期记忆，再做长期偏好  
4. `安全防护强化`：高危词表、场景模板、审计日志  

---

## 8. 当前版本结论
当前版本已完成从“问答 Demo”向“可用的正念双 Skill Agent”演进：

- Skill1 可多轮确认并结构化槽位
- Skill2 可触发脚本生成并支持可选 RAG 参考

但距离你的完整设计目标，仍缺“混合检索+重排、校验自愈闭环、三层记忆”三大系统模块。
