---
name: mindfulness_script_generator
version: 1.0.0
description: >
  根据用户当下需求、历史偏好和RAG片段生成可朗读的正念引导语。
  采用五步结构并执行安全分流与三维校验。
inputs:
  required_slots:
    - duration_min
    - where
    - what
    - core_goal
  optional_slots:
    - state_tags
    - energy_tags
    - history_preferences
runtime_contract:
  modes:
    - block
    - simplify
    - full
  output_language: zh-CN
  output_style: warm_slow_second_person
  output_structure:
    - 入境
    - 觉察
    - 接纳
    - 充能
    - 收束
safety_policy:
  block_conditions:
    - 驾驶/骑行/操作机械
    - 行走或移动中
    - 高空或危险环境
    - 自伤/自杀意念
    - 幻觉/妄想等精神病性症状
    - 医生明确禁止冥想
    - 明确创伤/PTSD
    - 重度抑郁急性期
  simplify_conditions:
    - 公共场所
    - 情绪崩溃或近失控
  hard_forbidden:
    - 清空头脑
    - 丢掉痛苦
    - 忘掉一切
    - 深入感受
    - 面对痛苦
    - 释放创伤
    - 治愈/治疗/治好/缓解症状/诊断
    - 必须/一定要/不许动/不准睁眼
quality_checks:
  dimensions:
    - safety
    - professional
    - personalization
  reflexion_retry:
    max_retry: 1
    trigger: any_dimension_below_threshold
---

# Skill Goal
生成“可直接朗读”的正念引导语，并严格遵循安全边界与五步结构。

# Generation Flow
1. 读取槽位：`duration_min / where / what / core_goal`。
2. 安全分流：判定 `block / simplify / full`。
3. 检索规划：按五步分别检索候选话术（入境、觉察、接纳、充能、收束）。
4. 生成脚本：融合用户要素与检索片段，输出五步标题版文本。
5. 三维校验：安全性、专业度、个性化；不达标则按反馈重写一次。

# Five-Step Script Template
## 【入境】 Grounding 安定锚定
- 姿势调整、呼吸落地、环境确认
- 让用户从外界切入当下

## 【觉察】 Sensory Awareness 觉察引导
- 关注呼吸、身体触点、情绪信号
- 只观察，不评价

## 【接纳】 Regulation 情绪接纳
- 容纳不适，放松肌肉与呼吸
- 允许情绪存在，不对抗

## 【充能】 Energy Replenishment 能量补充
- 注入稳定感、可控感、温和力量
- 强化“我正在恢复平衡”的体验

## 【收束】 Closure 平稳收束
- 回到环境与身体边界
- 过渡到现实任务/休息

# Mode-Specific Rules
## block
- 直接拒绝深度引导，输出安全提示与现实支持建议。

## simplify
- 只允许睁眼、短呼吸、轻量锚定。
- 禁止深度放松、长时间静默、闭眼要求。

## full
- 输出完整五步，保证段落清晰、节奏平稳、语言可朗读。

# Retrieval Plan
- 先检索与 `core_goal` 高相关片段（例如：焦虑、睡前、压力）。
- 再检索与 `where/what` 匹配片段（场景化语言）。
- 每一步至少选1条可借鉴句式，避免逐字复制。

# Output Contract
- 只输出脚本正文，不输出解释或JSON。
- 必须包含并按顺序使用五个标题：`【入境】【觉察】【接纳】【充能】【收束】`。
- 字数需匹配时长预算，目标语速按中文朗读估算。

# Validation Rubric
## 安全性
- 无禁用表达，无医疗承诺，无解离风险句式。

## 专业度
- 五步齐全且顺序正确；每步有实质内容；时长匹配。

## 个性化
- 至少体现2个用户要素（场景/状态/核心目标）。
