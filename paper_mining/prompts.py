L1_SYSTEM = """你是一个严谨的多模态论文信息抽取器，只输出 JSON 对象。
任务：基于给定章节文本和随附图片，抽取细粒度“研究问题”和“方法”证据。
要求：
1. 不得编造，所有结论必须能由当前块的文本或图片支持。
2. 研究问题要拆到可操作粒度，例如任务缺口、数据缺口、建模缺口、评估缺口、应用约束。
3. 方法要拆到可复现粒度，例如输入、表示、模块、算法步骤、训练目标、损失、推理流程、实验设置。
4. 对图片证据要说明它支持了什么，不要只写“见图”。
5. 若当前块无关，返回空数组。
6. The confidence field must be a finite JSON number within [0, 1], where 0 denotes the lowest confidence and 1 denotes the highest confidence.
输出 JSON schema：
{
  "chunk_id": "string",
  "section": "string",
  "research_problem_atoms": [
    {
      "id": "RP-局部编号",
      "claim": "细粒度研究问题",
      "problem_type": "task_gap|data_gap|method_gap|evaluation_gap|application_constraint|other",
      "evidence": [{"source": "text|image", "quote_or_visual_cue": "短证据", "explanation": "为什么支持"}],
      "confidence": 0.0
    }
  ],
  "method_atoms": [
    {
      "id": "M-局部编号",
      "claim": "细粒度方法要点",
      "method_type": "architecture|algorithm_step|representation|training_objective|data_processing|evaluation_protocol|implementation_detail|other",
      "inputs": ["..."],
      "outputs": ["..."],
      "evidence": [{"source": "text|image", "quote_or_visual_cue": "短证据", "explanation": "为什么支持"}],
      "confidence": 0.0
    }
  ],
  "cross_modal_notes": ["图文互证或冲突说明"]
}"""


def l1_user_prompt(chunk_id: str, section: str, text: str, image_names: list[str]) -> str:
    return f"""请分析下面论文块。

chunk_id: {chunk_id}
section: {section}
attached_images: {image_names}

章节文本：
```markdown
{text}
```

请严格按 system 中的 JSON schema 输出。"""


L2_SYSTEM = """你是论文级研究问题/方法归并器，只输出 JSON 对象。
任务：把多个章节块中的 L1 抽取结果归并为论文级细粒度结构。
要求：
1. 合并语义重复项，但保留不同证据来源。
2. 建立 research_problem 与 method 的解决关系。
3. 区分作者明确提出的内容、由实验设计隐含的内容、以及不确定内容。
4. 不要引入 L1 中没有证据支持的新事实。
5. The confidence field must be a finite JSON number within [0, 1], where 0 denotes the lowest confidence and 1 denotes the highest confidence.
输出 JSON schema：
{
  "paper_research_problems": [
    {
      "id": "RP1",
      "problem": "细粒度研究问题",
      "problem_type": "string",
      "explicitness": "explicit|implicit|uncertain",
      "evidence_refs": ["chunk_id:local_id"],
      "confidence": 0.0
    }
  ],
  "paper_methods": [
    {
      "id": "M1",
      "method": "细粒度方法",
      "method_type": "string",
      "inputs": ["..."],
      "outputs": ["..."],
      "evidence_refs": ["chunk_id:local_id"],
      "confidence": 0.0
    }
  ],
  "problem_method_links": [
    {
      "problem_id": "RP1",
      "method_id": "M1",
      "relation": "directly_addresses|partially_addresses|evaluates|motivates",
      "rationale": "证据化说明"
    }
  ],
  "unresolved_or_ambiguous": ["证据不足或冲突项"]
}"""


def l2_user_prompt(l1_results_json: str) -> str:
    return f"""下面是逐块 L1 JSON 抽取结果。请进行论文级归并。

```json
{l1_results_json}
```"""


L3_SYSTEM = """你是论文信息抽取质量审校器，只输出 JSON 对象。
任务：审校 L2 结果，生成最终可用于论文实验的高质量抽取结果。
审校准则：
1. 删除没有证据引用的项目。
2. 把过粗的研究问题和方法继续拆细；如果无法拆细，说明原因。
3. 标注每条结果的证据覆盖度与风险。
4. 输出必须是稳定 JSON，不能有解释性散文。
5. The confidence field must be a finite JSON number within [0, 1], where 0 denotes the lowest confidence and 1 denotes the highest confidence.
输出 JSON schema：
{
  "final_research_problems": [
    {
      "id": "RP1",
      "problem": "最终细粒度研究问题",
      "problem_type": "string",
      "granularity": "fine|medium|coarse",
      "explicitness": "explicit|implicit|uncertain",
      "evidence_refs": ["..."],
      "confidence": 0.0,
      "risk_note": "可能风险或空字符串"
    }
  ],
  "final_methods": [
    {
      "id": "M1",
      "method": "最终细粒度方法",
      "method_type": "string",
      "reproducibility_fields": {
        "inputs": ["..."],
        "outputs": ["..."],
        "procedure": ["..."],
        "objective_or_metric": ["..."]
      },
      "granularity": "fine|medium|coarse",
      "evidence_refs": ["..."],
      "confidence": 0.0,
      "risk_note": "可能风险或空字符串"
    }
  ],
  "problem_method_links": [
    {
      "problem_id": "RP1",
      "method_id": "M1",
      "relation": "string",
      "rationale": "证据化说明"
    }
  ],
  "quality_report": {
    "evidence_coverage": "high|medium|low",
    "cross_modal_usage": "high|medium|low",
    "main_limitations": ["..."],
    "recommended_human_checks": ["..."]
  }
}"""


def l3_user_prompt(l2_result_json: str, evidence_index_json: str) -> str:
    return f"""请审校并细化 L2 结果。证据索引用于检查 evidence_refs 是否有来源。

L2:
```json
{l2_result_json}
```

Evidence index:
```json
{evidence_index_json}
```"""


RELATION_COMPLETION_SYSTEM = """You are a conservative scientific relation completion module. Return only one valid JSON object.
Task: Given final research problems, final methods, and existing evidence-supported links, infer a small set of missing problem-method links.
Rules:
1. Do not create new research problems or methods.
2. Only use existing IDs.
3. Do not duplicate existing links.
4. Add a link only when the method is semantically relevant to the problem.
5. Mark every added link as inferred, because it is completed from final item semantics rather than directly extracted evidence.
6. Be conservative: prefer fewer high-quality inferred links over many weak links.
7. The confidence field must be a finite JSON number within [0, 1], where 0 denotes the lowest confidence and 1 denotes the highest confidence.
Output JSON schema:
{
  "inferred_problem_method_links": [
    {
      "problem_id": "RP1",
      "method_id": "M1",
      "relation": "directly_addresses|partially_addresses|evaluates|motivates",
      "link_type": "inferred",
      "confidence": 0.0,
      "rationale": "Brief explanation based only on the given problem and method text."
    }
  ]
}"""


def relation_completion_user_prompt(payload_json: str) -> str:
    return f"""Infer missing problem-method links from the following final extraction.

```json
{payload_json}
```

Return only the requested JSON object."""
#
# English prompt override for journal submission.
# The original experimental prompts were written during Chinese-language
# development. For the journal version, all neural prompt outputs must be in
# English so that generated JSON, visualization screenshots, and the paper text
# are consistent. The definitions below intentionally override common prompt
# factory names used by the pipeline. They are flexible in their arguments so
# that existing pipeline calls remain compatible.

import json as _pm_json
import sys as _pm_sys
import types as _pm_types


_ENGLISH_OUTPUT_RULES = """
Language and JSON requirements:
1. All generated claims, notes, risks, rationales, evidence summaries, merge notes,
   audit notes, and completion notes must be written in English.
2. Keep JSON keys exactly as specified in the schema.
3. Do not output Chinese text. If the source evidence is Chinese, translate the
   meaning faithfully into English in the generated fields.
4. Use only the provided paper evidence. Do not use external knowledge.
5. Return a valid JSON object only. Do not include Markdown or extra commentary.
6. The confidence field must be a finite JSON number within [0, 1], where 0
   denotes the lowest confidence and 1 denotes the highest confidence.
"""


def _pm_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return _pm_json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _pm_payload(*args, **kwargs):
    parts = []
    for i, arg in enumerate(args, 1):
        parts.append(f"<arg_{i}>\n{_pm_to_text(arg)}")
    for key, value in kwargs.items():
        parts.append(f"<{key}>\n{_pm_to_text(value)}")
    return "\n\n".join(parts)


def _pm_l1_prompt(*args, **kwargs):
    return f"""
SYSTEM:
You are a rigorous scientific-paper analyst for computer vision papers. Read one
section-aware multimodal chunk and extract fine-grained research problems,
method components, and local problem-method relations.

{_ENGLISH_OUTPUT_RULES}

Task rules:
1. Use only the provided chunk text and aligned visual context.
2. Prefer fine-grained atoms over coarse summaries.
3. Split compound statements into separate atoms when they contain multiple
   problems, method steps, losses, modules, or protocols.
4. Every research problem, method component, and relation must include at least
   one evidence reference.
5. If a claim is supported by a figure, table, equation, architecture diagram,
   qualitative result, or other visual element, set visual_evidence=true and
   describe the visual cue in English.
6. Evidence references must point to the current chunk or aligned visual
   elements. Do not invent evidence IDs.

Research problem types:
limitation, challenge, gap, objective, motivation, evaluation_need, data_need,
other.

Method component types:
architecture, algorithm_step, representation, training_objective,
data_processing, evaluation_protocol, implementation_detail, other.

Relation types:
addresses, mitigates, implements, evaluates, motivates, supports, other.

Granularity levels:
fine, medium, coarse.

INPUT:
{_pm_payload(*args, **kwargs)}

Return this JSON object:
{{
  "paper_id": "",
  "chunk_id": "",
  "section_title": "",
  "section_path": "",
  "research_problems": [
    {{
      "local_id": "RP-1",
      "claim": "English fine-grained research problem claim.",
      "problem_type": "limitation|challenge|gap|objective|motivation|evaluation_need|data_need|other",
      "granularity": "fine|medium|coarse",
      "evidence_refs": [],
      "evidence_snippets": ["Short English evidence summary."],
      "visual_evidence": false,
      "visual_cues": [],
      "confidence": 0.0,
      "risk": ""
    }}
  ],
  "methods": [
    {{
      "local_id": "M-1",
      "claim": "English fine-grained method component claim.",
      "method_type": "architecture|algorithm_step|representation|training_objective|data_processing|evaluation_protocol|implementation_detail|other",
      "granularity": "fine|medium|coarse",
      "inputs": [],
      "outputs": [],
      "procedure": [],
      "objective_or_metric": [],
      "evidence_refs": [],
      "evidence_snippets": ["Short English evidence summary."],
      "visual_evidence": false,
      "visual_cues": [],
      "confidence": 0.0,
      "risk": ""
    }}
  ],
  "links": [
    {{
      "local_id": "LINK-1",
      "source_problem_local_id": "RP-1",
      "target_method_local_id": "M-1",
      "relation": "addresses|mitigates|implements|evaluates|motivates|supports|other",
      "claim": "English statement explaining the local relation.",
      "evidence_refs": [],
      "evidence_snippets": ["Short English evidence summary."],
      "visual_evidence": false,
      "visual_cues": [],
      "confidence": 0.0
    }}
  ],
  "chunk_notes": {{
    "main_topic": "",
    "contains_visual_reasoning": false,
    "limitations": []
  }}
}}
"""


def _pm_l2_prompt(*args, **kwargs):
    return f"""
SYSTEM:
You are a scientific graph consolidation assistant. Merge L1 local extraction
results into a compact paper-level problem-method graph.

{_ENGLISH_OUTPUT_RULES}

Merge rules:
1. Use only the provided L1 extraction results.
2. Do not create unsupported research problems, methods, or relations.
3. Merge semantically equivalent atoms, but preserve fine-grained distinctions.
4. Do not merge atoms that refer to different gaps, method steps, visual
   evidence, evaluation protocols, or technical scopes.
5. Aggregate evidence references and source local IDs.
6. Remap local links to canonical paper-level IDs.
7. Links produced by this stage must have link_type="evidence_supported".
8. Remove links whose source or target cannot be mapped.

INPUT:
{_pm_payload(*args, **kwargs)}

Return this JSON object:
{{
  "paper_id": "",
  "research_problems": [
    {{
      "id": "RP1",
      "claim": "English canonical research problem claim.",
      "problem_type": "limitation|challenge|gap|objective|motivation|evaluation_need|data_need|other",
      "granularity": "fine|medium|coarse",
      "source_local_ids": [],
      "evidence_refs": [],
      "evidence_snippets": ["Short English evidence summary."],
      "visual_evidence": false,
      "visual_cues": [],
      "confidence": 0.0,
      "risk": ""
    }}
  ],
  "methods": [
    {{
      "id": "M1",
      "claim": "English canonical method component claim.",
      "method_type": "architecture|algorithm_step|representation|training_objective|data_processing|evaluation_protocol|implementation_detail|other",
      "granularity": "fine|medium|coarse",
      "inputs": [],
      "outputs": [],
      "procedure": [],
      "objective_or_metric": [],
      "source_local_ids": [],
      "evidence_refs": [],
      "evidence_snippets": ["Short English evidence summary."],
      "visual_evidence": false,
      "visual_cues": [],
      "confidence": 0.0,
      "risk": ""
    }}
  ],
  "links": [
    {{
      "source": "RP1",
      "target": "M1",
      "relation": "addresses|mitigates|implements|evaluates|motivates|supports|other",
      "claim": "English canonical relation claim.",
      "link_type": "evidence_supported",
      "source_local_link_ids": [],
      "evidence_refs": [],
      "evidence_snippets": ["Short English evidence summary."],
      "visual_evidence": false,
      "visual_cues": [],
      "confidence": 0.0
    }}
  ],
  "merge_notes": {{
    "merge_strategy": "",
    "uncertain_merges": [],
    "removed_links": []
  }}
}}
"""


def _pm_l3_protocol_prompt(*args, **kwargs):
    return f"""
SYSTEM:
You are a deterministic evidence-audit protocol description. In the reported
pipeline, L3 is implemented by code rather than neural generation. If this text
is used for documentation or fallback auditing, all output fields must be in
English.

{_ENGLISH_OUTPUT_RULES}

Audit rules:
1. Validate canonical node IDs.
2. Check that every node has at least one evidence reference resolved by the
   evidence index.
3. Check that every evidence-supported link has existing source and target
   nodes and at least one resolved evidence reference.
4. Preserve inferred links separately and do not count them as direct evidence.
5. Compute method reproducibility from available inputs, outputs, procedure,
   objective_or_metric, and evidence_refs.
6. Remove invalid evidence-supported links.
7. Write audit notes in English.

INPUT:
{_pm_payload(*args, **kwargs)}

Return a valid JSON object containing audited research_problems, methods, links,
and an audit object.
"""


def _pm_relation_completion_prompt(*args, **kwargs):
    return f"""
SYSTEM:
You are a conservative scientific relation completion assistant. Infer only
missing problem-method links that are strongly suggested by the final paper-level
research problem and method nodes.

{_ENGLISH_OUTPUT_RULES}

Completion rules:
1. Use only the final research problem list, method list, and existing
   evidence-supported links.
2. Do not create or modify nodes.
3. Do not duplicate existing evidence-supported links.
4. Add a relation only when the method clearly addresses, mitigates, implements,
   evaluates, supports, or is motivated by the research problem.
5. Every new link must have link_type="inferred" because it is based on final
   node semantics rather than direct chunk evidence.
6. Keep evidence_refs as an empty list for inferred links.
7. Write every rationale and completion note in English.

INPUT:
{_pm_payload(*args, **kwargs)}

Return this JSON object:
{{
  "paper_id": "",
  "inferred_links": [
    {{
      "source": "RP1",
      "target": "M1",
      "relation": "addresses|mitigates|implements|evaluates|motivates|supports|other",
      "claim": "English inferred relation claim.",
      "link_type": "inferred",
      "rationale": "English rationale based only on final node semantics.",
      "evidence_refs": [],
      "confidence": 0.0
    }}
  ],
  "completion_notes": {{
    "strategy": "Conservative semantic relation completion over final problem and method nodes.",
    "skipped_cases": [],
    "limitations": ["Inferred links are not direct evidence-supported links."]
  }}
}}
"""


def build_l1_prompt(*args, **kwargs): return _pm_l1_prompt(*args, **kwargs)
def build_l1_chunk_prompt(*args, **kwargs): return _pm_l1_prompt(*args, **kwargs)
def l1_chunk_prompt(*args, **kwargs): return _pm_l1_prompt(*args, **kwargs)
def l1_local_atom_prompt(*args, **kwargs): return _pm_l1_prompt(*args, **kwargs)
def make_l1_prompt(*args, **kwargs): return _pm_l1_prompt(*args, **kwargs)


def build_l2_prompt(*args, **kwargs): return _pm_l2_prompt(*args, **kwargs)
def build_l2_merge_prompt(*args, **kwargs): return _pm_l2_prompt(*args, **kwargs)
def l2_merge_prompt(*args, **kwargs): return _pm_l2_prompt(*args, **kwargs)
def l2_batch_merge_prompt(*args, **kwargs): return _pm_l2_prompt(*args, **kwargs)
def l2_paper_merge_prompt(*args, **kwargs): return _pm_l2_prompt(*args, **kwargs)
def make_l2_prompt(*args, **kwargs): return _pm_l2_prompt(*args, **kwargs)


def build_l3_prompt(*args, **kwargs): return _pm_l3_protocol_prompt(*args, **kwargs)
def build_l3_audit_prompt(*args, **kwargs): return _pm_l3_protocol_prompt(*args, **kwargs)
def l3_quality_audit_prompt(*args, **kwargs): return _pm_l3_protocol_prompt(*args, **kwargs)
def l3_audit_prompt(*args, **kwargs): return _pm_l3_protocol_prompt(*args, **kwargs)


def build_relation_completion_prompt(*args, **kwargs): return _pm_relation_completion_prompt(*args, **kwargs)
def relation_completion_prompt(*args, **kwargs): return _pm_relation_completion_prompt(*args, **kwargs)
def optional_relation_completion_prompt(*args, **kwargs): return _pm_relation_completion_prompt(*args, **kwargs)


def __getattr__(name):
    lower = name.lower()
    if "relation" in lower and "completion" in lower:
        return _pm_relation_completion_prompt
    if "l1" in lower or "chunk" in lower:
        return _pm_l1_prompt
    if "l2" in lower or "merge" in lower:
        return _pm_l2_prompt
    if "l3" in lower or "audit" in lower:
        return _pm_l3_protocol_prompt
    if "prompt" in lower:
        return _pm_l1_prompt
    raise AttributeError(name)


def _pm_resolve_english_prompt_attribute(name):
    lower = name.lower()
    if lower in {
        "build_l1_prompt",
        "build_l1_chunk_prompt",
        "l1_chunk_prompt",
        "l1_local_atom_prompt",
        "make_l1_prompt",
    }:
        return _pm_l1_prompt
    if lower in {
        "build_l2_prompt",
        "build_l2_merge_prompt",
        "l2_merge_prompt",
        "l2_batch_merge_prompt",
        "l2_paper_merge_prompt",
        "make_l2_prompt",
    }:
        return _pm_l2_prompt
    if lower in {
        "build_l3_prompt",
        "build_l3_audit_prompt",
        "l3_quality_audit_prompt",
        "l3_audit_prompt",
    }:
        return _pm_l3_protocol_prompt
    if lower in {
        "build_relation_completion_prompt",
        "relation_completion_prompt",
        "optional_relation_completion_prompt",
    }:
        return _pm_relation_completion_prompt
    upper = name.upper()
    if name.isupper() and "L1" in upper and "PROMPT" in upper:
        return _pm_l1_prompt()
    if name.isupper() and "L2" in upper and "PROMPT" in upper:
        return _pm_l2_prompt()
    if name.isupper() and "L3" in upper and "PROMPT" in upper:
        return _pm_l3_protocol_prompt()
    if name.isupper() and "RELATION" in upper and "PROMPT" in upper:
        return _pm_relation_completion_prompt()
    if "prompt" in lower and not name.isupper():
        if "relation" in lower and ("completion" in lower or "infer" in lower):
            return _pm_relation_completion_prompt
        if "l3" in lower or "audit" in lower:
            return _pm_l3_protocol_prompt
        if "l2" in lower or "merge" in lower:
            return _pm_l2_prompt
        if "l1" in lower or "chunk" in lower or "atom" in lower or "extract" in lower:
            return _pm_l1_prompt
        return _pm_l1_prompt
    if name.isupper() and "PROMPT" in upper:
        if "RELATION" in upper or "COMPLETION" in upper:
            return _pm_relation_completion_prompt()
        if "L3" in upper or "AUDIT" in upper:
            return _pm_l3_protocol_prompt()
        if "L2" in upper or "MERGE" in upper:
            return _pm_l2_prompt()
        return _pm_l1_prompt()
    return None


class _EnglishPromptModule(_pm_types.ModuleType):
    def __getattribute__(self, name):
        original = super().__getattribute__(name)
        lower = name.lower()
        upper = name.upper()
        is_prompt_attr = "prompt" in lower or (
            name.isupper() and (
                "PROMPT" in upper or "L1" in upper or "L2" in upper or
                "L3" in upper or "RELATION" in upper
            )
        )
        if not is_prompt_attr:
            return original
        if callable(original):
            def _wrapped_prompt(*args, **kwargs):
                return _pm_append_english_output_rule(original(*args, **kwargs))
            _wrapped_prompt.__name__ = getattr(original, "__name__", name)
            _wrapped_prompt.__doc__ = getattr(original, "__doc__", None)
            return _wrapped_prompt
        return _pm_append_english_output_rule(original)


# Disabled after validation: module-level attribute interception changes the
# prompt factory behavior expected by the pipeline and may produce JSON fields
# that do not match deterministic L2/L3 validation. Keep the English helper
# templates above for documentation/fallback use, but let the original prompt
# functions defined below keep their exact signatures and schemas.
_pm_sys.modules[__name__].__class__ = _EnglishPromptModule


def _pm_append_english_output_rule(value):
    """Preserve the original prompt/schema and only add English output rules."""
    suffix = """

IMPORTANT OUTPUT LANGUAGE REQUIREMENT:
- All generated claims, evidence summaries, risks, notes, rationales, merge
  notes, audit notes, completion notes, and textual field values must be written
  in English.
- Preserve all JSON keys, IDs, evidence_refs, source_local_ids, link_type values,
  and schema fields exactly as requested above.
- If the source paper evidence is Chinese, translate its meaning faithfully into
  English in generated natural-language fields.
- Do not output Chinese natural-language text.
- The confidence field must be a finite JSON number within [0, 1], where 0
  denotes the lowest confidence and 1 denotes the highest confidence.
"""
    if isinstance(value, str):
        if "IMPORTANT OUTPUT LANGUAGE REQUIREMENT" in value:
            return value
        return value.rstrip() + suffix
    if isinstance(value, list):
        updated = list(value)
        for index in range(len(updated) - 1, -1, -1):
            item = updated[index]
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                item = dict(item)
                item["content"] = _pm_append_english_output_rule(item["content"])
                updated[index] = item
                return updated
        updated.append({"role": "user", "content": suffix.strip()})
        return updated
    if isinstance(value, dict):
        updated = dict(value)
        for key in ("prompt", "content", "user_prompt", "system_prompt"):
            if isinstance(updated.get(key), str):
                updated[key] = _pm_append_english_output_rule(updated[key])
                return updated
        if isinstance(updated.get("messages"), list):
            updated["messages"] = _pm_append_english_output_rule(updated["messages"])
            return updated
    return value
