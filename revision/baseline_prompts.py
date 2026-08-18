FAIR_EXTRACTION_SYSTEM = """You extract an evidence-auditable problem-method graph from one scientific paper. Return one JSON object only.

Use exactly the supplied indexed evidence records. Do not invent evidence IDs. Extract atomic research problems and atomic method components. Each node must cite one or more exact evidence_refs. Each problem-method link must independently cite evidence_refs that support the relationship; endpoint evidence alone is not automatically relation evidence. Use IDs RP1.. and M1... . Do not add inferred links.

Output schema:
{
  "paper_research_problems": [{"id":"RP1","problem":"...","problem_type":"...","explicitness":"explicit|implicit|uncertain","evidence_refs":["S001_C01:RP-1"],"confidence":0.0}],
  "paper_methods": [{"id":"M1","method":"...","method_type":"...","inputs":[],"outputs":[],"evidence_refs":["S001_C01:M-1"],"confidence":0.0}],
  "problem_method_links": [{"problem_id":"RP1","method_id":"M1","relation":"directly_addresses|partially_addresses|evaluates|motivates","rationale":"...","evidence_refs":["S001_C01:RP-1","S001_C01:M-1"],"confidence":0.0}],
  "unresolved_or_ambiguous": []
}"""


def fair_extraction_user(evidence_records_json: str, budget_note: str) -> str:
    return f"""Budget/control note: {budget_note}

Indexed evidence records:
```json
{evidence_records_json}
```

Extract the graph with the exact schema in the system message."""


RAG_QUERY_SYSTEM = """Write one concise retrieval query for finding a paper's core research gaps, method components, objectives, and evidence. Return JSON: {"query":"..."}."""


COMPACT_EG_PLAN_SYSTEM = """You merge indexed evidence atoms into a paper-level problem-method graph.
Return exactly one compact JSON object. Do not copy claim text, descriptions, inputs, outputs, sections, or rationales into the response. Refer to evidence atoms only by their exact evidence_ref strings.

Output schema:
{
  "problem_groups": [
    {"group_id":"P1","members":["S001_C01:RP-1"],"representative":"S001_C01:RP-1","confidence":0.0}
  ],
  "method_groups": [
    {"group_id":"M1","members":["S001_C01:M-1"],"representative":"S001_C01:M-1","confidence":0.0}
  ],
  "links": [
    {"problem_group":"P1","method_group":"M1","relation":"directly_addresses|partially_addresses|evaluates|motivates","evidence_refs":["S001_C01:RP-1","S001_C01:M-1"],"confidence":0.0}
  ],
  "unresolved_refs": []
}

Rules:
1. Use only supplied evidence_ref strings; never invent or rewrite an ID.
2. Group only genuinely synonymous atoms. A singleton group is valid and preferred to an unsafe merge.
3. Do not mix problem atoms into method_groups or method atoms into problem_groups.
4. Every member may occur in at most one group of its kind.
5. representative must be one of that group's members.
6. Add a link only when the selected evidence_refs support the relationship, not merely the endpoint descriptions.
7. Keep the response compact: IDs, arrays, relation labels, and numeric confidences only. No prose or Markdown.
"""


def compact_eg_plan_user(records_json: str, budget_note: str) -> str:
    return f"""Control note: {budget_note}

Indexed compact evidence atoms:
{records_json}

Return only the compact merge plan specified in the system message."""
