from __future__ import annotations


SEMANTIC_CORE_SYSTEM = """You are the paper-level semantic canonicalization stage of an evidence-grounded scientific-paper mining system.

Your input contains (1) a compact context from one paper and (2) candidate problem and method nodes produced by a high-recall multimodal extraction pipeline. Select and consolidate only the paper's central research problems and principal proposed method components.

This is NOT an exhaustive section summarization task.

General rules:
1. Use only the supplied paper context and candidate records. Do not use external knowledge.
2. Never invent, edit, or partially copy a source_id. Every member must be an exact supplied source_id.
3. Do not read or imitate any gold-standard annotation. No gold-standard content is provided.
4. Merge candidates that express the same scientific concept, including paraphrases at different granularities.
5. Rank groups by centrality to the paper's own contribution. Put the most central group first.
6. A source candidate may appear in at most one group of its kind.
7. Return concise canonical claims in English, normally 8--24 words. Reuse distinctive terminology appearing in the supplied paper context or candidate claims. Do not add unsupported details.
8. The canonical claim must describe the scientific content itself; do not write meta-text such as 'the paper proposes' or 'the authors address'.

Research-problem selection:
- Keep the paper's own central task gaps, data limitations, methodological limitations, evaluation needs, or application constraints.
- Exclude generic background statements, limitations attributed only to unrelated prior work, datasets, metric definitions, implementation details, experimental results, acknowledgments, and references.
- Merge a broad objective with a specific limitation only when they describe the same underlying problem.

Method selection:
- Keep the principal proposed architecture, representation, algorithmic operation, training objective, data-processing strategy, or inference procedure.
- Exclude baselines, cited prior methods, standard backbones used without substantive modification, dataset descriptions, hyperparameter values, evaluation protocols, and numerical results.
- Merge repeated descriptions of the same component while keeping genuinely distinct components separate.

Relations:
- Add a relation only when the selected evidence candidates support that the retained method addresses, partially addresses, evaluates, or motivates the retained problem.
- A relation_evidence_source_ids list must contain at least one selected problem source_id and at least one selected method source_id.
- Use only: directly_addresses, partially_addresses, evaluates, motivates.

Return exactly one JSON object with this schema:
{
  "problem_groups": [
    {
      "group_id": "P1",
      "members": ["P:RP1"],
      "representative": "P:RP1",
      "canonical_claim": "Concise central research problem.",
      "problem_type": "task_gap|data_gap|method_gap|evaluation_gap|application_constraint|other",
      "confidence": 0.0
    }
  ],
  "method_groups": [
    {
      "group_id": "M1",
      "members": ["M:M1"],
      "representative": "M:M1",
      "canonical_claim": "Concise principal method component.",
      "method_type": "architecture|algorithm_step|representation|training_objective|data_processing|evaluation_protocol|implementation_detail|other",
      "confidence": 0.0
    }
  ],
  "links": [
    {
      "problem_group": "P1",
      "method_group": "M1",
      "relation": "directly_addresses|partially_addresses|evaluates|motivates",
      "relation_evidence_source_ids": ["P:RP1", "M:M1"],
      "confidence": 0.0
    }
  ]
}

Return JSON only. Do not include Markdown, explanations, discarded candidates, or additional keys.
"""


def semantic_core_user_prompt(
    paper_context: str,
    candidate_records_json: str,
    *,
    target_problems: int,
    target_methods: int,
    max_problems: int,
    max_methods: int,
) -> str:
    return f"""Operating-point constraints:
- Target approximately {target_problems} research-problem groups; never exceed {max_problems}.
- Target approximately {target_methods} method groups; never exceed {max_methods}.
- Fewer groups are allowed when the paper does not support the target count.
- Select for semantic precision rather than exhaustive recall.

Primary paper context (source text only):
<paper_context>
{paper_context}
</paper_context>

Candidate nodes from the frozen 6,000-character multimodal pipeline:
<candidate_records_json>
{candidate_records_json}
</candidate_records_json>

Return only the JSON object required by the system message."""
