DIRECT_EXTRACTION_SYSTEM = """You are a rigorous multimodal scientific-paper information extraction system.
Return only one valid JSON object. Do not include Markdown fences or explanatory prose.
Extract fine-grained research problems and methods from the provided paper content.
Every extracted item must include evidence. Evidence can be textual, visual, tabular, equation-based, or mixed."""


DIRECT_EXTRACTION_USER_TEMPLATE = """Task:
Given the following paper content and attached images, extract fine-grained research problems and methods.

Definitions:
1. A research problem should be fine-grained and operational. It may describe a task gap, data gap, methodological limitation, evaluation gap, or application constraint.
2. A method should be fine-grained and reproducible. It may describe model architecture, input/output representation, algorithmic steps, training objectives, inference procedures, data processing, implementation details, or evaluation protocols.
3. Do not provide a general summary. Extract atomic research-problem and method units.
4. Every extracted item must include evidence. Evidence can be a short text cue, a figure/table cue, or both.
5. If evidence is uncertain, mark explicitness as "uncertain" and reduce confidence.

Attached image names:
{image_names}

Paper content:
```markdown
{paper_text}
```

Output JSON schema:
{{
  "final_research_problems": [
    {{
      "id": "RP1",
      "problem": "A fine-grained research problem stated in one sentence.",
      "problem_type": "task_gap|data_gap|method_gap|evaluation_gap|application_constraint|other",
      "granularity": "fine|medium|coarse",
      "explicitness": "explicit|implicit|uncertain",
      "evidence_refs": [
        {{
          "source": "text|figure|table|equation|mixed",
          "location": "section/page/figure/table if available",
          "cue": "short textual or visual cue",
          "explanation": "why this evidence supports the extracted problem"
        }}
      ],
      "confidence": 0.0,
      "risk_note": "possible hallucination or ambiguity risk, or an empty string"
    }}
  ],
  "final_methods": [
    {{
      "id": "M1",
      "method": "A fine-grained method component stated in one sentence.",
      "method_type": "architecture|algorithm_step|representation|training_objective|data_processing|evaluation_protocol|implementation_detail|other",
      "reproducibility_fields": {{
        "inputs": ["input data, features, or conditions"],
        "outputs": ["outputs or predictions"],
        "procedure": ["step 1", "step 2", "step 3"],
        "objective_or_metric": ["loss function, optimization objective, metric, or empty string"]
      }},
      "granularity": "fine|medium|coarse",
      "evidence_refs": [
        {{
          "source": "text|figure|table|equation|mixed",
          "location": "section/page/figure/table if available",
          "cue": "short textual or visual cue",
          "explanation": "why this evidence supports the extracted method"
        }}
      ],
      "confidence": 0.0,
      "risk_note": "possible hallucination or ambiguity risk, or an empty string"
    }}
  ],
  "problem_method_links": [
    {{
      "problem_id": "RP1",
      "method_id": "M1",
      "relation": "directly_addresses|partially_addresses|evaluates|motivates",
      "rationale": "evidence-based explanation of how the method is related to the problem"
    }}
  ],
  "quality_report": {{
    "evidence_coverage": "high|medium|low",
    "cross_modal_usage": "high|medium|low",
    "main_limitations": ["limitations of the extraction"],
    "recommended_human_checks": ["items that should be manually checked"]
  }}
}}"""
