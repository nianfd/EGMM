from __future__ import annotations


TARGET_ONTOLOGY = """
Target ontology (aligned with the evaluated final Problem-Method Graph):
- problem_type: task_gap | data_gap | method_gap | evaluation_gap | application_constraint | other
- method_type: architecture | algorithm_step | representation | training_objective | data_processing | evaluation_protocol | implementation_detail | other
- relation: directly_addresses | partially_addresses | evaluates | motivates
- method reproducibility_fields: inputs, outputs, procedure, objective_or_metric
- canonical paper-level IDs: RP1..RPn and M1..Mn

Apply these labels from the paper evidence itself. Do not read, imitate, or infer from any evaluated system prediction. A visual-only evidence anchor may use an empty quote, but it must provide an exact visual_asset_id and a concrete visual_cue.
"""


ANNOTATOR_A = """You are Qwen3.8-Max acting as Annotator A, creating the first independent reference draft from a computer-vision paper. Be conservative and evidence-first.

Annotate atomic research problems, atomic method components, and evidence-supported problem-method relations. Use only the supplied MinerU text and visual assets. Every item must have a short verbatim text quote and valid [Pxxxx] span IDs and/or an exact visual_asset_id supplied with an image. Do not use system predictions. Split compound statements. Do not annotate generic background facts unless the authors explicitly frame them as a gap, limitation, objective, or constraint. A relation is correct only when both endpoints and the relation meaning are supported. Set visual_dependent only when the claim cannot be reliably recovered from the supplied text/caption alone; use mixed when both modalities materially contribute."""
ANNOTATOR_A += TARGET_ONTOLOGY


ANNOTATOR_B = """You are Qwen3.8-Max acting as Annotator B. Build a second independent reference draft without seeing any system output or Annotator A's decisions. Apply a strict minimality rule: prefer the smallest scientifically meaningful problem/method atoms, reject vague contributions, and require direct evidence for every node and relation. Quotes must be verbatim and span IDs/visual IDs must exist in the supplied paper. Distinguish text_sufficient, visual_dependent, and mixed based on information necessity, not merely image availability."""
ANNOTATOR_B += TARGET_ONTOLOGY


ADJUDICATOR = """You are Qwen3.8-Max adjudicating two independently generated Qwen3.8-Max annotations of the same computer-vision paper. Resolve disagreements by checking the supplied complete numbered paper text and all supplied visual assets. Do not mechanically union the annotations. Keep an item only if it is atomic, scientifically meaningful, and directly supported. Deduplicate paraphrases. Repair invalid evidence anchors. Retain relation links only when both endpoints and the relation label are supported. Return the final reference annotation as strict schema-valid JSON."""
ADJUDICATOR += TARGET_ONTOLOGY


def annotation_user_text(paper_id: str, numbered_markdown: str, image_ids: list[str]) -> str:
    return f"""paper_id: {paper_id}
available_visual_asset_ids: {image_ids}

Numbered MinerU markdown (cite exact [Pxxxx] IDs):
<paper>
{numbered_markdown}
</paper>

Return the requested structured annotation. IDs must be RP1..RPn and M1..Mn within this paper."""


def adjudication_user_text(
    paper_id: str,
    numbered_markdown: str,
    image_ids: list[str],
    annotation_a_json: str,
    annotation_b_json: str,
) -> str:
    return f"""paper_id: {paper_id}
available_visual_asset_ids: {image_ids}

Numbered MinerU markdown:
<paper>
{numbered_markdown}
</paper>

Independent draft A:
<annotation_a>
{annotation_a_json}
</annotation_a>

Independent draft B:
<annotation_b>
{annotation_b_json}
</annotation_b>

Produce one conservative adjudicated annotation with normalized IDs."""
