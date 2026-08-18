from __future__ import annotations

from pathlib import Path

from revision.annotation import number_markdown, normalize_visual_asset_ids, select_images, verify_annotation
from revision.annotation_schema import EvidenceAnchor, GoldLink, GoldMethod, GoldProblem, PaperGoldAnnotation, ReproducibilityFields, normalize_incomplete_gold_links
from revision.calibration import calibration_metrics, grouped_cross_validated_platt
from revision.dashscope_qwen38_client import Qwen38AnnotationClient, _json_object, _validate_with_conservative_gold_recovery
from revision.evidence import build_exact_evidence_map, relation_evidence_support, strict_visual_support
from revision.io_utils import canonical_ref, write_json
from revision.contracts import validate_condition_output
from revision.evaluation import evaluate_paper, greedy_match
from revision.runner import build_qwen_config
from revision.experiments import bm25_rank, compact_prompt_records, materialize_compact_merge_plan
from revision.resilient_qwenvl import ResilientQwenVLClient, _extract_complete_object_items
from revision.semantic_core import materialize_semantic_core_plan, select_primary_context
from revision.semantic_core_prompts import SEMANTIC_CORE_SYSTEM, semantic_core_user_prompt
from paper_mining.config import PipelineConfig
import paper_mining.pipeline as original_main_pipeline
from paper_mining.qwenvl_client import QwenVLClient
from revision.qwen38_main_aligned_client import Qwen38MainAlignedClient
from revision.qwen38_main_aligned_gold import (
    main_output_to_gold,
    run_main_aligned_gold_pipeline,
    runtime_alignment_report,
)
from revision.semantic_matching import SentenceEmbeddingMatcher, hungarian_match_from_scores
from scripts.build_paper_tables_1_2_3 import build_tables


def sample_annotation() -> PaperGoldAnnotation:
    anchor = EvidenceAnchor(span_ids=["P0001"], quote="problem", support_type="text")
    return PaperGoldAnnotation(
        paper_id="1",
        research_problems=[GoldProblem(id="RP1", claim="problem", problem_type="method_gap", evidence=[anchor], visual_dependency="text_sufficient")],
        methods=[GoldMethod(id="M1", claim="method", method_type="architecture", reproducibility_fields=ReproducibilityFields(), evidence=[anchor], visual_dependency="text_sufficient")],
        problem_method_links=[GoldLink(problem_id="RP1", method_id="M1", relation="directly_addresses", evidence=[anchor], visual_dependency="text_sufficient")],
    )


def test_numbering_and_annotation_verification() -> None:
    numbered, spans = number_markdown("problem\n\nmethod")
    result = verify_annotation(sample_annotation(), spans, set())
    assert numbered.startswith("[P0001]")
    assert result["valid"]


def test_gold_input_policy_never_samples_images(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("a.jpg", "b.png", "c.webp"):
        (image_dir / name).write_bytes(b"image-placeholder")
    selected = select_images("![second](images/b.png)", image_dir)
    assert [path.name for path in selected] == ["b.png", "a.jpg", "c.webp"]
    assert len(selected) == 3


def test_visual_asset_ids_are_normalized_only_to_manifest_entries() -> None:
    annotation = sample_annotation()
    actual = "a484947f89df1aed58ae7267e309bf9988b6c02849eaf03ced32ed67589aa7e2.jpg"
    typo = "a484947f89df1aed58ae7267e309bf9988b6c02846e02849eaf03ced32ed67589aa7e2.jpg"
    annotation.methods[0].evidence[0].visual_asset_ids = [typo]
    report = normalize_visual_asset_ids(annotation, {actual, "041e035be7d46db61c647caba375e66d664da0e300ba7046c6ba17c888a5d3f4.jpg"})
    assert annotation.methods[0].evidence[0].visual_asset_ids == [actual]
    assert report["actions"][0]["method"] == "unique_high_similarity_hash"
    assert report["unresolved"] == []

    no_extension = sample_annotation()
    no_extension.methods[0].evidence[0].visual_asset_ids = [Path(actual).stem]
    report = normalize_visual_asset_ids(no_extension, {actual})
    assert no_extension.methods[0].evidence[0].visual_asset_ids == [actual]
    assert report["actions"][0]["method"] == "unique_exact_stem"

    unrelated = sample_annotation()
    unrelated.methods[0].evidence[0].visual_asset_ids = ["ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"]
    report = normalize_visual_asset_ids(unrelated, {actual})
    assert report["unresolved"]
    assert unrelated.methods[0].evidence[0].visual_asset_ids[0] not in {actual}


def test_incomplete_gold_link_is_dropped_without_inventing_fields() -> None:
    payload = sample_annotation().model_dump(mode="json")
    payload["problem_method_links"] = [{"problem_id": "RP1", "method_id": "M1,"}]
    normalized, audit = normalize_incomplete_gold_links(payload)
    assert normalized["problem_method_links"] == []
    assert audit["action_count"] == 1
    assert set(audit["dropped_links"][0]["missing_fields"]) == {"relation", "evidence", "visual_dependency"}
    parsed, recovery = _validate_with_conservative_gold_recovery(PaperGoldAnnotation, payload)
    assert parsed.problem_method_links == []
    assert recovery is not None and recovery["applied"]


def test_complete_link_endpoint_punctuation_is_normalized_only_to_existing_id() -> None:
    payload = sample_annotation().model_dump(mode="json")
    payload["problem_method_links"][0]["method_id"] = "M1,"
    parsed, recovery = _validate_with_conservative_gold_recovery(PaperGoldAnnotation, payload)
    assert parsed.problem_method_links[0].method_id == "M1"
    assert recovery is not None
    assert recovery["endpoint_normalizations"][0]["normalized"] == "M1"


def test_saved_raw_answer_can_be_recovered_without_api(tmp_path) -> None:
    client = Qwen38AnnotationClient(
        api_key="dry-run", model="qwen3.8-max", cache_dir=tmp_path, dry_run=True,
    )
    kwargs = {
        "stage": "raw_recovery_test",
        "instructions": "return JSON",
        "user_text": "paper 1",
        "schema": PaperGoldAnnotation,
        "dry_run_value": sample_annotation(),
    }
    client.parse(**kwargs)
    cache_path = next((tmp_path / "raw_recovery_test").glob("*.json"))
    cache_path.unlink()
    payload = sample_annotation().model_dump(mode="json")
    payload["problem_method_links"] = [{"problem_id": "RP1", "method_id": "M1,"}]
    cache_path.with_suffix(".answer.txt").write_text(__import__("json").dumps(payload), encoding="utf-8")
    cache_path.with_suffix(".reasoning.txt").write_text("", encoding="utf-8")
    parsed, metadata = client.parse(**kwargs)
    assert parsed.problem_method_links == []
    assert metadata["recovered_from_saved_raw_answer"] is True
    assert metadata["local_schema_normalization"]["dropped_links"]


def test_exact_and_strict_visual_evidence() -> None:
    l1 = [{
        "chunk_id": "S001_C01",
        "section": "Method",
        "research_problem_atoms": [{"id": "RP-1", "claim": "gap", "evidence": [{"source": "text", "quote_or_visual_cue": "gap"}]}],
        "method_atoms": [{"id": "M-01", "claim": "module", "evidence": [{"source": "image", "quote_or_visual_cue": "three branches"}]}],
    }]
    index = {"S001_C01": {"images": ["fig1.png"]}}
    evidence_map = build_exact_evidence_map(l1, index)
    result = strict_visual_support({"evidence_refs": ["S001_C01:M-1"]}, evidence_map)
    assert result["has_strict_visual_support"]
    assert "S001_C01:M-1" in evidence_map
    assert relation_evidence_support({"evidence_refs": ["S001_C01:RP-1", "S001_C01:M-1"]}, evidence_map)["supported"]
    assert not relation_evidence_support({"evidence_refs": ["S001_C01:M-1"]}, evidence_map)["supported"]


def test_hierarchical_decimal_evidence_ids_are_exact() -> None:
    assert canonical_ref("S009_C01:RP-5.1-01") == "S009_C01:RP-5.1-1"
    assert canonical_ref("S009_C01:M-5.1-7") == "S009_C01:M-5.1-7"
    l1 = [{
        "chunk_id": "S009_C01",
        "section": "5.1 Synthetic data evaluation",
        "research_problem_atoms": [{"id": "RP-5.1-01", "claim": "gap", "evidence": []}],
        "method_atoms": [{"id": "M-5.1-7", "claim": "module", "evidence": []}],
    }]
    evidence_map = build_exact_evidence_map(l1, {"S009_C01": {"images": []}})
    assert "S009_C01:RP-5.1-1" in evidence_map
    assert "S009_C01:M-5.1-7" in evidence_map


def test_wrong_kind_evidence_is_scored_as_quality_warning(tmp_path) -> None:
    l1 = [{
        "chunk_id": "S030_C01",
        "section": "Ablation",
        "research_problem_atoms": [{"id": "RP-1", "claim": "gap", "evidence": [], "confidence": 0.8}],
        "method_atoms": [{"id": "M-1", "claim": "module", "evidence": [], "confidence": 0.8}],
    }]
    final = {
        "final_research_problems": [{
            "id": "RP1", "problem": "gap", "problem_type": "method_gap",
            "granularity": "fine", "explicitness": "explicit",
            "evidence_refs": ["S030_C01:M-1"], "confidence": 0.8, "risk_note": "",
        }],
        "final_methods": [{
            "id": "M1", "method": "module", "method_type": "architecture",
            "reproducibility_fields": {"inputs": [], "outputs": [], "procedure": [], "objective_or_metric": []},
            "granularity": "fine", "evidence_refs": ["S030_C01:M-1"],
            "confidence": 0.8, "risk_note": "",
        }],
        "problem_method_links": [],
        "quality_report": {},
    }
    write_json(tmp_path / "01_l1_chunk_results.json", l1)
    write_json(tmp_path / "02_evidence_index.json", {"S030_C01": {"section": "Ablation", "images": []}})
    write_json(tmp_path / "04_final_extraction.json", final)
    write_json(tmp_path / "condition_info.json", {
        "condition": "added_visual_masked", "model": "qwen-vl-max", "dry_run": False,
    })
    report = validate_condition_output(tmp_path, "added_visual_masked")
    assert report["valid"]
    assert any("wrong atom kind" in warning for warning in report["warnings"])
    assert report["quality_warning_count"] == 1


def test_resilient_l1_recovers_complete_objects_from_truncated_json(tmp_path) -> None:
    client = ResilientQwenVLClient(
        api_key="unused", base_url="https://example.invalid", model="qwen-vl-max",
        cache_dir=tmp_path, dry_run=False, verbose=False,
    )
    system = "system"
    user = "user"
    cache_path = client.cache_path_for(
        "l1_chunk_extract", system, user, [], {"chunk_id": "S008_C01"},
    )
    cache_path.parent.mkdir(parents=True)
    cache_path.with_suffix(".raw.txt").write_text(
        '{"research_problem_atoms":[{"id":"RP-1","claim":"gap","evidence":[]}],'
        '"method_atoms":[{"id":"M-1","claim":"complete","evidence":[]},'
        '{"id":"M-2","claim":"truncated',
        encoding="utf-8",
    )
    result = client.chat_json(
        "l1_chunk_extract", system, user, [], {"chunk_id": "S008_C01"},
    )
    assert [item["id"] for item in result["research_problem_atoms"]] == ["RP-1"]
    assert [item["id"] for item in result["method_atoms"]] == ["M-1"]
    assert cache_path.exists()
    audit = __import__("json").loads(cache_path.with_suffix(".recovery.json").read_text(encoding="utf-8"))
    assert audit["method"] == "local_complete_object_salvage"


def test_resilient_compact_plan_salvage_preserves_complete_groups() -> None:
    broken = (
        '{"problem_groups":[{"group_id":"P1","members":["S1:RP-1"],'
        '"representative":"S1:RP-1","confidence":0.8}],'
        '"method_groups":[{"group_id":"M1","members":["S1:M-1"],'
        '"representative":"S1:M-1","confidence":0.9}],'
        '"links":[{"problem_group":"P1","method_group":"M1","relation":"directly_addresses",'
        '"evidence_refs":["S1:RP-1","S1:M-1"],"confidence":0.8},'
        '{"problem_group":"P1","method_group":"M1","relation":"truncated'
    )
    found, links = _extract_complete_object_items(broken, "links")
    assert found and len(links) == 1
    client = ResilientQwenVLClient(
        api_key="unused", base_url="https://example.invalid", model="qwen-vl-max",
        cache_dir=Path("unused"), dry_run=False, verbose=False,
    )
    plan = client._local_recovery("added_bm25_rag_text_compact_plan_v2", broken)
    assert plan is not None
    assert len(plan["problem_groups"]) == 1
    assert len(plan["method_groups"]) == 1
    assert len(plan["links"]) == 1


def test_matching_and_grouped_calibration() -> None:
    match = greedy_match(["spectral denoising limitation"], ["limitation of spectral denoising"], 0.4)
    assert len(match.pairs) == 1
    rows = [
        {"paper_id": str(index), "confidence": 0.8 if index % 2 else 0.2, "correct": index % 2}
        for index in range(1, 11)
    ]
    calibrated, model = grouped_cross_validated_platt(rows, folds=5)
    assert len(calibrated) == len(rows)
    assert model["grouping"] == "paper_id"
    assert calibration_metrics(calibrated, "calibrated_confidence")["count"] == 10


def test_qwen38_main_aligned_client_reuses_original_chat_json_contract(tmp_path) -> None:
    client = Qwen38MainAlignedClient(
        api_key="dry-run",
        base_url="https://example.invalid",
        model="qwen3.8-max",
        cache_dir=tmp_path,
        dry_run=True,
        verbose=False,
    )
    result = client.chat_json(
        stage="l1_chunk_extract",
        system_prompt=original_main_pipeline.L1_SYSTEM,
        user_text="chunk",
        images=[],
        extra_cache_key={"chunk_id": "S001_C01"},
    )
    assert result["dry_run"] is True
    assert Qwen38MainAlignedClient.chat_json is QwenVLClient.chat_json


def test_main_aligned_gold_is_deterministic_projection_of_main_output() -> None:
    l1 = [{
        "chunk_id": "S001_C01",
        "section": "Method",
        "research_problem_atoms": [{
            "id": "RP-1", "claim": "fine-grained gap", "problem_type": "method_gap",
            "evidence": [{"source": "text", "quote_or_visual_cue": "gap evidence"}],
        }],
        "method_atoms": [{
            "id": "M-1", "claim": "fine-grained module", "method_type": "architecture",
            "evidence": [{"source": "image", "quote_or_visual_cue": "three branches"}],
        }],
    }]
    index = {"S001_C01": {"section": "Method", "images": ["fig1.png"]}}
    final = {
        "final_research_problems": [{
            "id": "RP1", "problem": "fine-grained gap", "problem_type": "method_gap",
            "evidence_refs": ["S001_C01:RP-1"],
        }],
        "final_methods": [{
            "id": "M1", "method": "fine-grained module", "method_type": "architecture",
            "evidence_refs": ["S001_C01:M-1"],
            "reproducibility_fields": {
                "inputs": ["x"], "outputs": ["y"], "procedure": ["run"],
                "objective_or_metric": ["loss"],
            },
        }],
        "problem_method_links": [{
            "problem_id": "RP1", "method_id": "M1", "relation": "directly_addresses",
        }],
        "quality_report": {},
    }
    gold = main_output_to_gold("1", final, l1, index)
    assert [(item.id, item.claim) for item in gold.research_problems] == [("RP1", "fine-grained gap")]
    assert [(item.id, item.claim) for item in gold.methods] == [("M1", "fine-grained module")]
    assert gold.methods[0].visual_dependency == "visual_dependent"
    assert gold.problem_method_links[0].relation == "directly_addresses"


def test_exact_original_pipeline_runs_in_main_aligned_dry_run(tmp_path) -> None:
    paper_dir = tmp_path / "1"
    image_dir = paper_dir / "images"
    image_dir.mkdir(parents=True)
    (paper_dir / "full.md").write_text(
        "# Test Paper\n\n## Abstract\nA problem and a method.\n",
        encoding="utf-8",
    )
    output_dir = paper_dir / "outputs" / "revision_annotations_main_aligned"
    config = PipelineConfig(
        paper_dir=paper_dir,
        markdown_path=paper_dir / "full.md",
        image_dir=image_dir,
        output_dir=output_dir,
        cache_dir=output_dir / "cache",
        api_key="dry-run",
        model="qwen3.8-max",
        dry_run=True,
        verbose=False,
    )
    original_client = original_main_pipeline.QwenVLClient
    metadata = run_main_aligned_gold_pipeline(config)
    assert original_main_pipeline.QwenVLClient is original_client
    assert (output_dir / "human_gold.json").is_file()
    assert metadata["annotation_status"] == "qwen38_max_main_method_aligned_gold"
    alignment = runtime_alignment_report(config)
    assert alignment["max_chars_per_chunk"] == 9000
    assert alignment["max_images_per_chunk"] == 4


def test_semantic_hungarian_uses_dummies_and_maximizes_valid_match_count() -> None:
    scores = [
        [0.90, 0.80],
        [0.85, 0.10],
    ]
    result = hungarian_match_from_scores(scores, threshold=0.70)
    assert {(p, g) for p, g, _ in result.pairs} == {(0, 1), (1, 0)}
    assert result.unmatched_predictions == []
    assert result.unmatched_gold == []

    rejected = hungarian_match_from_scores([[0.69]], threshold=0.70)
    assert rejected.pairs == []
    assert rejected.unmatched_predictions == [0]
    assert rejected.unmatched_gold == [0]


def test_sentence_embedding_matcher_caches_normalized_vectors(tmp_path) -> None:
    import numpy as np

    class FakeEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            mapping = {
                "spectral noise limitation": np.array([1.0, 0.0], dtype=np.float32),
                "limitation caused by spectral noise": np.array([0.98, 0.20], dtype=np.float32),
                "unrelated method": np.array([0.0, 1.0], dtype=np.float32),
            }
            values = np.stack([mapping[text] for text in texts])
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            return values

    encoder = FakeEncoder()
    matcher = SentenceEmbeddingMatcher(
        "fake-model",
        cache_path=tmp_path / "embeddings.sqlite3",
        encoder=encoder,
        show_progress=False,
    )
    result = matcher.match(
        ["spectral noise limitation", "unrelated method"],
        ["limitation caused by spectral noise"],
        threshold=0.80,
    )
    assert [(p, g) for p, g, _ in result.pairs] == [(0, 0)]
    matcher.encode(["spectral noise limitation"])
    assert encoder.calls == 1
    matcher.close()


def test_semantic_evaluator_accepts_paraphrases_and_scores_conditional_links(tmp_path) -> None:
    import numpy as np

    paper = tmp_path / "1"
    condition_dir = paper / "outputs" / "major_revision_additions" / "added_chunk_6000"
    gold_dir = paper / "outputs" / "revision_annotations_main_aligned"
    prediction_problem = "Spectral corruption degrades hyperspectral observations"
    gold_problem = "Hyperspectral images are impaired by band-wise noise"
    prediction_method = "A transformer unmixing network separates clean signal from corruption"
    gold_method = "The approach restores clean spectra using transformer-based unmixing"
    write_json(condition_dir / "01_l1_chunk_results.json", [{
        "chunk_id": "S001_C01", "section": "Method",
        "research_problem_atoms": [{
            "id": "RP-1", "claim": prediction_problem,
            "evidence": [{"source": "text", "quote_or_visual_cue": "spectral corruption"}],
        }],
        "method_atoms": [{
            "id": "M-1", "claim": prediction_method,
            "evidence": [{"source": "text", "quote_or_visual_cue": "transformer unmixing"}],
        }],
    }])
    write_json(condition_dir / "02_evidence_index.json", {
        "S001_C01": {"section": "Method", "images": []},
    })
    write_json(condition_dir / "04_final_extraction.json", {
        "final_research_problems": [{
            "id": "RP1", "problem": prediction_problem,
            "problem_type": "data_gap", "evidence_refs": ["S001_C01:RP-1"],
        }],
        "final_methods": [{
            "id": "M1", "method": prediction_method,
            "method_type": "architecture", "evidence_refs": ["S001_C01:M-1"],
        }],
        "problem_method_links": [{
            "problem_id": "RP1", "method_id": "M1", "relation": "directly_addresses",
            "link_type": "evidence_supported",
            "evidence_refs": ["S001_C01:RP-1", "S001_C01:M-1"],
        }],
    })
    anchor = {"span_ids": ["P0001"], "quote": "support", "support_type": "text"}
    write_json(gold_dir / "human_gold.json", {
        "paper_id": "1",
        "research_problems": [{
            "id": "RP1", "claim": gold_problem, "problem_type": "data_gap",
            "evidence": [anchor], "visual_dependency": "text_sufficient",
        }],
        "methods": [{
            "id": "M1", "claim": gold_method, "method_type": "architecture",
            "reproducibility_fields": {}, "evidence": [anchor],
            "visual_dependency": "text_sufficient",
        }],
        "problem_method_links": [{
            "problem_id": "RP1", "method_id": "M1", "relation": "directly_addresses",
            "evidence": [anchor], "visual_dependency": "text_sufficient",
        }],
    })

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            mapping = {
                prediction_problem: np.array([1.0, 0.0], dtype=np.float32),
                gold_problem: np.array([0.98, 0.20], dtype=np.float32),
                prediction_method: np.array([0.0, 1.0], dtype=np.float32),
                gold_method: np.array([0.20, 0.98], dtype=np.float32),
            }
            values = np.stack([mapping[text] for text in texts])
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            return values

    matcher = SentenceEmbeddingMatcher("fake", encoder=FakeEncoder(), show_progress=False)
    lexical = evaluate_paper(
        paper, "added_chunk_6000", gold_dir / "human_gold.json",
        matching_method="lexical",
    )
    semantic = evaluate_paper(
        paper, "added_chunk_6000", gold_dir / "human_gold.json",
        matching_method="semantic", semantic_matcher=matcher,
        problem_similarity_threshold=0.70, method_similarity_threshold=0.70,
    )
    assert lexical["problem"]["f1"] == 0.0
    assert semantic["problem"]["f1"] == 1.0
    assert semantic["method"]["f1"] == 1.0
    assert semantic["link"]["f1"] == 1.0
    assert semantic["link_conditional_on_matched_nodes"]["f1"] == 1.0
    matcher.close()


def test_tables_1_2_3_enforce_100_plus_30_cohorts(tmp_path) -> None:
    structural_path = tmp_path / "metrics_summary_core.csv"
    structural_conditions = [
        "baseline_oneshot_mineru", "added_bm25_rag_text", "added_global_eg_merge",
        "proposed_full", "added_chunk_6000", "added_chunk_9000", "added_chunk_12000",
    ]
    from revision.io_utils import write_csv

    write_csv(structural_path, [{
        "condition": condition, "num_successful_papers": 100,
        "NET": 1.0, "AMG": 1.0, "ESGC": 0.8,
        "TLP": 0.7, "EGMR": 0.6, "BEGQ": 0.75,
    } for condition in structural_conditions])
    semantic_dir = tmp_path / "semantic"
    semantic_conditions = [
        "added_chunk_6000", "original_main", "original_baseline_oneshot_mineru",
        "added_global_eg_merge", "added_bm25_rag_text", "added_semantic_core_6000",
    ]
    write_json(semantic_dir / "summary.json", [{
        "condition": condition, "paper_count": 30,
        "gold_statuses": ["qwen38_max_main_method_aligned_gold"],
        "problem_micro": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
        "method_micro": {"f1": 0.7},
        "link_micro": {"f1": 0.1},
        "link_conditional_on_matched_nodes_micro": {"f1": 0.2},
        "exact_node_reference_rate": 1.0,
    } for condition in semantic_conditions])
    write_json(semantic_dir / "matching_protocol.json", {
        "matcher": "dense_cosine_hungarian_with_explicit_unmatched_dummies",
        "problem_similarity_threshold": 0.70,
        "method_similarity_threshold": 0.70,
    })
    manifest = build_tables(structural_path, semantic_dir, tmp_path / "tables")
    assert manifest["cohorts"] == {
        "table_1_structural": 100,
        "table_2_semantic_reference": 30,
        "table_3_chunk_ablation": 100,
    }
    latex = (tmp_path / "tables" / "article_tables_1_2_3.tex").read_text(encoding="utf-8")
    assert "Revised proposed method (6k multimodal)" in latex
    assert "6,000 (proposed)" in latex
    assert manifest["revised_main_condition"] == "added_chunk_6000"
    assert manifest["semantic_reference"]["chunk_chars"] == 9000
    assert manifest["semantic_reference"]["regeneration_required"] is False
    assert "One-shot MinerU & N/A & N/A" in latex
    assert "Thirty-paper semantic evaluation" in latex


def test_revision_extraction_is_locked_to_qwen_vl_max(tmp_path) -> None:
    (tmp_path / "full.md").write_text("# Test", encoding="utf-8")
    (tmp_path / "images").mkdir()
    accepted = build_qwen_config(
        tmp_path, tmp_path / "accepted", api_key="dry-run", base_url="https://example.invalid",
        model="qwen-vl-max", max_chars_per_chunk=9000, overlap_chars=900,
        max_images_per_chunk=4, request_timeout=1, max_retries=1,
        max_tokens=10, temperature=0.1, dry_run=True, quiet=True,
    )
    assert accepted.model == "qwen-vl-max"
    try:
        build_qwen_config(
            tmp_path, tmp_path / "out", api_key="dry-run", base_url="https://example.invalid",
            model="qwen3-vl-plus", max_chars_per_chunk=9000, overlap_chars=900,
            max_images_per_chunk=4, request_timeout=1, max_retries=1,
            max_tokens=10, temperature=0.1, dry_run=True, quiet=True,
        )
    except ValueError as exc:
        assert "locked to 'qwen-vl-max'" in str(exc)
    else:
        raise AssertionError("A non-qwen-vl-max backend was not rejected")


def test_gold_generation_is_locked_to_qwen38_max(tmp_path) -> None:
    value = _json_object('```json\n{"paper_id":"1"}\n```')
    assert value["paper_id"] == "1"
    client = Qwen38AnnotationClient(
        api_key="dry-run", model="qwen3.8-max", cache_dir=tmp_path, dry_run=True,
    )
    parsed, metadata = client.parse(
        stage="test", instructions="return JSON", user_text="paper 1",
        schema=PaperGoldAnnotation, dry_run_value=sample_annotation(),
    )
    assert parsed.paper_id == "1"
    assert metadata["model_requested"] == "qwen3.8-max"
    assert metadata["enable_thinking"] is True
    try:
        Qwen38AnnotationClient(api_key="dry-run", model="qwen3-vl-plus", dry_run=True)
    except ValueError as exc:
        assert "locked to 'qwen3.8-max'" in str(exc)
    else:
        raise AssertionError("A non-qwen3.8-max annotation backend was not rejected")


def test_bm25_ranks_relevant_chunk_first() -> None:
    ranked = bm25_rank(
        "research problem limitation proposed method architecture",
        [("relevant", "The proposed method addresses the research limitation with a new architecture."),
         ("irrelevant", "Acknowledgments and author affiliations.")],
        k=2,
    )
    assert ranked[0] == "relevant"


def test_compact_merge_plan_materializes_metric_schema() -> None:
    records = [
        {"evidence_ref": "S1:RP-1", "kind": "problem", "claim": "a visual grounding gap", "type": "method_gap", "inputs": [], "outputs": [], "section": "Introduction", "confidence": 0.8, "evidence": [{"verbose": "omitted"}]},
        {"evidence_ref": "S2:RP-2", "kind": "problem", "claim": "the visual grounding gap", "type": "method_gap", "inputs": [], "outputs": [], "section": "Method", "confidence": 0.9, "evidence": []},
        {"evidence_ref": "S2:M-1", "kind": "method", "claim": "an evidence fusion module", "type": "architecture", "inputs": ["text", "images"], "outputs": ["graph"], "section": "Method", "confidence": 0.9, "evidence": []},
    ]
    compact = compact_prompt_records(records)
    assert "evidence" not in compact[0]
    plan = {
        "problem_groups": [{"group_id": "P1", "members": ["S1:RP-1", "S2:RP-2"], "representative": "S2:RP-2", "confidence": 0.9}],
        "method_groups": [{"group_id": "M1", "members": ["S2:M-1"], "representative": "S2:M-1", "confidence": 0.9}],
        "links": [{"problem_group": "P1", "method_group": "M1", "relation": "directly_addresses", "evidence_refs": ["S2:RP-2", "S2:M-1"], "confidence": 0.8}],
        "unresolved_refs": [],
    }
    result, audit = materialize_compact_merge_plan(plan, records)
    assert result["paper_research_problems"][0]["evidence_refs"] == ["S1:RP-1", "S2:RP-2"]
    assert result["paper_methods"][0]["inputs"] == ["text", "images"]
    assert result["problem_method_links"][0]["problem_id"] == "RP1"
    assert result["problem_method_links"][0]["method_id"] == "M1"
    assert audit["input_record_count"] == 3


def test_semantic_core_materialization_is_bounded_and_evidence_exact() -> None:
    records = [
        {
            "source_id": "P:RP1", "kind": "problem", "claim": "low-light images obscure object boundaries",
            "type": "data_gap", "sections": ["Abstract"], "evidence_refs": ["S003_C01:RP-1"],
            "confidence": 0.9,
        },
        {
            "source_id": "P:RP2", "kind": "problem", "claim": "object boundaries are ambiguous in low light",
            "type": "data_gap", "sections": ["Introduction"], "evidence_refs": ["S004_C01:RP-2"],
            "confidence": 0.8,
        },
        {
            "source_id": "M:M1", "kind": "method", "claim": "dual-branch boundary-aware fusion network",
            "type": "architecture", "sections": ["Method"], "evidence_refs": ["S006_C01:M-1"],
            "confidence": 0.9, "inputs": ["low-light image"], "outputs": ["restored image"],
        },
        {
            "source_id": "M:M2", "kind": "method", "claim": "boundary consistency objective",
            "type": "training_objective", "sections": ["Method"], "evidence_refs": ["S007_C01:M-2"],
            "confidence": 0.8, "inputs": [], "outputs": [],
        },
    ]
    plan = {
        "problem_groups": [{
            "group_id": "P1", "members": ["P:RP1", "P:RP2", "P:DOES_NOT_EXIST"],
            "representative": "P:RP1", "canonical_claim": "Low-light ambiguity obscures object boundaries",
            "problem_type": "data_gap", "confidence": 0.9,
        }],
        "method_groups": [
            {
                "group_id": "M1", "members": ["M:M1"], "representative": "M:M1",
                "canonical_claim": "A dual-branch boundary-aware fusion network restores low-light images",
                "method_type": "architecture", "confidence": 0.9,
            },
            {
                "group_id": "M2", "members": ["M:M2"], "representative": "M:M2",
                "canonical_claim": "Boundary consistency guides training",
                "method_type": "training_objective", "confidence": 0.8,
            },
        ],
        "links": [{
            "problem_group": "P1", "method_group": "M1", "relation": "directly_addresses",
            "relation_evidence_source_ids": ["P:RP1", "M:M1"], "confidence": 0.9,
        }],
    }
    result, audit = materialize_semantic_core_plan(
        plan, records, target_problems=1, target_methods=2, max_problems=1, max_methods=2,
    )
    assert len(result["paper_research_problems"]) == 1
    assert len(result["paper_methods"]) == 2
    assert result["paper_research_problems"][0]["evidence_refs"] == [
        "S003_C01:RP-1", "S004_C01:RP-2",
    ]
    assert set(result["problem_method_links"][0]["evidence_refs"]) == {
        "S003_C01:RP-1", "S006_C01:M-1",
    }
    assert audit["retained_problem_count"] <= audit["budgets"]["max_problems"]
    assert audit["retained_method_count"] <= audit["budgets"]["max_methods"]


def test_semantic_core_rejects_relation_without_both_endpoint_kinds() -> None:
    records = [
        {"source_id": "P:RP1", "kind": "problem", "claim": "a central research gap", "type": "method_gap", "sections": ["Abstract"], "evidence_refs": ["S1_C01:RP-1"], "confidence": 0.8},
        {"source_id": "M:M1", "kind": "method", "claim": "a central proposed architecture", "type": "architecture", "sections": ["Method"], "evidence_refs": ["S2_C01:M-1"], "confidence": 0.8, "inputs": [], "outputs": []},
    ]
    plan = {
        "problem_groups": [{"group_id": "P1", "members": ["P:RP1"], "canonical_claim": "A central research gap", "problem_type": "method_gap"}],
        "method_groups": [{"group_id": "M1", "members": ["M:M1"], "canonical_claim": "A central proposed architecture", "method_type": "architecture"}],
        "links": [{"problem_group": "P1", "method_group": "M1", "relation": "directly_addresses", "relation_evidence_source_ids": ["M:M1"]}],
    }
    result, _ = materialize_semantic_core_plan(
        plan, records, target_problems=1, target_methods=1, max_problems=1, max_methods=1,
    )
    assert result["problem_method_links"] == []


def test_semantic_core_context_and_prompt_do_not_expose_gold() -> None:
    markdown = "# Paper\nfront\n# Abstract\ncentral claim\n# 1 Introduction\nproblem\n# 2 Method\nmethod\n# References\nsecret bibliography"
    context = select_primary_context(markdown, max_chars=10000)
    assert "central claim" in context
    assert "secret bibliography" not in context
    prompt = semantic_core_user_prompt(
        context, "[]", target_problems=6, target_methods=8, max_problems=10, max_methods=12,
    )
    assert "never exceed 10" in prompt
    assert "never exceed 12" in prompt
    assert "Do not read or imitate any gold-standard annotation" in SEMANTIC_CORE_SYSTEM
