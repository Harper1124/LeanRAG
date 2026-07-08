from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from multimodal.graph.mm_graph_loader import MMGraph, load_mm_graph


DEFAULT_CONTEXT_AGGREGATION = {
    "enabled": True,
    "mode": "media_projection_lca",
    "max_media_projection_entities": 8,
    "max_text_projection_entities": 8,
    "max_lca_input_entities": 16,
    "max_page_context_nodes": 8,
    "max_aggregation_context_chars": 12000,
    "allow_single_letter_entities_for_projection": False,
    "filter_stopword_entities": True,
    "page_context": {
        "use_neighbor_pages": False,
        "allow_hop2_pages_as_background": True,
        "max_neighbor_page_context_nodes": 2,
    },
    "lca": {
        "method": "query_graph_describe_reuse",
        "fallback_to_page_context": True,
    },
}
STOPWORD_ENTITIES = {"a", "an", "the", "to", "at", "it", "by", "for", "in", "as", "is", "on", "of", "and", "or", "with", "from"}
FORMULA_TYPES = {"formula_symbol", "variable", "metric", "math_symbol"}


def multimodal_context_aggregation(
    question: str,
    evidence_package: dict,
    merged_candidates: list[dict[str, Any]],
    graph: MMGraph,
    db,
    global_config: dict,
    query_info: dict | None = None,
) -> dict[str, Any]:
    del query_info
    config = _context_config(global_config)
    result = {
        "enabled": bool(config["enabled"]),
        "mode": config["mode"],
        "anchor_nodes": [],
        "projected_nodes": [],
        "filtered_projection_candidates": [],
        "lca_input_entities": [],
        "lca_result": {"method": "", "input_entities": [], "lca_nodes": [], "paths": [], "describe_preview": ""},
        "page_context_nodes": [],
        "kept_media_nodes": [],
        "aggregation_context": "",
        "aggregation_context_added": False,
        "errors": [],
    }
    if not result["enabled"]:
        return result

    node_lookup = _candidate_lookup(merged_candidates, graph)
    anchors = _select_anchor_nodes(evidence_package, node_lookup)
    result["anchor_nodes"] = [_slim_node(anchor) for anchor in anchors]
    result["kept_media_nodes"] = [_slim_node(anchor) for anchor in anchors if anchor.get("node_type") == "media"]

    projected_by_id: dict[str, dict[str, Any]] = {}
    page_context_by_id: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        node_type = anchor.get("node_type")
        if node_type == "media":
            projections = _project_media(anchor, merged_candidates, graph, config, result)
            if not projections:
                for node in _page_context_for_anchor(anchor, graph, config):
                    page_context_by_id[node["node_id"]] = node
            for projection in projections:
                _accept_projection(projection, projected_by_id, result, question, config)
        elif node_type == "text":
            projections = _project_text(anchor, graph, config, result)
            if not projections:
                for node in _page_context_for_anchor(anchor, graph, config):
                    page_context_by_id[node["node_id"]] = node
            for projection in projections:
                _accept_projection(projection, projected_by_id, result, question, config)
        elif node_type == "entity":
            projection = _projection_record(anchor, anchor, "entity_anchor", 1.0, "entity anchor")
            _accept_projection(projection, projected_by_id, result, question, config)
        elif node_type == "page":
            for node in _page_context_for_anchor(anchor, graph, config):
                page_context_by_id[node["node_id"]] = node

    result["projected_nodes"] = sorted(projected_by_id.values(), key=lambda item: item.get("score", 0), reverse=True)
    result["lca_input_entities"] = _lca_input_entities(result["projected_nodes"], int(config["max_lca_input_entities"]))
    result["page_context_nodes"] = list(page_context_by_id.values())[: int(config["max_page_context_nodes"])]
    if result["lca_input_entities"]:
        result["lca_result"] = _run_original_leanrag_context(question, db, global_config, result["lca_input_entities"], config)
        result["aggregation_context"] = str(result["lca_result"].get("describe") or result["lca_result"].get("describe_preview") or "")
        result["aggregation_context"] = result["aggregation_context"][: int(config["max_aggregation_context_chars"])]
        result["aggregation_context_added"] = bool(result["aggregation_context"])
    elif result["page_context_nodes"]:
        result["aggregation_context"] = _format_page_context(result["page_context_nodes"], int(config["max_aggregation_context_chars"]))
        result["aggregation_context_added"] = bool(result["aggregation_context"])
    return result


def merge_aggregation_context(evidence_package: dict, context_agg: dict, config: dict | None = None) -> dict:
    del config
    merged = dict(evidence_package)
    for key, value in evidence_package.items():
        if isinstance(value, list):
            merged[key] = list(value)
    merged["aggregation_context"] = context_agg.get("aggregation_context", "") if context_agg.get("enabled") else ""
    merged["projection_context"] = context_agg
    selected = list(merged.get("all_selected_nodes", []))
    seen = {item.get("node_id") for item in selected}
    for node in context_agg.get("page_context_nodes", []):
        if node.get("node_id") not in seen:
            selected.append(node)
            seen.add(node.get("node_id"))
    merged["all_selected_nodes"] = selected
    return merged


def is_valid_projection_entity(entity_node: dict, context: dict | None = None) -> bool:
    return _entity_filter_reason(entity_node, context) is None


def _project_media(anchor: dict, merged_candidates: list[dict[str, Any]], graph: MMGraph, config: dict, result: dict) -> list[dict[str, Any]]:
    source_id = anchor.get("node_id")
    projections: list[dict[str, Any]] = []
    for candidate in merged_candidates:
        if candidate.get("node_type") != "entity":
            continue
        debug = candidate.get("debug") or {}
        paths = [debug] + [path for path in debug.get("expansion_paths", []) if isinstance(path, dict)]
        for path in paths:
            if path.get("from_anchor") == source_id and path.get("edge_type") == "entity_link_media":
                projections.append(_projection_record(anchor, candidate, "entity_link_media", _path_score(path, 1.0), "expanded entity_link_media hop=1" if path.get("hop") == 1 else "expanded entity_link_media"))
                break
    existing = {item["projected_node_id"] for item in projections}
    for edge, neighbor_id in graph.get_neighbor_edges(source_id, edge_types={"entity_link_media"}, direction="both"):
        node = graph.get_node(neighbor_id)
        if node and node.get("node_type") == "entity" and node.get("node_id") not in existing:
            projections.append(_projection_record(anchor, node, "entity_link_media", float(edge.get("weight", 1.0) or 1.0), "graph entity_link_media neighbor"))
            existing.add(node.get("node_id"))
    for name in (anchor.get("metadata") or {}).get("attached_entity_names") or []:
        node = _find_entity_by_name(graph, str(name))
        if node and node.get("node_id") not in existing:
            projections.append(_projection_record(anchor, node, "attached_entity_names", 0.8, "media attached_entity_names"))
            existing.add(node.get("node_id"))
    media_text = _node_text(anchor)
    for node in graph.nodes.values():
        if node.get("node_type") != "entity" or node.get("node_id") in existing:
            continue
        if _name_in_text(_entity_name(node), media_text):
            projections.append(_projection_record(anchor, node, "media_text_exact_match", 0.7, "entity name exact match in media text"))
            existing.add(node.get("node_id"))
    projections.sort(key=lambda item: item["score"], reverse=True)
    return projections[: int(config["max_media_projection_entities"])]


def _project_text(anchor: dict, graph: MMGraph, config: dict, result: dict) -> list[dict[str, Any]]:
    del result
    projections = []
    source_id = anchor.get("node_id")
    for edge, neighbor_id in graph.get_neighbor_edges(source_id, edge_types={"text_mentions_entity"}, direction="both"):
        node = graph.get_node(neighbor_id)
        if node and node.get("node_type") == "entity":
            projections.append(_projection_record(anchor, node, "text_mentions_entity", float(edge.get("weight", 1.0) or 1.0), "graph text_mentions_entity"))
    source_hash = (anchor.get("raw_ref") or {}).get("hash_code")
    for node in graph.nodes.values():
        if node.get("node_type") != "entity":
            continue
        source_ids = str((node.get("raw_ref") or {}).get("source_id") or "").split("|")
        if source_hash and source_hash in source_ids:
            projections.append(_projection_record(anchor, node, "entity_source_id", 0.8, "entity source_id matches text hash"))
        elif _name_in_text(_entity_name(node), _node_text(anchor)):
            projections.append(_projection_record(anchor, node, "text_exact_match", 0.6, "entity name exact match in text"))
    return _dedupe_projections(projections)[: int(config["max_text_projection_entities"])]


def _accept_projection(projection: dict[str, Any], projected_by_id: dict[str, dict[str, Any]], result: dict, question: str, config: dict) -> None:
    entity_node = projection.get("_entity_node") or {}
    reason = _entity_filter_reason(entity_node, {"question": question, "config": config, "source_node_id": projection.get("source_node_id")})
    if reason:
        result["filtered_projection_candidates"].append({"node_id": projection.get("projected_node_id"), "entity_name": _entity_name(entity_node), "reason": reason, "source_node_id": projection.get("source_node_id")})
        return
    public = {key: value for key, value in projection.items() if key != "_entity_node"}
    current = projected_by_id.get(public["projected_node_id"])
    if current is None or public["score"] > current.get("score", 0):
        projected_by_id[public["projected_node_id"]] = public


def _entity_filter_reason(entity_node: dict, context: dict | None = None) -> str | None:
    context = context or {}
    config = context.get("config") or {}
    name = _entity_name(entity_node).strip()
    lowered = name.lower()
    if len(name) < 2:
        if _allow_single_letter(entity_node, context, config):
            return None
        return "single_letter_entity_filtered" if re.fullmatch(r"[A-Za-z]", name) else "low_information_entity_filtered"
    if re.fullmatch(r"\W+", name) or re.fullmatch(r"\d+", name):
        return "low_information_entity_filtered"
    if config.get("filter_stopword_entities", True) and lowered in STOPWORD_ENTITIES:
        return "stopword_entity_filtered"
    if re.fullmatch(r"[A-Za-z]", name) and not _allow_single_letter(entity_node, context, config):
        return "single_letter_entity_filtered"
    return None


def _allow_single_letter(entity_node: dict, context: dict, config: dict) -> bool:
    if config.get("allow_single_letter_entities_for_projection"):
        return True
    entity_type = str((entity_node.get("metadata") or {}).get("entity_type") or "").lower()
    if entity_type in FORMULA_TYPES:
        return True
    name = re.escape(_entity_name(entity_node))
    text = str(context.get("question") or "") + "\n" + str(context.get("nearby_text") or "")
    return re.search(rf"\b(variable|matrix|node|model)\s+{name}\b", text, re.IGNORECASE) is not None


def _run_original_leanrag_context(question: str, db, global_config: dict, input_entities: list[dict[str, Any]], config: dict) -> dict[str, Any]:
    method = (config.get("lca") or {}).get("method", "query_graph_describe_reuse")
    result = {"method": method, "input_entities": input_entities, "lca_nodes": [], "paths": [], "describe_preview": "", "describe": "", "errors": []}
    if method != "query_graph_describe_reuse":
        result["errors"].append("only query_graph_describe_reuse is implemented for Phase 5")
        return result
    try:
        required = {"working_dir", "chunks_file", "use_llm_func", "embeddings_func", "level_mode", "topk"}
        missing = sorted(key for key in required if key not in global_config)
        if missing:
            raise KeyError(f"query_graph config missing keys: {missing}")
        from query_graph import query_graph

        describe, _ = query_graph(global_config, db, question)
        result["describe"] = str(describe)
        result["describe_preview"] = str(describe)[:1000]
    except Exception as exc:
        result["errors"].append(str(exc))
        fallback = "Projected LeanRAG entities:\n" + json.dumps(input_entities, ensure_ascii=False, indent=2)
        result["describe"] = fallback
        result["describe_preview"] = fallback[:1000]
    return result


def _select_anchor_nodes(evidence_package: dict, node_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = []
    for key in ("all_selected_nodes", "visual_evidence", "table_evidence", "entity_evidence", "text_evidence", "page_evidence"):
        ordered.extend(evidence_package.get(key) or [])
    seen = set()
    anchors = []
    for item in ordered:
        node_id = item.get("node_id")
        if not node_id or node_id in seen or item.get("node_type") not in {"media", "entity", "text", "page"}:
            continue
        seen.add(node_id)
        merged = dict(node_lookup.get(node_id) or {})
        merged.update(item)
        anchors.append(merged)
    return anchors


def _candidate_lookup(merged_candidates: list[dict[str, Any]], graph: MMGraph) -> dict[str, dict[str, Any]]:
    lookup = {node_id: dict(node) for node_id, node in graph.nodes.items()}
    for candidate in merged_candidates or []:
        node_id = candidate.get("node_id")
        if not node_id:
            continue
        base = dict(lookup.get(node_id) or {})
        base.update(candidate)
        lookup[node_id] = base
    return lookup


def _page_context_for_anchor(anchor: dict, graph: MMGraph, config: dict) -> list[dict[str, Any]]:
    page_id = anchor.get("page_id")
    if page_id is None:
        return []
    nodes = [
        _slim_node(node)
        for node in graph.nodes.values()
        if node.get("page_id") == page_id and node.get("node_type") in {"text", "media", "page"} and node.get("node_id") != anchor.get("node_id")
    ]
    nodes.sort(key=lambda item: (0 if item.get("node_type") == "text" else 1, str(item.get("node_id"))))
    return nodes[: int(config["max_page_context_nodes"])]


def _lca_input_entities(projected_nodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for item in projected_nodes:
        node_id = item.get("projected_node_id")
        if node_id in seen:
            continue
        seen.add(node_id)
        rows.append({"node_id": node_id, "entity_name": item.get("entity_name"), "score": item.get("score"), "source_node_id": item.get("source_node_id"), "projection_type": item.get("projection_type")})
    return rows[:limit]


def _projection_record(source: dict, entity: dict, projection_type: str, score: float, reason: str) -> dict[str, Any]:
    return {
        "source_node_id": source.get("node_id"),
        "source_node_type": source.get("node_type"),
        "projected_node_id": entity.get("node_id"),
        "projected_node_type": "entity",
        "entity_name": _entity_name(entity),
        "projection_type": projection_type,
        "score": float(score),
        "reason": reason,
        "accepted": True,
        "_entity_node": entity,
    }


def _dedupe_projections(projections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {}
    for item in projections:
        node_id = item.get("projected_node_id")
        if node_id not in by_id or item.get("score", 0) > by_id[node_id].get("score", 0):
            by_id[node_id] = item
    return sorted(by_id.values(), key=lambda item: item.get("score", 0), reverse=True)


def _find_entity_by_name(graph: MMGraph, name: str) -> dict[str, Any] | None:
    normalized = _normalize_name(name)
    for node in graph.nodes.values():
        if node.get("node_type") == "entity" and _normalize_name(_entity_name(node)) == normalized:
            return node
    return None


def _entity_name(node: dict[str, Any]) -> str:
    raw = node.get("raw_ref") or {}
    return str(raw.get("entity_name") or str(node.get("text_for_embedding") or "").split("\n")[0] or node.get("node_id", "").split("::")[-1])


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _name_in_text(name: str, text: str) -> bool:
    if not name:
        return False
    if re.fullmatch(r"[A-Za-z0-9_ -]+", name):
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name.lower())}(?![A-Za-z0-9_])", text.lower()) is not None
    return name in text


def _node_text(node: dict[str, Any]) -> str:
    raw = node.get("raw_ref") or {}
    return "\n".join(str(part or "") for part in [node.get("text_for_embedding"), node.get("caption"), node.get("ocr_text"), node.get("summary"), raw.get("table_markdown"), raw.get("table_html")])


def _path_score(path: dict, fallback: float) -> float:
    return float(path.get("edge_weight", fallback) or fallback)


def _slim_node(node: dict[str, Any]) -> dict[str, Any]:
    return {"node_id": node.get("node_id"), "node_type": node.get("node_type"), "page_id": node.get("page_id"), "source": node.get("source"), "score": node.get("score"), "debug": node.get("debug") or {}, "raw_ref": node.get("raw_ref") or {}, "metadata": node.get("metadata") or {}, "text_for_embedding": node.get("text_for_embedding", "")}


def _format_page_context(nodes: list[dict[str, Any]], max_chars: int) -> str:
    return json.dumps([_slim_node(node) for node in nodes], ensure_ascii=False, indent=2)[:max_chars]


def _context_config(config: dict | None) -> dict[str, Any]:
    value = dict(DEFAULT_CONTEXT_AGGREGATION)
    value["page_context"] = dict(DEFAULT_CONTEXT_AGGREGATION["page_context"])
    value["lca"] = dict(DEFAULT_CONTEXT_AGGREGATION["lca"])
    section = {}
    if isinstance(config, dict):
        section = config.get("context_aggregation") if isinstance(config.get("context_aggregation"), dict) else {}
        if isinstance(config.get("multimodal"), dict) and isinstance(config["multimodal"].get("context_aggregation"), dict):
            section = config["multimodal"]["context_aggregation"]
    for key, item in section.items():
        if key in {"page_context", "lca"} and isinstance(item, dict):
            value[key].update(item)
        else:
            value[key] = item
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 multimodal context aggregation for one anchor node.")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--anchor_node_id", required=True)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    graph = load_mm_graph(Path(args.nodes).parent, node_file=args.nodes, edge_file=args.edges)
    anchor = graph.get_node(args.anchor_node_id)
    if not anchor:
        raise SystemExit(f"anchor not found: {args.anchor_node_id}")
    candidate = {**anchor, "score": 1.0, "source": "direct_recall", "retrievers": ["manual_anchor"]}
    evidence_package = {"all_selected_nodes": [candidate], "visual_evidence": [candidate] if anchor.get("node_type") == "media" else [], "table_evidence": [], "entity_evidence": [candidate] if anchor.get("node_type") == "entity" else [], "text_evidence": [candidate] if anchor.get("node_type") == "text" else [], "page_evidence": [candidate] if anchor.get("node_type") == "page" else []}
    result = multimodal_context_aggregation(args.query, evidence_package, [candidate], graph, None, {"context_aggregation": {"enabled": True}}, {})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
