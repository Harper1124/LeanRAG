import os
import json
try:  # package import (python -m multimodal.media_graph_pipeline)
    from ._utils import split_string_by_multi_markers, _handle_single_entity_extraction, \
        _handle_single_relationship_extraction, clean_str, pack_user_ass_to_openai_messages
except ImportError:  # legacy direct script execution
    from _utils import split_string_by_multi_markers, _handle_single_entity_extraction, \
        _handle_single_relationship_extraction, clean_str, pack_user_ass_to_openai_messages
from collections import Counter, defaultdict
from prompt import PROMPTS
import asyncio
import re
import copy

"""
GraphRAG 风格的实体/关系抽取入口。

这个文件读取 file_chunk.py 生成的 chunk JSON，然后分两步调用 LLM：
1. entity_extraction：从每个 chunk 中抽取实体和实体描述。
2. relation_extraction：基于第一步抽出的实体列表，再抽取实体之间的关系。

最终输出：
- entity.jsonl：给 build_graph.py 作为原始实体输入。
- relation.jsonl：给 build_graph.py 作为原始关系输入。
"""


def get_chunk(chunk_file):
    """读取 chunk 文件，并转换成 {hash_code: text}，hash_code 会作为 source_id。"""
    doc_name=os.path.basename(chunk_file).rsplit(".",1)[0]
    with open(chunk_file, "r") as f:
            corpus=json.load(f)
    chunks = {item["hash_code"]: item["text"] for item in corpus}
    return chunks

async def triple_extraction(chunks, use_llm_func, output_dir=None, write_func=None):
    """
    对所有 chunk 异步执行实体和关系抽取。

    use_llm_func 通常来自 InstanceManager.generate_text_asy，内部会管理并发请求。
    这里先抽实体，再把同一 chunk 内的实体列表放入关系抽取 prompt，从而减少无关关系。
    """
    
    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings

    
    already_processed = 0
    already_entities = 0
    already_relations = 0
    ordered_chunks = list(chunks.items())
    async def _process_single_content_entity(chunk_key_dp,use_llm_func):           # for each chunk, run the func
        """处理单个 chunk 的实体抽取，并把 LLM 输出解析成 maybe_nodes。"""
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        entity_extract_prompt = PROMPTS["entity_extraction"]        # give 3 examples in the prompt context
        relation_extract_prompt = PROMPTS["relation_extraction"]
        continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"] 
        context_base_entity = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["META_ENTITY_TYPES"])
    )
        entity_extract_max_gleaning=1
        hint_prompt = entity_extract_prompt.format(**context_base_entity, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_entity["record_delimiter"], context_base_entity["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_entity["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    
    entity_results = await asyncio.gather(
        *[_process_single_content_entity(c,use_llm_func) for c in ordered_chunks]
    )
    print()  # clear the progress bar

    # Keep one row per chunk evidence. Cross-chunk name/endpoint fusion belongs to
    # the deterministic merge stage; overwriting a dict key here loses source_id.
    all_entities = []
    for item in entity_results:
        for values in item[0].values():
            all_entities.extend(values)
    context_entities = {key[0]: list(x[0].keys()) for key, x in zip(ordered_chunks, entity_results)}
    already_processed = 0
    async def _process_single_content_relation(chunk_key_dp,use_llm_func):           # for each chunk, run the func
        """处理单个 chunk 的关系抽取；只在该 chunk 已抽到的实体集合内找关系。"""
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        entity_extract_prompt = PROMPTS["entity_extraction"]        # give 3 examples in the prompt context
        relation_extract_prompt = PROMPTS["relation_extraction"]
        continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"] 
        entities = context_entities[chunk_key]
        context_base_relation = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entities=",".join(entities)
            )
        entity_extract_max_gleaning=1
        hint_prompt = relation_extract_prompt.format(**context_base_relation, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_relation["record_delimiter"], context_base_relation["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_relation["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    relation_results = await asyncio.gather(
        *[_process_single_content_relation(c,use_llm_func) for c in ordered_chunks]
    )
    print()
    all_relations = []
    for item in relation_results:
        for values in item[1].values():
            all_relations.append(values)
    save_entity=[]
    save_relation=[]
    save_entity.extend(copy.deepcopy(all_entities))
    save_relation.extend(copy.deepcopy(all_relations))
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if write_func is None:
            from tools.utils import write_jsonl as write_func
        write_func(save_entity, os.path.join(output_dir, "entity.jsonl"))
        write_func(save_relation, os.path.join(output_dir, "relation.jsonl"))
    return save_entity, save_relation
   
            
    
    
    
    
    
if __name__ == "__main__":
    from tools.utils import InstanceManager

    MODEL = "qwen3_14b"
    num=4
    instanceManager=InstanceManager(
        url="http://xxx",
        ports=[8001 for i in range(num)],
        gpus=[i for i in range(num)],
        generate_model=MODEL,
        startup_delay=30
    )
    use_llm=instanceManager.generate_text_asy
    chunk_file="/data/zyz/LeanRAG/datasets/mix/mix_chunk.json"
    chunks=get_chunk(chunk_file)
    output_dir="ttt"
    loop = asyncio.get_event_loop()
    loop.run_until_complete(triple_extraction(chunks, use_llm,output_dir))

    
