from collections import Counter, defaultdict
from dataclasses import field
import json
import os
import time
import logging
import numpy as np
from openai import OpenAI
import pymysql
import tiktoken
from tqdm import tqdm
import yaml
from tools.utils import InstanceManager
from openai import  OpenAI
from database_utils import build_vector_search,search_vector_search,find_tree_root,\
    search_nodes_link,search_nodes,search_community,search_chunks,get_text_units,find_path
from prompt import GRAPH_FIELD_SEP, PROMPTS
from itertools import combinations

logger=logging.getLogger(__name__)

"""
LeanRAG 查询阶段主入口。

这个文件负责把一个用户问题转成最终答案，核心步骤是：
1. 对 query 做 embedding。
2. 在 Milvus Lite 向量索引中召回 top-k 相关实体或聚合实体。
3. 通过 MySQL 中的 parent 字段向上寻找实体所在的层级路径。
4. 收集路径上的聚合实体摘要、相关边关系、原文 chunk。
5. 将这些结构化证据拼成 prompt 上下文，调用 LLM 生成回答。
"""

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']
EMBEDDING_MODEL = config['glm']['model']
EMBEDDING_URL = config['glm']['base_url']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

def embedding(texts: list[str]) -> np.ndarray:
    """使用和建图阶段一致的 embedding 服务，把查询文本转换成向量。"""
    model_name = EMBEDDING_MODEL
    client = OpenAI(
        api_key=EMBEDDING_MODEL,
        base_url=EMBEDDING_URL
    ) 
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)

tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_text(text, max_tokens=4096):
    """按 token 截断文本，避免超过模型或 embedding 服务的输入长度限制。"""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def get_reasoning_chain(global_config,entities_set):
    """
    根据召回实体构造推理路径。

    对 top-k 召回实体两两组合：
    - 先用 find_tree_root 找到每个实体从自身到根节点的 parent 链。
    - 找到两条链的交汇位置，拼成一条从实体 A 到实体 B 的层级路径。
    - 再查询路径节点之间是否存在 relations，收集这些关系描述作为推理证据。

    返回值：
    - reasoning_path：实体/聚合实体名称组成的路径列表。
    - reasoning_path_information_description：路径上关系的文本描述。
    """
    maybe_edges=list(combinations(entities_set,2))
    reasoning_path=[]
    reasoning_path_information=[]
    db_name=global_config['working_dir'].split("/")[-1]
    information_record=[]
    for edge in maybe_edges:
        a_path=[]
        b_path=[]
        node1=edge[0]
        node2=edge[1]
        node1_tree=find_tree_root(db_name,node1)
        node2_tree=find_tree_root(db_name,node2)
        
        # if node1_tree[1]!=node2_tree[1] :
        #     print("debug")
        for index,(i,j) in enumerate(zip(node1_tree,node2_tree)):
            if i==j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i!=j :
                a_path.append(i)
                b_path.append(j)
            
            
        reasoning_path.append(a_path+[b_path[len(b_path)-1-i] for  i in range(len(b_path))]) 
        a_path=list(set(a_path))
        b_path=list(set(b_path))
        for maybe_edge in list(combinations(a_path+b_path,2)):
            if maybe_edge[0]==maybe_edge[1]:
                continue
            information=search_nodes_link(maybe_edge[0],maybe_edge[1],global_config['working_dir'])
            if information==None:
                continue
            information_record.append(information)
            reasoning_path_information.append([maybe_edge[0],maybe_edge[1],information[2]])
    # columns=['src_tgt','tgt_src','path_description']
    # reasoning_path_information_description="\t\t".join(columns)+"\n"
    temp_relations_information=list(set([information[2] for information in reasoning_path_information]))
    reasoning_path_information_description="\n".join(temp_relations_information)  
    return  reasoning_path,reasoning_path_information_description

def get_entity_description(global_config,entities_set,mode=0):
    """
    将向量召回得到的实体结果格式化成表格文本。

    entities_set 中每个元素通常是：
    (entity_name, parent, description, source_id)
    这部分信息会直接进入最终 RAG prompt。
    """
    
    
    
    columns=['entity_name','parent','description']
    entity_descriptions="\t\t".join(columns)+"\n"
    entity_descriptions+="\n".join([information[0]+"\t\t"+information[1]+"\t\t"+information[2] for information in entities_set])

    return entity_descriptions
        
def get_aggregation_description(global_config,reasoning_path,if_findings=False):
    """
    根据推理路径收集聚合实体的社区摘要。

    reasoning_path 中既可能包含原始实体，也可能包含上层聚合实体；这里取出所有路径节点，
    到 communities 表中查询 LLM 在建图阶段生成的聚合描述。
    """
    
    aggregation_results=[]
    
    communities=set([community for each_path in reasoning_path for community in each_path])
    for community in communities:
        temp=search_community(community,global_config['working_dir'])
        if temp=="":
            continue
        aggregation_results.append(temp)
    if if_findings:
        columns=['entity_name','entity_description','findings']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1])+"\t\t"+information[2] for information in aggregation_results])
    else:
        columns=['entity_name','entity_description']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1]) for information in aggregation_results])
    return aggregation_descriptions,communities

def format_text_units(text_units):
    """
    Format retrieved source chunks as structured evidence blocks.

    Each evidence block keeps multimodal metadata so the answer prompt can
    distinguish text, table, and image-derived descriptions.
    """
    evidence_blocks = []
    for index, item in enumerate(text_units, start=1):
        evidence_blocks.append({
            "id": index,
            "modality": item.get("modality", "text"),
            "page": item.get("page"),
            "asset_path": item.get("asset_path"),
            "summary": item.get("summary", ""),
            "text": item.get("text", ""),
        })
    return json.dumps(evidence_blocks, ensure_ascii=False, indent=2)

def query_graph(global_config,db,query):
    """
    单次问答主流程。

    输入：
    - global_config：包含 working_dir、chunks_file、topk、level_mode、LLM 函数等配置。
    - db：MySQL 连接对象；当前函数主要通过 database_utils 内部函数重新连接查询。
    - query：用户问题。

    输出：
    - describe：本次喂给 LLM 的结构化证据上下文，便于调试。
    - response：LLM 基于证据生成的最终回答。
    """
    use_llm_func: callable = global_config["use_llm_func"]
    embedding: callable=global_config["embeddings_func"]
    b=time.time()
    level_mode=global_config['level_mode']
    topk=global_config['topk']
    chunks_file=global_config["chunks_file"]
    # 第一步：向量召回。level_mode 控制只搜原始实体、只搜聚合实体，还是全层级都搜。
    entity_results=search_vector_search(global_config['working_dir'],embedding(query),topk=topk,level_mode=level_mode)
    v=time.time()
    res_entity=[i[0]for i in entity_results]
    chunks=[i[-1]for i in entity_results]
    entity_descriptions=get_entity_description(global_config,entity_results)
    # 第二步：在层级树上寻找召回实体之间的连接路径，并取出路径相关关系。
    reasoning_path,reasoning_path_information_description=get_reasoning_chain(global_config,res_entity)
    # reasoning_path,reasoning_path_information_description=get_path_chain(global_config,res_entity)
    # 第三步：获取路径上聚合实体的摘要，这些摘要相当于“上层语义证据”。
    aggregation_descriptions,aggregation=get_aggregation_description(global_config,reasoning_path)
    # chunks=search_chunks(global_config['working_dir'],aggregation)
    # 第四步：根据召回实体的 source_id 回查原文 chunk，补充最细粒度证据。
    text_units=get_text_units(global_config['working_dir'],chunks,chunks_file,k=5)
    structured_text_units=format_text_units(text_units)
    describe=f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {structured_text_units}
    """
    e=time.time()
    
    # print(describe)
    # 第五步：把结构化证据填入回答模板，交给生成模型输出最终答案。
    sys_prompt =PROMPTS["rag_response"].format(context_data=describe)
    response=use_llm_func(query,system_prompt=sys_prompt)
    g=time.time()
    print(f"embedding time: {v-b:.2f}s")
    print(f"query time: {e-v:.2f}s")
    
    print(f"response time: {g-e:.2f}s")
    return describe,response
if __name__=="__main__":
    # 示例入口：连接 MySQL，配置图谱目录和 chunk 文件，然后对单个 query 做检索生成。
    db = pymysql.connect(host='localhost', user='root',port=4321,
                    passwd='123',  charset='utf8mb4')
    global_config={}
    WORKING_DIR = f"/data/zyz/trag_ds/exp/lean_full_cs10_top10_chunk5/mix"
    global_config['chunks_file']="/data/zyz/trag_ds/hi_ex/mix/kv_store_text_chunks.json"
    global_config['embeddings_func']=embedding
    global_config['working_dir']=WORKING_DIR
    global_config['topk']=10
    global_config['level_mode']=1
    num=4
    instanceManager=InstanceManager(
        url="http://xxx",
        ports=[8001 for i in range(num)],
        gpus=[i for i in range(num)],
        generate_model="qwen3_32b",
        startup_delay=30
    )
    
    global_config['use_llm_func']=instanceManager.generate_text
    query="What is the maturity date of the credit agreement?"
    topk=10
    ref,response=query_graph(global_config,db,query)
    print(ref)
    print("#"*20)
    print(response)
    db.close()
    
