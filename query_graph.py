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
from openai import  OpenAI
from tools.utils import response as use_llm_func
from database_utils import build_vector_search,search_vector_search,find_tree_root,search_nodes_link,search_nodes,search_community
from prompt import GRAPH_FIELD_SEP, PROMPTS
from itertools import combinations

logger=logging.getLogger(__name__)
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
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def get_reasoning_chain(global_config,db,entities_set):
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
        node1_tree=find_tree_root(db,db_name,node1)
        node2_tree=find_tree_root(db,db_name,node2)
        
        # if node1_tree[1]!=node2_tree[1] :
        #     print("debug")
        for index,(i,j) in enumerate(zip(node1_tree,node2_tree)):
            if i!=j :
                a_path.append(i)
                b_path.append(j)
            if i==j:
                a_path.append(i)
                break
        reasoning_path.append(a_path+[b_path[len(b_path)-1-i] for  i in range(len(b_path))]) 
        a_path=list(set(a_path))
        b_path=list(set(b_path))
        for maybe_edge in list(combinations(a_path+b_path,2)):
            if maybe_edge[0]==maybe_edge[1]:
                continue
            information=search_nodes_link(maybe_edge[0],maybe_edge[1],db,global_config['working_dir'])
            if information==None:
                continue
            information_record.append(information)
            reasoning_path_information.append([maybe_edge[0],maybe_edge[1],information[2]])
    # columns=['src_tgt','tgt_src','path_description']
    # reasoning_path_information_description="\t\t".join(columns)+"\n"
    temp_relations_information=list(set([information[2] for information in reasoning_path_information]))
    reasoning_path_information_description="\n".join(temp_relations_information)  
    return  reasoning_path,reasoning_path_information_description

def get_entity_description(global_config,db,entities_set,mode=0):
    
    
    
    columns=['entity_name','parent','description']
    entity_descriptions="\t\t".join(columns)+"\n"
    entity_descriptions+="\n".join([information[0]+"\t\t"+information[1]+"\t\t"+information[2] for information in entities_set])
    '''
    两个节点在同一个实体下导致失去联系，这种情况只会在底层实体中出现
    0609 记录 首先实体在底层中时，直接返回他们关系即可，而在聚合实体中则不会出现不返回关系的情况
    其次,返回所有节点信息 或 所有节点信息加关系这个信息量太大，可能会导致噪声
    目前采取方法将底层图加入到数据库中
    '''
    # if mode==0:
    #     e_p=[[information[0],information[4]] for information in entity_informations]
    #     parent_counter = Counter(parent for _, parent in e_p)
    #     parent_to_entities = defaultdict(list)
    #     for entity, parent in e_p:
    #         if parent_counter[parent] > 2:
    #             parent_to_entities[parent].append(entity)
    return entity_descriptions
        
def get_aggregation_description(global_config,db,reasoning_path,if_findings=False):
    
    aggregation_results=[]
    
    communities=set([community for each_path in reasoning_path for community in each_path])
    for community in communities:
        temp=search_community(community,db,global_config['working_dir'])
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
    return aggregation_descriptions
def query_graph(global_config,db,query):
    use_llm_func: callable = global_config["use_llm_func"]
    b=time.time()
    level_mode=global_config['level_mode']
    topk=global_config['topk']
    port=global_config['port']
    entity_results=search_vector_search(global_config['working_dir'],embedding(query),topk=topk,level_mode=level_mode)
    v=time.time()
    res_entity=[i[0]for i in entity_results]
    entity_descriptions=get_entity_description(global_config,db,entity_results)
    reasoning_path,reasoning_path_information_description=get_reasoning_chain(global_config,db,res_entity)
    aggregation_descriptions=get_aggregation_description(global_config,db,reasoning_path)
    describe=f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    """
    e=time.time()
    
    # print(describe)
    sys_prompt =PROMPTS["rag_response"].format(context_data=describe)
    response=use_llm_func(query,system_prompt=sys_prompt,port=port)
    g=time.time()
    print(f"embedding time: {v-b:.2f}s")
    print(f"query time: {e-v:.2f}s")
    
    print(f"response time: {g-e:.2f}s")
    return describe,response
if __name__=="__main__":
    db = pymysql.connect(host='localhost', user='root',
                      passwd='123',  charset='utf8mb4')
    global_config={}
    WORKING_DIR = f"/cpfs04/user/zhangyaoze/workspace/trag/exp/mix_vector_search/legal"
    global_config['use_llm_func']=use_llm_func
    global_config['embeddings_func']=embedding
    global_config['working_dir']=WORKING_DIR
    global_config['port']=8001
    global_config['topk']=10
    global_config['level_mode']=1
    
    query="What is the maturity date of the credit agreement?"
    topk=10
    ref,response=query_graph(global_config,db,query)
    print(ref)
    print("#"*20)
    print(response)
    # beginning=time.time()
    # for i in range(10):
    #     print(query_graph(global_config,db,query))
    # end=time.time()
    # print(f"total time: {end-beginning:.2f}s")
    db.close()
    