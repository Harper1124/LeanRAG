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
from openai import AsyncOpenAI, OpenAI
from _cluster_utils import Hierarchical_Clustering
from tools.utils import write_jsonl
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
def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    global TOTAL_TOKEN_COST
    global TOTAL_API_CALL_COST

    openai_async_client =OpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # -----------------------------------------------------
    retry_time = 3
    try:
        # logging token cost
        cur_token_cost = len(tokenizer.encode(messages[0]['content']))
        TOTAL_TOKEN_COST += cur_token_cost
        # logging api call cost
        TOTAL_API_CALL_COST += 1
        # request
        response =openai_async_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    except Exception as e:
        print(f"Retry for Error: {e}")
        retry_time -= 1
        response = ""
    
    if response == "":
        return response
    return response.choices[0].message.content
def get_reasoning_chain(global_config,db,entities_set):
    maybe_edges=list(combinations(entities_set,2))
    reasoning_path=[]
    reasoning_path_information=[]
    information_record=[]
    for edge in maybe_edges:
        a_path=[]
        b_path=[]
        node1=edge[0]
        node2=edge[1]
        node1_tree=find_tree_root(db,global_config['working_dir'],node1)
        node2_tree=find_tree_root(db,global_config['working_dir'],node2)
        
        # if node1_tree[1]!=node2_tree[1] :
        #     print("debug")
        for index,(i,j) in enumerate(zip(node1_tree,node2_tree)):
            if i!=j :
                a_path.append(i)
                b_path.append(j)
                if (i,j) not in information_record and (j,i)not in information_record:
                    information=search_nodes_link(i,j,db,global_config['working_dir'],index-1)
                    if information!=None:
                        reasoning_path_information.append(information)
                        information_record.append((i,j))
            if i==j:
                a_path.append(i)
                break
        reasoning_path.append(a_path+[b_path[len(b_path)-1-i] for  i in range(len(b_path))]) 
    # columns=['src_tgt','tgt_src','path_description']
    # reasoning_path_information_description="\t\t".join(columns)+"\n"
    reasoning_path_information_description="\n".join([information[2] for information in reasoning_path_information])  
    return  reasoning_path,reasoning_path_information_description
def get_entity_description(global_config,db,entities_set):
    entity_informations=search_nodes(entities_set,db,global_config['working_dir'])
    columns=['entity_name','parent','description']
    entity_descriptions="\t\t".join(columns)+"\n"
    entity_descriptions+="\n".join([information[0]+"\t\t"+information[4]+"\t\t"+information[1] for information in entity_informations])
    return entity_descriptions
        
def get_aggregation_description(global_config,db,reasoning_path,if_findings=False):
    communities=set()
    aggregation_results=[]
    for each_path in reasoning_path:
        for community in each_path[1:-1]:
            communities.add(community)
    for community in communities:
        aggregation_results.append(search_community(community,db,global_config['working_dir']))
    if if_findings:
        columns=['entity_name','entity_description','findings']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1])+"\t\t"+information[2] for information in aggregation_results])
    else:
        columns=['entity_name','entity_description']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1]) for information in aggregation_results])
    return aggregation_descriptions
def query_graph(global_config,db,query,topk=10):
    use_llm_func: callable = global_config["use_llm_func"]
    b=time.time()
    res_entity=search_vector_search(global_config['working_dir'],embedding(query),topk=topk)
    v=time.time()
    entity_descriptions=get_entity_description(global_config,db,res_entity)
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
    
    print(describe)
    sys_prompt =PROMPTS["response"].format(context_data=describe)
    response=use_llm_func(query,system_prompt=sys_prompt)
    g=time.time()
    print(f"embedding time: {v-b:.2f}s")
    print(f"query time: {e-v:.2f}s")
    
    print(f"response time: {g-e:.2f}s")
    return response
if __name__=="__main__":
    db = pymysql.connect(host='localhost', user='root',
                      passwd='123', charset='utf8')
    global_config={}
    WORKING_DIR = f"ttt"
    global_config['use_llm_func']=deepseepk_model_if_cache
    global_config['embeddings_func']=embedding
    global_config['working_dir']=WORKING_DIR
    
    query="What's the relationship between  Research and Israel?"
    topk=10
    print(query_graph(global_config,db,query,topk=topk))
    # beginning=time.time()
    # for i in range(10):
    #     print(query_graph(global_config,db,query))
    # end=time.time()
    # print(f"total time: {end-beginning:.2f}s")
    db.close()
    