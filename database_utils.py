import json
import os
import numpy as np
from pymilvus  import MilvusClient
import pymysql
from collections import Counter

"""
建图和查询共用的数据访问层。

这里同时管理两类存储：
1. Milvus Lite：保存实体/聚合实体的向量，用于 query -> top-k 实体召回。
2. MySQL：保存实体层级树、关系边、聚合社区摘要，用于沿 parent 找路径和取证据。

build_graph.py 会调用 build_vector_search / create_db_table_mysql / insert_data_to_mysql。
query_graph.py 会调用 search_vector_search / find_tree_root / search_nodes_link /
search_community / get_text_units 等函数。
"""

def build_vector_search(data,working_dir):
    """
    将多层实体写入 Milvus Lite 向量索引。

    data 的结构来自 Hierarchical_Clustering.perform_clustering：
    - 前几层通常是 list[entity_dict]。
    - 最顶层有时是单个 dict。

    每个实体会被拍平成一条 Milvus 记录，并额外写入 id 和 level。
    level=0 表示原始实体，level 越大表示越高层的聚合实体。
    """
   
    milvus_client = MilvusClient(uri=f"{working_dir}/milvus_demo.db")
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )
    
    collection_name = "entity_collection"
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=1024,
        index_params=index_params,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
    )
    id=0
    flatten=[]
    print("dealing data level")
    for level,sublist in enumerate(data):
        if type(sublist) is not list:
            item=sublist
            item['id']=id
            id+=1
            item['level']=level
            if len(item['vector'])==1:
                item['vector']=item['vector'][0]
            flatten.append(item)
        else:
            for item in sublist:
                item['id']=id
                id+=1
                item['level']=level
                if len(item['vector'])==1:
                    item['vector']=item['vector'][0]
                flatten.append(item)
        print(level)
        # embedding = emb_text(description)
   
    piece=10
    
    for indice in range(len(flatten)//piece +1):
        start = indice * piece
        end = min((indice + 1) * piece, len(flatten))
        data_batch = flatten[start:end]
        milvus_client.insert(
            collection_name="entity_collection",
            data=data_batch
        )
    # milvus_client.insert(
    #         collection_name=collection_name,
    #         data=data
    #     )

def search_vector_search(working_dir,query,topk=10,level_mode=2):
    # 按 query embedding 在 Milvus 中召回 top-k 实体。
    # level_mode:
    #   0 -> 只搜索原始实体节点，也就是 level == 0。
    #   1 -> 只搜索聚合实体节点，也就是 level > 0。
    #   2 -> 搜索所有层级节点。
    '''
    level_mode: 0: 原始节点
                1: 聚合节点
                2: 所有节点
    '''
    if level_mode==0:
        filter_filed=" level == 0 "
    elif level_mode==1:
        filter_filed=" level > 0 "
    # elif level_mode==2:
    #     filter_filed=" level < 58736"
    else:
        filter_filed=""
    dataset=os.path.basename(working_dir)
    if os.path.exists(f"{working_dir}/milvus_demo.db"):
        print(f"{working_dir}milvus_demo.db already exists, using it")
        milvus_client = MilvusClient(uri=f"{working_dir}/milvus_demo.db")
    else:
        raise FileNotFoundError(f"milvus_demo.db not found under {working_dir}")
    collection_name = "entity_collection"
    # query_embedding = emb_text(query)
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=query,
        limit=topk,
        params={"metric_type": "IP", "params": {}},
        filter=filter_filed,
        output_fields=["entity_name", "description","parent","level","source_id"],
    )
    # print(search_results)
    extract_results=[(i['entity']['entity_name'],i["entity"]["parent"],i["entity"]["description"],i["entity"]["source_id"])for i in search_results[0]]
    # print(extract_results)
    return extract_results
def create_db_table_mysql(working_dir):
    """
    为当前数据集重建 MySQL 数据库和三张表。

    数据库名取 working_dir 的最后一级目录名，例如 /path/to/mix -> mix。
    注意：这里会 drop database，因此是“重建”语义，适合建图阶段重新生成索引时使用。
    """
    con = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',  charset='utf8mb4')
    cur=con.cursor()
    dbname=os.path.basename(working_dir)
    
    cur.execute(f"drop database if exists {dbname};")
    cur.execute(f"create database {dbname} character set utf8mb4;")
    
    # 使用库
    cur.execute(f"use {dbname};")
    cur.execute("drop table if exists entities;")
    # 建表
    cur.execute("create table entities\
        (entity_name varchar(500), description varchar(10000),source_id varchar(1000),\
            degree int,parent varchar(1000),level int ,INDEX en(entity_name))character set utf8mb4 COLLATE utf8mb4_unicode_ci;")
    
    cur.execute("drop table if exists relations;")
    cur.execute("create table relations\
        (src_tgt varchar(190),tgt_src varchar(190), description varchar(10000),\
            weight int,level int ,INDEX link(src_tgt,tgt_src))character set utf8mb4 COLLATE utf8mb4_unicode_ci;")
    
    
    cur.execute("drop table if exists communities;")
    cur.execute("create table communities\
        (entity_name varchar(500), entity_description varchar(10000),findings text,INDEX en(entity_name)\
             )character set utf8mb4 COLLATE utf8mb4_unicode_ci ;")
    cur.close()
    con.close()
    
def insert_data_to_mysql(working_dir):
    """
    将建图阶段生成的 JSON 文件导入 MySQL。

    读取文件：
    - all_entities.json -> entities 表。
    - generate_relations.json -> relations 表。
    - community.json -> communities 表。
    """
    dbname=os.path.basename(working_dir)
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',database=dbname,  charset='utf8mb4')
    cursor = db.cursor()
    
    entity_path=os.path.join(working_dir,"all_entities.json")
    with open(entity_path,"r")as f:
        val=[]
        for level,entitys in enumerate(f):
            local_entity=json.loads(entitys)
            if type(local_entity) is not dict:
                for entity in json.loads(entitys):
                    # entity=json.load(entity_l)
                    
                    entity_name=entity['entity_name']
                    description=entity['description']
                    # if "|Here" in description:
                    #     description=description.split("|Here")[0]
                    source_id="|".join(entity['source_id'].split("|")[:5])
                   
                    degree=entity['degree']
                    parent=entity['parent']
                    val.append((entity_name,description,source_id,degree,parent,level))
            else:
                entity=local_entity
                entity_name=entity['entity_name']
                description=entity['description']
                source_id="|".join(entity['source_id'].split("|")[:5])
                degree=entity['degree']
                parent=entity['parent']
                val.append((entity_name,description,source_id,degree,parent,level))
        sql = "INSERT INTO entities(entity_name, description, source_id, degree,parent,level) VALUES (%s,%s,%s,%s,%s,%s)"
        try:
        # 执行sql语句
            cursor.executemany(sql,tuple(val))
            # 提交到数据库执行
            db.commit()
        except Exception as e:
            # 发生错误时回滚
            db.rollback()
            print(e)
            print("insert entities error")
         
    relation_path=os.path.join(working_dir,"generate_relations.json")
    with open(relation_path,"r")as f:
        val=[]
        for relation_l in f:
            relation=json.loads(relation_l)
            src_tgt=relation['src_tgt']
            tgt_src=relation['tgt_src']
            description=relation['description']
            weight=relation['weight']
            level=relation['level']
            val.append((src_tgt,tgt_src,description,weight,level))
        sql = "INSERT INTO relations(src_tgt, tgt_src, description,  weight,level) VALUES (%s,%s,%s,%s,%s)"
        try:
        # 执行sql语句
            cursor.executemany(sql,tuple(val))
            # 提交到数据库执行
            db.commit()
        except Exception as e:
            # 发生错误时回滚
            db.rollback()
            print(e)
            print("insert relations error")
        
    community_path=os.path.join(working_dir,"community.json")
    with open(community_path,"r")as f:
        val=[]
        for community_l in f:
            community=json.loads(community_l)
            title=community['entity_name']
            summary=community['entity_description']
            findings=str(community['findings'])
           
            val.append((title,summary,findings))
        sql = "INSERT INTO communities(entity_name, entity_description,  findings ) VALUES (%s,%s,%s)"
        try:
        # 执行sql语句
            cursor.executemany(sql,tuple(val))
            # 提交到数据库执行
            db.commit()
        except Exception as e:
            # 发生错误时回滚
            db.rollback()
            print(e)
            print("insert communities error")
def find_tree_root(working_dir,entity):
    """
    从某个实体开始不断查 parent，返回该实体到根节点的路径。

    query_graph.py 会用两条 parent 链寻找召回实体之间的共同祖先。
    """
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',  charset='utf8mb4')
    dbname=os.path.basename(working_dir)
    res=[entity]
    cursor = db.cursor()
    db_name=os.path.basename(working_dir)
    depth_sql=f"select max(level) from {db_name}.entities"
    cursor.execute(depth_sql)
    depth=cursor.fetchall()[0][0]
    i=0
    
    while i< depth:
        sql=f"select parent from {db_name}.entities where entity_name=%s "
        
        cursor.execute(sql,(entity))
        ret=cursor.fetchall()
        # print(ret)
        i+=1
        if len(ret)==0:
            break
        entity=ret[0][0]
        res.append(entity)
    # res=list(set(res))
    # res = list(dict.fromkeys(res))

    return res

def find_path(entity1,entity2,working_dir,level,depth=5):
    """
    在同一 level 的 relations 表中用递归 SQL 寻找 entity1 到 entity2 的最短路径。

    当前 query_graph.py 默认使用 parent 链方式；这个函数保留给需要同层多跳检索的策略。
    """
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',  charset='utf8mb4')
    db_name=os.path.basename(working_dir)
    cursor = db.cursor()

    query = f"""
        WITH RECURSIVE path_cte AS (
            SELECT 
                src_tgt,
                tgt_src,
                 CAST(CONCAT(src_tgt, '|', tgt_src) AS CHAR(5000)) AS path,
                1 AS depth
            FROM {db_name}.relations
            WHERE src_tgt = %s
              AND level = %s

            UNION ALL

            SELECT 
                p.src_tgt,
                t.tgt_src,
                CONCAT(p.path, '|', t.tgt_src),
                p.depth + 1
            FROM path_cte p
            JOIN {db_name}.relations t ON p.tgt_src = t.src_tgt
            WHERE NOT FIND_IN_SET(
                  CONVERT(t.tgt_src USING utf8mb4) COLLATE utf8mb4_unicode_ci,
                  CONVERT(p.path USING utf8mb4) COLLATE utf8mb4_unicode_ci
              )
              AND level = %s
              AND p.depth < %s
        )
        SELECT path
        FROM path_cte
        WHERE tgt_src = %s
        ORDER BY depth ASC
        LIMIT 1;
    """
    cursor.execute(query, (entity1,level,level,depth,entity2))
    result = cursor.fetchone()

    if result:
            return result[0].split('|')  # 返回节点列表
    else:
        return None

def search_nodes_link(entity1,entity2,working_dir,level=0):
    """
    查询两个节点之间是否存在直接关系。

    关系可能以 entity1 -> entity2 或 entity2 -> entity1 保存，所以这里会双向查询。
    """
    # cursor = db.cursor()
    # db_name=os.path.basename(working_dir)
    # sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s and level=%s"
    # cursor.execute(sql,(entity1,entity2,level))
    # ret=cursor.fetchall()
    # if len(ret)==0:
    #     sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s and level=%s"
    #     cursor.execute(sql,(entity2,entity1,level))
    #     ret=cursor.fetchall()
    # if len(ret)==0:
    #     return None
    # else:
    #     return ret[0]
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',  charset='utf8mb4')
    cursor = db.cursor()
    db_name=os.path.basename(working_dir)
    sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s "
    cursor.execute(sql,(entity1,entity2))
    ret=cursor.fetchall()
    if len(ret)==0:
        sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s "
        cursor.execute(sql,(entity2,entity1))
        ret=cursor.fetchall()
    if len(ret)==0:
        return None
    else:
        return ret[0]
def search_chunks(working_dir,entity_set):
    """根据实体名查询 source_id；source_id 对应 chunk 文件里的 hash_code。"""
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',  charset='utf8mb4')
    res=[]
    db_name=os.path.basename(working_dir)
    cursor = db.cursor()
    for entity in entity_set:
        if entity=='root':
            continue
        sql=f"select source_id from {db_name}.entities where entity_name=%s "
        cursor.execute(sql,(entity,))
        ret=cursor.fetchall()
        res.append(ret[0])
    return res
def search_nodes(entity_set,working_dir):
    """查询一批原始实体节点的完整表记录，目前主要用于调试或备用检索流程。"""
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',  charset='utf8mb4')
    res=[]
    db_name=os.path.basename(working_dir)
    cursor = db.cursor()
    for entity in entity_set:
        sql=f"select * from {db_name}.entities where entity_name=%s and level=0"
        cursor.execute(sql,(entity,))
        ret=cursor.fetchall()
        res.append(ret[0])
    return res
def _flatten_chunk_ids(chunks_set):
    chunk_ids = []
    for chunks in chunks_set:
        if chunks is None:
            continue
        if isinstance(chunks, (list, tuple)):
            chunks = chunks[0] if chunks else ""
        chunks = str(chunks)
        if "|" in chunks:
            temp_chunks = chunks.split("|")
        else:
            temp_chunks = [chunks]
        chunk_ids += [chunk.strip() for chunk in temp_chunks if chunk.strip()]
    return chunk_ids


def _normalize_evidence_chunk(item):
    text = item.get("text", "")
    modality = item.get("modality") or item.get("type") or "text"
    summary = item.get("summary") or text[:240]
    return {
        "hash_code": item.get("hash_code"),
        "modality": modality,
        "page": item.get("page"),
        "asset_path": item.get("asset_path"),
        "summary": summary,
        "text": text,
    }


def get_text_units(working_dir,chunks_set,chunks_file,k=5):
    """
    根据召回实体携带的 source_id 回查原文 chunk。

    一个实体可能来自多个 chunk，source_id 之间用 | 拼接。这里会统计 chunk hash 出现频次，
    优先选择被多个召回实体共同指向的 chunk，因为它们更可能是高价值证据。
    """
    db_name=os.path.basename(working_dir)
    chunks_list=_flatten_chunk_ids(chunks_set)
    counter = Counter(chunks_list)

    # 筛选出出现多次的元素
    # duplicates = [item for item, count in counter.items() if count > 2]
    duplicates = [item for item, _ in sorted(
    [(item, count) for item, count in counter.items() if count > 1],
    key=lambda x: x[1],
    reverse=True
        )[:k]]
    if len(duplicates)< k:
        used = set(duplicates)
        for item, _ in counter.items():
            if item not in used:
                duplicates.append(item)
                used.add(item)
            if len(duplicates) == k:
                break
    
    with open (chunks_file,'r',encoding='utf-8')as f:
        chunks_data= json.load(f)
    chunks_dict={item["hash_code"]: _normalize_evidence_chunk(item) for item in chunks_data}

    text_units=[]
    for chunks in duplicates:
        evidence = chunks_dict.get(chunks)
        if evidence is None:
            continue
        evidence["score"] = counter.get(chunks, 0)
        text_units.append(evidence)
    return text_units
    
def search_community(entity_name,working_dir):
    """按聚合实体名称查询 communities 表，返回建图阶段生成的聚合摘要和 findings。"""
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',  charset='utf8mb4')
    db_name=os.path.basename(working_dir)
    cursor = db.cursor()
    sql=f"select * from {db_name}.communities where entity_name=%s"
    cursor.execute(sql,(entity_name,))
    ret=cursor.fetchall()
    if len(ret)!=0:
        return ret[0]
    else:
        return ""
            # return ret[0]
def insert_origin_relations(working_dir):
    """
    额外导入原始关系到 relations 表的辅助函数。

    主流程 insert_data_to_mysql 主要导入聚合关系 generate_relations.json；这个函数用于把
    原始 relation.jsonl 也补充进 MySQL，方便对比实验或调试底层边。
    """
    dbname=os.path.basename(working_dir)
    db = pymysql.connect(host='localhost',port=4321, user='root',
                      passwd='123',database=dbname,  charset='utf8mb4')
    cursor = db.cursor()
    # relation_path=os.path.join(f"datasets/{dbname}","relation.jsonl")
    # relation_path=os.path.join(f"/data/zyz/reproduce/HiRAG/eval/datasets/{dbname}/test")
    relation_path=os.path.join(f"hi_ex/{dbname}","relation.jsonl")
    # relation_path=os.path.join(f"32b/{dbname}","relation.jsonl")
    with open(relation_path,"r")as f:
        val=[]
        for relation_l in f:
            relation=json.loads(relation_l)
            src_tgt=relation['src_tgt']
            tgt_src=relation['tgt_src']
            if len(src_tgt)>190 or len(tgt_src)>190:
                print(f"src_tgt or tgt_src too long: {src_tgt} {tgt_src}")
                continue
            description=relation['description']
            weight=relation['weight']
            level=0
            val.append((src_tgt,tgt_src,description,weight,level))
        sql = "INSERT INTO relations(src_tgt, tgt_src, description,  weight,level) VALUES (%s,%s,%s,%s,%s)"
        try:
        # 执行sql语句
            cursor.executemany(sql,tuple(val))
            # 提交到数据库执行
            db.commit()
        except Exception as e:
            # 发生错误时回滚
            db.rollback()
            print(e)
            print("insert relations error")
if __name__ == "__main__":
    working_dir='exp/compare_hirag_opt1_commonkg_32b/mix'
    # build_vector_search()
    # search_vector_search()
    create_db_table_mysql(working_dir)
    insert_data_to_mysql(working_dir)
    insert_origin_relations(working_dir)
    # print(find_tree_root(working_dir,'Policies'))
    # print(search_nodes_link('Innovation Policy Network','document',working_dir,0))
    # from query_graph import embedding
    # topk=200
    # query=embedding("mary")
    # milvus_client = MilvusClient(uri=f"/cpfs04/user/zhangyaoze/workspace/trag/ttt/milvus_demo.db")
    # collection_name = "entity_collection"
    # # query_embedding = emb_text(query)
    # search_results = milvus_client.search(
    #     collection_name=collection_name,
    #     data=query,
    #     limit=topk,
    #     filter=' level ==1 ',
    #     params={"metric_type": "L2", "params": {}},
    #     output_fields=["entity_name", "description","vector","level"],
    # )
    # print(len(search_results[0]))
    # for entity in search_results[0]:
    #     if entity['entity']['level']!=1:
    #         print(entity)
        
    # search_results2 = milvus_client.search(
    #     collection_name=collection_name,
    #     data=[vec],
    #     limit=topk,
    #     params={"metric_type": "L2", "params": {}},
    #     output_fields=["entity_name", "description","vector"],
    # )
    # recall=search_results2[0][0]['entity']['vector']
    # print(recall==vec)
