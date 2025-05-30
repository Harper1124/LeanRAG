import json
import os
from pymilvus  import MilvusClient
import ollama
import pymysql
def emb_text(text):
    response = ollama.embeddings(model="bge-m3:latest", prompt=text)
    return response["embedding"]
def build_vector_search(data,working_dir):
    # with open(json_file_path, 'r') as file:
    #     for line in file:
    #         # 去除空行
    #         if line.strip():
    #             data = json.loads(line.strip())
    #             break
    milvus_client = MilvusClient(uri=f"{working_dir}/milvus_demo.db")
    index_params = milvus_client.prepare_index_params()

# Add indexes
    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )
    piece=10
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
    for i,item in enumerate(data):
        # entity_name = item['entity_name']
        # embedding = emb_text(entity_name)
        # item['vector']=embedding
        item['id']=i
        # embedding = emb_text(description)
    for indice in range(len(data)//piece +1):
        start = indice * piece
        end = min((indice + 1) * piece, len(data))
        data_batch = data[start:end]
        milvus_client.insert(
            collection_name=collection_name,
            data=data_batch
        )
    # milvus_client.insert(
    #         collection_name=collection_name,
    #         data=data
    #     )
def search_vector_search(working_dir,query,topk=10):
    milvus_client = MilvusClient(uri=f"{working_dir}/milvus_demo.db")
    collection_name = "entity_collection"
    # query_embedding = emb_text(query)
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=query,
        limit=topk,
        params={"metric_type": "IP", "params": {}},
        output_fields=["entity_name", "description"],
    )
    extract_results=[i['entity']['entity_name']for i in search_results[0]]
    # print(extract_results)
    return extract_results
def create_db_table_mysql(working_dir):
    con = pymysql.connect(host='localhost', user='root',
                      passwd='123', charset='utf8')
    cur=con.cursor()
    dbname=os.path.basename(working_dir)
    cur.execute(f"drop database if exists {dbname};")
    cur.execute(f"create database {dbname} character set utf8;")
    
    # 使用库
    cur.execute(f"use {dbname};")
    cur.execute("drop table if exists entities;")
    # 建表
    cur.execute("create table entities\
        (entity_name varchar(500), description varchar(10000),source_id varchar(1000),\
            degree int,parent varchar(1000),level int ,INDEX en(entity_name))character set utf8;")
    
    cur.execute("drop table if exists relations;")
    cur.execute("create table relations\
        (src_tgt varchar(500),tgt_src varchar(500), description varchar(10000),\
            weight int,level int ,INDEX link(src_tgt,tgt_src))character set utf8;")
    
    cur.execute("drop table if exists communities;")
    cur.execute("create table communities\
        (entity_name varchar(500), entity_description varchar(10000),findings text,INDEX en(entity_name)\
             )character set utf8;")
    cur.close()
    con.close()
    
def insert_data_to_mysql(working_dir):
    dbname=os.path.basename(working_dir)
    db = pymysql.connect(host='localhost', user='root',
                      passwd='123',database=dbname, charset='utf8')
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
            val.append((src_tgt,tgt_src,description,weight,level+1))
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
def find_tree_root(db,working_dir,entity):
    res=[entity]
    cursor = db.cursor()
    
    depth_sql=f"select max(level) from {working_dir}.entities"
    cursor.execute(depth_sql)
    depth=cursor.fetchall()[0][0]#树的深度为max(level)+1，因为level是从0开始的，但最后一层的parent为root，没什么意义，所以就这样算了，不进行+1
    i=0
    
    while i< depth:
        sql=f"select parent from {working_dir}.entities where entity_name='{entity}' and level={i} "
        cursor.execute(sql)
        ret=cursor.fetchall()
        # print(ret)
        i+=1
        entity=ret[0][0]
        res.append(entity)
    return res
def search_nodes_link(entity1,entity2,db,working_dir,level):
    cursor = db.cursor()
    sql=f"select * from {working_dir}.relations where src_tgt='{entity1}' and tgt_src='{entity2}' and level={level}"
    cursor.execute(sql)
    ret=cursor.fetchall()
    if len(ret)==0:
        sql=f"select * from {working_dir}.relations where src_tgt='{entity2}' and tgt_src='{entity1}' and level={level}"
        cursor.execute(sql)
        ret=cursor.fetchall()
    if len(ret)==0:
        return None
    else:
        return ret[0]

def search_nodes(entity_set,db,working_dir):
    res=[]
    cursor = db.cursor()
    for entity in entity_set:
        sql=f"select * from {working_dir}.entities where entity_name='{entity}' and level=0"
        cursor.execute(sql)
        ret=cursor.fetchall()
        res.append(ret[0])
    return res
def search_community(entity_name,db,working_dir):
    
    cursor = db.cursor()
    sql=f"select * from {working_dir}.communities where entity_name='{entity_name}'"
    cursor.execute(sql)
    ret=cursor.fetchall()
    return ret[0]
            # return ret[0]
if __name__ == "__main__":
    working_dir='ttt'
    # build_vector_search()
    # search_vector_search()
    create_db_table_mysql(working_dir)
    insert_data_to_mysql(working_dir)
    # print(find_tree_root(working_dir,'Policies'))
    # print(search_nodes_link('Innovation Policy Network','document',working_dir,0))