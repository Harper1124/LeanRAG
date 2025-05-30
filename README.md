# TRAG
## 构建逻辑
#### 获取三元组和实体描述
根据CommonKG流程构建三元组及实体和关系描述，用于TRAG建图及索引
#### 建图
将已提取的实体描述和关系描述放用于聚类及关系生成，构建一个树形的结构
#### 检索
根据query查询图中topk个实体，然后根据树形构建两两节点间的路径，根据路上的同层节点间关系以及路径上的聚合实体作为信息，返回给llm，用于内容生成
## 执行步骤
### Step 1:准备本地服务
启动vllm脚本，开启本地llm服务
```
bash vllm_start.sh num 
```
num为卡数，脚本中执行的为单卡运行单个模型，部署别的模型可以进行修改，但需要修改对应的配置文件

启动mysql服务
其中database_utils.py脚本中的创建和插入数据库的用户 密码需要自行修改。
create_kg/config/create_kg_conf_test.yaml 中控制的为生成三元组所用的llm和url
config.yaml 中控制的为构建RAG所需的llm和url

### Step2:获取实体及关系描述
流程同CommonKG相同，配置好配置文件后，运行create_kg.py，对应生成的文件中会为六元组
然后运行get_data.py的process_triple，将获取到的三元组进行预处理，生成实体描述和关系描述
其中file_path为生成的六元组文件，output_path为working_dir
该脚本会生成实体描述和关系描述，分别对应于entity.jsonl和relation.jsonl
### Step3:建图
```
python build_HiRAG_graph.py
```
其中working_dir即为entity.jsonl和relation.jsonl的路径
num为卡数，与启动脚本时的num相对应
过程中会将enetiy的description存入milvus中，
entity,aggregation_entity,generated_relation存入mysql中，database对应为workingdir
### Step4:检索
```
python query_HiRAG_graph.py
```
其中tokp对应于前topk个实体，query为检索的query
## 文件结构

```
├── config                          # 配置文件
│   ├── create_kg_conf_test.yaml
│   └── set_config.py
├── create_kg
│   ├──config                          # 配置文件
│         ├── create_kg_conf_test.yaml
│         └── set_config.py
│   ├── create_kg.py  ##生成六元组 subject,suject_description,relation,relation_description,object,object_description
│   └── ···
├── datasets           #用于benchmark测试
├── logs               #vllm脚本以及create_kg的日志
├── tools
│   ├── io_file.py
│   ├── logger_factory.py
│   ├── tools.py
│   ├── _utils.py
│   └── utils.py
├── build_HiRAG_graph.py                        # 进行HiRAG建图
├── query_HiRAG_graph.py                        # 进行HiRAG检索
├── _cluster_utils.py                           # 实现聚类算法
├── database_utils.py                           # 实现图数据持续化保存至数据库，以及索引时从数据库取数据
├── prompt.py                                   # 过程中的所有prompt
├── _cluster_utils.py                           # 实现聚类算法
├── config.yaml                                 # 配置文件
├── README.md
├── requirements.txt

```

