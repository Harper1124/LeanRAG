# LeanRAG

LeanRAG 是一个轻量级 RAG（Retrieval-Augmented Generation）框架，提供从文档切片、知识图谱构建，到图谱检索和问答生成的完整流程。

---

## 📌 使用流程

### **Step 1: 文档切片**
在 `file_chunk.py` 中对文档进行切片：
- 切片大小：`1024`
- 滑动步长：`128`（即按 128 的步长进行滑动窗口切分）

切分后的 `chunk` 文件中每个字典包含两个属性：
- `hash_code`：根据 `text` 内容使用哈希计算，用于溯源
- `text`：切片的文本内容

---

### **Step 2: 获取三元组和实体描述**

目前提供两种知识图谱抽取方法：

#### **方法 1：CommonKG**
基于 Wikipedia 实体，先定义**头实体 list**，然后从文档中抽取三元组。

**使用方法：**
1. 编辑配置文件：  
   `CommonKG/config/create_kg_conf_test.yaml`  
   填入模型的 `url` 和 `name`，以及切分后的 `chunk` 文件路径。
2. 执行抽取：  
   ```bash
   python CommonKG/create_kg.py
   ```
    抽取结果会保存在 output_dir 中。
3. 处理带描述的 6 元组：
    ```bash
    python CommonKG/deal_triple.py 
    ```
    输出包括：

    - entity.jsonl

    - relation.jsonl
#### **方法 2：GraphRAG**
依赖 LLM 的能力，通过在 Prompt 中给出几个示例，进行 Few-shot 抽取。

**使用方法：**

1. 编辑 GraphExtraction/chunk.py，填写 url 和 model。
    chunk_file 文件与 CommonKG 相同，均为第一步切分后的文件。

2. 对抽取结果去重：
    ```bash
    python GraphExtraction/deal_triple.py 
    ```

    输出包括：

    - entity.jsonl

    - relation.jsonl
### **Step 3: 建图**
- 将已提取的实体描述和关系描述进行聚类及关系生成；

- 构建一个树形结构图谱，支持后续的检索和问答。
### **Step 4: 检索**
1. 选择正确的 `chunks_file` 文件；

2. 根据 `query` 查询图中 **Top-K** 个实体；

3. 根据树形结构生成两两节点间的路径；

4. 将路径上的同层节点关系及聚合实体信息返回给 LLM，用于生成最终答案。





















