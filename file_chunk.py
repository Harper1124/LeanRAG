import json
import tiktoken
from hashlib import md5

"""
文档切块入口。

LeanRAG 后续的实体抽取、关系抽取、回溯原文证据都依赖这里生成的 chunk。
每个 chunk 会保存两个字段：
- hash_code：由 chunk 文本计算出的稳定 ID，后续作为 source_id 使用。
- text：真正送入抽取模型的文本片段。
"""

def compute_mdhash_id(content, prefix: str = ""):
    """根据文本内容生成 md5 标识；同一段文本会得到同一个 hash，便于跨阶段追踪。"""
    return prefix + md5(content.encode()).hexdigest()


def chunk_documents(
    docs,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
):
    """
    将文档列表按 token 滑动窗口切块。

    max_token_size 控制单个 chunk 的最大长度；overlap_token_size 控制相邻 chunk
    的重叠长度，用来降低实体或关系刚好落在边界上而被切断的概率。
    """
    ENCODER = tiktoken.get_encoding(model_name)
    # 批量编码比逐条处理更快；这里用的是 OpenAI/tiktoken 的 cl100k_base 编码。
    tokens_list = ENCODER.encode_batch(docs, num_threads=16)

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token_ids = []
        lengths = []

        # 步长 = 窗口大小 - 重叠大小；例如 1024/128 时，每次向前滑动 896 token。
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk = tokens[start : start + max_token_size]
            chunk_token_ids.append(chunk)
            lengths.append(len(chunk))

        # 解码所有 chunk
        chunk_texts = ENCODER.decode_batch(chunk_token_ids)

        for i, text in enumerate(chunk_texts):
            results.append({
                # "tokens": lengths[i],
                "hash_code": compute_mdhash_id(text), ##使用hash进行编码
                "text": text.strip().replace("\n", ""),
                # "chunk_order_index": i,
            })

    return results
if __name__ == "__main__":
    # 这里是 README 中 Step 1 的示例入口：对 mix 数据集进行切块。
    max_token_size=1024
    overlap_token_size=128
    original_text_file="datasets/mix/mix.jsonl"
    chunk_text_file="datasets/mix/mix_chunk.json"
    dataset='mix'
    data=[]
    with open(original_text_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    # 原始 jsonl 每条样本含有 context；这里只切正文，问题字段不参与知识图谱构建。
    data = [item['context'] for item in data if 'input' in item]
    results = chunk_documents(
        data,
        max_token_size=max_token_size,
        overlap_token_size=overlap_token_size,
    )
    # 输出文件会被 CommonKG 或 GraphExtraction 作为 chunk_file 读取。
    with open(f'datasets/{dataset}/{dataset}_chunk.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
