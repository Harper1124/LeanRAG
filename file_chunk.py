import json
import tiktoken
from hashlib import md5
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
def chunk_documents(
    docs,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens_list = ENCODER.encode_batch(docs, num_threads=16)

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token_ids = []
        lengths = []

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
    max_token_size=1024
    overlap_token_size=128
    original_text_file="datasets/mix/mix_unique_contexts.json"
    chunk_text_file="datasets/mix/mix_chunk.json"
    with open(original_text_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = chunk_documents(
        data,
        max_token_size=max_token_size,
        overlap_token_size=overlap_token_size,
    )
    with open(f'datasets/{dataset}/{dataset}_chunk.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
