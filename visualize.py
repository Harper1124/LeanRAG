import json
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib import rcParams
# sudo apt update
#sudo apt install fonts-wqy-zenhei fonts-noto-cjk ##安装字体

rcParams['font.sans-serif'] = ["WenQuanYi Zen Hei"]   # 中文字体
rcParams['axes.unicode_minus'] = False   
def get_entity_list(entity_path):
    entity_list = []
    with open(entity_path, 'r', encoding='utf-8') as f:
        for line in f:
            entity = json.loads(line)
            entity_list.append(entity['entity_name'])
    return entity_list
def get_relation_list(relation_path):
    relation_list = []
    with open(relation_path, 'r', encoding='utf-8') as f:
        for line in f:
            relation = json.loads(line)
            relation_list.append((relation['src_tgt'], relation['tgt_src']))
    return relation_list
def get_subgraph_layered(G, query_node_list, max_per_hop1=10,max_per_hop2=3):
    nodes = set()
    first_hop = set()
    second_hop = set()

    for query_node in query_node_list:
        nodes.add(query_node)
        neighbors1 = list(G.neighbors(query_node))[:max_per_hop1]
        first_hop.update(neighbors1)

        for n1 in neighbors1:
            neighbors2 = list(G.neighbors(n1))
            # 去掉已加入节点，避免重复
            neighbors2_filtered = [n for n in neighbors2 if n not in nodes and n not in first_hop and n not in second_hop]
            second_hop.update(neighbors2_filtered[:max_per_hop2])

    # 合并三层节点
    nodes.update(first_hop)
    nodes.update(second_hop)
    return nodes

if __name__ == '__main__':
    entity_path = 'entity.jsonl'
    relation_path = 'relation.jsonl'
    MAX_NODES = 5
    entity_list = get_entity_list(entity_path)
    relation_list = get_relation_list(relation_path)
    G = nx.Graph()
    G.add_nodes_from(entity_list)
    G.add_edges_from(relation_list)
    query_node=["力旺電子","矽智財平台"]
    query_color=["#8ECFC9","#FFBE7A","#FA7F6F","#82B0D2","#F7E1ED","#C497B2","#A9B8C6"]
    
    subnodes= get_subgraph_layered(G, query_node, max_per_hop1=20,max_per_hop2=20)
    subG = G.subgraph(subnodes).copy()
    pos = nx.spring_layout (subG, k=0.8,seed=1)  # 使用spring布局,k越大越稀疏
    
    
    num_nodes = len(subG.nodes)
    fig, ax = plt.subplots(figsize=(8,6))
    
    
    node_colors = []
    node_sizes = []
    for node in subG.nodes():
        if node in query_node:
            node_colors.append(random.choice(query_color))  # 红色高亮
            node_sizes.append(2000)
        else:
            node_colors.append("#7499F7")
            node_sizes.append(200)
    ax.set_facecolor("#BFD3F3") 
    nx.draw_networkx_nodes(subG, pos, ax=ax,
                       node_color='none',
                       node_size=[s * 1.02 for s in node_sizes],
                       edgecolors='white', linewidths=1,)
    nx.draw_networkx_nodes(subG, pos, ax=ax,
                       node_color=node_colors,
                       node_size=node_sizes,
                       edgecolors='white', linewidths=2,alpha=0.8)
    # 绘图
    nx.draw_networkx_edges(subG, pos, ax=ax,
            edge_color='gray')

    # 设置标签偏移（上方显示）
    label_pos = {node: (x, y + 0.08) for node, (x, y) in pos.items()}
    label=nx.draw_networkx_labels(subG, label_pos, ax=ax,font_size=9, font_color='black',font_family='sans-serif')

    # plt.title("2-hop Subgraph")
    # plt.axis('off')
    # plt.gca().set_facecolor('#55403E')  # 设置绘图区背景色
    fig.patch.set_facecolor("#E6F2FF")  # 整体背景色
    # ax.patch.set_facecolor("#55403E") 
    plt.gcf().set_facecolor(fig.get_facecolor())
    plt.gca().set_facecolor(ax.get_facecolor())
    plt.savefig('graph.png',
            format='png',
            dpi=1200,
            facecolor=fig.get_facecolor(),
            bbox_inches='tight')  # 避免边界裁剪丢背景色

    plt.show()