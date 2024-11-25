import networkx as nx
import torch
from pathlib import Path
import os
import pickle
import graphein
from torch_geometric.utils import index_to_mask, to_dense_batch
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.ml import GraphFormatConvertor
from graphein.protein.edges.distance import add_k_nn_edges
from graphein.protein.features.nodes.geometry import add_virtual_beta_carbon_vector, add_sequence_neighbour_vector, VECTOR_FEATURE_NAMES
import pandas as pd
import torch_geometric.transforms as T
import einops
from torch_geometric.data import Data, Batch
from torch_geometric.data import InMemoryDataset, Data, Batch
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from functools import partial

class PDBToPyGPretransform:
    def __init__(
        self,
        k: int = 10,
        undirected: bool = True,
        type1_features=(
            'virtual_c_beta_vector',
            'sequence_neighbour_vector_n_to_c',
            'sequence_neighbour_vector_c_to_n'
        ),
        divide_coords_by: float = 1.0,
    ):
        self.config = ProteinGraphConfig(
            edge_construction_functions=[] if k is None else [
                partial(add_k_nn_edges, k=k, long_interaction_threshold=0)
            ],
            node_metadata_functions=[
                amino_acid_one_hot
            ]
        )
        self.undirected = undirected
        if not undirected:
            raise NotImplementedError()
        self.type1_features = type1_features
        self.divide_coords_by = divide_coords_by

    def __call__(self, path: Path, label: int):
        g = construct_graph(config=self.config, path=str(path), verbose=False)

        if self.type1_features:
            if 'virtual_c_beta_vector' in self.type1_features:
                add_virtual_beta_carbon_vector(g)
            if 'sequence_neighbour_vector_n_to_c' in self.type1_features:
                add_sequence_neighbour_vector(g)
            if 'sequence_neighbour_vector_c_to_n' in self.type1_features:
                add_sequence_neighbour_vector(g, n_to_c=False)
                
        g.graph['label'] = label
        return g

def process_pdb_file(args):
    pdb_file_path, transform, save_path, label = args
    try:
        if os.path.exists(save_path):
            print(f"{save_path} already exists. Skipping...")
            return  # 如果目标文件已经存在，直接跳过

        graph = transform(Path(pdb_file_path), label)
        
        if graph is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"Saved transformed graph with label {label} to {save_path}")
        else:
            print(f"Failed to process {pdb_file_path}")
    except Exception as e:
        print(f"Error processing {pdb_file_path}: {e}")

def main():
    base_dir = '/data/a/xiaoyao/dataset3/val_set_interface'
    save_dir = '/data/a/xiaoyao/dataset3/val_set_interface_graph'  
    df_dir = '/data/a/xiaoyao/dataset3/filtered_val_df.csv'
    df = pd.read_csv(df_dir)
    os.makedirs(save_dir, exist_ok=True)

    transform = PDBToPyGPretransform()

    tasks = []
    for subdir in os.listdir(base_dir):
        pdb_dir = os.path.join(base_dir, subdir)
        output_dir = os.path.join(save_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing directory: {subdir}")

        for pdb_file in os.listdir(pdb_dir):
            if pdb_file.endswith(".pdb"):
                pdb_file_path = os.path.join(pdb_dir, pdb_file)
                save_path = os.path.join(output_dir, os.path.splitext(pdb_file)[0] + '.pkl')
                
                # 检查文件是否已存在
                if os.path.exists(save_path):
                    print(f"{save_path} already exists. Skipping...")
                    continue
                
                # 设置label!!!!!!!! 
                # This must be achieved
                #label = df.loc[df['pdb_id'] == os.path.splitext(pdb_file)[0], 'interface'].values[0]
                
                tasks.append((pdb_file_path, transform, save_path, label))

    # 使用多进程池处理任务
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(process_pdb_file, tasks), total=len(tasks)):
            pass

if __name__ == "__main__":
    main()
