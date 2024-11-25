import torch
from equiformer_pytorch import Equiformer
import torch.nn as nn
import os
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pickle
from graphein.ml import GraphFormatConvertor
from graphein.protein.features.nodes.geometry import VECTOR_FEATURE_NAMES
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import os
import math
import pickle
from torch_geometric.data import Data, Dataset
import numpy as np
from collections import Counter
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from functools import partial

#load encoder 
encoder = Equiformer(
    dim=(128, 64),
    dim_in=(20, 3),
    num_degrees=2,
    input_degrees=2,
    heads=2,
    dim_head=(32, 16),
    depth=2,
    num_neighbors=10,
    num_edge_tokens=2,
    edge_dim=16,
    gate_attn_head_outputs=False
).cuda()

checkpoint_path = './best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cuda')
encoder.load_state_dict(checkpoint['encoder_state_dict'])


#dataset
class ProteinDataset(Dataset):
    def __init__(
        self, 
        root_dir,  # 接受一个或多个 root_dir
        pdb_folders=None,
        type1_features=('virtual_c_beta_vector', 'sequence_neighbour_vector_c_to_n', 'sequence_neighbour_vector_n_to_c'),
        coord_fill_value=0.0,
        indices=None
    ):
        self.pdb_folders = pdb_folders if pdb_folders is not None else []
        self.pdb_folders.extend([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        self.convertor = GraphFormatConvertor(
            src_format='nx',
            dst_format='pyg',
            columns=[
                'coords',
                'node_id',
                'amino_acid_one_hot',
                'edge_index',
                *VECTOR_FEATURE_NAMES
            ]
        )
        self.type1_features = type1_features
        self.coord_fill_value = coord_fill_value

    def __len__(self):
        return len(self.pdb_folders)

    def load_data(self, pdb_folder, file_name):
        file_path = os.path.join(pdb_folder, file_name)
        with open(file_path, 'rb') as f:
            g = pickle.load(f)  
        pyg_data = self.convertor(g)

        data = Data(
            f=pyg_data.amino_acid_one_hot.float(),
            pos=pyg_data.coords.float(),
            node_id=pyg_data.node_id,
            edge_index=pyg_data.edge_index
        )

        # add type-1 features
        for feat_name in self.type1_features:
            if hasattr(pyg_data, feat_name):
                feat = getattr(pyg_data, feat_name)
                setattr(data, feat_name, feat.float())
            else:
                feat = torch.tensor(
                    np.array([g.nodes[node][feat_name] for node in g.nodes]),
                    dtype=torch.float                
                )
                setattr(data, feat_name, feat)

        # Add edge feature about chain id
        chain_id = [node_id.split(':', 1)[0] for node_id in pyg_data.node_id]
        num_edges = data.edge_index.shape[1]

        edge_attr = torch.zeros(num_edges, 1, dtype=torch.float)

        for i in range(num_edges):
            source_node = data.edge_index[0, i].item()
            target_node = data.edge_index[1, i].item()

            if chain_id[source_node] != chain_id[target_node]:
                edge_attr[i] = 1

        data.edge_attr = edge_attr
        
        if 'label' in g.graph:
            data.y = torch.tensor([g.graph['label']], dtype=torch.float)
        else:
            raise ValueError("Label not found in graph metadata (g.graph).")

        return data

    def __getitem__(self, idx):
        pass

#training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0

    with tqdm(train_loader, total=len(train_loader), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

        for batch_data in tepoch:
            #  GPU
            if batch_data is None:
                continue 
            feats = {key: val.to(device) for key, val in batch_data['inputs'].items() if isinstance(val, torch.Tensor)}
            coors = batch_data['coors'].to(device)
            mask = batch_data['mask'].to(device)
            edges = batch_data['edges'].to(device)
            labels = batch_data['labels'].to(device)

            # Forward pass through encoder
            out = encoder(feats, coors, mask, edges=edges)
            type1_embedding = out.type0
            mask_expanded = mask.unsqueeze(-1)

            # Masked embedding
            masked_type1_embedding = type1_embedding * mask_expanded
            masked_sum = masked_type1_embedding.sum(dim=1)
            valid_node_count = mask.sum(dim=1, keepdim=True)
            mean_pooling = masked_sum / valid_node_count # this is the output of encoder, dimension 
            
            


