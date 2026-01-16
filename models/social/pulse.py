import torch as t
from torch import nn
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
import networkx as nx
from torch_scatter import scatter_mean
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class PULSE(BaseModel):
    def __init__(self, data_handler):
        super(PULSE, self).__init__(data_handler)

        # Config - Hyperparameter
        self.device = configs['device']
        self.layer_num = configs['model']['layer_num']
        self.layer_num_s = configs['model']['layer_num_s']
        self.reg_weight = configs['model']['reg_weight'] 
        self.mask_ratio = configs['model']['mask_ratio'] # for Community-Aware SSL
        self.temperature = configs['model']['temp'] # for Community-Aware SSL
        self.cl_weight = configs['model']['cl_weight'] # for Community-Aware SSL

        self.trn_mat = self._coo_to_sparse_tensor(data_handler.trn_mat)
        self.trust_mat = self._coo_to_sparse_tensor(data_handler.trust_mat)

        # # Load User-Community Membership
        cluster_info = data_handler._load(f"{data_handler.group_info_file_leiden}_theta_1_5_{configs['data']['name']}.pkl")
        self.cluster_info = self._normalize_sparse_matrix_3(self._coo_to_sparse_tensor(cluster_info))

        # Community Embedding (E_C)
        self.group_embeds = nn.Parameter(init(t.empty(self.cluster_info.size(1), self.embedding_size))) # |Community| * d

        # Item Embedding (E_I)
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size))) # |Item| * d

        # Social Gating Network
        self.layer1 = nn.Parameter(init(t.empty(self.embedding_size*2, self.embedding_size))) # (2d) * d (W_1)
        self.layer2 = nn.Parameter(init(t.empty(self.embedding_size, 1))) # (d * 1) (W_2)

        self.is_training = True
        self.final_embeds = None
        
        # Load User-Item Interaction -> Adjacency Matrix -> Normalization
        A = self._create_adj_matrix(self.trn_mat)
        self.adj = self._normalize_sparse_matrix(A)

    def _propagate(self, adj, embeds):
        """Message Aggregation During GCN"""
        propagated_embeds = t.sparse.mm(adj, embeds)
        return propagated_embeds

    def socially_connected_item_aggregator(self, trust_mat, adj, embeds):
        # User Behavior Embedding
        embeds = t.sparse.mm(adj, embeds)[:self.user_num]
        
        # RBF Kernel-based Adjustment
        user_norm_embeds = embeds / (t.norm(embeds, p=2, dim=1, keepdim=True) + 1e-8) 
        indices = trust_mat._indices()

        user_i = indices[0]  
        user_j = indices[1] 
        cosine_sim = (user_norm_embeds[user_i] * user_norm_embeds[user_j]).sum(dim=1) 
        user_norm_sim = (1 + cosine_sim) / 2

        if 'sigma' in configs['model']:
            sigma = configs['model']['sigma']
            squared_user_embeds = (embeds ** 2).sum(dim=1)

            dist_squared = squared_user_embeds[user_i] - 2 * (embeds[user_i] * embeds[user_j]).sum(dim=1) + squared_user_embeds[user_j]
            kernel_values = t.exp(-dist_squared / (2 * sigma ** 2))

            adjusted_values = user_norm_sim * kernel_values
        else:
            adjusted_values = user_norm_sim

        # Based on adjusted weights, propagate socially-connected item information
        trust_influence_mat = self._normalize_sparse_matrix_2(adjusted_values, trust_mat)

        for j in range(self.layer_num_s): 
            user_embeds_socially_connected_item_wise = self._propagate(trust_influence_mat, embeds)

        return user_embeds_socially_connected_item_wise
    
    def _gatingNet(self, user_embeds_social_wise, user_embeds_socially_connected_item_wise):
        combined_embeds = t.cat([user_embeds_social_wise, user_embeds_socially_connected_item_wise], dim=1)

        # Pass through the gating network
        hidden = t.relu(combined_embeds @ self.layer1)   # Shape: (n, d)
        alpha = t.sigmoid(hidden @ self.layer2)

        user_embeds = alpha*user_embeds_social_wise + (1-alpha) * user_embeds_socially_connected_item_wise 

        return user_embeds

    def forward(self, mask_ratio=0): 
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        
        A_group = self.cluster_info.to(self.device)
        # for Community-Aware SSL
        if mask_ratio > 0: 
            indices = A_group._indices()  
            values = A_group._values()    
            nnz = indices.size(1)

            num_masked = int(nnz * mask_ratio)
            num_remain = nnz - num_masked

            # Randomly select indices to mask
            remain_indices = t.randperm(nnz)[:num_remain].to(self.device)

            new_indices = indices[:, remain_indices]
            new_values = values[remain_indices]
            A_group = t.sparse_coo_tensor(new_indices, new_values, A_group.size(), device=self.device).coalesce()
        
        user_embeds_social_community_wise = self._propagate(A_group, self.group_embeds)
        user_embeds_socially_connected_item_wise = self.socially_connected_item_aggregator(self.trust_mat, self.adj, 
                                                    t.concat([t.zeros_like(user_embeds_social_community_wise), self.item_embeds.detach()], axis=0))
        self.user_embeds = self._gatingNet(user_embeds_social_community_wise, user_embeds_socially_connected_item_wise)
        
        interaction_embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        interaction_embeds_list = [interaction_embeds]        
        interaction_adj = self.adj
        
        for i in range(self.layer_num): 
            interaction_embeds = self._propagate(interaction_adj, interaction_embeds_list[-1])
            interaction_embeds_list.append(interaction_embeds)
        
        embeds = sum(interaction_embeds_list)
        
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        # Community-Aware SSL
        user_embeds1, _ = self.forward(mask_ratio=self.mask_ratio)
        user_embeds2, _ = self.forward(mask_ratio=self.mask_ratio)
        anc_embeds1 = user_embeds1[ancs]
        anc_embeds2 = user_embeds2[ancs]

        cl_loss = self.cl_weight * cal_infonce_loss(anc_embeds1, anc_embeds2, anc_embeds2, self.temperature) / anc_embeds2.shape[0]
        loss = bpr_loss + reg_loss + cl_loss

        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

    def _coo_to_sparse_tensor(self, coo_mat):
        coo_mat = coo_mat.tocoo()
        indices = t.stack([
            t.from_numpy(coo_mat.row).long(),
            t.from_numpy(coo_mat.col).long()
        ])
        values = t.from_numpy(coo_mat.data).float()
        shape = coo_mat.shape
        return t.sparse_coo_tensor(indices, values, shape, device=self.device).coalesce()

    def _create_adj_matrix(self, trn_mat):
        user_num = self.user_num
        item_num = self.item_num
        total_num = user_num + item_num

        trn_indices = trn_mat._indices()
        trn_values = trn_mat._values()

        user_indices = trn_indices[0, :]
        item_indices = trn_indices[1, :] + user_num 

        upper_indices = t.stack([user_indices, item_indices])
        lower_indices = t.stack([item_indices, user_indices])

        combined_indices = t.cat([upper_indices, lower_indices], dim=1)
        combined_values = t.cat([trn_values, trn_values], dim=0)

        A = t.sparse_coo_tensor(combined_indices, combined_values, (total_num, total_num), device=self.device).coalesce()
        return A

    def _normalize_sparse_matrix(self, mat):
        degree = t.sparse.sum(mat, dim=1).to_dense()
        degree_inv_sqrt = t.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0  

        values = mat.values()
        indices = mat.indices()

        row = indices[0]
        col = indices[1]

        norm_values = degree_inv_sqrt[row] * values * degree_inv_sqrt[col]
        normalized_mat = t.sparse_coo_tensor(indices, norm_values, mat.size(), device=self.device)

        return normalized_mat.coalesce()
    
    def _normalize_sparse_matrix_2(self, weight, trust_mat):
        degree = t.sparse.sum(trust_mat, dim=1).to_dense()
        degree_inv_sqrt = t.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0  

        values = weight
        indices = trust_mat.indices()

        row = indices[0]
        col = indices[1]

        norm_values = degree_inv_sqrt[row] * values * degree_inv_sqrt[col]
        normalized_mat = t.sparse_coo_tensor(indices, norm_values, trust_mat.size(), device=self.device)

        return normalized_mat.coalesce()
    
    def _normalize_sparse_matrix_3(self, mat):
        degree = t.sparse.sum(mat, dim=1).to_dense()

        values = mat.values()
        indices = mat.indices()

        row = indices[0]
        # col = indices[1]

        norm_values = values / (degree[row] + 1e-8)
        normalized_mat = t.sparse_coo_tensor(indices, norm_values, mat.size(), device=self.device)

        return normalized_mat.coalesce()