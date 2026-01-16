import torch as t
from torch import nn
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LIGHTGCN(BaseModel):
    def __init__(self, data_handler):
        super(LIGHTGCN, self).__init__(data_handler)

        self.device = configs['device']
        self.trn_mat = self._coo_to_sparse_tensor(data_handler.trn_mat)

        A = self._create_adj_matrix(self.trn_mat)
        self.adj = self._normalize_sparse_matrix(A)

        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        self.is_training = True
        self.final_embeds = None

        param_list = [
            ('user_embeds', self.user_embeds),
            ('item_embeds', self.item_embeds)
        ]
        total_params = 0
        for name, param in param_list:
            num_params = param.numel()
            print(f"{name}: {num_params}")
            total_params += num_params
        print(f"Total parameters: {total_params}")

        # # Add this to print total number of trainable parameters
        # total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Total trainable parameters: {total_params}")

    
    def _coo_to_sparse_tensor(self, coo_mat):
        coo_mat = coo_mat.tocoo()
        indices = t.tensor([coo_mat.row, coo_mat.col], dtype=t.long)
        values = t.tensor(coo_mat.data, dtype=t.float32)
        shape = coo_mat.shape
        return t.sparse_coo_tensor(indices, values, shape, device=self.device).coalesce()

    def _create_adj_matrix(self, trn_mat):
        user_num = self.user_num
        item_num = self.item_num
        total_num = user_num + item_num

        trn_indices = trn_mat._indices()
        trn_values = trn_mat._values()

        user_indices = trn_indices[0, :]
        item_indices = trn_indices[1, :] + user_num  # item index shift

        upper_indices = t.stack([user_indices, item_indices])
        lower_indices = t.stack([item_indices, user_indices])

        combined_indices = t.cat([upper_indices, lower_indices], dim=1)
        combined_values = t.cat([trn_values, trn_values], dim=0)

        A = t.sparse_coo_tensor(combined_indices, combined_values, (total_num, total_num), device=self.device).coalesce()
        
        return A

    def _normalize_sparse_matrix(self, mat):
        degree = t.sparse.sum(mat, dim=1).to_dense()
        degree[degree == 0] = 1e-8
        degree_inv_sqrt = t.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0

        values = mat.values()
        indices = mat.indices()

        row = indices[0]
        col = indices[1]

        norm_values = degree_inv_sqrt[row] * values * degree_inv_sqrt[col]
        normalized_mat = t.sparse_coo_tensor(indices, norm_values, mat.size(), device=self.device)

        return normalized_mat.coalesce()

    def _propagate(self, adj, embeds):
        propagated_embeds = t.sparse.mm(adj, embeds).to(self.device)
        return propagated_embeds
    
    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0).to(self.device)
        embeds_list = [embeds]

        for i in range(self.layer_num):
            embeds = self._propagate(self.adj, embeds_list[-1])
            embeds_list.append(embeds)
        
        embeds = sum(embeds_list)# / len(embeds_list)
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
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
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