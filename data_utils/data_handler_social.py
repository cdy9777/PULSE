import pickle
import numpy as np
from scipy.sparse import coo_matrix
from config.configurator import configs
from data_utils.datasets_social import PairwiseTrnData, AllRankTstData
import torch.utils.data as data


class DataHandlerSocial:
    def __init__(self):
        dataset_name = configs['data']['name']
        predir = f'./datasets/social/{dataset_name}/'
        if predir is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in predefined paths.")

        self.trn_file = predir + 'trn_mat.pkl'
        self.val_file = predir + 'val_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'
        self.trust_file = predir + 'trust_mat.pkl'
        self.group_info_file = predir + 'group_info'
        self.group_info_file_leiden = predir + 'leiden'

    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file

        Args:
            file (string): path of the file to load

        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat

    def _load(self, path):
        with open(path, 'rb') as fs:
            data = pickle.load(fs)
        return data

    def _save(self, data, path):
        with open(path, 'wb') as fs:
            pickle.dump(data, fs)

    def load_data(self):
        trn_mat = self._load(self.trn_file)
        val_mat = self._load(self.val_file)
        tst_mat = self._load(self.tst_file)
        trust_mat = self._load(self.trust_file)

        self.trn_mat = trn_mat
        self.val_mat = val_mat
        self.trust_mat = trust_mat
        
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
        
        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
    
        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)

        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
