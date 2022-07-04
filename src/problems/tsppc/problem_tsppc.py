from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.tsppc.state_tsppc import StateTSPPC
from utils2.beam_search import beam_search


# import TSPPC generator
import sys, os
sys.path.append(os.path.join('..', '..', 'utils'))
import utils.problem_generator as probgen


class TSPPC(object):

    NAME = 'tsppc'

    @staticmethod
    def get_costs(dataset, pi):  # pi:[batch_size, graph_size]
        assert (pi[:, 0]==0).all(), "not starting at depot"
        dataset = torch.cat([dataset['depot'].reshape(-1, 1, 2), dataset['loc']], dim = 1)  # [batch, graph_size+1, 2]
        d = dataset.gather(1, pi.unsqueeze(-1).expand(-1, -1, 2))  # [batch, graph_size+1, 2]
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None


    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPPCDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSPPC.initialize(*args, **kwargs)
        
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)
        
        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )
        state = TSPPC.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
            )        

        return beam_search(state, beam_size, propose_expansions)

def make_instance(args):
    loc, att_mask, group_mask, *args = args
    return {
        'loc': torch.tensor(loc[1:], dtype=torch.float),
        'depot': torch.tensor(loc[0], dtype=torch.float),
        'att_mask': torch.tensor(att_mask, dtype=torch.bool),
        'group_mask': torch.tensor(group_mask, dtype=torch.bool)
    }


class TSPPCDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=None, offset=0, distribution=None):
        super(TSPPCDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            data = probgen.preprocessed_tsppcs(n_problems=num_samples, n_nodes=size, distr=distribution)

        if num_samples is None:
            self.data = [make_instance(args) for args in data[offset:]]
        else:
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
