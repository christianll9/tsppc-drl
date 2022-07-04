import torch
from typing import NamedTuple
from utils2.boolmask import mask_long2bool


class StateTSPPC(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    predecessors: torch.Tensor
    successors: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            predecessors=self.predecessors[key],
            successors=self.successors[key]
        )


    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot'][:, None, :]
        loc = input['loc']

        batch_size, n_loc, _ = loc.size()

        constr_indices = torch.nonzero(input['att_mask'][:, 1:, 1:]) + 1

        # Assuming all datasets in the batch have the same amount of constraints!
        successors   = constr_indices[:,1].reshape((batch_size, -1)) # [batch_size, n_constr]
        predecessors = constr_indices[:,2].reshape((batch_size, -1)) # [batch_size, n_constr]

        return StateTSPPC(
            coords=torch.cat((depot, loc), -2),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=depot,  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            successors=successors,
            predecessors=predecessors
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)


        assert self.visited_.dtype == torch.uint8
        visited_ = self.visited_.scatter(-1, selected[:, :, None], 1)
        self.successors[self.predecessors == selected] = 0 # remove successors ...

        
        return self._replace(
            prev_a=selected, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, 
        )

    def all_finished(self):
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, 1, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        n_loc = self.visited_.size(-1) - 1  # num of nodes without depot
        batch_size = self.visited_.size(0)

        if self.i == 0:  # [0,1...1]
            return torch.cat([torch.zeros(batch_size, 1, 1, dtype=torch.uint8, device=self.visited_.device),
                            torch.ones(batch_size, 1, n_loc, dtype=torch.uint8, device=self.visited_.device)], dim=-1) > 0

        
        mask_loc = torch.clone(self.visited_[:,0,:])
        n_succ = self.successors.size(1) # num of successors

        b_ids = torch.arange(self.ids.size(0)).view(self.ids.size(0), 1)
        mask_loc[b_ids.repeat(1, n_succ), self.successors] = 1        

        return (mask_loc > 0)[:, None, :]  # return true/false

    def construct_solutions(self, actions):
        return actions
