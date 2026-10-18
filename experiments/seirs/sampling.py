"""SEIRS coefficient and tensor perturbation sampling."""

import numpy as np
import torch
from .fields import calc_grids, calc_X, ode2vec


class DataSampler(object):

    def __init__(self, config):
        self.config = dict(config)
        self.odes = []
        self.grid_shape = self.config["grid_shape"]
        self.grids = calc_grids(
            grid_limits=self.config["grid_limits"], grid_shape=self.config["grid_shape"]
        )
        self.max_order = 0
        self.dimension = 0
        self.extra_term_dicts = {}
        self.X = calc_X(self.grids)

    def get_batch(self, batch_size, indexes=None, turb=True, turb_scale=None):
        vecs = []
        if indexes is None:
            indexes = np.random.randint(len(self.odes), size=(batch_size,))
        elif isinstance(indexes, (int, float)):
            indexes = np.array([indexes])
        if turb_scale is None:
            turb_scale = self.config["turb_scale"]
        for index in indexes:
            if turb:
                self.odes[index].random_coef()
            else:
                self.odes[index].reset_coef()
            vec = ode2vec(ode=self.odes[index], grids=self.grids)
            if turb:
                vec += turb_scale * np.random.randn(*vec.shape)
            vecs.append(vec)
        self.odes[index].reset_coef()
        vecs = np.array(vecs)
        vecs = vecs.transpose(0, 4, 1, 2, 3)
        return (
            torch.tensor(vecs, dtype=torch.float32),
            [self.odes[index].name for index in indexes],
        )

    def add_odes(self, odes):
        for ode in odes:
            self.odes.append(ode)
            if self.max_order < ode.max_order:
                self.max_order = ode.max_order
            if self.dimension < ode.dimension:
                self.dimension = ode.dimension
            self.extra_term_dicts.update(ode.extra_term_dicts)
            print(f"ODE {ode.name} added to Data Sampler!")

    def remove_all_odes(self):
        self.odes = []
        print("All ODEs removed")
