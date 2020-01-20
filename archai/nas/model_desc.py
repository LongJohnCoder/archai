from enum import Enum
from typing import Mapping, Optional, List
import pathlib
import os
import torch

import yaml

from ..common.common import expdir_abspath


"""
Note: All classes in this file needs to be deepcopy compatible because
      descs are used as template to create copies by macro builder.
"""

class ConvMacroParams:
    """Holds parameters that may be altered by macro architecture"""

    def __init__(self, ch_in:int, ch_out:int) -> None:
        self.ch_in, self.ch_out = ch_in, ch_out

class OpDesc:
    """Op description that is in each edge
    """
    def __init__(self, name:str, params:dict, in_len:int,
                 trainables:Optional[Mapping],
                 children:Optional[List['OpDesc']]=None)->None:
        self.name = name
        self.in_len = in_len
        self.params = params # parameters specific to op needed to construct it
        self.trainables = trainables # TODO: make this private due to clear_trainable
        # if op is keeping any child op then it should save it in children
        # this was we can control state_dict of children
        self.children = children

    def clear_trainables(self)->None:
        self.trainables = None
        if self.children is not None:
            for child in self.children:
                child.clear_trainables()

    def state_dict(self)->dict:
        return  {
                    'trainables': self.trainables,
                    'children': [child.state_dict() if child is not None else None
                                 for child in self.children] \
                                     if self.children is not None else None
                }

    def load_state_dict(self, state_dict)->None:
        if state_dict is not None:
            self.trainables = state_dict['trainables']
            c, cs = self.children, state_dict['children']
            assert (c is None and cs is None) or \
                   (c is not None and cs is not None and len(c) == len(cs))
            if c is not None:
                for cx, csx in zip(c, cs):
                    assert (cx is None and csx is None) or \
                            (cx is not None and csx is not None)
                    if cx is not None:
                        cx.load_state_dict(csx)

class EdgeDesc:
    """Edge description between two nodes in the cell
    """
    def __init__(self, op_desc:OpDesc, index:int, input_ids:List[int])->None:
        assert op_desc.in_len == len(input_ids)
        self.op_desc = op_desc
        self.index = index
        self.input_ids = input_ids

class NodeDesc:
    def __init__(self, edges:List[EdgeDesc]) -> None:
        self.edges = edges

class AuxTowerDesc:
    def __init__(self, ch_in:int, n_classes:int) -> None:
        self.ch_in = ch_in
        self.n_classes = n_classes

class CellType(Enum):
    Regular = 'regular'
    Reduction  = 'reduction'

class CellDesc:
    def __init__(self, cell_type:CellType, index:int, nodes:List[NodeDesc],
            s0_op:OpDesc, s1_op:OpDesc, out_nodes:int, node_ch_out:int,
            alphas_from:int, max_final_edges:int, cell_post_op:str)->None:
        assert s0_op.params['conv'].ch_out == s1_op.params['conv'].ch_out
        assert s0_op.params['conv'].ch_out == node_ch_out

        self.cell_type = cell_type
        self.index = index
        self.nodes = nodes
        self.s0_op, self.s1_op = s0_op, s1_op
        self.out_nodes, self.node_ch_out = out_nodes, node_ch_out
        self.alphas_from = alphas_from # cell index with which we share alphas
        self.max_final_edges = max_final_edges
        self.cell_ch_out = out_nodes * node_ch_out
        self.conv_params = ConvMacroParams(node_ch_out, node_ch_out)
        self.cell_post_op = cell_post_op

    def all_empty(self)->bool:
        return len(self.nodes)==0 or all((len(n.edges)==0 for n in self.nodes))
    def all_full(self)->bool:
        return len(self.nodes)>0 and all((len(n.edges)>0 for n in self.nodes))


class ModelDesc:
    def __init__(self, stem0_op:OpDesc, stem1_op:OpDesc, pool_op:OpDesc,
                 ds_ch:int, n_classes:int, cell_descs:List[CellDesc],
                 aux_tower_descs:List[Optional[AuxTowerDesc]])->None:
        assert len(cell_descs) == len(aux_tower_descs)
        self.stem0_op, self.stem1_op, self.pool_op = stem0_op, stem1_op, pool_op
        self.ds_ch = ds_ch
        self.n_classes = n_classes
        self.cell_descs = cell_descs
        self.aux_tower_descs = aux_tower_descs

    def has_aux_tower(self)->bool:
        return any(self.aux_tower_descs)

    def all_empty(self)->bool:
        return len(self.cell_descs)==0 or \
             all((c.all_empty() for c in self.cell_descs))
    def all_full(self)->bool:
        return len(self.cell_descs)>0 and \
            all((c.all_full() for c in self.cell_descs))

    def state_dict(self, clear=False)->dict:
        state_dict = {}
        for ci, cell_desc in enumerate(self.cell_descs):
            sd_cell = state_dict[ci] = {}
            for ni, node in enumerate(cell_desc.nodes):
                sd_node = sd_cell[ni] = {}
                for ei, edge_desc in enumerate(node.edges):
                    sd_node[ei] = edge_desc.op_desc.state_dict()
                    if clear:
                        edge_desc.op_desc.clear_trainables()
        return state_dict

    def load_state_dict(self, state_dict:dict)->None:
        for ci, cell_desc in enumerate(self.cell_descs):
            sd_cell = state_dict[ci]
            for ni, node in enumerate(cell_desc.nodes):
                sd_node = sd_cell[ni]
                for ei, edge_desc in enumerate(node.edges):
                    edge_desc.op_desc.load_state_dict(sd_node[ei])

    def save(self, filename:str)->Optional[str]:
        yaml_filepath = expdir_abspath(filename)
        if yaml_filepath:
            if not yaml_filepath.endswith('.yaml'):
                yaml_filepath += '.yaml'

            # clear so PyTorch state is not saved in yaml
            state_dict = self.state_dict(clear=True)
            pt_filepath = ModelDesc._pt_filepath(yaml_filepath)
            torch.save(state_dict, pt_filepath)
            # save yaml
            pathlib.Path(yaml_filepath).write_text(yaml.dump(self))
            # restore state
            self.load_state_dict(state_dict)

        return yaml_filepath

    @staticmethod
    def _pt_filepath(desc_filepath:str)->str:
        return str(pathlib.Path(desc_filepath).with_suffix('.pth'))

    @staticmethod
    def load(yaml_filename:str)->'ModelDesc':
        yaml_filepath = expdir_abspath(yaml_filename)
        if not yaml_filepath or not os.path.exists(yaml_filepath):
            raise RuntimeError("Model description file is not found."
                "Typically this file should be generated from the search."
                "Please copy this file to '{}'".format(yaml_filepath))
        with open(yaml_filepath, 'r') as f:
            model_desc = yaml.load(f, Loader=yaml.Loader)

        # look for pth file that should have pytorch parameters state_dict
        pt_filepath = ModelDesc._pt_filepath(yaml_filepath)
        if os.path.exists(pt_filepath):
            state_dict = torch.load(pt_filepath, map_location=torch.device('cpu'))
            model_desc.load_state_dict(state_dict)
        # else no need to restore weights
        return model_desc
