from overrides import overrides

from  ..nas.model_desc import ModelDesc, CellDesc, CellDesc, NodeDesc, OpDesc, \
                              EdgeDesc, CellType
from ..nas.micro_builder import MicroBuilder
from ..nas.operations import Op
from .petridish_op import PetridishOp, PetridishFinalOp


class PetridishMicroBuilder(MicroBuilder):
    @overrides
    def register_ops(self) -> None:
        Op.register_op('petridish_normal_op',
                    lambda op_desc, alphas, affine:
                        PetridishOp(op_desc, alphas, False, affine))
        Op.register_op('petridish_reduction_op',
                    lambda op_desc, alphas, affine:
                        PetridishOp(op_desc, alphas, True, affine))
        Op.register_op('petridish_final_op',
                    lambda op_desc, alphas, affine:
                        PetridishFinalOp(op_desc, affine))


    @overrides
    def build(self, model_desc:ModelDesc, search_iteration:int)->None:
        for cell_desc in model_desc.cell_descs:
            self._build_cell(cell_desc, search_iteration)

    def _build_cell(self, cell_desc:CellDesc, search_iteration:int)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # remove all empty nodes
        for i in reversed(range(len(cell_desc.nodes))):
            if len(cell_desc.nodes[i].edges)==0:
                cell_desc.nodes.pop(i)

        # for each search iteration i, we will operate on node i
        # cell falls short of i-th node, then add it
        if len(cell_desc.nodes) == search_iteration:
            cell_desc.nodes.append(NodeDesc(edges=[]))

        # if we don't have node to operate, then it's no go
        assert len(cell_desc.nodes) >= search_iteration+1

        # At each iteration i we pick the node i and add petridish op to it
        # NOTE: Where is it enforced that the cell already has 1 node. How is that node created?
        input_ids = list(range(search_iteration + 2)) # all previous states are input
        op_desc = OpDesc('petridish_reduction_op' if reduction else 'petridish_normal_op',
                            params={
                                'conv': cell_desc.conv_params,
                                # specify strides for each input, later we will
                                # give this to each primitive
                                '_strides':[2 if reduction and j < 2 else 1 \
                                           for j in input_ids],
                            }, in_len=len(input_ids), trainables=None, children=None)
        # add our op to last node
        node = cell_desc.nodes[search_iteration]
        edge = EdgeDesc(op_desc, index=len(node.edges),
                        input_ids=input_ids)
        node.edges.append(edge)
