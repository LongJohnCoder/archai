from overrides import overrides

from  ..nas.model_desc import ModelDesc, CellDesc, OpDesc, \
                              EdgeDesc, CellType
from ..nas.micro_builder import MicroBuilder
from ..nas.operations import Op
from .petridish_op import PetridishOp, PetridishFinalOp, TempIdentityOp


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
        Op.register_op('temp_identity_op',
                    lambda op_desc, alphas, affine:
                        TempIdentityOp(op_desc))

    @overrides
    def seed(self, model_desc:ModelDesc)->None:
        for cell_desc in model_desc.cell_descs:
            # add identity op for all empty nodes after search_iteration
            for i in range(len(cell_desc.nodes())):
                if len(cell_desc.nodes()[i].edges)==0:
                    op_desc = OpDesc('temp_identity_op',
                        params={'conv': cell_desc.conv_params},
                        in_len=1, trainables=None, children=None)
                    edge = EdgeDesc(op_desc, index=0, input_ids=[i-1])
                    cell_desc.nodes()[i].edges.append(edge)

    @overrides
    def build(self, model_desc:ModelDesc, search_iteration:int)->None:
        for cell_desc in model_desc.cell_descs:
            self._build_cell(cell_desc, search_iteration)

    def _build_cell(self, cell_desc:CellDesc, search_iteration:int)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # for each search iteration i, we will operate on node i
        node = cell_desc.nodes()[search_iteration]

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
        edge = EdgeDesc(op_desc, index=len(node.edges), input_ids=input_ids)

        # overwrite previously added temp identity operator if any
        id_edges = [(i, edge) for i, edge in enumerate(node.edges)
                    if edge.op_desc.name == 'temp_identity_op']
        assert len(id_edges) <= 1
        if len(id_edges):
            node.edges[id_edges[0][0]] = edge
        else:
            node.edges.append(edge)

