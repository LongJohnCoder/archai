from overrides import overrides

from ..nas.micro_builder import MicroBuilder
from ..nas.operations import Op
from ..nas.model_desc import ModelDesc, CellDesc, CellType, OpDesc, EdgeDesc
from .xnas_op import XnasOp

class XnasMicroBuilder(MicroBuilder):
    @overrides
    def register_ops(self) -> None:
        Op.register_op('xnas_op',
                       lambda op_desc, alphas, affine:
                           XnasOp(op_desc, alphas, affine))

    @overrides
    def build(self, model_desc:ModelDesc, search_iteration:int)->None:
        assert search_iteration==0, 'Multiple iterations for xnas is not supported'
        for cell_desc in model_desc.cell_descs:
            self._build_cell(cell_desc)

    def _build_cell(self, cell_desc:CellDesc)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add xnas op for each edge
        for i, node in enumerate(cell_desc.nodes):
            for j in range(i+2):
                op_desc = OpDesc('xnas_op',
                                    params={
                                        'conv': cell_desc.conv_params,
                                        'stride': 2 if reduction and j < 2 else 1
                                    }, in_len=1, trainables=None, children=None)
                edge = EdgeDesc(op_desc, len(node.edges), input_ids=[j])
                node.edges.append(edge)



