from typing import Optional, Tuple, List
from copy import deepcopy

from overrides import EnforceOverrides

from ..common.config import Config
from .model_desc import ModelDesc, OpDesc, CellType, NodeDesc, EdgeDesc, \
                        CellDesc, AuxTowerDesc, ConvMacroParams
from ..common.common import get_logger

class MacroBuilder(EnforceOverrides):
    def __init__(self, conf_model_desc: Config, aux_tower:bool,
                 template:Optional[ModelDesc]=None)->None:
        # region conf vars
        conf_data = conf_model_desc['dataset']
        self.ds_name = conf_data['name']
        self.ds_ch = conf_data['channels']
        self.n_classes = conf_data['n_classes']
        self.init_ch_out = conf_model_desc['init_ch_out']
        self.n_cells = conf_model_desc['n_cells']
        self.n_nodes = conf_model_desc['n_nodes']
        self.out_nodes = conf_model_desc['out_nodes']
        self.stem_multiplier = conf_model_desc['stem_multiplier']
        self.aux_weight = conf_model_desc['aux_weight']
        self.max_final_edges = conf_model_desc['max_final_edges']
        self.cell_post_op = conf_model_desc['cell_post_op']
        # endregion

        self.aux_tower = aux_tower
        self.template = template

        self._set_templates()
        self._set_op_names()

    def _set_templates(self)->None:
        self.normal_template,  self.reduction_template = None, None
        if self.template is not None:
            for cell_desc in self.template.cell_descs:
                if self.normal_template is None and \
                        cell_desc.cell_type==CellType.Regular:
                  self.normal_template = cell_desc
                if self.reduction_template is None and \
                        cell_desc.cell_type==CellType.Reduction:
                    self.reduction_template = cell_desc

    def model_desc(self)->ModelDesc:
        # create stems
        stem0_op, stem1_op = self._get_model_stems()

        # create cell descriptions
        cell_descs, aux_tower_descs = self._get_cell_descs(
            stem0_op.params['conv'].ch_out, self.max_final_edges)

        pool_op = OpDesc(self.pool_op_name,
                         params={'conv': cell_descs[-1].conv_params},
                         in_len=1, trainables=None)

        return ModelDesc(stem0_op, stem1_op, pool_op, self.ds_ch,
                         self.n_classes, cell_descs, aux_tower_descs)

    def _set_op_names(self)->None:
        if self.ds_name == 'cifar10':
            self.stem0_op_name, self.stem1_op_name, self.pool_op_name = \
                'stem_cifar', 'stem_cifar', 'pool_cifar'
        elif self.ds_name == 'imagenet':
            self.stem0_op_name, self.stem1_op_name, self.pool_op_name = \
                 'stem0_imagenet', 'stem1_imagenet', 'pool_imagenet'
        else:
            raise NotImplementedError(
                f'Stem and pool ops for "{self.ds_name}" are not supported yet')

    def _get_cell_descs(self, stem_ch_out:int, max_final_edges:int)\
            ->Tuple[List[CellDesc], List[Optional[AuxTowerDesc]]]:
        cell_descs, aux_tower_descs = [], []
        reduction_p = False
        pp_ch_out, p_ch_out, ch_out = stem_ch_out, stem_ch_out, self.init_ch_out

        first_normal, first_reduction = -1, -1
        for ci in range(self.n_cells):
            # find cell type and output channels for this cell
            # also update if this is our first cell from which alphas will be shared
            reduction = self._is_reduction(ci)
            if reduction:
                ch_out, cell_type = ch_out*2, CellType.Reduction
                first_reduction = ci if first_reduction < 0 else first_reduction
                alphas_from = first_reduction
            else:
                cell_type = CellType.Regular
                first_normal = ci if first_normal < 0 else first_normal
                alphas_from = first_normal

            s0_op, s1_op = self._get_cell_stems(
                ch_out, p_ch_out, pp_ch_out, reduction_p)

            nodes:List[NodeDesc] = [NodeDesc(edges=[]) for _ in range(self.n_nodes)]

            cell_descs.append(CellDesc(
                cell_type=cell_type, index=ci,
                nodes=nodes,
                s0_op=s0_op, s1_op=s1_op,
                out_nodes=self.out_nodes, node_ch_out=ch_out,
                alphas_from=alphas_from,
                max_final_edges=max_final_edges, cell_post_op=self.cell_post_op
            ))
            # add any nodes from the template to the just added cell
            self._add_template_nodes(cell_descs[-1])
            # add aux tower
            aux_tower_descs.append(self._get_aux_tower(cell_descs[-1]))

            # we concate all channels so next cell node gets channels from all nodes
            pp_ch_out, p_ch_out = p_ch_out, cell_descs[-1].cell_ch_out
            reduction_p = reduction

        return cell_descs, aux_tower_descs

    def _add_template_nodes(self, cell_desc:CellDesc)->None:
        """For specified cell, copy nodes from the template """

        if self.template is None:
            return

        logger = get_logger()

        # select cell template
        reduction = cell_desc.cell_type == CellType.Reduction
        cell_template = self.reduction_template if reduction else self.normal_template

        if cell_template is None:
            return

        if len(cell_desc.nodes) > len(cell_template.nodes):
            cell_desc.nodes = cell_desc.nodes[:len(cell_template.nodes)]
            logger.warn('cell nodes trimmed, cell nodes={}, template nodes={}'.format(
                len(cell_desc.nodes), len(cell_template.nodes)))

        # copy each template node to cell
        for node, template_node in zip(cell_desc.nodes, cell_template.nodes):
            edges_copy = deepcopy(template_node.edges)
            nl = len(node.edges)
            for ei, ec in enumerate(edges_copy):
                # TODO: need method on EdgeDesc for for this
                ec.op_desc.params['conv'] = deepcopy(cell_desc.conv_params)
                ec.op_desc.clear_trainables() # TODO: check for compatibility?
                ec.index = ei + nl
            node.edges.extend(edges_copy)

    def _is_reduction(self, cell_index:int)->bool:
        return cell_index in [self.n_cells//3, 2*self.n_cells//3]

    def _get_cell_stems(self, ch_out: int, p_ch_out: int, pp_ch_out:int,
                   reduction_p: bool)->Tuple[OpDesc, OpDesc]:
        # TODO: investigate why affine=False for search but True for test
        s0_op = OpDesc('prepr_reduce' if reduction_p else 'prepr_normal',
                    params={
                        'conv': ConvMacroParams(pp_ch_out, ch_out)
                    }, in_len=1, trainables=None)

        s1_op = OpDesc('prepr_normal',
                    params={
                        'conv': ConvMacroParams(p_ch_out, ch_out)
                    }, in_len=1, trainables=None)
        return s0_op, s1_op

    def _get_aux_tower(self, cell_desc:CellDesc) -> Optional[AuxTowerDesc]:
        # TODO: shouldn't we be adding aux tower at *every* 1/3rd?
        if self.aux_tower and    \
                self.aux_weight > 0.0 and   \
                cell_desc.index == 2*self.n_cells//3:
            return AuxTowerDesc(cell_desc.cell_ch_out, self.n_classes)
        return None

    def _get_model_stems(self)->Tuple[OpDesc, OpDesc]:
        # TODO: weired not always use two different stemps as in original code
        # TODO: why do we need stem_multiplier?
        # TODO: in original paper stems are always affine
        conv_params = ConvMacroParams(self.ds_ch,
                                      self.init_ch_out*self.stem_multiplier)
        stem0_op = OpDesc(name=self.stem0_op_name, params={'conv': conv_params},
                          in_len=1, trainables=None)
        stem1_op = OpDesc(name=self.stem1_op_name, params={'conv': conv_params},
                          in_len=1, trainables=None)
        return stem0_op, stem1_op
