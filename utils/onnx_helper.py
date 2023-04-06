import re
from typing import Dict, Tuple, Union

import onnx
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from onnx import ModelProto, ValueInfoProto, GraphProto


__verbose__: bool = True
"""
Whether log information of successful operations
"""


def _verbose(self, *args, sep=' ', end='\n', file=None):
    if __verbose__:
        print(self, *args, sep=sep, end=end, file=file)


def model_override_io_shapes(
        model: ModelProto,
        input_shapes: Dict[str, Tuple[Union[str, int]]] = None,
        output_shapes: Dict[str, Tuple[Union[str, int]]] = None,
):
    """
    Override the shapes of inputs/outputs of the model graph (in-place operation).
    @param model: model to perform the operation on
    @param input_shapes: a dict with keys as input/output names and values as shape tuples
    @param output_shapes: the same as input_shapes
    """
    def _override_shapes(
            shape_list_old: RepeatedCompositeFieldContainer[ValueInfoProto],
            shape_dict_new: Dict[str, Tuple[Union[str, int]]]):
        for value_info in shape_list_old:
            if value_info.name in shape_dict_new:
                name = value_info.name
                dims = value_info.type.tensor_type.shape.dim
                assert len(shape_dict_new[name]) == len(dims), \
                    f'Number of given and existing dimensions mismatch: {name}'
                for i, dim in enumerate(shape_dict_new[name]):
                    if isinstance(dim, int):
                        dims[i].dim_param = ''
                        dims[i].dim_value = dim
                    else:
                        dims[i].dim_value = 0
                        dims[i].dim_param = dim
                _verbose(f'| override shape of \'{name}\' with {list(shape_dict_new[name])}')

    if input_shapes is not None:
        _override_shapes(model.graph.input, input_shapes)
    if output_shapes is not None:
        _override_shapes(model.graph.output, output_shapes)


def model_add_prefixes(
        model: ModelProto,
        initializer_prefix=None,
        value_info_prefix=None,
        node_prefix=None,
        dim_prefix=None,
        ignored_pattern=None,
):
    """
    Adds prefixes to names inside the given ONNX model graph, including sub-graphs (in-place operation).
    This method is a complete version of the official onnx.compose.add_prefix API, which does not consider sub-graphs.
    """
    initializers = set()
    value_infos = set()

    def _record_initializers_and_value_infos_recursive(subgraph):
        # Record names in current graph
        for initializer in subgraph.initializer:
            if re.match(ignored_pattern, initializer.name):
                continue
            initializers.add(initializer.name)
        for value_info in subgraph.value_info:
            if re.match(ignored_pattern, value_info.name):
                continue
            value_infos.add(value_info.name)
        for node in subgraph.node:
            # For 'If' and 'Loop' nodes, do recording recursively
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _record_initializers_and_value_infos_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _record_initializers_and_value_infos_recursive(body)

    def _add_prefixes_recursive(subgraph):
        # Add prefixes in current graph
        if initializer_prefix is not None:
            for initializer in subgraph.initializer:
                if re.match(ignored_pattern, initializer.name):
                    continue
                new_name = initializer_prefix + initializer.name
                _verbose('| add prefix:', initializer.name, '->', new_name)
                initializer.name = new_name
        
        for value_info in subgraph.value_info:
            if dim_prefix is not None:
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_param is None or dim.dim_param == '' or re.match(ignored_pattern, dim.dim_param):
                        continue
                    new_dim_param = dim_prefix + dim.dim_param
                    _verbose('| add prefix:', dim.dim_param, '->', new_dim_param)
                    dim.dim_param = new_dim_param

            if value_info_prefix is None or re.match(ignored_pattern, value_info.name):
                continue
            new_name = value_info_prefix + value_info.name
            _verbose('| add prefix:', value_info.name, '->', new_name)
            value_info.name = new_name
        
        if node_prefix is not None:
            for node in subgraph.node:
                if re.match(ignored_pattern, node.name):
                    continue
                new_name = node_prefix + node.name
                _verbose('| add prefix:', node.name, '->', new_name)
                node.name = new_name
        
        for node in subgraph.node:
            # For 'If' and 'Loop' nodes, add prefixes recursively
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _add_prefixes_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _add_prefixes_recursive(body)
            
            # For each node, rename its inputs and outputs
            for io_list in [node.input, node.output]:
                for i, io_value in enumerate(io_list):
                    if io_value in initializers and initializer_prefix is not None:
                        new_value = initializer_prefix + io_value
                        _verbose('| add prefix:', io_value, '->', new_value)
                        io_list.pop(i)
                        io_list.insert(i, new_value)
                    if io_value in value_infos and value_info_prefix is not None:
                        new_value = value_info_prefix + io_value
                        _verbose('| add prefix:', io_value, '->', new_value)
                        io_list.pop(i)
                        io_list.insert(i, new_value)

    _record_initializers_and_value_infos_recursive(model.graph)
    _add_prefixes_recursive(model.graph)


def graph_fold_back_to_squeeze(graph: GraphProto):
    """
    Fold the substructures of 'Shape', 'Gather', 'Equal', 'If' to one single 'Squeeze' node.
    This can unify the different behaviors between aten::squeeze and onnx:Squeeze.
    """
    def _graph_fold_back_to_squeeze_recursive(subgraph: GraphProto):
        # Do folding in sub-graphs recursively.
        for node in subgraph.node:
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _graph_fold_back_to_squeeze_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _graph_fold_back_to_squeeze_recursive(body)

        # Do folding in current graph.
        i_shape = 0
        while i_shape < len(subgraph.node):
            if subgraph.node[i_shape].op_type == 'Shape':
                shape_node = subgraph.node[i_shape]
                shape_out = shape_node.output[0]
                i_gather = i_shape + 1
                while i_gather < len(subgraph.node):
                    if subgraph.node[i_gather].op_type == 'Gather' and subgraph.node[i_gather].input[0] == shape_out:
                        gather_node = subgraph.node[i_gather]
                        gather_out = gather_node.output[0]
                        i_equal = i_gather + 1
                        while i_equal < len(subgraph.node):
                            if subgraph.node[i_equal].op_type == 'Equal' and (
                                    subgraph.node[i_equal].input[0] == gather_out
                                    or subgraph.node[i_equal].input[1] == gather_out):
                                equal_node = subgraph.node[i_equal]
                                equal_out = equal_node.output[0]
                                i_if = i_equal + 1
                                while i_if < len(subgraph.node):
                                    if subgraph.node[i_if].op_type == 'If' \
                                            and subgraph.node[i_if].input[0] == equal_out:
                                        # Found the substructure to be folded.
                                        if_node = subgraph.node[i_if]
                                        # Create 'Squeeze' node.
                                        squeeze_node = onnx.helper.make_node(
                                            op_type='Squeeze',
                                            inputs=[
                                                *list(shape_node.input),
                                                # For ONNX opset >= 13, axes should be an input instead of an attribute.
                                                gather_node.input[1]  # Use 'indices' input of 'Gather'
                                            ],
                                            outputs=if_node.output,
                                            name=shape_node.name.replace('Shape', 'Squeeze')
                                        )
                                        # Replace 'Shape', 'Gather', 'Equal', 'If' with 'Squeeze'.
                                        subgraph.node.insert(i_shape, squeeze_node)
                                        subgraph.node.remove(shape_node)
                                        subgraph.node.remove(gather_node)
                                        subgraph.node.remove(equal_node)
                                        subgraph.node.remove(if_node)
                                        _verbose(
                                            f'| fold nodes: [\'{shape_node.name}\', \'{gather_node.name}\', '
                                            f'\'{equal_node.name}\', \'{if_node.name}\'] -> \'{squeeze_node.name}\'')
                                        break
                                    i_if += 1
                                else:
                                    break
                            i_equal += 1
                        else:
                            break
                    i_gather += 1
                else:
                    break
            i_shape += 1

    _graph_fold_back_to_squeeze_recursive(graph)


def graph_remove_unused_values(graph: GraphProto):
    used_values = set()

    def _record_usage_recursive(subgraph: GraphProto):
        for node in subgraph.node:
            # For 'If' and 'Loop' nodes, do recording recursively
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _record_usage_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _record_usage_recursive(body)
            # For each node, record its inputs and outputs
            for io_list in [node.input, node.output]:
                for io_value in io_list:
                    used_values.add(io_value)

    def _clean_unused_recursively(subgraph):
        # Do cleaning in sub-graphs recursively.
        for node in subgraph.node:
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _clean_unused_recursively(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _clean_unused_recursively(body)

        # Do cleaning in current graph.
        i = 0
        while i < len(subgraph.initializer):
            name = subgraph.initializer[i].name
            if name not in used_values:
                subgraph.initializer.pop(i)
                _verbose(f'| remove unused initializer: {name}')
            else:
                i += 1
        i = 0
        while i < len(subgraph.value_info):
            name = subgraph.value_info[i].name
            if name not in used_values:
                subgraph.value_info.pop(i)
                _verbose(f'| remove unused value info: {name}')
            else:
                i += 1

    _record_usage_recursive(graph)
    _clean_unused_recursively(graph)
