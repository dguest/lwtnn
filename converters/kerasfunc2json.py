#!/usr/bin/env python3
#
# Converter from Keras sequential NN to JSON
"""____________________________________________________________________
Variable specification file

In additon to the standard Keras architecture and weights files, you
must provide a "variable specification" json file.

Here `scale` and `offset` account for any scaling and shifting to the
input variables in preprocessing. The "default" value is optional.

If no file is provided, a template will be generated.
"""

import argparse
import json
import h5py
from collections import Counter
import sys, os
import importlib
from keras_layer_converters_common import skip_layers

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.arch_file, 'r') as arch_file:
        arch = json.load(arch_file)
    if not args.variables_file:
        _build_variables_file(args)
        sys.exit(0)
    with open(args.variables_file, 'r') as variables_file:
        variables = json.load(variables_file)

    _check_version(arch)

    with h5py.File(args.hdf5_file, 'r') as h5:
        for group in h5:
            if group == "model_weights":
                sys.exit(
                    "The weight file has been saved incorrectly.\n"
                    "Please see https://github.com/lwtnn/lwtnn/wiki/Keras-Converter#saving-keras-models \n"
                    "on how to correctly save weights.")
        layers, node_dict = _get_layers_and_nodes(arch, h5)
    input_layer_arch = arch['config']['input_layers']
    nodes = _build_node_list(node_dict, input_layer_arch)

    vars_per_input = _get_vars_per_input(input_layer_arch, node_dict)
    out_dict = {
        'layers': layers, 'nodes': nodes,
        'inputs': _parse_inputs(
            variables['inputs'],
            vars_per_input[1]),
        'input_sequences': _parse_inputs(
            variables['input_sequences'],
            vars_per_input[2]),
        'outputs': _parse_outputs(
            variables['outputs'],
            arch['config']['output_layers'],
            node_dict),
    }
    print(json.dumps(out_dict, indent=2, sort_keys=True))

def _check_version(arch):
    if arch["class_name"] != "Model":
        sys.exit("this is not a graph, try using keras2json")
    global BACKEND
    if 'backend' not in arch:
        sys.stderr.write(
            'WARNING: no backend found for this architecture!\n'
            'Defaulting to theano.\n')
        BACKEND="theano"
    else:
        BACKEND = arch['backend']
    global KERAS_VERSION
    if 'keras_version' not in arch:
        sys.stderr.write(
            'WARNING: no version number found for this architecture!\n'
            'Defaulting to version 1.2.\n')
        KERAS_VERSION=1
    else:
        major, minor, *bugfix = arch['keras_version'].split('.')
        KERAS_VERSION=int(major)
        config_tmp = (
            "lwtnn converter being configured for keras (v{}.{}).\n")
        sys.stderr.write(config_tmp.format(major, minor))


def _get_args():
    parser = argparse.ArgumentParser(
        description="Converter from Keras saved NN to JSON",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('arch_file', help='architecture json file')
    parser.add_argument('hdf5_file', help='Keras weights file')
    parser.add_argument('variables_file', help='variable spec as json',
                        nargs='?')
    return parser.parse_args()

def _build_variables_file(args):
    with open(args.arch_file, 'r') as arch_file:
        arch = json.load(arch_file)

    _check_version(arch)

    with h5py.File(args.hdf5_file, 'r') as h5:
        layers, node_dict = _get_layers_and_nodes(arch, h5)
    input_layer_arch = arch['config']['input_layers']
    vars_per_input = _get_vars_per_input(input_layer_arch, node_dict)
    def get_input(n):
        return {'name': 'variable_{}'.format(n), 'scale': 1, 'offset':0}
    def build_inputs(input_items):
        inputs = []
        for nodenum, n_vars in input_items:
            the_input = {
                'name': 'node_{}'.format(nodenum),
                'variables': [get_input(n) for n in range(n_vars)]
            }
            inputs.append(the_input)
        return inputs
    inputs = build_inputs(sorted(vars_per_input[1].items()))
    seq_inputs = build_inputs(sorted(vars_per_input[2].items()))
    outputs = []
    for kname, kid, ks in arch['config']['output_layers']:
        node = node_dict[(kname, kid)]
        output = {
            'name': '{}_{}'.format(kname, kid),
            'labels': ['out_{}'.format(x) for x in range(node.n_outputs)]
        }
        outputs.append(output)
    out_dict = {
        'inputs': inputs, 'input_sequences': seq_inputs,
        'outputs': outputs}
    print(json.dumps(out_dict, indent=2, sort_keys=True))


# __________________________________________________________________________
# master layer converter / inputs function

# output names
LAYERS = 'layers'
NODES = 'nodes'

class Node:
    # TODO: make the node object a wrapper on the Keras layer dictionary
    def __init__(self, layer, idx):
        self.layer_type = layer['class_name'].lower()
        self.name = layer['name']
        self.idx = idx
        self.sources = []
        self.number = None
        self.layer_number = None
        self.n_outputs = None
        self.dims = None
        inbound = layer['inbound_nodes']
        if self.layer_type != "inputlayer":
            for sname, sidx, *something in inbound[idx]:
                for some_stuff in something:
                    assert not some_stuff
                self.sources.append( (sname, sidx) )
        else:
            shape = layer['config']['batch_input_shape']
            self.n_outputs = shape[-1]
            self.dims = len(shape) - 1
        self.keras_layer = layer

    def __str__(self):
        parents = ', '.join(( str(x) for x in self.sources))
        tmp = '{} <- [{}]'
        if self.number is not None:
            tmp = '{}{{{n}}} <- [{}]'
        return tmp.format( (self.name, self.idx), parents, n=self.number)

    def get_key(self):
        return (self.name, self.idx)

    def __lt__(self, other):
        return self.get_key() < other.get_key()

# We want to build one node per _inbound_ node
def _build_node_dict(network):
    layers = {l['name']: l for l in network['config']['layers']}
    nodes = {}

    # first get the nodes that something points to
    # source nodes are inbound nodes
    for layer in layers.values():
        for sink in layer['inbound_nodes']:
            for kname, kid, *something in sink:
                nodes[(kname, kid)] = Node(layers[kname], kid)

    # get the output nodes now
    for kname, kid, something in network['config']['output_layers']:
        id_tup = (kname, kid)
        if id_tup not in nodes:
            nodes[id_tup] = Node(layers[kname], kid)

    # now we collapse the node references
    for node in nodes.values():
            source_nodes = []
            for source in node.sources:
                source_nodes.append(nodes[source])
            node.sources = source_nodes

    # Remove the nodes and sources that are of type skip_layers
    removed_nodes = set()
    for node_index, node in nodes.items():
        if node.layer_type in skip_layers:
            removed_nodes.add(node_index)
        else:
            new_sources = []
            for source in node.sources:
                new_sources.append(_get_valid_sources(source))
            node.sources = new_sources
    for node_index in removed_nodes:
        del nodes[node_index]
    return nodes

def _get_valid_sources(node_source):
    """Function to get valid sources for a node.
        Apply this recursively"""
    if node_source.layer_type not in skip_layers:
        return node_source
    else:
        assert len(node_source.sources) == 1
        #@Todo: Check that this will work with two skip_layers in a row
        return _get_valid_sources(node_source.sources[0])


def _number_nodes(node_dict):
    for number, node in enumerate(sorted(node_dict.values())):
        node.number = number

def _build_layer(output_layers, node_key, h5, node_dict, layer_dict):
    node = node_dict[node_key]
    if node.n_outputs is not None:
        return

    for source in node.sources:
        _build_layer(output_layers, source.get_key(), h5,
                     node_dict, layer_dict)

    # special cases for merge layers
    if node.layer_type in ['concatenate','merge']:
        node.n_outputs = 0
        for source in node.sources:
            node.n_outputs += source.n_outputs
        return
    else:
        assert len(node.sources) == 1, "in {}".format(node.layer_type)

    # if this layer is already defined, just add the node count and
    # continue
    if node.name in layer_dict:
        node.n_outputs = layer_dict[node.name]['n_outputs']
        node.layer_number = layer_dict[node.name]['pos']
        return

    layer_type = node.layer_type

    if KERAS_VERSION == 1:
        keras_layer_converters = "keras_v1_layer_converters"
    elif KERAS_VERSION == 2:
        keras_layer_converters = "keras_v2_layer_converters"
    else:
        sys.exit("We don't support Keras version {}.\n"
          "Pleas open an issue at https://github.com/lwtnn").format(KERAS_VERSION)

    _send_recieve_meta_info = getattr(importlib.import_module(keras_layer_converters),
      "_send_recieve_meta_info")
    layer_converters = getattr(importlib.import_module(keras_layer_converters),
      "layer_converters")

    _send_recieve_meta_info(BACKEND)
    convert = layer_converters[layer_type]


    # build the out layer
    n_inputs = sum(s.n_outputs for s in node.sources)
    layer_config = node.keras_layer['config']
    out_layer, node.n_outputs = convert(
        h5, layer_config, n_inputs, layer_type)
    layer_number = len(output_layers)
    layer_dict[node.name] = {
        'n_outputs': node.n_outputs,
        'pos': layer_number}
    node.layer_number = layer_number
    output_layers.append(out_layer)

_node_type_map = {
    'batchnormalization': 'feed_forward',
    'merge': 'concatenate',       # <- v1
    'concatenate': 'concatenate', # <- v2
    'inputlayer': 'input',
    'dense': 'feed_forward',
    'activation': 'feed_forward',
    'lstm': 'sequence',
    'gru': 'sequence',
    'timedistributed': 'time_distributed',
    'softmax': 'feed_forward',
}

def _build_node_list(node_dict, input_layer_arch):
    """
    no effort is made to sort this list in any way, but the ordering
    is important because each node contains indices for other nodes
    """
    node_list = []
    input_map = {}
    for kname, kid, ks in input_layer_arch:
        node = node_dict[(kname, kid)]
        submap = input_map.setdefault(node.dims, {})
        n_in = len(submap)
        submap[kname] = n_in

    for node in sorted(node_dict.values()):
        node_type = _node_type_map[node.layer_type]
        out_node = {'type': node_type}
        if node.sources:
            out_node['sources'] = [n.number for n in node.sources]

        if node_type == 'input':
            out_node['sources'] = [input_map[node.dims][node.name]]
            out_node['size'] = node.n_outputs
            if node.dims > 1:
                out_node['type'] = 'input_sequence'
        elif node_type in ['feed_forward', 'sequence', 'time_distributed']:
            out_node['layer_index'] = node.layer_number
        node_list.append(out_node)
    return node_list

def _get_layers_and_nodes(network, h5):
    node_dict = _build_node_dict(network)
    _number_nodes(node_dict)

    output_layers = []
    layer_meta = {}
    for node_key in node_dict:
        _build_layer(output_layers, node_key, h5, node_dict, layer_meta)

    return output_layers, node_dict

# TODO: rename this, and make it return a mapping to the nodes rather
# than to the number of variables in the nodes.
def _get_vars_per_input(input_layer_arch, node_dict):
    vars_per_input = {1: {}, 2: {}}
    for lname, lidx, something in input_layer_arch:
        assert lidx == 0 and something == 0
        node = node_dict[(lname, lidx)]
        nodenum = len(vars_per_input[node.dims])
        vars_per_input[node.dims][nodenum] = node.n_outputs
    return vars_per_input

def _parse_inputs(input_list, vars_per_input):
    nodes = []
    for input_number, node in enumerate(input_list):
        inputs = []
        node_name = node['name']
        for val in node['variables']:
            var_info = {x: val[x] for x in ['offset', 'scale', 'name']}
            default = val.get("default")
            if default is not None:
                var_info['default'] = default
            inputs.append(var_info)

        assert vars_per_input[input_number] == len(inputs)

        nodes.append({'name': node_name, 'variables': inputs})
    return nodes

def _parse_outputs(user_outputs, output_layers, node_dict):
    outputs = {}
    assert len(user_outputs) == len(output_layers)
    for num, (usr, ker) in enumerate(zip(user_outputs, output_layers)):
        kname, kid, ks = ker
        assert ks == 0
        node = node_dict[(kname, kid)]
        assert node.n_outputs == len(usr['labels'])
        assert usr['name'] not in outputs
        output = {
            'node_index': node.number,
            'labels': usr['labels']
        }
        outputs[usr['name']] = output
    return outputs

def _parse_miscellaneous(variables):
    if 'miscellaneous' in variables:
        misc_dict = {}
        for key, val in variables['miscellaneous'].items():
            misc_dict[str(key)] = str(val)
        out['miscellaneous'] = misc_dict
    return out

if __name__ == '__main__':
    _run()
