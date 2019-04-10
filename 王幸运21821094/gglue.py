import torch.nn as nn
import torch

import buffer
import collections
import sys
import traceback

def uniquefy(seq):
    unique = []
    [unique.append(item) for item in seq if item not in unique]
    return unique

def find_same_channel_index(channel_list, channel):
    for idx, channel_b in enumerate(channel_list):
        if channel == channel_b:
            return idx

    assert False
    return -1

def get_buffer(x):
    if isinstance(x, Gglue.SimpleChannel):
        return x.buffer
    return x

class Gglue:
    def cals_backwards(self, forward_relationship):
        backward_relationship={}

        for frm, to in forward_relationship.items():
            if to in backward_relationship:
                backward_relationship[to].add(frm)
            else:
                backward_relationship[to] = {frm}

        return backward_relationship

    class Channel:
        pass

    class NodeChannel(Channel):
        def __init__(self, node_name, channel_id):
            self.node_name = node_name
            self.channel_id = channel_id

        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

        def __ne__(self, other):
            return not self.__eq__(other)

    class SimpleChannel(Channel):
        def __init__(self, buffer, groundinput=False):
            self.buffer = buffer
            self.groundinput = groundinput
            self.position = None

        def __eq__(self, other):
            return isinstance(other, self.__class__) and self.groundinput == other.groundinput and self.buffer.is_same(other.buffer)

        def __ne__(self, other):
            return not self.__eq__(other)

        def __str__(self):
            if self.groundinput:
                source_value="ground"
            else:
                source_value="input"

            return "SC({}, '{}')".format(str(self.buffer), source_value)

        def is_hidden(self):
            return self.buffer.is_hidden()

    def calc_nodes(self):
        self.nodes = set()

        for node in self.networks:
            self.nodes.add(node)

    def __init__(self, networks, forward_relationship):
        self.networks = networks
        self.forward_relationship = forward_relationship
        self.groundinput = False

        self.calc_nodes()
        self.backward_relationship = self.cals_backwards(self.forward_relationship)


        self.active_nodes = None

    def finilize_connection(self, active_nodes, ):
        for name, network in self.networks.items():
            if name not in active_nodes:
                continue
            # we just do it for active nodes

            if name in self.backward_relationship:
                for idx_in, buffer_in in enumerate(network.buffers_in):
                    found = False
                    for in_network in self.backward_relationship[name]:

                        in_network_daptor = self.networks[in_network]
                        for ind_out, buffer_out in enumerate(in_network_daptor.buffers_out):
                            if buffer_in.is_same(buffer_out):
                                if found:
                                    raise Exception

                                found = True

                                if in_network in active_nodes:
                                    network.buffers_in_proxies[idx_in] = \
                                        Gglue.NodeChannel(in_network, ind_out)
                                else:
                                    if self.groundinput:
                                        network.buffers_in_proxies[idx_in] = Gglue.SimpleChannel(network.buffers_in[idx_in], True)




                                    # path dependant
    def generate_superNet(self, graph_nodes=None):
        class superModule(nn.Module):

            def __init__(self, gglue, graph_nodes):
                super(superModule, self).__init__()

                if graph_nodes is None:
                    sorted_list = gglue.sorted_list
                else:

                    #TODO, check if connected graph
                    sorted_list = list(filter(lambda x: x in graph_nodes, gglue.sorted_list))

                self.inputs_translation_table = [None] * len(sorted_list)
                self.networks_list = nn.ModuleList([gglue.networks[name].net for name in sorted_list])

                name_2_id_map = {name:sorted_list.index(name) for name in sorted_list}


                def get_translation_table(buffers_in_proxies):

                    tt = [None] * len(buffers_in_proxies)

                    for buffer_in_id, buffer_in_proxy in enumerate(buffers_in_proxies):
                        if isinstance(buffer_in_proxy, Gglue.NodeChannel):
                            if buffer_in_proxy.node_name in sorted_list:
                                tt[buffer_in_id] = (name_2_id_map[buffer_in_proxy.node_name], buffer_in_proxy.channel_id)


                        elif isinstance(buffer_in_proxy, Gglue.SimpleChannel):
                            real_input_id = find_same_channel_index(gglue.needed_buffers_in, buffer_in_proxy)
                            tt[buffer_in_id] = (-1, real_input_id)


                        else:
                            assert False
                            real_input_id = buffer.find_same_buffer_index(gglue.needed_buffers_in, buffer_in_proxy)
                            tt[buffer_in_id] = (-1, real_input_id)

                    return tt

                for node_id, node_name in enumerate(sorted_list):


                    node = gglue.networks[node_name]


                    self.inputs_translation_table[node_id] = get_translation_table(node.buffers_in_proxies)

                local_gt_layer_buffers_in = gglue.get_local_gt_needed_buffers(graph_nodes)
                self.gt_layer_tt = get_translation_table(local_gt_layer_buffers_in)

            def forward(self, real_input):
                outputs = [None] * len(self.networks_list)

                def get_input_info(translation_table):
                    inputs = [None] * len(translation_table)

                    for i in range(len(inputs)):
                        input_info = translation_table[i]

                        if input_info[0] < 0:
                            assert input_info[1] < real_input.size()[1]
                            inputs[i] = real_input[:, input_info[1]:input_info[1]+1, :, :]
                        else:
                            assert input_info[0] < len(outputs)
                            assert input_info[1] < outputs[input_info[0]].size()[1]
                            inputs[i] = outputs[input_info[0]][:, input_info[1]:input_info[1]+1, :, :]

                    return inputs


                for node_id in range(len(self.networks_list)):
                    input_array = get_input_info(self.inputs_translation_table[node_id])

                    node_input = torch.cat(input_array, dim=1)
                    net = self.networks_list[node_id]
                    outputs[node_id] = net(node_input)

                output = torch.cat(get_input_info(self.gt_layer_tt), dim=1)


                return output

        return superModule(self, graph_nodes)

    def caclulate_buffer_indices(self, targer_buffers, given_buffers):
        result = [None] * len(targer_buffers)

        def simplify(x):
            if isinstance(x, Gglue.SimpleChannel):
                return x.buffer
            else:
                return x
        # TODO check for duplicates
        given_buffers_simple = [simplify(x) for x in given_buffers]
        for i, t_buffer in enumerate(targer_buffers):
            result[i] = buffer.find_same_buffer_index2(given_buffers_simple, t_buffer)

        return result

    def caclulate_fixed_buffer_indices(self, targer_fixed_buffers, given_buffers):
        result = [None] * len(targer_fixed_buffers)
        for i, t_buffer in enumerate(targer_fixed_buffers):
            assert False
            # Todo get final value instead just permutation
            result[i] = buffer.find_same_buffer_index(given_buffers, t_buffer)

        return result

    def caclulate_input_buffer_indices(self, targer_buffers):
        return self.caclulate_buffer_indices(targer_buffers, self.needed_buffers_in)

    def caclulate_output_buffer_indices(self, targer_buffers):
        return self.caclulate_buffer_indices(targer_buffers, self.needed_buffers_out)

    def caclulate_fixed_output(self, targer_fixed_buffers):
        return self.caclulate_fixed_buffer_indices(targer_fixed_buffers, self.needed_buffers_out)

            #path dependent
    def calculate_needed_buffers(self, active_nodes):
        self.needed_buffers_in = []
        self.needed_buffers_out = []



        # all nodes need inputs
        for node in active_nodes:
            for buffer_f in self.networks[node].buffers_in_proxies:
                if not isinstance(buffer_f, Gglue.NodeChannel):
                    self.needed_buffers_in.append(buffer_f)

        self.needed_buffers_in = uniquefy(self.needed_buffers_in)

        for idx, buffer_in in enumerate( self.needed_buffers_in):
            buffer_in.position = idx

        #print("\n".join(str(x) for x in self.needed_buffers_in))

        self.gt_layer_buffers_in2 = []
        # only (current) roots have output
        for node in self.root_nodes:
            for idx, buffer in enumerate(self.networks[node].buffers_out):
                if not isinstance(buffer, Gglue.NodeChannel):
                    self.needed_buffers_out.append(buffer)
                    self.gt_layer_buffers_in2.append(Gglue.NodeChannel(node, idx))

        pass

    def filter_nodes(self, nodes, graph_nodes):
        if graph_nodes is not None:
            return list(filter(lambda x: x in graph_nodes, nodes))
        else:
            return list(nodes)

    def get_local_gt_needed_buffers(self, graph_nodes):
        local_root_nodes = self.filter_nodes(self.root_nodes, graph_nodes)
        if graph_nodes:
            assert len(local_root_nodes) == 1

        local_gt_layer_buffers_in = []

        for node in local_root_nodes:
            for idx, buf in enumerate(self.networks[node].buffers_out):
                if not isinstance(buf, Gglue.NodeChannel):
                    local_gt_layer_buffers_in.append(Gglue.NodeChannel(node, idx))

        return local_gt_layer_buffers_in

    def get_validate_nodes(self, nodes):
        bad_nodes = sorted([node for node in nodes if node not in self.networks])

        if len(bad_nodes) > 0:
            print("Following nodes are invalid: {}".format(",".join(bad_nodes)))


        return [node for node in nodes if node in self.networks]

    def get_all_nodes(self):
        return list(self.networks.keys())

    def set_active_nodes(self, nodes):
        self.active_nodes = self.get_validate_nodes(nodes)


        self.finilize_connection(self.active_nodes)
        self.topological_sort(self.active_nodes)
        self.calculate_needed_buffers(self.active_nodes)

    def get_subset_forward_relationship(self, active_nodes):
        return {k: v for (k, v) in self.forward_relationship.items() if k in active_nodes and v in active_nodes}

    def topological_sort(self, active_nodes):
        root_nodes = set()
        nonroot_nodes = set()

        sub_forward_relationship = self.get_subset_forward_relationship(active_nodes)
        sub_backward_relationship = self.cals_backwards(sub_forward_relationship)

        for node in active_nodes:
            if node not in sub_forward_relationship:
                root_nodes.add(node)

        for node in active_nodes:
            if node not in root_nodes:
                nonroot_nodes.add(node)

        touched = set()

        sorted_list = []

        def process_element(e):
            if e is not touched:
                touched.add(e)
                sorted_list.append(e)

                if e in sub_backward_relationship:
                    for ne in sub_backward_relationship[e]:
                        process_element(ne)

        for root in root_nodes:
            process_element(root)

        self.root_nodes = root_nodes

        sorted_list = list(reversed(sorted_list))
        self.sorted_list = sorted_list

