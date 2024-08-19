from node import Node
from typing import Dict, List, Union


class Network:
    """
    Base network class that provides basic network functionality for directed and undirected networks. We will later
    create other network classes that inherit the basic functionality from this class.
    """
    def __init__(self, undirected=True, allow_self_edges=False):
        """
        Creates an empty network. There is nothing to do here for you.

        :param undirected: True if the network has undirected edges, False if the network is directed (optional)
        :param allow_self_edges: True if nodes are allowed to have edges to themselves, False otherwise (optional)
        """
        # dictionary containing the Nodes of the network
        # key = node ID, value = Node-object
        self.nodes = dict()                                                     # type: Dict[Union[int, str], Node]
        # flags for the edges
        self.undirected = undirected                                            # type: bool
        self.allow_self_edges = allow_self_edges                                # type: bool

    def print(self):
        """
        Prints the network as a sorted adjacency list, taking into account whether the network is directed or not.
        For example:
          1 <--> '2'
        '0' <--> no edges
        '1' <--> no edges
        '2' <--> 1
        """
        # network information consisting of the directness and whether the network allows self-edges
        network_type = ', '.join(['undirected' if self.undirected else 'directed',
                                  ('no ' if not self.allow_self_edges else '') + 'self-edges allowed'])     # type: str
        # represent whether the node has directed or undirected edges depending on the network type
        edge_type = '<-->' if self.undirected else '-->'                                                    # type: str
        # the length of the longest node ID plus same additional spacing
        max_length = max([len(str(node)) for node in self.nodes.values()], default=0) + 2                   # type: int

        # print the basic network configuration, and optionally whether the network is empty
        print(f'Network ({network_type}):', 'empty' if not self.nodes else '')
        # print the nodes in order of their identifier (implicitly making use of node.__lt__)
        for node in sorted(self.nodes.values()):
            # get a sorted list of the connected nodes and format them as a comma-separated list
            neighbours = ', '.join(str(neighbour) for neighbour in sorted(node.neighbour_nodes))            # type: str
            # format the right-aligned node label, using join helps to maintain the visual distinction between 1 and '1'
            label = ''.join([' ' * (max_length - len(str(node))), str(node)])                               # type: str
            # print the node information
            print(label, edge_type, neighbours if neighbours else 'no edges')

    def size(self) -> int:
        """
        :return: number of nodes in the network
        """
        return len(self.nodes)

    def number_edges(self) -> int:
        """
        Computes the current number of edges in the network. In undirected networks, the edges A <=> B and B <=> A only
        count as one edge, not two. In directed networks, the two edges are counted separately. A <=> A (if self-edges
        are allowed) counts as a single edge in both directed and undirected networks.

        :return: the number of edges in the network
        """
        edges = 0               # type: float

        # iterate over all edges
        for node_1 in self.nodes.values():
            for node_2 in node_1.neighbour_nodes:
                # in a directed network, each edge counts individually (as do self-edges)
                # in an undirected network, A <=> B and B <=> A only counts as one edge, hence adding 0.5 instead of 1
                edges += 0.5 if self.undirected and node_1 != node_2 else 1

        return int(edges)

    def add_node(self, node: Node):
        """
        Adds a node to the network.

        :param node: Node-object
        :raises: ValueError (with custom message) if there is already a node with the same identifier in the network.
                 (This avoids accidentally overwriting an existing nodes and its potentially already existing edges.)
        """
        # make sure the node does not yet exist in the network
        if node.identifier in self.nodes:
            raise ValueError(f'Network add node: TThe network already has node with identifier: {node.identifier}')

        # add the node to the network
        self.nodes[node.identifier] = node

    def get_node(self, node: Union[int, str, Node]) -> Node:
        """
        Retrieves the node from the network. If a Node-object instead of a node identifier is provided, return the Node
        object existing in the network with the same identifier rather than the Node-object that was provided as input.

        :param node: node ID or Node-object
        :return: Node object in the network with the matching identifier
        :raise: KeyError (with custom message) if the node (if provided as an object) does not exist in the network or
                if there is no node with that ID (if an ID was provided) in the network
        """
        # get the node identifier depending on the type of the input
        identifier = node.identifier if isinstance(node, Node) else node        # type: Union[int, str]

        # case: the node does not exist
        if identifier not in self.nodes:
            raise KeyError(f'Network get node: There is no matching node ({identifier}) in the network.')

        # return the node object stored in the network to ensure edge consistency
        return self.nodes[identifier]

    def edge_exists(self, node_1: Union[str, int, Node], node_2: Union[str, int, Node]) -> bool:
        """
        Checks if an edge exists between the two nodes. The directedness of the edge depends on the network type. In a
        directed network, node_1 is the source and node_2 the target: node_1 --> node_2. In an undirected network, both
        node_1 --> node_2 and node_2 --> node_1 need to exist.

        :param node_1: identifier or Node-object of the first node in the edge
        :param node_2: identifier or Node-object of the second node in the edge
        :return: True if the edge exists between the two nodes, False otherwise
        :raises: KeyError if either node does not exist in the network (automatically from get_node)
        """
        # retrieve the Node objects from the network, if they exist
        node_1 = self.get_node(node_1)                              # type: Node
        node_2 = self.get_node(node_2)                              # type: Node

        # case undirected network: node_1 --> node_2 and node_2 --> node_1 need to exist
        if self.undirected:
            return node_1.has_edge_to(node_2) and node_2.has_edge_to(node_1)

        # case directed network: only node_1 --> node_2 has to exist
        return node_1.has_edge_to(node_2)

    def add_edge(self, node_1: Union[str, int, Node], node_2: Union[str, int, Node]):
        """
        Adds an edge between two nodes, the directedness of the edge depends on the network type. In a directed network,
        node_1 is the source and node_2 the target: node_1 --> node_2. In an undirected network, both
        node_1 --> node_2 and node_2 --> node_1 need to be created.

        :param node_1: identifier or Node-object of the first node in the edge
        :param node_2: identifier or Node-object of the second node in the edge
        :raises: KeyError if either node does not exist in the network (automatically from get_node)
        :raises: ValueError if the edge already exists (automatically from add_edge in the Node class)
        :raises: ValueError (with custom message) if it is a self-edge and the network does not allow self-edges
        """
        # retrieve the Node objects from the network, if they exist
        node_1 = self.get_node(node_1)                              # type: Node
        node_2 = self.get_node(node_2)                              # type: Node

        # check for self-edges
        if node_1 == node_2 and not self.allow_self_edges:
            raise ValueError('Network add edge: the network should not contain self-edges')

        # always (try to) add node_1 --> node_2
        node_1.add_edge(node_2)

        # only (try to) add node_2 --> node_1 if the network is undirected (and it's not a self-edge)
        if self.undirected and node_1 != node_2:
            node_2.add_edge(node_1)

    def remove_edge(self, node_1: Union[str, int, Node], node_2: Union[str, int, Node]):
        """
        Removes an edge between two nodes, the directedness of the edge removal depends on the network type. In a
        directed network, node_1 is the source and node_2 the target: node_1 --> node_2. In an undirected network, both
        node_1 --> node_2 and node_2 --> node_1 need to be removed.

        :param node_1: identifier or Node-object of the first node in the edge
        :param node_2: identifier or Node-object of the second node in the edge
        :raises: KeyError if either node does not exist in the network (automatically from get_node)
        :raises: ValueError if the edge does not exist (automatically from remove_edge in the Node-class)
        """
        # retrieve the Node objects from the network, if they exist
        node_1 = self.get_node(node_1)                              # type: Node
        node_2 = self.get_node(node_2)                              # type: Node

        # always (try to) remove node_1 --> node_2
        node_1.remove_edge(node_2)

        # only (try to) remove node_2 --> node_1 if the network is undirected, and it's not a self-edge
        if self.undirected and node_1 != node_2:
            node_2.remove_edge(node_1)
        
        return

    def max_degree(self) -> int:
        """
        :return: highest node degree in the network
        """
        return max([node.degree() for node in self.nodes.values()], default=0)

    def degree_histogram(self) -> List[int]:
        """
        Computes the current degree distribution of the network as a histogram. Make sure that both degree 0 and the
        maximum degree are included!

        :return: the histogram as a list of integers
        """
        # initialize the histogram with degree counts of 0, making sure that both degree 0 and max degree are included
        histogram = [0] * (self.max_degree() + 1)           # type: list[int]

        # count the occurrence of each node degree
        for node in self.nodes.values():
            histogram[node.degree()] += 1

        return histogram

    def normalized_degree_distribution(self) -> List[float]:
        """
        Computes the current normalized degree distribution of the network as a histogram.

        :return: the normalized histogram as a list of decimal numbers
        """
        # edge case: empty network
        if not self.size():
            return [1.0]

        return [degree_count / self.size() for degree_count in self.degree_histogram()]

    def export(self, file_path: str, delimiter='\t'):
        """
        Writes edges of the network into the specified file, which overwrites the current content, if any. The format
        consists of two columns, where each row represents an edge between the node ID in the first column and the
        node ID in the second column. E.g.:

        node1   node2
        node1   node3

        In undirected networks, include both A <=> B and B <=> A. You do not need to handle potential file opening
        errors. Make sure your export function is consistent with the FileNetwork class.

        :param file_path: path to the output network file
        :param delimiter: the delimiter that separates the two columns, default is a tab
        """
        with open(file_path, 'w') as file:
            for node_1 in sorted(self.nodes.values()):
                for node_2 in sorted(node_1.neighbour_nodes):
                    file.write(f'{node_1.identifier}{delimiter}{node_2.identifier}\n')
