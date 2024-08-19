from typing import Set, Union


class Node:
    def __init__(self, identifier: Union[str, int], name: str = None):
        """
        Creates a Node-object with the given identifier.

        :param identifier: node ID
        """
        self.identifier = identifier  # type: Union[str, int]
        self.name = name  # type: str
        self.neighbour_nodes = set()  # type: Set[Node]                                              # type: Set[Node]

    def __str__(self) -> str:
        """
        This overwrites the basic __str__ function that each class has. It is called everytime a class is converted
        implicitly or explicitly to a string, e.g. when using str(node), Python internally calls node.__str__().

        :return: string representation of the node
        """
        return self.name if self.name else str(self.identifier)
    
    def __eq__(self, obj) -> bool:
        """
        This overwrites the basic __eq__ function that every class has. It is called by Python everytime the object
        is checked for equality with another object, e.g. when using node_1 == node_2 anywhere in the code.

        :param obj: some other object that is checked for equality with this node
        :return: True if the other object is a Node and has the same identifier, False otherwise
        """
        return isinstance(obj, Node) and self.identifier == obj.identifier

    def __lt__(self, obj) -> bool:
        """
        This overwrites the basic __lt__ function that every class has. It is called by Python everytime the object
        is checked for 'less than' with another object, e.g. when using node_1 < node_2 or when sorting a list of
        nodes. Together with __eq__ this is all that is needed to sort.

        :param obj: some other object that is checked for 'less than' with this node
        :return: True if the other object is not a Node, or if the other object's identifier is less than this one's,
                 False otherwise
        """
        # sort Nodes before other objects
        if not (isinstance(obj, Node)):
            return True
        # when the other object is also a Node, sort by identifier with numerical identifiers before string identifiers
        return (isinstance(self.identifier, str), self.identifier) < (isinstance(obj.identifier, str), obj.identifier)

    def __hash__(self) -> int:
        """
        Computes a hash value for this Node-object from its identifier. Together with the __eq__ function, this makes it
        possible for Node-objects to be stored in sets and other data structures that require objects to be hashable.

        :return: hash value of this Node-object
        """
        return hash(self.identifier)

    def has_edge_to(self, node) -> bool:
        """
        :param node: Node-object
        :return: True if this node has an edge to the other node, False otherwise
        """
        return node in self.neighbour_nodes

    def add_edge(self, node):
        """
        Adds an edge to the other node by adding it to the neighbour-nodes.

        :param node: Node-object
        :raises: ValueError (with custom message) if the edge already exists
        """
        if self.has_edge_to(node):
            raise ValueError(f'Add edge error: The edge from Node {self} to Node {node} already exists.')
        self.neighbour_nodes.add(node)

    def remove_edge(self, node):
        """
        Removes the edge to the other node, if that edge exists, by removing the other node from the neighbour nodes.

        :param node: Node-object
        :raises: ValueError (with custom message) if the edge does not exist
        """
        if not self.has_edge_to(node):
            # raise ValueError(f'Remove edge error: The edge from Node {self} to Node {node} does not exist.')
            return
        self.neighbour_nodes.discard(node)

    def degree(self) -> int:
        """
        :return: the degree of this node (= number of neighbouring nodes)
        """
        return len(self.neighbour_nodes)
