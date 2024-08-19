from node import Node
from network import Network


class FileNetwork(Network):
    """
    Creates a network from a network file, where each row consists of two columns, denoting two nodes that
    are connected by an edge.
    """
    def __init__(self, file_path: str, delimiter='\t', undirected=True, allow_self_edges=False):
        """
        Specification:
        - Skip rows with no, one, or more than 2 columns.
        - Do not produce errors when adding data to the network (e.g. adding an edge more than once).
        - It is fine if the network file is empty, the result is just an empty network.
        - If an unhandled error occurs while opening the file (e.g. the file does not exist), that is okay.

        :param file_path: path to the network file
        :param delimiter: symbol between columns in a row, default is a tab
        :param undirected: True if the network has undirected edges, False if the network is directed (optional)
        :param allow_self_edges: True if nodes are allowed to have edges to themselves, False otherwise (optional)
        """
        # initialise the Network class from which FileNetwork inherits its data structure and other functions
        Network.__init__(self, undirected=undirected, allow_self_edges=allow_self_edges)

        # open the file for reading and process it line by line
        with open(file_path, 'r') as file:
            for line in file:
                # convert the line content to columns using the delimiter, after removing the newline character and
                # other surrounding whitespace, which prevents empty identifiers and the like
                columns = line.strip().split(delimiter)         # type: list[str]

                # skip lines with the wrong number of columns
                if len(columns) != 2:
                    continue

                # create and add the nodes, if they do not exist yet
                for identifier in columns:
                    if identifier not in self.nodes:
                        self.add_node(Node(identifier))

                # get the two node identifiers
                identifier_1, identifier_2 = columns

                # handle self-edges
                if not allow_self_edges and identifier_1 == identifier_2:
                    continue

                # add the edge if it does not exist yet
                if not self.edge_exists(identifier_1, identifier_2):
                    self.add_edge(identifier_1, identifier_2)
