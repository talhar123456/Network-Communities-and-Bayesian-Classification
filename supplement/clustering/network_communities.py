from typing import Dict, List, Set, Tuple, Union
from network_file import FileNetwork
from node import Node
from network import Network
# TODO: whatever other imports you need


def triangles(network: Network, node_1: Node, node_2: Node) -> int:
    """
    This function computes the number of triangles to which the edge between node_1 and node_2 contributes. Correctly
    handle the case where the edge does not exist.

    :param network: the network containing node_1 and node_2
    :param node_1: first node in the edge
    :param node_2: second node in the edge
    :return: number of triangles to which the edge between node_1 and node_2 contributes, if the edge exists at all
    """
    triangles_count = 0
    
    # Get the neighbor nodes of node_1 and node_2
    neighbors_1 = network.get_node(node_1).neighbour_nodes
    neighbors_2 = network.get_node(node_2).neighbour_nodes
    
    # Calculate the number of triangles
    for neighbor in neighbors_1.intersection(neighbors_2):
        triangles_count += 1
    
    return triangles_count


def edge_clustering_coefficient(network: Network, node_1: Node, node_2: Node) -> float:
    """
    This function computes the edge-clustering coefficient for an edge between nodes node_1 and node_2.

    :param network: the Network-object containing node_1 and node_2
    :param node_1: first node in the edge
    :param node_2: second node in the edge
    :raise: ValueError (with a custom message) if there is no undirected edge between node_1 and node_2
    :return: edge clustering coefficient
    """
    # Calculate the minimum of degrees of node_1 and node_2
    min_degree = min(network.get_node(node_1).degree(), network.get_node(node_2).degree())
    
    # Check if one of the nodes has a degree of 1
    if min_degree == 1:
        return float('inf')
    
    # Calculate the number of triangles to which the edge contributes
    triangles_count = triangles(network, node_1, node_2)
    
    # Calculate the edge-clustering coefficient
    return (triangles_count + 1) / (min_degree - 1)


def get_sorted_edge_list(network: Network) -> List[Tuple[Node, Node]]:
    """
    Generates a sorted list of all unique edges in an undirected network. (This is for reproducibility.)

    Specification:
    - Each undirected edge A <=> B and B <=> A is only included once, and the version included is the one that comes
      first in lexicographical order, so A <=> B will be included in the edge list and B <=> A is not.
    - The list should not contain duplicates.
    - The list is sorted in lexicographical order.
    - You may assume that the network identifiers are either all numerical or all strings.

    Example: [(Ben, Ellen), (Ben, John), (Catelyn, Eddard), (Catelyn, Jennifer),...]

    :returns: lexicographically sorted list of edges
    :raise: ValueError (with a custom message) if the network is directed and/or allows self-edges
    """
    edge_list = set()
    
    for node_id, node in network.nodes.items():
        for neighbor_id in node.neighbour_nodes:
            # Avoid duplicates by sorting the node IDs
            edge_list.add(tuple(sorted((node_id, neighbor_id))))
    
    return sorted(edge_list)


def decompose(network: Network) -> List[Tuple[Node, Node, float]]:
    """
    This function decomposes an undirected network using the edge clustering coefficient. Make sure that you do not
    'destroy' the only copy of the original network. Or, if you 'destroy' it in this function, you need to 'restore' it
    to its original condition at the end of the function.

    Approach:
    1. Get a sorted list of all edges in the network.
    2. Iterate until all edges are deleted. In each iteration:
       i.   Calculate the current edge coefficient for each remaining edge.
       ii.  Find the edge with the smallest coefficient, store it in a list and then remove it from the network. If
            there are multiple edges with the smallest coefficient, choose the one that comes first in
            lexicographical order.

            Print the following:
            Step X: removed node_1 --> node_2 with ECC: coefficient
       iii. Repeat i. and ii. until all edges are deleted.

    :param network: a Network object for which to compute the decomposition
    :return: list of edges in order of deletion and their respective edge clustering coefficient at time of deletion,
             e.g. [(node_1, node_2, 0.3), (node_5, node_7, 0.5),...]
    :raise: ValueError if the network is directed and/or allows self-edges (automatically from get_sorted_edge_list)
    """
    deleted_edges = []

    # Get the sorted list of unique edges
    edge_list = get_sorted_edge_list(network)

    while edge_list:
        min_coefficient = float('inf')
        edge_to_delete = None

        for edge in edge_list:
            coefficient = edge_clustering_coefficient(network, edge[0], edge[1])
            if coefficient < min_coefficient:
                min_coefficient = coefficient
                edge_to_delete = edge

        # Check if edge_to_delete is None
        if edge_to_delete is None:
            break

        # Remove the edge from the network and add it to the deleted edges list
        network.remove_edge(edge_to_delete[0], edge_to_delete[1])
        deleted_edges.append((edge_to_delete[0], edge_to_delete[1], min_coefficient))

        # Remove the edge from the edge list
        edge_list.remove(edge_to_delete)

    return deleted_edges


def classify(community: Set[Node]) -> str:
    """
    Use the definitions by Radicchi given on the sheet to classify the community as strong or weak, or as not a
    community.

    :param community: a set of nodes making up the community
    :raises: ValueError (with a custom message) if the community is empty
    :return: the classification of the community as a string (one of: 'strong', 'weak' or 'none')
    """
    if not community:
        raise ValueError("Community is empty")

    total_internal_links = 0
    total_external_links = 0

    # Calculate internal and external links for each node in the community
    for node in community:
        internal_links = 0
        external_links = 0

        for neighbor in node.neighbour_nodes:
            if neighbor in community:
                internal_links += 1
            else:
                external_links += 1

        total_internal_links += internal_links
        total_external_links += external_links

    # Check the classification criteria
    if total_internal_links > total_external_links:
        return "strong"
    elif total_internal_links == total_external_links:
        return "weak"
    else:
        return "none"


def find_community(communities: List[Set[Node]], node: Node) -> Set[Node]:
    """
    Find the community the node belongs to and return it. If the node is not part of an existing community,
    establish a new community only containing the node.

    :param communities: list of already established communities (= sets of Nodes)
    :param node: a node in the network
    :return: community containing the node
    """
    for community in communities:
        for member in community:
            if member.identifier == node.identifier:
                return community

    # If the node is not part of any existing community, create a new one
    new_community = {node}
    communities.append(new_community)
    return new_community


def rebuild(edges: List[Tuple[Node, Node, float]]):
    """
    This function takes the output of 'decompose' and uses it to build communities.

    Approach:
    Iterate over the edges in reverse order (last deleted edge first). In each iteration:
    i.   If the edge is not connected to an existing community, create a new community.
    ii.  If the edge has a single node in common with an existing community, then add the other node to the
         community as well.
    iii. If both nodes of the edge are already part of the same community, there is nothing to do.
    iv.  If they are, however, part of two different communities, merge the two communities into one. In this case,
         print the current communities with their classification:
         Step X:
            none: [Hanna, Peter]
            strong: [Catelyn, Eddard, Jennifer, Kate]
            strong: [Ben, Ellen, John]

    :param edges: list of deleted edges in order of deletion (first deleted edge first)
    """
    communities = []

    for edge in reversed(edges):
        node_1, node_2, coefficient = edge

        community_1 = find_community(communities, node_1)
        
        # Ensure node_2 is a Node object
        if isinstance(node_2, str):
            node_2 = Node(node_2)
        
        community_2 = find_community(communities, node_2)

        if community_1 != community_2:
            # Merge the two communities
            merged_community = community_1.union(community_2)
            communities.remove(community_1)
            communities.remove(community_2)
            communities.append(merged_community)

            print("Step {}: Merged communities:".format(len(edges) - len(communities) + 1))
            for comm in communities:
                classification = classify(comm)
                print(f"{classification}: {comm}")

        # If both nodes are already part of the same community, do nothing

    # Print the final communities
    print("Final communities:")
    for comm in communities:
        classification = classify(comm)
        print(f"{classification}: {comm}")



def exercise_1():
    # Define the file path for the network.tsv file
    file_path = "network.tsv"

    # Create a FileNetwork object to read the network from the file
    file_network = FileNetwork(file_path)

    # Decompose the network
    deleted_edges = decompose(file_network)

    # Rebuild the network
    rebuild(deleted_edges)

    # Print out the steps of the decomposition
    for step, (node_1, node_2, coefficient) in enumerate(deleted_edges, start=1):
        print(f"Step {step}: Removed edge between {node_1} and {node_2} with ECC: {coefficient}")

if __name__ == "__main__":
        exercise_1()
