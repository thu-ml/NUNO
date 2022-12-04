import numpy as np
from .node import KDTreeNode


class KDTree:
    """
    A KD-Tree for splitting given scattered data (point set)
    to several more uniformly distributed subdomains.

    Usage
    ----------
    1. Construct a KD-Tree:
        `kd_tree = KDTree(points, dim, ...)`.
        Then the tree will immediately start to divide... 
    2. You can then use `get_subdomain_points` for resulting points 
        in each subdomain or `get_subdomain_bounding_boxes` for 
        the bounding box of each subdomain.
    """
    def __init__(self, points, dim, n_subdomains=32, n_blocks=8,
        smallest_points=8, max_depth=None, group=1, return_indices=False):
        """
        Build the KD-Tree for subdomain dividing.

        Parameters
        ----------
        points : list<point>
            A `list` of points, e.g., `[[1, 2], [1, 2.5], ..]`.
            We should note that `numpy.ndarray` is NOT allowed.
        dim : int 
            The dimensionality of the points, e.g., `2`. 
        n_subdomains : int, optional
            The number of subdomains generated.
            The default is `32`.
        n_blocks : int, optional
            The number of blocks to approximate the 
            distribution of point set in the subdomain, 
            which is expected to be greater than `2 ** dim`.
            The default is `8`.
        smallest_points: int, optional
            The least points contained in each subdomain, 
            which is expected to be greater than `n_blocks`.
            The default is `8`.
        max_depth: int, optional
            The maximum number of splits allowed for a subdomain, 
            which is strictly expected to be greater than `log2(n_subdomains)`.
            The default is `None` which means no restriction.
        group: int, optional
            The number of groups where subdomains are splitted 
            separately. The groups are first obtained by 
            splitting original subdomains iteratively according to 
            the median like conventional KD-Trees.
            The default is `1` which means no grouping.
        return_indices: bool, optional
            Whether return the indices of the points in each subdomain
            w.r.t. the original list `points`. The indices can be obtained
            by calling `get_subdomain_indices`.
            The default is `False` which means no indices are returned.
        """
        self.return_indices = return_indices
        self.dim = dim
        if return_indices:
            points = [list(points[i]) + [i] for i in range(len(points))]
        self.nodes = [KDTreeNode(0, points, dim, n_blocks, smallest_points, max_depth)]
        self.overall_bbox = self.nodes[0].bbox
        # Preprocessing
        self.preprocess()
        # To generate groups
        if group > 1:
            self.nodes = self.split(self.nodes, group, simple=True)
        tot_nodes = []
        # Visit each group separately
        for node in self.nodes:
            tot_nodes = tot_nodes + self.split([node], 
                n_subdomains // max(group, 1), simple=False)
        self.nodes = tot_nodes
        # Postprocessing
        self.postprocess()

    def split(self, nodes, n_subdomains, simple=False):
        while len(nodes) < n_subdomains:
            ind = list(range(len(nodes)))
            # Filter the dead and off-ready node
            ind = [i for i in ind if nodes[i].alive and nodes[i].ready]
            if len(ind) == 0:
                print("Warning: early termination because all the nodes are not alive. Perhaps the parameters are inconsistent.")
                break
            # Simple split on the median
            if simple:
                node_utilities = [len(nodes[i].points) for i in ind] 
                node_chosen = np.argmax(node_utilities)
                # Splitting
                son_a, son_b = nodes[ind[node_chosen]].simple_split()
            # Normal split via KL divergence
            else:
                node_utilities = [nodes[i].overall_kl *
                    len(nodes[i].points) for i in ind] 
                node_chosen = np.argmax(node_utilities)
                # Splitting
                son_a, son_b = nodes[ind[node_chosen]].split()
            nodes[ind[node_chosen]] = son_a
            nodes.append(son_b)
        return nodes

    def preprocess(self):
        '''
        Implement this function as what you need.
        '''
        pass
    
    def postprocess(self):
        '''
        Implement this function as what you need.
        '''
        pass
    
    def get_subdomain_points(self):
        if self.return_indices:
            return [node.points_np[:, :-1] for node in self.nodes]
        else:
            return [node.points_np for node in self.nodes]
    
    def get_subdomain_bounding_boxes(self):
        return [node.get_bounding_box() for node in self.nodes]
    
    def get_subdomain_indices(self):
        '''
        If `return_indices == False`, return `None`.
        '''
        if self.return_indices:
            return [node.points_np[:, -1].astype(int) 
                for node in self.nodes]
        else:
            return None
