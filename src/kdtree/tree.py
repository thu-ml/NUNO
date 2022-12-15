import numpy as np
from .node import KDTreeNode


class KDTree:
    """
    A KD-Tree for splitting given scattered data (point set)
    to several more uniformly distributed subsets (and subdomains).

    Usage
    ----------
    1. Construct a KD-Tree:
        `kd_tree = KDTree(points, dim, ...)`.
        Then immediately call `solve()` to start splitting... 
    2. You can call `get_subdomain_points()` for resulting points in each subdomain, 
        `get_subdomain_indices()` for the indices of the points in each subdomain,
        `get_subdomain_borders()` for the boundaries of
        each subdomain, or `get_subdomain_bounding_boxes()` for 
        the bounding box of the points in each subdomain.
    """
    def __init__(self, points, dim, n_subdomains=32, n_blocks=8,
        smallest_points=8, max_depth=None, group=1, 
        return_indices=False, overall_borders=None):
        """
        Build the KD-Tree for subdomain splitting.

        Parameters
        ----------
        points : list<point>
            A `list` of points, e.g., `[[1, 2, 0.25], [1, 2.5, 3], ...]` (3D).
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
        overall_borders: list<tuple>, optional
            The boundaries of the overall domain. If not specified, 
            the default behavior is to use the bounding box of `points`.
            E.g.: `[(0, 1), (0.5, 0.75), (0.1, 0.9)]` (3D).
        """
        self.return_indices = return_indices
        self.dim = dim
        self.smallest_points = smallest_points
        self.n_subdomains = n_subdomains
        self.group = group
        if overall_borders is None:
            points_np = np.array(points)
            overall_borders = [(np.min(points_np[:, i]), 
            np.max(points_np[:, i])) for i in range(self.dim)]
        if return_indices:
            points = [list(points[i]) + [i] for i in range(len(points))]
        self.nodes = [KDTreeNode(0, points, dim, n_blocks, 
            smallest_points, max_depth, overall_borders)]
        self.overall_bbox = self.nodes[0].bbox

    def solve(self, prepro=False, postpro=False):
        '''
        KD-Tree iteratively splitting until the number of 
        subdomains exceeds `n_subdomains` or all the subdomains
        are not separable (dead, e.g, some exceeds `max_depth`).

        Parameters
        ----------
        prepro : bool, optional
            whether to call the customized preprocessing function.
            The default is `False`.
        postpro : bool, optional
            whether to call the customized postprocessing function.
            The default is `False`.
        '''
        # Customized preprocessing function
        if prepro:
            self.preprocess()
        # To generate groups
        if self.group > 1:
            self.nodes = self.split(self.nodes, self.group, simple=True)
        tot_nodes = []
        # Visit each group separately
        for node in self.nodes:
            tot_nodes = tot_nodes + self.split([node], 
                self.n_subdomains // max(self.group, 1), simple=False)
        self.nodes = tot_nodes
        # Customized postprocessing function
        if postpro:
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
        '''
        Remember to call `solve()` before calling this function.
        '''
        if self.return_indices:
            return [node.points_np[:, :-1] for node in self.nodes]
        else:
            return [node.points_np for node in self.nodes]
    
    def get_subdomain_bounding_boxes(self):
        '''
        Format: [[(np.min(nodes.points_dim_i), 
            np.max(nodes.points_dim_i)) for i in range(self.dim)]
            for node in self.nodes]
        Remember to call `solve()` before calling this function.
        '''
        return [node.get_bounding_box() for node in self.nodes]
    
    def get_subdomain_borders(self, shrink=False, 
        shrink_tol=0.25, shrink_proportion=0.25):
        '''
        Format: [[(nodes.border_min_dim_i, 
            nodes.border_max_dim_i) for i in range(self.dim)]
            for node in self.nodes]
        Remember to call `solve()` before calling this function.

        Parameters
        ----------
        shrink : bool, optional
            Whether to shrink the borders so that they are more compact
            with the points in the subdomain.
            The default is `False`.
        shrink_tol : float, optional
            Shrinking happens when the ration of the scale of (one-side) empty void
            to the scale of the bounding box exceeds `shrink_tol` 
            (handing multiple dimensions one by one).
            The default is `0.25`.
        shrink_proportion : float, optional
            The empty void will shrink to `shrink_proportion` times 
            its original length.
            The default is `0.25`.
        '''
        borders = [node.get_borders() for node in self.nodes]
        if not shrink:
            return borders
        bboxes = [node.get_bounding_box() for node in self.nodes]
        for j in range(len(borders)):
            border = borders[j]
            bbox = bboxes[j]
            for i in range(self.dim):
                new_l, new_r = border[i]
                if bbox[i][0] - border[i][0] > (bbox[i][1] - bbox[i][0]) * shrink_tol:
                    new_l = bbox[i][0] - (bbox[i][0] - border[i][0]) * shrink_proportion
                if border[i][1] - bbox[i][1] > (bbox[i][1] - bbox[i][0]) * shrink_tol:
                    new_r = bbox[i][1] + (border[i][1] - bbox[i][1]) * shrink_proportion
                border[i] = (new_l, new_r)
        return borders

    def get_subdomain_indices(self):
        '''
        Remember to call `solve()` before calling this function.
        Reminder: if `return_indices == False`, it will return `None`.
        '''
        if self.return_indices:
            return [node.points_np[:, -1].astype(int) 
                for node in self.nodes]
        else:
            return None
    
    def sort_nodes_by_n_points(self):
        '''
        Sort the subdomains by the number of their points (ascending order).
        '''
        self.nodes.sort(key=lambda node: len(node.points))
