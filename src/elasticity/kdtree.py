import numpy as np
from ..kdtree.tree import KDTree as BaseKDTree


class KDTree(BaseKDTree):
    '''
    Implement `postprocess` for elasticity experiment.
    '''

    def postprocess(self):
        '''
        Split the subdomain with too much void
        '''
        # Check if a subdomain contain any void (empty area)
        def check_sparse(node):
            # Neglect small subdomains
            if (node.bbox[0][1] - node.bbox[0][0]) * 5 < self.overall_bbox[0][1] - self.overall_bbox[0][0] and \
            (node.bbox[1][1] - node.bbox[1][0]) * 5 < self.overall_bbox[1][1] - self.overall_bbox[1][0]:
                return False, None, None
            for i in range(self.dim):
                # Find the void in the subdomain
                n_blocks = 5
                block_points = [0] * n_blocks
                for point in node.points:
                    # Indexing the block
                    block_points[min(int((point[i] - node.bbox[i][0]) 
                        / (node.bbox[i][1] - node.bbox[i][0]) 
                        * n_blocks), n_blocks - 1)] += 1
                # Split in the void
                split_val = None
                for j in range(n_blocks):
                    if block_points[j] == 0:
                        split_val = node.bbox[i][0] + \
                    (node.bbox[i][1] - node.bbox[i][0]) / n_blocks * (j + 0.5)
                if split_val is not None:
                    return True, i, split_val
            return False, None, None

        def spread_points(node_idx, i, points, spread_dict):
            for point in points:
                indices = [j for j in range(len(self.nodes)) if j != node_idx]
                dists = [
                    (0. if self.nodes[idx].bbox[i][0] <= point[i] <= 
                        self.nodes[idx].bbox[i][1] else min(
                        abs(self.nodes[idx].bbox[i][0] - point[i]),
                        abs(self.nodes[idx].bbox[i][1] - point[i]) 
                    )) + (0. if self.nodes[idx].bbox[1-i][0] <= point[1-i] <= 
                    self.nodes[idx].bbox[1-i][1] else min(
                        abs(self.nodes[idx].bbox[1-i][0] - point[1-i]),
                        abs(self.nodes[idx].bbox[1-i][1] - point[1-i]) 
                    ))
                    for idx in indices
                ]
                nearest_idx_idx = np.argmin(dists)
                nearest_idx = indices[nearest_idx_idx]
                if nearest_idx not in spread_dict:
                    spread_dict[nearest_idx] = [point.tolist()]
                else:
                    spread_dict[nearest_idx].append(point.tolist())

        def split_node_by_index(node_idx, dim_splitted, split_val, spread_dict):
            node = self.nodes[node_idx]
            i = dim_splitted
            n_left = np.sum(node.points_np[:, i] <= split_val)
            n_right = np.sum(node.points_np[:, i] > split_val)
            preserve_part = node.points_np[node.points_np[:, i] <= split_val]
            eliminate_part = node.points_np[node.points_np[:, i] > split_val]
            if n_left < n_right:
                preserve_part, eliminate_part = eliminate_part, preserve_part
            # Spread `eliminate_part` to nearest subdomains
            spread_points(node_idx, i, eliminate_part, spread_dict)
            # Preserve `preserve_part` if there are enough points
            if len(preserve_part) >= self.smallest_points:
                self.nodes[node_idx].replace_points(preserve_part.tolist())
                return True
            else:
                spread_points(node_idx, i, preserve_part, spread_dict)
                return False

        change = True
        max_iter = 1000
        iter_cnt = 0
        while change:
            change = False
            # node_idx -> list of points to add (spreading)
            spread_dict = {}
            for i in range(len(self.nodes)):
                is_sparse, dim_splitted, split_val = \
                    check_sparse(self.nodes[i])
                if is_sparse:
                    change = True
                    break
            if is_sparse:
                is_preserve = split_node_by_index(i, dim_splitted, split_val, spread_dict)
                # apply spreading
                for k, v in spread_dict.items():
                    self.nodes[k].add_points(v)
                if not is_preserve:
                    self.nodes.pop(i)
            # Avoid dead loops
            iter_cnt += 1
            if iter_cnt >= max_iter:
                break
        
        # If the number of nodes are less than `n_subdomains`,
        # split new nodes
        self.nodes = self.split(self.nodes, self.n_subdomains, simple=False)