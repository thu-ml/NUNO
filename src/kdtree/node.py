import numpy as np


BIG_CONSTANT = 1e100


class KDTreeNode:
    '''
    A Tree Node used by `KDTree`.
    '''
    def __init__(self, depth, points, dim, n_blocks, smallest_points, max_depth, borders):
        self.depth = depth
        self.points = points
        self.points_np = np.array(points)
        self.dim = dim
        self.n_blocks = n_blocks
        self.smallest_points = smallest_points
        self.max_depth = max_depth
        self.borders = borders
        self.alive = True
        # Calculate the bounding box
        self.bbox = [(np.min(self.points_np[:, i]), 
            np.max(self.points_np[:, i])) for i in range(self.dim)]
        self.choose_split()
        self.ready = True   # if ready to be splitted
    
    def cal_kl_divergence(self, point_set, bbox):
        '''
        Calculate the KL divergence of the distribution of 
        a point set w.r.t to the uniform distribution.\n
        The shape of discrete blocks and calculation result are returned.
        '''
        # Specify the discrete blocks 
        # which is used to approximate the distribution of the point set:
        # shape: K1 x K2 x...x Kd \approx n_blocks
        # s.t. Ki / Kj \approx bbox_len_i / bbox_len_j for all i,j
        shape = [None] * self.dim
        shape[0] = self.n_blocks * np.prod([(bbox[0][1] - bbox[0][0]) / 
            (bbox[j][1] - bbox[j][0]) for j in range(1, self.dim)])
        shape[0] = shape[0] ** (1 / self.dim)
        for j in range(1, self.dim):
            shape[j] = (bbox[j][1] - bbox[j][0]) / \
                (bbox[0][1] - bbox[0][0]) * shape[0]
        shape = [max(int(np.round(l)), 1) for l in shape]
        # Count the discrete distribution of the point set
        block_proportions = np.zeros(shape)
        for point in point_set:
            # Indexing the block
            ind = []
            for j in range(self.dim):
                ind.append(min(int((point[j] - bbox[j][0]) 
                    / (bbox[j][1] - bbox[j][0]) * shape[j]), shape[j] - 1))
            ind = tuple(ind)
            block_proportions[ind] += 1
        block_proportions = block_proportions / len(point_set)
        # Calculate the KL divergence:
        # KL(counted distribution || uniform distribution)
        kl_result = block_proportions * \
            np.log(block_proportions * np.prod(shape), 
                out=np.zeros_like(block_proportions), where=(block_proportions!=0)) 
        kl_result = np.sum(kl_result)
        return shape, kl_result

    def try_split(self, i):
        '''
        Try to split in the i-th dimension,
        and return the best possible KL.
        '''
        cond_kls = []
        split_points = []
        ok = False
        # Enumerate the split point
        i_shape = self.K_shape[i]
        for j in range(i_shape-1):
            # Find split point
            # Note: avoid rounding errors
            split_val = (j+1) / i_shape * (self.bbox[i][1] - self.bbox[i][0]) \
                + self.bbox[i][0]
            split_point = min(self.points_np[:, i].searchsorted(
                split_val + 1e-8) + 1, len(self.points))
            split_points.append(split_point)
            # KL of left part
            left_points = self.points_np[:split_point, :]
            # Check if the number of points is too small
            if split_point < max(self.smallest_points, self.n_blocks):
                cond_kls.append(BIG_CONSTANT)
                continue
            _, left_kl = self.cal_kl_divergence(
                left_points,
                [(np.min(left_points[:, k]), 
                    np.max(left_points[:, k])) for k in range(self.dim)]
            )
            left_tot_proportion = split_point / len(self.points)
            # KL of right part
            right_points = self.points_np[split_point:, :]
            # Check if the number of points is too small
            if len(self.points) - split_point < max(self.smallest_points, self.n_blocks):
                cond_kls.append(BIG_CONSTANT)
                continue
            _, right_kl = self.cal_kl_divergence(
                right_points,
                [(np.min(right_points[:, k]), 
                    np.max(right_points[:, k])) for k in range(self.dim)]
            )
            right_tot_proportion = (len(self.points) - split_point) / len(self.points)
            # Conditional KL
            cond_kl = left_tot_proportion * left_kl + right_tot_proportion * right_kl
            cond_kls.append(cond_kl)
            ok = True

        # Return values on the split
        return ok, split_points[np.argmin(cond_kls)]

    def choose_split(self):
        # Corner cases
        if (self.max_depth is not None and self.depth >= self.max_depth) or \
            len(self.points) < 2 * max(self.smallest_points, self.n_blocks):
            self.set_dead()
            return
        # Split the dimension with the biggest scale
        self.dim_chosen = np.argmax([e[1] - e[0] for e in self.bbox])
        self.points.sort(key=lambda x: x[self.dim_chosen])
        self.points_np = np.array(self.points)
        # Calculate the overall KL divergence
        self.K_shape, self.overall_kl = self.cal_kl_divergence(
            self.points, self.bbox)
        # Choose the split point which maximize
        # the difference of KL divergence after splitting (i.e., KL gain)
        ok, self.split_chosen = self.try_split(self.dim_chosen)
        if not ok:
            self.set_dead()
    
    def set_dead(self):
        '''
        Set current node inactive (dead)
        '''
        self.overall_kl = None
        self.alive = False

    def split(self):
        '''
        Split according to the best KL gain
        '''
        borders_l, borders_r = self.borders[:], self.borders[:]
        split_val = self.points[self.split_chosen][self.dim_chosen]
        borders_l[self.dim_chosen] = (
            borders_l[self.dim_chosen][0], split_val)
        borders_r[self.dim_chosen] = (
            split_val, borders_r[self.dim_chosen][1]
        )
        return KDTreeNode(self.depth + 1, self.points[:self.split_chosen], 
                self.dim, self.n_blocks, self.smallest_points, 
                self.max_depth, borders_l), \
            KDTreeNode(self.depth + 1, self.points[self.split_chosen:], 
                self.dim, self.n_blocks, self.smallest_points, 
                self.max_depth, borders_r)
    
    def simple_split(self):
        '''
        Split according to the median
        '''
        m = len(self.points) >> 1
        borders_l, borders_r = self.borders[:], self.borders[:]
        split_val = self.points[m][self.dim_chosen]
        borders_l[self.dim_chosen] = (
            borders_l[self.dim_chosen][0], split_val)
        borders_r[self.dim_chosen] = (
            split_val, borders_r[self.dim_chosen][1]
        )
        return KDTreeNode(self.depth + 1, self.points[:m], 
                self.dim, self.n_blocks, self.smallest_points, 
                self.max_depth, borders_l), \
            KDTreeNode(self.depth + 1, self.points[m:], 
                self.dim, self.n_blocks, self.smallest_points, 
                self.max_depth, borders_r)

    def get_bounding_box(self):
        return self.bbox

    def get_borders(self):
        return self.borders

    def add_points(self, new_points):
        '''
        The `new_points` should be an instance of `list`,
        instead of `numpy.ndarray`.
        WARNING: this function will not re-calculate the 
        `self.borders`, so it may be incorrect after calling
        this function.
        '''
        self.points.extend(new_points)
        self.points_np = np.array(self.points)
        # Re-calculate the bounding box
        self.bbox = [(np.min(self.points_np[:, i]), 
            np.max(self.points_np[:, i])) for i in range(self.dim)]
        # To be ready, we need to re-call `choose_split`
        self.ready = False
    
    def replace_points(self, new_points):
        '''
        The `new_points` should be an instance of `list`,
        instead of `numpy.ndarray`.
        WARNING: this function will not re-calculate the 
        `self.borders`, so it may be incorrect after calling
        this function.
        '''
        self.points = new_points
        self.points_np = np.array(self.points)
        # Re-calculate the bounding box
        self.bbox = [(np.min(self.points_np[:, i]), 
            np.max(self.points_np[:, i])) for i in range(self.dim)]
        # To be ready, we need to re-call `choose_split`
        self.ready = False
