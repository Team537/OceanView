# map_management/kdtree_manager.py
from scipy.spatial import KDTree

class KDTreeManager:
    """
    Builds and manages the KD tree for efficient spatial searches.
    """

    def __init__(self, branch_manager):
        self.branch_manager = branch_manager
        self.kdtree = None
        self._build_kdtree()

    def _build_kdtree(self):
        base_poles = self.branch_manager.get_base_poles()
        if not base_poles:
            raise ValueError("No base poles available to build KDTree.")
        self.kdtree = KDTree(base_poles)

    def build_kdtree(self):
        """Public method to rebuild the KDTree, useful after alliance changes."""
        self._build_kdtree()

    def search_branches(self, query_point, k=4, valid_levels=['L2', 'L3', 'L4']):
        """
        Search for the k closest base poles to the query_point and retrieve their levels.

        Args:
            query_point (tuple): (x, z) coordinates.
            k (int): Number of closest base poles to find.
            valid_levels (list): Levels to retrieve.

        Returns:
            dict: Branch names mapped to their level positions.
        """
        if not self.kdtree:
            raise ValueError("KDTree has not been built.")

        distances, indices = self.kdtree.query(query_point, k=k)

        # Ensure indices is iterable
        if k == 1:
            indices = [indices]

        results = {}
        branches = self.branch_manager.get_branches()
        for idx in indices:
            branch = branches[idx]
            levels = branch.get_all_levels()
            filtered_levels = {level: pos for level, pos in levels.items() if level in valid_levels}
            results[branch.name] = filtered_levels

        return results
