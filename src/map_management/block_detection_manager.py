from scipy.spatial import KDTree
import numpy as np

class BlockDetectionManager:
    """
    Detects which scoring locations are blocked by algae vs. coral.
    Returns only unblocked or algae-blocked. Excludes coral-blocked entirely.
    """
    def __init__(self, branch_manager, algae_block_threshold=0.5, coral_block_threshold=0.5):
        """
        Params: 
            branch_manager: An instance of your BranchManager (for scoring positions).
            algae_block_threshold: Distance threshold under which a location is blocked by algae, in meters.
            coral_block_threshold: Distance threshold under which a location is blocked by coral, in meters.
        """
        self.branch_manager = branch_manager
        self.algae_block_threshold = algae_block_threshold
        self.coral_block_threshold = coral_block_threshold

        # Internal KDTree references
        self.algae_kdtree = None
        self.coral_kdtree = None

    def build_obstacle_kdtrees(self, algae_positions, coral_positions):
        """
        Build two separate KDTrees for algae and coral.

        Params:
            algae_positions: List of dicts [{'x':..., 'y':..., 'z':...}, ...]
            coral_positions: List of dicts [{'x':..., 'y':..., 'z':...}, ...]
        """

        # --------- ALGAE TREE --------- #
        if not algae_positions:
            self.algae_kdtree = None
        else:
            algae_pts = np.array([[p['x'], p['y'], p['z']] for p in algae_positions])
            self.algae_kdtree = KDTree(algae_pts)

        # --------- CORAL TREE --------- #
        if not coral_positions:
            self.coral_kdtree = None
        else:
            coral_pts = np.array([[p['x'], p['y'], p['z']] for p in coral_positions])
            self.coral_kdtree = KDTree(coral_pts)

    def get_block_status(self, x, y, z):
        """
        Check a single (x,y,z) scoring location:
          1) If within coral threshold => return 'coral'
          2) Else if within algae threshold => return 'algae'
          3) Else => return 'available'
        """

        # ------ Check CORAL first ------ #
        if self.coral_kdtree is not None:
            coral_dist, _ = self.coral_kdtree.query([x, y, z])
            if coral_dist < self.coral_block_threshold:
                return 'coral'  # We will exclude these entirely

        # ------ Check ALGAE next ------ #
        if self.algae_kdtree is not None:
            algae_dist, _ = self.algae_kdtree.query([x, y, z])
            if algae_dist < self.algae_block_threshold:
                return 'algae'

        # If neither coral nor algae are too close
        return 'available'

    def check_blocked_locations(self, valid_levels=('L2', 'L3', 'L4')):
        """
        Returns only available & algae-blocked locations, excluding coral-blocked.

        Returns:
            A dictionary with two keys: 
                {
                "available": [ (branch_name, level), ... ],
                "algae_blocked": [ (branch_name, level), ... ]
                }
        """
        result = {
            "available": [],
            "algae_blocked": []
        }

        branches = self.branch_manager.get_branches()  # list of Branch objects

        for branch in branches:
            branch_name = branch.name

            # For each desired level:
            for level in valid_levels:
                level_pos = branch.get_level_position(level)
                if level_pos is None:
                    # e.g. if the level doesn't exist for some reason
                    continue
                x, y, z = level_pos

                # Determine block status
                block_status = self.get_block_status(x, y, z)

                if block_status == 'available':
                    result["available"].append((branch_name, level, level_pos))
                elif block_status == 'algae':
                    result["algae_blocked"].append((branch_name, level, level_pos))
                # If block_status == 'coral', do nothing => exclude from final

        return result
