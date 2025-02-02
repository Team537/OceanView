# map_management/branch_manager.py
import yaml
from map_management.branch import Branch

class BranchManager:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.branches = []
        self.base_poles = []
        self.base_pole_names = []
        self.center_x = 0.0
        self.center_z = 0.0
        self.reef_radius = 0.0
        self.radius_offset = 0.0
        self.level_heights = {}
        self.alliance = "red"  # Default alliance
        self._load_branches()

    def _load_branches(self):
        # Load YAML data
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Extract branch settings from 'branchSettings'
        branch_settings = data.get('branch_settings', {})
        self.level_heights = {
            'L2': branch_settings.get('L2_height'),
            'L3': branch_settings.get('L3_height'),
            'L4': branch_settings.get('L4_height')
        }
        self.radius_offset = branch_settings.get('radius_offset')

        # Extract reef center position based on current alliance
        center_pose_key = f"{self.alliance}Pose"
        reef_center = data['center_of_reef'].get(center_pose_key, {})
        self.center_x = reef_center.get('x', 0.0)
        self.center_z = reef_center.get('z', 0.0)
        self.reef_radius = data['center_of_reef'].get('radius', 0.54901342212)

        # Load all branches
        branch_data = data.get('branches', [])
        for branch_info in branch_data:
            # Validate branch_info has necessary keys
            if 'name' not in branch_info or 'angle_deg' not in branch_info:
                raise KeyError("Each branch must have 'name' and 'angle_deg' keys.")

            branch = Branch(
                name=branch_info['name'],
                angle_deg=branch_info['angle_deg'],
                center_x=self.center_x,
                center_z=self.center_z,
                reef_radius=self.reef_radius,
                radius_offset=self.radius_offset,
                level_heights=self.level_heights,
                alliance=self.alliance
            )
            self.branches.append(branch)
            self.base_poles.append((branch.base_x, branch.base_z))
            self.base_pole_names.append(branch.name)

    def get_branches(self):
        return self.branches

    def get_base_poles(self):
        return self.base_poles

    def get_base_pole_names(self):
        return self.base_pole_names

    def set_alliance(self, alliance):
        """
        Update the alliance and reload branches.
        :param alliance: "red" or "blue"
        """
        alliance = alliance.lower()
        if alliance not in ['red', 'blue']:
            raise ValueError("Alliance must be 'red' or 'blue'")
        self.alliance = alliance
        self.branches.clear()
        self.base_poles.clear()
        self.base_pole_names.clear()
        self._load_branches()
