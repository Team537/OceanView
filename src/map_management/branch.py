# map_management/branch.py
import math

class Branch:
    def __init__(self, name, angle_deg, center_x, center_z, reef_radius, radius_offset, level_heights, alliance="red"):
        self.name = name
        self.angle_deg = angle_deg
        self.alliance = alliance.lower()
        self.angle_rad = math.radians(angle_deg)
        self.center_x = center_x
        self.center_z = center_z
        self.reef_radius = reef_radius
        self.radius_offset = radius_offset
        self.level_heights = level_heights  # Dictionary: {'L2': height, 'L3': height, 'L4': height}

        # Rotate the angle by 180 degrees for blue alliance
        if self.alliance == "blue":
            self.angle_rad = math.radians(angle_deg + 180)
            # Normalize angle_rad between 0 and 2*pi
            self.angle_rad = self.angle_rad % (2 * math.pi)

        # Calculate base position
        self.base_x = self.center_x + self.reef_radius * math.cos(self.angle_rad)
        self.base_z = self.center_z + self.reef_radius * math.sin(self.angle_rad)

        # Calculate positions for each level
        self.level_positions = self._calculate_level_positions()

    def _calculate_level_positions(self):
        positions = {}
        for idx, (level, height) in enumerate(self.level_heights.items(), start=1):
            x = self.base_x + idx * self.radius_offset * math.cos(self.angle_rad)
            z = self.base_z + idx * self.radius_offset * math.sin(self.angle_rad)
            positions[level] = (x, height, z)
        return positions

    def get_level_position(self, level):
        return self.level_positions.get(level)

    def get_all_levels(self):
        return self.level_positions
