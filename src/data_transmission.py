from networktables import NetworkTables
import threading
        

# Initialize NetworkTables
ROBORIO_IP = "roborio-XXX-frc.local"  # Replace XXX with your team number
NetworkTables.initialize(server=ROBORIO_IP)

# Get the Vision table
vision_table = NetworkTables.getTable("Vision")


def send_data_to_robotrio(bounding_boxes, positions):
    """
    Uploads the given data to the roborio.

    Args:
        bounding_boxes (list): List of bounding boxes.
        positions (tuple): List of object's positions.
    """
    # Verify that the Raspberry PI is connected to the network tables.
    if not NetworkTables.isConnected:
        print("CONNECTION ERROR: Raspberry PI is not connected to the RobotRIO!")
        return
    
    # Send data to network tables.
    vision_table.putNumber("NumTargets", len(bounding_boxes))
    for i, (box, position) in enumerate(zip(bounding_boxes, positions)):
        vision_table.putNumberArray(f"Targets{i}_BoundingBox", box)
        vision_table.putNumberArray(f"Targets{i}_BoundingBox", position)
