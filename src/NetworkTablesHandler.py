from networktables import NetworkTables

class NetworkTablesHandler:

    # -- Settings -- #
    TEAM_NUMBER = 537

    def __init__(self, main_controller):
        self.main_controller = main_controller
        self.vision_table = None

    def start_listening(self):
        print("Starting NetworkTables listener")

        # Initialize the connection to NetworkTables using the RoboRIO address.
        NetworkTables.initialize(server=f"roborio-{self.TEAM_NUMBER}-frc.local")
        self.vision_table = NetworkTables.getTable("Vision")

    def upload_data(self, positions):
        """
        Uploads the given data to the RoboRIO.

        Args:
            positions (tuple): List of object's positions.
        """

        # Verify that the Raspberry Pi is connected to NetworkTables.
        if not NetworkTables.isConnected:
            print("CONNECTION ERROR: Raspberry Pi is not connected to the RobotRIO!")
            return
        
        # Send the number of targets.
        self.vision_table.putNumber("NumTargets", len(positions))
        
        # Upload the positions of each target.
        for i, position in enumerate(positions):

            # Ensure position is a tuple of x, y, z
            x, y, z = position
            
            # Upload the position values as an array
            self.vision_table.putNumberArray(f"Target{i}_Positions", [x, y, z])

