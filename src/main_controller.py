# src/main_controller.py
import threading
import cv2
import time
import logging

from map_management.branch_manager import BranchManager
from map_management.kdtree_manager import KDTreeManager
from block_detection_manager import BlockDetectionManager

from depthai_pipeline import DepthAIPipeline
from opencv_processor import OpenCVProcessor

from file_handling.image_saver import ImageSaver

from data_transmission.flask_server_handler import FlaskServerHandler
from data_transmission.tcp_receiver import TCPReceiver
from data_transmission.udp_sender import UDPSender

class MainController:

    # -- Flags -- #
    capture_input_frame = False
    capture_output_frame = False
    capture_depth_frame = False

    # -- Constants -- #
    ROBORIO_IP = "10.5.37.2"
    ROBORIO_PORT = 5000

    def __init__(self):

        # Processing
        self.depthai_pipeline = DepthAIPipeline()
        self.opencv_processor = OpenCVProcessor(self.depthai_pipeline)

        # File Management
        self.image_saver = ImageSaver()

        # Data Transmission
        self.tcp_receiver = TCPReceiver(self, ip=self.ROBORIO_IP)
        self.udp_sender = UDPSender(ip=self.ROBORIO_IP)
        self.flask_server_handler = FlaskServerHandler(self.ROBORIO_PORT)

        # Map Management
        self.branch_manager = BranchManager('config/scoring_positions.yml')
        self.kdtree_manager = KDTreeManager(self.branch_manager)

        self.block_detector = BlockDetectionManager(branch_manager=self.branch_manager,algae_block_threshold=0.5,coral_block_threshold=0.5)

        # Initialize robot_pose
        self.robot_pose = {
            "x": 0,
            "y": 0,
            "z": 0,
            "pitch" : 0,
            "roll" : 0,
            "yaw": 0
        }

    def start(self):
        # Setup the DepthAI Pipeline
        self.depthai_pipeline.start_pipeline()
        self.opencv_processor.start_processor()

        # Start Data Transmission
        self.tcp_receiver.start()

        # Start the Flask server in a separate thread
        threading.Thread(target=self.flask_server_handler.run, daemon=True).start()

        # Start the main program loop
        self.main_loop()

    def main_loop(self):
        try:
            while True:

                # Define a query point based on robot's current position
                query_point = (self.robot_pose['x'], self.robot_pose['z'])

                # Search for the closest branches
                closest_branches = self.kdtree_manager.search_branches(query_point, k=4)

                # Print the results
                print("\nClosest Branches and Their Levels:")
                for branch_name, levels in closest_branches.items():
                    print(f"Branch {branch_name}:")
                    for level, pos in levels.items():
                        print(f"  {level}: {pos}")

                # Get the next frame(s) from DepthAI
                color_frame = self.depthai_pipeline.get_frame()
                depth_frame = self.depthai_pipeline.get_depth_frame()

                # If no frames are active, wait until a frame is available
                if color_frame is None or depth_frame is None:
                    time.sleep(0.01)  # Prevent the CPU from running as fast as possible.
                    continue

                # Process the new frame
                processed_frame, algae_positions, coral_positions = self.opencv_processor.process_frame(color_frame, depth_frame)

                # Build KD-trees from new obstacle data
                self.block_detector.build_obstacle_kdtrees(algae_positions=algae_positions, coral_positions=coral_positions)

                # Check which scoring locations are available or blocked by algae
                blocked_result = self.block_detector.check_blocked_locations(valid_levels=['L2','L3','L4'])
                available = blocked_result["available"]
                algae_blocked = blocked_result["algae_blocked"]

                # Prepare the algae_positions list for JSON (already in desired format)
                # Ensure that algae_positions is a list of dicts with "x", "y", "z"

                # Send the data to RoboRIO
                self.udp_sender.upload_data(
                    available=available,
                    algae_blocked=algae_blocked,
                    algae_positions=algae_positions
                )

                # Display the processed video frame
                self.flask_server_handler.update_frame(processed_frame)

                # Display the frames - Disable when running on PI.
                cv2.imshow("Input Frame", color_frame)
                cv2.imshow("Output Frame", processed_frame)

                # If told to save the images, save them, and toggle the capture flags.
                if self.capture_input_frame:
                    self.image_saver.save_image(color_frame, "input_frame")
                    self.capture_input_frame = False
                if self.capture_output_frame:
                    self.image_saver.save_image(processed_frame, "output_frame")
                    self.capture_output_frame = False
                if self.capture_depth_frame:
                    self.image_saver.save_image(depth_frame, "depth_frame")
                    self.capture_depth_frame = False

        except KeyboardInterrupt:
            print("Stopping System")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        finally:
            cv2.destroyAllWindows()
            self.depthai_pipeline.stop_pipeline()
            self.udp_sender.close()
            self.tcp_receiver.stop()

    def save_frames(self, save_input_frame, save_output_frame, save_depth_frame):
        """
        Saves the specified frames to the file.
        """
        self.capture_input_frame = save_input_frame
        self.capture_output_frame = save_output_frame
        self.capture_depth_frame = save_depth_frame

    def update_robot_pose(self, pose):
        """
        Updates the robot's stored position. This is used to determine whether or not certain targets are blocked.
        """
        self.robot_pose = pose

    def switch_alliance(self, alliance):
        """
        Switches the alliance between red and blue.
        """
        logging.info(f"Switching alliance to {alliance.capitalize()} alliance.")

        try:
            # Update the alliance in BranchManager
            self.branch_manager.set_alliance(alliance)
            logging.info(f"Alliance set to {alliance.capitalize()} in BranchManager.")

            # Rebuild the KDTree
            self.kdtree_manager.build_kdtree()
            logging.info("KDTree rebuilt successfully.")

        except Exception as e:
            logging.error(f"Failed to switch alliance: {e}")
        else:
            print(f"Alliance switched to {alliance.capitalize()} successfully.")

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create and start the main controller
    controller = MainController()
    controller.start()