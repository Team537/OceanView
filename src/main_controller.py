import threading
import cv2
import time
 
import depthai_pipeline
import opencv_processor
import image_saver
import flask_server_handler

import tcp_receiver
import udp_sender

class MainController:

    # -- Flags -- #
    capture_input_frame = False
    capture_output_frame = False
    capture_depth_frame = False

    # -- Constants -- #
    ROBORIO_IP = "10.5.37.2"
    ROBORIO_PORT = 5000

    # -- Storage -- #
    robot_pose = {
        "x": 0,
        "y": 0,
        "z": 0,
        "pitch" : 0,
        "roll" : 0,
        "yaw": 0
    }

    def __init__(self):

        # Processing
        self.depthai_pipeline = depthai_pipeline()
        self.opencv_processor = opencv_processor(self.depthai_pipeline)

        # File Management
        self.image_saver = image_saver()

        # Data Transmission
        self.tcp_receiver = tcp_receiver(self, ip=self.ROBORIO_IP)
        self.udp_sender = udp_sender(ip=self.ROBORIO_IP)
        self.flask_server_handler = flask_server_handler(self.ROBORIO_PORT)

    def start(self):

        # Setup the DepthAI Pipeline
        self.depthai_pipeline.start_pipeline()
        self.opencv_processor.start_processor()

        self.tcp_receiver.start()

        # Start the Flask server in a separate thread
        threading.Thread(target=self.video_stream_handler.run, daemon=True).start()
        
        # Start the main program loop
        self.main_loop()

    def main_loop(self):
        try:
            while True:

                # Get the next frame(s) from DepthAI
                color_frame = self.depthai_pipeline.get_frame()
                depth_frame = self.depthai_pipeline.get_depth_frame()

                # If no frames are active, wait until a frame is available
                if color_frame is None or depth_frame is None:
                    time.sleep(0.01) # Prevent the CPU from running as fast as possible.
                    continue

                # Process the new frame
                processed_frame, positions = self.opencv_processor.process_frame(color_frame, depth_frame)

                # Upload data to the robotRIO
                self.udp_sender.upload_data(positions)

                # Display the processed video frame
                self.flask_server_handler.update_frame(processed_frame)

                # Display the frames - Disable when running on PI.
                cv2.imshow("Input Frame", color_frame)
                cv2.imshow("Output Frame", processed_frame)

                # If told to save the images, save them, and toggle the capture flags.
                if self.capture_input_frame:
                    self.image_saver.save_image(color_frame, "input_frame ")
                    self.capture_input_frame = False
                if self.capture_output_frame:
                    self.image_saver.save_image(processed_frame, "output_frame ")
                    self.capture_output_frame = False
                if self.capture_depth_frame:
                    self.image_saver.save_image(depth_frame, "depth_frame ")
                    self.capture_depth_frame = False

                # TODO: Sync time between roboRIO and Raspberry Pi if needed.

        except KeyboardInterrupt:
            print("Stopping System")
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

if __name__ == "__main__":

    # Create and start the main controller
    controller = MainController()
    controller.start()