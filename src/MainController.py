import threading
import cv2
import time
 
import DepthAIPipeline
import OpenCVProcessor
import NetworkTablesHandler
import ImageSaver
import FlaskServerHandler

class MainController:

    # -- Flags -- #
    capture_input_frame = False
    capture_output_frame = False
    capture_depth_frame = False

    def __init__(self):
        self.depthai_pipeline = DepthAIPipeline()
        self.opencv_processor = OpenCVProcessor(self.depthai_pipeline)
        self.network_tables_handler = NetworkTablesHandler(self)
        self.image_saver = ImageSaver()
        self.video_stream_handler = FlaskServerHandler()

    def start(self):

        # Setup the DepthAI Pipeline and NetworkTables
        self.depthai_pipeline.start_pipeline()
        self.opencv_processor.start_processor()
        self.network_tables_handler.start_listening()

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
                self.network_tables_handler.upload_data(positions)

                # Display the processed video frame
                self.video_stream_handler.update_frame(processed_frame)

                # Display the frames - Disable when running on PI.
                cv2.imshow("Input Frame", color_frame)
                cv2.imshow("Output Frame", processed_frame)

                # If told to save the images, save them, and toggle the capture flags.
                if self.capture_input_frame:
                    self.image_saver.save_image(color_frame, "input_frame ")
                    self.capture_input_frame = False

                elif self.capture_output_frame:
                    self.image_saver.save_image(processed_frame, "output_frame ")
                    self.capture_output_frame = False
                    
                elif self.capture_depth_frame:
                    self.image_saver.save_image(depth_frame, "depth_frame ")
                    self.capture_depth_frame = False

                # TODO: Sync time between roboRIO and Raspberry Pi if needed.

        except KeyboardInterrupt:
            print("Stopping System")
        finally:
            cv2.destroyAllWindows()
            self.depthai_pipeline.stop_pipeline()

    def save_input_frame(self):
        """
        Saves the next frame captured by the color camera to the file.
        """
        self.capture_input_frame = True

    def save_output_frame(self):
        """
        Saves the next processed frame captured by the color camera to the file.
        """
        self.capture_output_frame = True

    def save_depth_frame(self):
        """
        Saves the next frame captured by the depth camera to the file.
        """
        self.capture_depth_frame = True

if __name__ == "__main__":

    # Create and start the main controller
    controller = MainController()
    controller.start()