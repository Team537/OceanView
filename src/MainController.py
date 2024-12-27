import DepthAIPipeline
import OpenCVProcessor
import NetworkTablesHandler
import cv2

class MainController:
    def __init__(self):
        self.depthai_pipeline = DepthAIPipeline()
        self.opencv_processor = OpenCVProcessor(self.depthai_pipeline)
        self.network_tables_handler = NetworkTablesHandler(self)

    def start(self):

        # Setup the DepthAI Pipeline and NetworkTables
        self.depthai_pipeline.start_pipeline()
        self.opencv_processor.start_processor()
        self.network_tables_handler.start_listening()

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
                    continue

                # Process the new frame
                processed_frame, positions = self.opencv_processor.process_frame(color_frame, depth_frame)

                # Display the frames - Disable when running on PI.
                cv2.imshow("Input Frame", color_frame)
                cv2.imshow("Output Frame", processed_frame)

                # TODO: Sync time between roboRIO and Raspberry Pi if needed.
                # Upload data to the robotRIO
                self.network_tables_handler.upload_data(positions)

        except KeyboardInterrupt:
            print("Stopping System")
        finally:
            cv2.destroyAllWindows()
            self.depthai_pipeline.stop_pipeline()

if __name__ == "__main__":

    # Create and start the main controller
    controller = MainController()
    controller.start()