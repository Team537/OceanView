import cv2
import depthai as dai
import numpy as np
from depthai_pipeline import create_pipeline
from opencv_processing import process_frame, calculate_distance

def main():

    # Create the DepthAI pipeline
    pipeline = create_pipeline()

    # Start the device with the pipeline
    with dai.Device(pipeline) as device:

        # Get the output queues for video and depth streams
        video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:

            # Get the latest frames from DepthAI
            frame = video_queue.get().getCvFrame()
            depth_frame = depth_queue.get().getFrame()

            # Process the frame (detection and bounding boxes)
            processed_frame, bounding_boxes = process_frame(frame, lower_bound=np.array([0, 120, 70]), upper_bound=np.array([10, 255, 255]))

            # For each detected object, calculate its distance
            for obj in bounding_boxes:
                distance = calculate_distance(depth_frame, obj)
                print(f"Object at {obj} is {distance:.2f} meters away")

                # Display the distance
                x, y, w, h = obj
                cv2.putText(processed_frame, f"{distance:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the processed frames
            cv2.imshow("Processed RGB Frame", processed_frame)
            cv2.imshow("Depth Frame", depth_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Close all of the OpenCV created during the program's operation.
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()