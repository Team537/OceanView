import cv2
import numpy as np

class OpenCVProcessor:

    # -- Constants -- #
    # HSV Thresholds
    LOWER_BOUND = np.array([0, 0, 0])
    UPPER_BOUND = np.array([255, 255, 255])

    # Canny Edge Detection
    LOWER_THRESHOLD = 400
    UPPER_THRESHOLD = 500

    # -- Settings -- #
    camera_intrinsics = None

    def __init__(self, depthai_pipeline):
        self.depthai_pipeline = depthai_pipeline

    def start_processor(self):
        
        # Assuming depthai_pipeline is correctly initialized and started,
        # and camera_intrinsics is set using the DepthAI device calibration
        self.camera_intrinsics = self.get_color_camera_intrinsics()

    def mask_image(self, frame):
        """
        Creates a mask to detect objects within the specified color range.
        
        Args:
            frame (numpy.ndarray): The HSV frame.
        
        Returns:
            numpy.ndarray: The binary mask of the detected color.
        """

        # Convert the image to HSV so that colors within the specified HSV range can be filtered out.
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        threshold_image = cv2.inRange(hsv_image, self.LOWER_BOUND, self.UPPER_BOUND)

        # Filter out detection noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
        threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

        # Return the processed frame
        return threshold_image

    def find_contours(self, mask):
        """
        Finds contours in a binary mask.
        
        Args:
            mask (numpy.ndarray): The binary mask.
        
        Returns:
            list: A list of contours.
        """

        # Find the edges in the image.
        edges = cv2.Canny(mask, self.LOWER_THRESHOLD, self.UPPER_THRESHOLD)

        # Find the contours of the image.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # TODO: Add feature to filter our contours if they are too small.

        # Return the found contours.
        return contours

    def extract_bounding_boxes(self, contours):
        """
        Converts contours to bounding boxes.
        
        Args:
            contours (list): List of contours.
        
        Returns:
            list: List of bounding boxes (x, y, w, h).
        """
        bounding_boxes = []

        for contour in contours:
            approximatedContour = cv2.approxPolyDP(contour, 3, True)
            bounding_boxes.append(cv2.boundingRect(approximatedContour))
        
        # Return the detected bounding boxes
        return bounding_boxes

    def calculate_positions(self, depth_frame, object_bboxes):
        """
        Calculate the position of an object from the camera using depth data.
        
        Args:
            depth_frame (numpy.ndarray): The depth frame to calculate distance from.
            object_bboxes (list): A list of the bounding boxes of the detected objects (x, y, w, h).
        
        Returns:
            list: List of positions (x, y, distance).
        """

        # Create a new array to store the positions of objects.
        positions = []

        for x, y, w, h in object_bboxes:

            # Calculate the center of the object (bounding box) 
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Extract the depth value from the depth frame at the center of the object
            depth_value = depth_frame[center_y, center_x]
            
            # Convert the depth value from raw disparity to actual distance (this may vary based on your camera)
            # TODO: Research and tune this value.
            distance_in_meters = depth_value / 1000.0 

            # Calculate the other 2 coordinates of the object.
            objectX = ((center_x - self.camera_intrinsics[0][2]) * distance_in_meters) / self.camera_intrinsics[0][0]
            objectY = ((center_y - self.camera_intrinsics[1][2]) * distance_in_meters) / self.camera_intrinsics[1][1]

            # Add the calculated position to the array of positions.
            positions.append((objectX, objectY, distance_in_meters))

        # Return the object's positions.
        return positions


    def draw_bounding_boxes(self, frame, bounding_boxes, positions, color=(0, 255, 0), thickness=2):
        """
        Draws bounding boxes on the frame and shows each object's estimated position.
        
        Args:
            frame (numpy.ndarray): The frame to draw on.
            bounding_boxes (list): List of bounding boxes to draw.
            positions (list): List of the calculated positions of the objects.
            color (tuple): The color of the bounding boxes.
            thickness (int): The thickness of the bounding box lines.
        
        Returns:
            numpy.ndarray: The frame with bounding boxes drawn.
        """
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(frame, f"({positions[0]:.2f}m, {positions[1]:.2f}m, {positions[2]:.2f}m)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    def process_frame(self, color_frame, depth_frame):
        """
        Full image processing pipeline: converts to HSV, creates a mask, detects contours, 
        draws bounding boxes, and finds the 3D position of the detected objects.
        
        Args:
            color_frame (numpy.ndarray): The RGB frame to process.
            depth_frame (numpy.ndarray): The depth frame to process.
        
        Returns:
            numpy.ndarray: The processed frame with bounding boxes drawn.
            list: List the 3D positions of detected objects.
        """
        
        # Filter our the specified colors from the image.
        masked_image = self.mask_image(color_frame)

        # Find the contours of the image.
        contours = self.find_contours(masked_image)

        # Extract bounding boxes from the image.
        bounding_boxes = self.extract_bounding_boxes(contours)

        # Calculate the position of the detected objects.
        positions = self.calculate_positions(depth_frame, bounding_boxes)

        # Draw bounding boxes on the image.
        processed_frame = self.draw_bounding_boxes(color_frame, bounding_boxes, positions)

        # Return the processed frame and found bounding boxes.
        return processed_frame, positions