import cv2
import numpy as np


class OpenCVProcessor:
    # ------------------------------------------------
    #               CONSTANTS & THRESHOLDS
    # ------------------------------------------------
    # HSV Thresholds
    ALGAE_LOWER_BOUND = np.array([0, 0, 0])
    ALGAE_UPPER_BOUND = np.array([255, 255, 255])

    CORAL_LOWER_BOUND = np.array([0, 0, 0])
    CORAL_UPPER_BOUND = np.array([255, 255, 255])

    # Canny Edge Detection
    ALGAE_LOWER_THRESHOLD = 400
    ALGAE_UPPER_THRESHOLD = 500

    CORAL_LOWER_THRESHOLD = 400
    CORAL_UPPER_THRESHOLD = 500

    def __init__(self, depthai_pipeline):
        """
        Initialize with a DepthAI pipeline for camera access and intrinsics.
        """
        self.depthai_pipeline = depthai_pipeline
        self.camera_intrinsics = None

    def start_processor(self):
        """
        Start the DepthAI pipeline (assumed already initialized) and
        retrieve camera intrinsics.
        """
        self.camera_intrinsics = self.depthai_pipeline.get_color_camera_intrinsics()

        # ------------------------------------------------

    #               HELPER FUNCTIONS
    # ------------------------------------------------

    def mask_image(self, frame_bgr, lower_bound, upper_bound):
        """
        Creates a binary mask to detect objects within the specified HSV range.

        Args:
            frame_bgr (numpy.ndarray): The BGR color frame.
            lower_bound (numpy.ndarray): Lower bound HSV value.
            upper_bound (numpy.ndarray): Upper bound HSV value.

        Returns:
            numpy.ndarray: The binary mask of the detected color.
        """
        # Convert the image to HSV
        hsv_image = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2HSV)

        # Create a binary mask based on HSV bounds
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Morphological operations (closing + opening) to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def find_contours(self, mask, lower_threshold, upper_threshold):
        """
        Finds contours in a binary mask using Canny edge detection.

        Args:
            mask (numpy.ndarray): The binary mask.
            lower_threshold (int): Lower Canny Edge Detection Threshold.
            upper_threshold (int): Upper Canny Edge Detection Threshold.

        Returns:
            list: A list of valid contours.
        """
        # 1. Edge detection
        edges = cv2.Canny(mask, lower_threshold, upper_threshold)

        # 2. Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Optionally filter out very small contours by area
        # filtered_contours = []
        # for c in contours:
        #     area = cv2.contourArea(c)
        #     if area > 100:  # adjust as needed
        #         filtered_contours.append(c)
        # return filtered_contours

        return contours

    def extract_bounding_boxes(self, contours):
        """
        Converts contours to bounding boxes (x, y, w, h).

        Args:
            contours (list): List of contours.

        Returns:
            list: List of bounding boxes in the form (x, y, w, h).
        """
        bounding_boxes = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            bounding_boxes.append(cv2.boundingRect(approx))

        return bounding_boxes

    def calculate_positions(self, depth_frame, object_bboxes):
        """
        Calculate the position of objects from the camera using depth data.

        Args:
            depth_frame (numpy.ndarray): The depth frame to calculate distances from.
            object_bboxes (list): A list of bounding boxes of the detected objects (x, y, w, h).

        Returns:
            list: List of dictionaries with keys {"x", "y", "z"} for each object.
        """
        positions = []

        if self.camera_intrinsics is None:
            raise ValueError(
                "Camera intrinsics must be set before calling calculate_positions()."
            )

        fx = self.camera_intrinsics[0][0]  # Focal length x
        fy = self.camera_intrinsics[1][1]  # Focal length y
        cx = self.camera_intrinsics[0][2]  # Optical center x
        cy = self.camera_intrinsics[1][2]  # Optical center y

        for x, y, w, h in object_bboxes:
            # Calculate the center of the bounding box
            # TODO: Check if it is more accurate to get the position of the top of the bounding box.
            center_x = x + w // 2
            center_y = y + h // 2

            # Depth value at the center of the bounding box (in mm, e.g.)
            depth_value = depth_frame[center_y, center_x]

            # Convert to meters (assumes depth_value is in millimeters)
            distance_m = depth_value / 1000.0

            # Project to 3D space
            obj_x = (center_x - cx) * distance_m / fx
            obj_y = (center_y - cy) * distance_m / fy

            positions.append({"x": obj_x, "y": obj_y, "z": distance_m})

        return positions

    def draw_bounding_boxes(
        self, frame, bounding_boxes, positions, color=(0, 255, 0), thickness=2
    ):
        """
        Draws bounding boxes on the frame and shows each object's estimated position.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            bounding_boxes (list): List of bounding boxes (x, y, w, h).
            positions (list): Corresponding list of positions (dict with x, y, z).
            color (tuple): The color of the bounding boxes.
            thickness (int): The thickness of the bounding box outline.

        Returns:
            numpy.ndarray: The frame with bounding boxes drawn.
        """
        for bbox, pos in zip(bounding_boxes, positions):
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Show the (x,y,z) in meters above the bounding box
            label = f"({pos['x']:.2f}m, {pos['y']:.2f}m, {pos['z']:.2f}m)"
            cv2.putText(
                frame,
                label,
                (x, max(0, y - 10)),  # Make sure text doesn't go off-frame
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return frame

    # ------------------------------------------------
    #               MAIN PIPELINE
    # ------------------------------------------------

    def process_frame(self, color_frame_bgr, depth_frame):
        """
        Full image processing pipeline:
        1. Mask for algae and coral
        2. Find contours and bounding boxes
        3. Calculate 3D positions
        4. Draw bounding boxes and position info

        Args:
            color_frame_bgr (numpy.ndarray): The BGR color frame from the camera.
            depth_frame (numpy.ndarray): The depth frame aligned to the color frame.

        Returns:
            tuple:
                processed_frame (numpy.ndarray): The frame with bounding boxes and text drawn.
                algae_positions (list): List of 3D positions for the algae detections.
                coral_positions (list): List of 3D positions for the coral detections.
        """

        # 1. Create binary masks
        algae_mask = self.mask_image(
            color_frame_bgr, self.ALGAE_LOWER_BOUND, self.ALGAE_UPPER_BOUND
        )
        coral_mask = self.mask_image(
            color_frame_bgr, self.CORAL_LOWER_BOUND, self.CORAL_UPPER_BOUND
        )

        # 2. Find contours
        algae_contours = self.find_contours(
            algae_mask, self.ALGAE_LOWER_THRESHOLD, self.ALGAE_UPPER_THRESHOLD
        )
        coral_contours = self.find_contours(
            coral_mask, self.CORAL_LOWER_THRESHOLD, self.CORAL_UPPER_THRESHOLD
        )

        # 3. Extract bounding boxes
        algae_bboxes = self.extract_bounding_boxes(algae_contours)
        coral_bboxes = self.extract_bounding_boxes(coral_contours)

        # 4. Calculate 3D positions
        algae_positions = self.calculate_positions(depth_frame, algae_bboxes)
        coral_positions = self.calculate_positions(depth_frame, coral_bboxes)

        # 5. Draw bounding boxes on the image
        processed_frame = color_frame_bgr.copy()
        processed_frame = self.draw_bounding_boxes(
            processed_frame,
            algae_bboxes,
            algae_positions,
            color=(0, 255, 0),
            thickness=2,
        )
        processed_frame = self.draw_bounding_boxes(
            processed_frame,
            coral_bboxes,
            coral_positions,
            color=(255, 0, 255),
            thickness=2,
        )

        return processed_frame, algae_positions, coral_positions
