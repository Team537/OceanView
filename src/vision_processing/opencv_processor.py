import cv2
import numpy as np


class OpenCVProcessor:
    """
    This class handles processing of frames using OpenCV.
    It performs tasks such as color masking, contour detection, calculating 3D positions,
    and drawing bounding boxes with estimated object positions.
    """
    # HSV thresholds for detecting algae.
    ALGAE_LOWER_BOUND = np.array([81, 110, 120])
    ALGAE_UPPER_BOUND = np.array([99, 255, 255])
    # HSV thresholds for detecting coral.
    CORAL_LOWER_BOUND = np.array([0, 0, 0])
    CORAL_UPPER_BOUND = np.array([255, 255, 255])
    # Canny edge detection thresholds for algae.
    ALGAE_LOWER_THRESHOLD = 400
    ALGAE_UPPER_THRESHOLD = 500
    # Canny edge detection thresholds for coral.
    CORAL_LOWER_THRESHOLD = 400
    CORAL_UPPER_THRESHOLD = 500

    def __init__(self, depthai_pipeline):
        """
        Initialize the processor with a DepthAIPipeline instance.

        Args:
            depthai_pipeline (DepthAIPipeline): The pipeline for accessing frames.
        """
        self.depthai_pipeline = depthai_pipeline
        # This will store the effective (undistorted) camera intrinsics computed from the calibration.
        self.camera_intrinsics = None

    def start_processor(self):
        """
        Retrieve the raw calibration data (intrinsics and distortion coefficients)
        and compute the effective intrinsics for the undistorted preview image using OpenCV.
        """
        # Get the raw intrinsics and distortion coefficients.
        raw_intrinsics = self.depthai_pipeline.get_color_camera_intrinsics()
        distCoeffs = self.depthai_pipeline.get_color_camera_distortion()

        # Define the sensor (and preview) size. In our pipeline, we set these to 1080x720.
        sensor_size = (1080, 720)
        preview_size = sensor_size

        # Compute the optimal new camera matrix for the undistorted image.
        # The alpha parameter (here set to 1) determines how much of the original image is retained.
        # (alpha=1 retains all pixels, alpha=0 crops out regions with no valid pixels)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(raw_intrinsics, distCoeffs, preview_size, 1, preview_size)
        # Save the effective intrinsics for later use in position calculation.
        self.camera_intrinsics = new_camera_matrix

    def mask_image(self, frame_bgr, lower_bound, upper_bound):
        """
        Convert a BGR image to HSV and create a binary mask based on specified HSV bounds.

        Args:
            frame_bgr (numpy.ndarray): Input BGR image.
            lower_bound (numpy.ndarray): Lower HSV threshold.
            upper_bound (numpy.ndarray): Upper HSV threshold.

        Returns:
            numpy.ndarray: Binary mask image.
        """
        hsv_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        # Use morphological operations to reduce noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def find_contours(self, mask, lower_threshold, upper_threshold):
        """
        Use Canny edge detection to find contours in the binary mask.

        Args:
            mask (numpy.ndarray): Binary mask image.
            lower_threshold (int): Lower threshold for Canny.
            upper_threshold (int): Upper threshold for Canny.

        Returns:
            list: List of contours.
        """
        edges = cv2.Canny(mask, lower_threshold, upper_threshold)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def extract_bounding_boxes(self, contours):
        """
        Convert contours to bounding boxes.

        Args:
            contours (list): List of contours.

        Returns:
            list: List of bounding boxes in (x, y, w, h) format.
        """
        bounding_boxes = []
        for contour in contours:
            # Approximate the contour to reduce noise.
            approx = cv2.approxPolyDP(contour, 3, True)
            bounding_boxes.append(cv2.boundingRect(approx))
        return bounding_boxes

    def calculate_positions(self, color_frame, depth_frame, object_bboxes):
        """
        Calculate the 3D position of each detected object.
        This method uses the effective (undistorted) intrinsics computed earlier.

        Args:
            color_frame (numpy.ndarray): The undistorted color frame.
            depth_frame (numpy.ndarray): The depth frame.
            object_bboxes (list): List of bounding boxes (x, y, w, h) for detected objects.

        Returns:
            list: List of dictionaries with 3D positions (x, y, z in meters).
        """
        positions = []
        if self.camera_intrinsics is None:
            raise ValueError("Effective camera intrinsics not set. Call start_processor() first.")

        # ---------------------------------------------------------------------
        # 1. Determine the scaling between the sensor (preview) resolution and
        #    the actual displayed image sizes.
        # ---------------------------------------------------------------------
        # Our pipeline was configured for a sensor/preview size of 1080x720.
        sensor_width, sensor_height = 1080, 720
        # Get the size of the current color frame (may be resized if necessary).
        color_h, color_w = color_frame.shape[:2]
        # Get the size of the depth frame (as set in depth.setInputResolution()).
        depth_h, depth_w = depth_frame.shape[:2]

        # Compute scale factors between the sensor resolution and the color frame size.
        scale_color_x = color_w / sensor_width
        scale_color_y = color_h / sensor_height

        # Using the effective intrinsics computed earlier (which are for the sensor size),
        # we adjust them for the actual color frame size.
        fx = self.camera_intrinsics[0, 0] * scale_color_x
        fy = self.camera_intrinsics[1, 1] * scale_color_y
        cx = self.camera_intrinsics[0, 2] * scale_color_x
        cy = self.camera_intrinsics[1, 2] * scale_color_y

        # Compute the scaling between the color frame and depth frame.
        scale_depth_x = depth_w / color_w
        scale_depth_y = depth_h / color_h

        # ---------------------------------------------------------------------
        # 2. For each object bounding box, determine the object center,
        #    sample depth using bilinear interpolation, and reproject to 3D.
        # ---------------------------------------------------------------------
        for bbox in object_bboxes:
            x, y, w, h = bbox
            # Calculate the center of the bounding box in the color image.
            center_color_x = x + (w / 2.0)
            center_color_y = y + (h / 2.0)
            # Map the center coordinates from the color image to the depth image.
            center_depth_x = center_color_x * scale_depth_x
            center_depth_y = center_color_y * scale_depth_y

            # Ensure the mapped coordinates are within bounds of the depth frame.
            if center_depth_x < 0 or center_depth_x >= depth_w - 1 or \
                    center_depth_y < 0 or center_depth_y >= depth_h - 1:
                continue

            # Use bilinear interpolation to get a subpixel depth value.
            x0 = int(np.floor(center_depth_x))
            x1 = min(x0 + 1, depth_w - 1)
            y0 = int(np.floor(center_depth_y))
            y1 = min(y0 + 1, depth_h - 1)
            dx = center_depth_x - x0
            dy = center_depth_y - y0

            d00 = float(depth_frame[y0, x0])
            d01 = float(depth_frame[y0, x1])
            d10 = float(depth_frame[y1, x0])
            d11 = float(depth_frame[y1, x1])
            depth_value = (d00 * (1 - dx) * (1 - dy) +
                           d01 * dx * (1 - dy) +
                           d10 * (1 - dx) * dy +
                           d11 * dx * dy)

            # Skip if no valid depth is available.
            if depth_value == 0:
                continue

            # Convert depth value from millimeters to meters.
            distance_m = depth_value / 1000.0

            # -----------------------------------------------------------------
            # 3. Reproject the object center to 3D space using the pinhole camera model:
            #       X = (u - cx) * Z / fx
            #       Y = (v - cy) * Z / fy
            # where (u, v) is the center of the object in the color image.
            # -----------------------------------------------------------------
            obj_x = (center_color_x - cx) * distance_m / fx
            obj_y = (center_color_y - cy) * distance_m / fy

            positions.append({"x": obj_x, "y": obj_y, "z": distance_m})

        return positions

    def draw_bounding_boxes(self, frame, bounding_boxes, positions, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes and the corresponding 3D position (x, y, z) onto the frame.

        Args:
            frame (numpy.ndarray): Image on which to draw.
            bounding_boxes (list): List of bounding boxes (x, y, w, h).
            positions (list): List of 3D positions for each bounding box.
            color (tuple): Color of the bounding box.
            thickness (int): Thickness of the drawn rectangle.

        Returns:
            numpy.ndarray: The image with drawn bounding boxes and labels.
        """
        for bbox, pos in zip(bounding_boxes, positions):
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            label = f"({pos['x']:.2f}m, {pos['y']:.2f}m, {pos['z']:.2f}m)"
            cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def process_frame(self, color_frame_bgr, depth_frame):
        """
        Full image processing pipeline:
          1. Create masks for algae and coral based on HSV thresholds.
          2. Detect contours in the masks.
          3. Extract bounding boxes from the contours.
          4. Calculate 3D positions of objects using depth information.
          5. Draw bounding boxes and display the estimated 3D positions.

        Args:
            color_frame_bgr (numpy.ndarray): The undistorted color frame (BGR).
            depth_frame (numpy.ndarray): The depth frame (aligned to the color frame).

        Returns:
            tuple: (processed_frame, algae_positions, coral_positions)
                   processed_frame: The color frame with bounding boxes drawn.
                   algae_positions: List of 3D positions for algae detections.
                   coral_positions: List of 3D positions for coral detections.
        """
        # 1. Create binary masks for algae and coral using HSV thresholds.
        algae_mask = self.mask_image(color_frame_bgr, self.ALGAE_LOWER_BOUND, self.ALGAE_UPPER_BOUND)
        coral_mask = self.mask_image(color_frame_bgr, self.CORAL_LOWER_BOUND, self.CORAL_UPPER_BOUND)

        # 2. Find contours in both masks.
        algae_contours = self.find_contours(algae_mask, self.ALGAE_LOWER_THRESHOLD, self.ALGAE_UPPER_THRESHOLD)
        coral_contours = self.find_contours(coral_mask, self.CORAL_LOWER_THRESHOLD, self.CORAL_UPPER_THRESHOLD)

        # 3. Convert contours into bounding boxes.
        algae_bboxes = self.extract_bounding_boxes(algae_contours)
        coral_bboxes = self.extract_bounding_boxes(coral_contours)

        # 4. Calculate the 3D positions of each detection using the color and depth frames.
        algae_positions = self.calculate_positions(color_frame_bgr, depth_frame, algae_bboxes)
        coral_positions = self.calculate_positions(color_frame_bgr, depth_frame, coral_bboxes)

        # 5. Draw the bounding boxes and position labels onto a copy of the color frame.
        processed_frame = color_frame_bgr.copy()
        processed_frame = self.draw_bounding_boxes(processed_frame, algae_bboxes, algae_positions, color=(0, 255, 0))
        processed_frame = self.draw_bounding_boxes(processed_frame, coral_bboxes, coral_positions, color=(255, 0, 255))

        return processed_frame, algae_positions, coral_positions