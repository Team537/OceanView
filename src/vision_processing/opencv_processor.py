import cv2
import numpy as np


class OpenCVProcessor:
    # ------------------------------------------------
    #               CONSTANTS & THRESHOLDS
    # ------------------------------------------------
    # HSV Thresholds
    ALGAE_LOWER_BOUND = np.array([81,110,120])
    ALGAE_UPPER_BOUND = np.array([99,255,255])

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
        hsv_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

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

    def calculate_positions(self, color_frame, depth_frame, object_bboxes):
        """
        Calculate the position of objects from the camera using both the color and depth frames.
        This version accounts for different image resolutions and uses bilinear interpolation
        to take advantage of subpixel depth values.

        Args:
            color_frame (numpy.ndarray): The color (preview) frame.
            depth_frame (numpy.ndarray): The depth frame.
            object_bboxes (list): A list of bounding boxes of the detected objects (x, y, w, h).

        Returns:
            list: List of dictionaries with keys {"x", "y", "z"} for each object (in meters).
        """
        positions = []
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics must be set before calling calculate_positions().")

        # -------------------------------------------------------------------------
        # 1. Figure out the scaling between the various image sizes.
        #
        #    (a) The intrinsics returned by calibration are for the full sensor image.
        #        In this pipeline the color camera is set to 1080p (1920x1080).
        #        (Change these if your sensor is different.)
        sensor_width, sensor_height = 1080, 720

        #    (b) Get the color frame size (the preview image may be downscaled)
        color_h, color_w = color_frame.shape[:2]

        #    (c) Get the depth frame size (set by depth.setInputResolution)
        depth_h, depth_w = depth_frame.shape[:2]

        #    (d) Scale the calibration intrinsics from sensor size to the preview size.
        scale_color_x = color_w / sensor_width
        scale_color_y = color_h / sensor_height

        fx = self.camera_intrinsics[0][0]
        fy = self.camera_intrinsics[1][1]
        cx = self.camera_intrinsics[0][2]
        cy = self.camera_intrinsics[1][2]

        # Effective intrinsics for the color preview:
        fx_eff = fx * scale_color_x
        fy_eff = fy * scale_color_y
        cx_eff = cx * scale_color_x
        cy_eff = cy * scale_color_y

        #    (e) Compute the scale factors from the color (preview) image to the depth image.
        scale_depth_x = depth_w / color_w
        scale_depth_y = depth_h / color_h

        # -------------------------------------------------------------------------
        # 2. For each detected object, map its center from the color image to the depth image,
        #    sample the depth using bilinear interpolation (for subpixel accuracy),
        #    and then reproject using the effective intrinsics.
        for bbox in object_bboxes:
            x, y, w, h = bbox

            # Center of bounding box in the color image (use float arithmetic)
            center_color_x = x + (w / 2.0)
            center_color_y = y + (h / 2.0)

            # Map center from color image coordinates to depth image coordinates.
            center_depth_x = center_color_x * scale_depth_x
            center_depth_y = center_color_y * scale_depth_y

            # --- Use bilinear interpolation on the depth frame ---
            # Make sure we’re within bounds
            if center_depth_x < 0 or center_depth_x >= depth_w - 1 or \
                    center_depth_y < 0 or center_depth_y >= depth_h - 1:
                continue  # Skip this bbox if mapping goes out of bounds

            # Compute the surrounding integer coordinates:
            x0 = int(np.floor(center_depth_x))
            x1 = min(x0 + 1, depth_w - 1)
            y0 = int(np.floor(center_depth_y))
            y1 = min(y0 + 1, depth_h - 1)
            # The fractional parts:
            dx = center_depth_x - x0
            dy = center_depth_y - y0

            # Get the four neighboring depth values.
            d00 = float(depth_frame[y0, x0])
            d01 = float(depth_frame[y0, x1])
            d10 = float(depth_frame[y1, x0])
            d11 = float(depth_frame[y1, x1])
            # Bilinear interpolation:
            depth_value = (d00 * (1 - dx) * (1 - dy) +
                           d01 * dx * (1 - dy) +
                           d10 * (1 - dx) * dy +
                           d11 * dx * dy)

            # If no valid depth is found, skip this detection.
            if depth_value == 0:
                continue

            # Convert from millimeters to meters (if your depth unit is mm)
            distance_m = depth_value / 1000.0

            # ---------------------------------------------------------------------
            # 3. Reproject the object center to 3D space.
            # Note: We use the original (color) center along with the effective intrinsics.
            # The standard pinhole projection gives:
            #      X = (u - cx_eff) * Z / fx_eff
            #      Y = (v - cy_eff) * Z / fy_eff
            obj_x = (center_color_x - cx_eff) * distance_m / fx_eff
            obj_y = (center_color_y - cy_eff) * distance_m / fy_eff

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
        algae_positions = self.calculate_positions(color_frame_bgr, depth_frame, algae_bboxes)
        coral_positions = self.calculate_positions(color_frame_bgr, depth_frame, coral_bboxes)

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
