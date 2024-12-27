import cv2
import numpy as np

# ------------------------------------
# Settings
# ------------------------------------
# HSV Thresholds
LOWER_BOUND = np.array([0, 0, 0])
UPPER_BOUND = np.array([255, 255, 255])

# Canny Edge Detection
LOWER_THRESHOLD = 400
UPPER_THRESHOLD = 500

def mask_image(frame):
    """
    Creates a mask to detect objects within the specified color range.
    
    Args:
        frame (numpy.ndarray): The HSV frame.
    
    Returns:
        numpy.ndarray: The binary mask of the detected color.
    """

    # Convert the image to HSV so that colors within the specified HSV range can be filtered out.
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold_image = cv2.inRange(hsv_image, LOWER_BOUND, UPPER_BOUND)

    # Filter out detection noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

    # Return the processed frame
    return threshold_image

def find_contours(mask):
    """
    Finds contours in a binary mask.
    
    Args:
        mask (numpy.ndarray): The binary mask.
    
    Returns:
        list: A list of contours.
    """

    # Find the edges in the image.
    edges = cv2.Canny(mask, LOWER_THRESHOLD, UPPER_BOUND)

    # Find the contours of the image.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # TODO: Add feature to filter our contours if they are too small.

    # Return the found contours.
    return contours

def extract_bounding_boxes(contours):
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

def draw_bounding_boxes(frame, bounding_boxes, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on the frame.
    
    Args:
        frame (numpy.ndarray): The frame to draw on.
        bounding_boxes (list): List of bounding boxes to draw.
        color (tuple): The color of the bounding boxes.
        thickness (int): The thickness of the bounding box lines.
    
    Returns:
        numpy.ndarray: The frame with bounding boxes drawn.
    """
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame

def process_frame(frame):
    """
    Full image processing pipeline: converts to HSV, creates a mask, detects contours, 
    and draws bounding boxes.
    
    Args:
        frame (numpy.ndarray): The RGB frame to process.
        lower_bound (numpy.ndarray): Lower HSV bound for the mask.
        upper_bound (numpy.ndarray): Upper HSV bound for the mask.
    
    Returns:
        numpy.ndarray: The processed frame with bounding boxes drawn.
        list: List of bounding boxes.
    """
    
    # Filter our the specified colors from the image.
    masked_image = mask_image(frame)

    # Find the contours of the image.
    contours = find_contours(masked_image)

    # Extract bounding boxes from the image.
    bounding_boxes = extract_bounding_boxes(contours)

    # Draw bounding boxes on the image.
    processed_frame = draw_bounding_boxes(frame, bounding_boxes)

    # Return the processed frame and found bounding boxes.
    return processed_frame


def calculate_distance(depth_frame, object_bbox):
    """
    Calculate the distance of an object from the camera using depth data.
    
    Args:
        depth_frame (numpy.ndarray): The depth frame to calculate distance from.
        object_bbox (tuple): The bounding box of the detected object (x, y, w, h).
    
    Returns:
        float: The average distance of the object from the camera in meters.
    """

    x, y, w, h = object_bbox
    # Calculate the center of the object (bounding box)
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Extract the depth value from the depth frame at the center of the object
    depth_value = depth_frame[center_y, center_x]
    
    # Convert the depth value from raw disparity to actual distance (this may vary based on your camera)
    # TODO: Research and tune this value.
    distance_in_meters = depth_value / 1000.0 
    
    return distance_in_meters