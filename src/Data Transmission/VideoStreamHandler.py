from flask import Flask, Response
import threading
import cv2

class VideoStreamHandler:
    def __init__(self):
        self.app = Flask(__name__)
        self.frame_lock = threading.Lock()  # Ensures thread-safe access to the frame
        self.current_frame = None  # Stores the latest frame

        # Add routes
        self.app.add_url_rule("/camera-stream", view_func=self.video_feed, methods=["GET"])

    def run(self):
        """Start sending data to the server."""
        self.app.run(host="0.0.0.0", port=5000)

    def update_frame(self, new_frame):
        """
        Update the frame to be streamed.

        Params:
            new_frame: The new frame that will be displayed.
        """
        with self.frame_lock:
            self.current_frame = new_frame

    def generate_frames(self):
        """Generate frames to stream to the client."""
        while True:
            with self.frame_lock:
                if self.current_frame is None:
                    continue  # Wait until a frame is available

                # Convert the current frame into a jpg image.
                _, buffer = cv2.imencode('.jpg', self.current_frame)

                # Converts the encoded JPEG buffer into bytes for transmission over HTTP.
                frame = buffer.tobytes()
            
            # Sends chunks of data to the client. The format includes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    def video_feed(self):
        """Route to stream video frames."""
        
        # Stream output to the client.
        return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')