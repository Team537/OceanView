import socket
import threading
import json

class TCPReceiver:
    def __init__(self, main_controller, ip="0.0.0.0", port=5801):
        """
        Initializes the TCP server.

        Params:
            main_controller: The main controller running the program.
            ip: The IP address to bind to (default: all interfaces).
            port: The port number to listen on.
        """
        self.main_controller = main_controller
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.running = False
        self.client_thread = None

    def start(self):
        """
        Starts the TCP server to listen for incoming data.
        This method runs in a separate thread to allow graceful stopping.
        """
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(5)  # Allow up to 5 queued connections
        print(f"TCPReceiver started on {self.ip}:{self.port}")

        # Start the server loop in a separate thread
        threading.Thread(target=self.server_loop, daemon=True).start()

    def server_loop(self):
        """
        The main server loop that waits for and handles incoming connections.
        """
        try:
            while self.running:
                try:
                    # Timeout so we can check `self.running` periodically
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Connection received from {client_address}")

                    # Handle this client in a separate thread
                    self.client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket,), daemon=True
                    )
                    self.client_thread.start()

                except socket.timeout:
                    continue  # Just loop again and check self.running

        except Exception as e:
            print(f"Error in server loop: {e}")
        finally:
            self.stop()

    def handle_client(self, client_socket):
        """
        Handles communication with a connected client (e.g., RoboRIO).
        Expects JSON-formatted data and parses it.

        Params:
            client_socket: The socket object for the client connection.
        """
        try:
            with client_socket:
                while self.running:
                    data = client_socket.recv(1024)  # Up to 1024 bytes per chunk
                    if not data:
                        print("Client disconnected.")
                        break

                    # Decode (optionally specify UTF-8 explicitly): data.decode("utf-8").strip()
                    message = data.decode().strip()
                    try:
                        parsed_data = json.loads(message)
                        print(f"Received JSON: {parsed_data}")

                        # Process the parsed data
                        self.process_data(parsed_data)

                    except json.JSONDecodeError:
                        print(f"Invalid JSON received: {message}")

        except Exception as e:
            print(f"Error in client communication: {e}")
        finally:
            print("Client handling thread terminated.")

    def process_data(self, parsed_data):
        """
        Processes the parsed JSON data, calling main_controller methods as needed.

        :param parsed_data: Dictionary (or list) from the JSON.
        """
        # 1) If "capture" data is present, handle it
        if "capture" in parsed_data and parsed_data["capture"] is not None:
            # Example: the 'capture' might contain frames to be saved
            capture_info = parsed_data["capture"]
            # e.g. {"inputFrame": "...", "outputFrame": "...", "depthFrame": "..."}
            input_frame  = capture_info.get("inputFrame")
            output_frame = capture_info.get("outputFrame")
            depth_frame  = capture_info.get("depthFrame")

            self.main_controller.save_frames(
                input_frame, output_frame, depth_frame
            )

        # 2) If "robot_pose" is present, handle it
        #    This might match what your RoboRIO sends, e.g.:
        #    {
        #      "pose": {"x": 1.23, "y": 4.56, "heading_rad": 0.78},
        #      "timestamp": 1670000000.0
        #    }
        if "robot_pose" in parsed_data:
            robot_pose = parsed_data["robot_pose"]
            print(f"Updating robot's position to: {robot_pose}")
            # If your main_controller expects x, y, heading, do something like:
            self.main_controller.update_robot_pose(robot_pose)

        # 3) If the robot sends simpler keys, e.g. "x", "y", "heading":
        #    If your JSON is like: {"x": 1.23, "y": 4.56, "heading": 0.78}, use:
        """
        if "x" in parsed_data and "y" in parsed_data and "heading" in parsed_data:
            x = parsed_data["x"]
            y = parsed_data["y"]
            heading = parsed_data["heading"]
            self.main_controller.update_robot_pose({"x": x, "y": y, "heading": heading})
        """

    def stop(self):
        """
        Stops the TCP server and releases all resources.
        """
        print("Stopping TCPReceiver...")
        self.running = False
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join()
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                print(f"Error closing server socket: {e}")
        print("TCPReceiver stopped.")