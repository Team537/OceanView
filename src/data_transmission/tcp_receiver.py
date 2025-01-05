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
        self.running = False  # Flag to control the server loop
        self.client_thread = None  # Thread for client handling

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
                    self.server_socket.settimeout(1.0)  # Prevent blocking indefinitely
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Connection received from {client_address}")

                    # Handle the client in a separate thread
                    self.client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket,), daemon=True
                    )
                    self.client_thread.start()
                except socket.timeout:
                    continue  # Allow loop to check the `running` flag
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
                    data = client_socket.recv(1024)  # Receive up to 1024 bytes
                    if not data:
                        print("Client disconnected.")
                        break

                    # Decode and process the received data
                    message = data.decode().strip()
                    try:
                        parsed_data = json.loads(message)  # Parse JSON
                        print(f"Received JSON: {parsed_data}")
                        self.process_data(parsed_data)  # Custom method to handle parsed data
                    except json.JSONDecodeError:
                        print(f"Invalid JSON received: {message}")

        except Exception as e:
            print(f"Error in client communication: {e}")
        finally:
            print("Client handling thread terminated.")

    def process_data(self, parsed_data):
        """
        Processes the parsed JSON data.
        Param:
            parsed_data: The parsed JSON data as a dictionary or list.
        """
        print(f"Processing data: {parsed_data}")

        # Save images to the file if told to do so.
        if parsed_data.capture is not None:
            self.main_controller.save_frames(parsed_data.capture.inputFrame, parsed_data.capture.outputFrame, parsed_data.capture.depthFrame)

        # Update the robot's position if told to do so.
        if parsed_data.robotPose is not None:
            print("Updating robot's position to: {parsed_data.robotPose}")
            self.main_controller.update_robot_pose(parsed_data.robotPose)

    def stop(self):
        """
        Stops the TCP server and releases all resources.
        """
        print("Stopping TCPReceiver...")
        self.running = False
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join()  # Wait for the client thread to finish
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                print(f"Error closing server socket: {e}")
        print("TCPReceiver stopped.")
