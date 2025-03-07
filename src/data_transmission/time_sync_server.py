import socket
import json
import time
import threading

class TimeSyncServer:
    """
    TimeSyncServer listens for UDP time synchronization requests from the RoboRIO.
    For each request, it captures:
      - t2: The time (in ns) when the request was received.
      - t3: The time (in ns) immediately before sending the response.
    It responds with these timestamps as a JSON object.
    """
    
    def __init__(self, ip="0.0.0.0", port=6000):
        """
        Initializes the TimeSyncServer.
        
        :param ip: The IP address to bind to (default "0.0.0.0" binds to all interfaces).
        :param port: The UDP port number on which the server will listen.
        """
        self.ip = ip
        self.port = port
        self.running = False
        self.thread = None

    def start(self):
        """
        Starts the time synchronization server in a background thread.
        """
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="TimeSyncServerThread")
        self.thread.start()
        print(f"TimeSyncServer started on {self.ip}:{self.port}")

    def _run(self):
        """
        Main server loop that listens for UDP requests, captures timestamps (in ns),
        and sends a JSON response with t2 and t3.
        """
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind((self.ip, self.port))
        print("TimeSyncServer is now listening for UDP requests.")
        
        # Set a timeout to allow periodic checks of self.running
        udp_socket.settimeout(1.0)
        
        while self.running:
            try:
                data, addr = udp_socket.recvfrom(2048)
                # Capture the time immediately when the packet is received (T2)
                t2 = time.monotonic_ns()
                # (Processing of the packet can occur here, if needed)
                # Capture the time immediately before sending the response (T3)
                t3 = time.monotonic_ns()
                
                # Create a JSON response with t2 and t3
                response = {"t2": t2, "t3": t3}
                json_response = json.dumps(response)
                
                # Send the response back to the requester
                udp_socket.sendto(json_response.encode("utf-8"), addr)
                print(f"Responded to {addr} with {json_response}")
            except socket.timeout:
                continue  # Loop again to check self.running
            except Exception as e:
                print(f"Error in TimeSyncServer: {e}")
        
        udp_socket.close()
        print("TimeSyncServer socket closed.")

    def stop(self):
        """
        Stops the time synchronization server and waits for the thread to finish.
        """
        print("Stopping TimeSyncServer...")
        self.running = False
        if self.thread is not None:
            self.thread.join()
        print("TimeSyncServer stopped.")