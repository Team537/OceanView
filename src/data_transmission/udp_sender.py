import socket
import json

class UDPSender:
    """
    A class to handle sending data via UDP, with built-in packet numbering
    for tracking packet loss.
    """

    def __init__(self, ip="0.0.0.0", port=5800):
        """
        Initializes the UDPSender.

        Params:
            ip: The IP address of the target (default: 0.0.0.0 for all interfaces).
            port: The port number on which the target is listening.
        """
        # Create a UDP socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Initialize packet counter to track the number of packets sent
        self.packet_number = 0

        # Store the target address (RoboRIO's IP and port)
        self.roborio_ip = ip
        self.port = port

    def build_upload_data(self, available, algae_blocked, algae_positions):
        """
        Build a dictionary with the three arrays stored in parallel.
        
        Params:
            available: a list of dicts, e.g., [{"branch": "B1", "level": "L2"}, ...]
            algae_blocked: a list of dicts, e.g., [{"branch": "B2", "level": "L3"}, ...]
            algae_positions: a list of dicts, e.g., [{"x": 1.23, "y": 0.45, "z": 2.34}, ...]
        Returns: 
            A dictionary ready for JSON encoding.
        """
        data = {
            "packet_number": self.packet_number,
            "available": available,
            "algae_blocked": algae_blocked,
            "algae_positions": algae_positions
        }
        self.packet_number += 1
        return data

    def upload_data(self, available, algae_blocked, algae_positions):
        """
        Sends data as a JSON-encoded UDP packet to the RoboRIO.
        
        Params:
            available: list of scoring positions available (each dict contains branch and level info).
            algae_blocked: list of scoring positions that are blocked by algae.
            algae_positions: list of raw algae positions (each dict contains x, y, z).
        """
        try:
            data = self.build_upload_data(available, algae_blocked, algae_positions)

            json_data = json.dumps(data)
            self.udp_socket.sendto(json_data.encode('utf-8'), (self.roborio_ip, self.port))
        except json.JSONDecodeError as e:
            print(f"Error encoding data to JSON: {e}")
        except Exception as e:
            print(f"Error sending data: {e}")

    def close(self):
        """
        Closes the UDP socket.
        """
        try:
            self.udp_socket.close()
            print("UDP socket closed successfully.")
        except Exception as e:
            print(f"Error closing UDP socket: {e}")
