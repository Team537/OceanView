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

    def upload_data(self, data):
        """
        Sends data as a JSON-encoded UDP packet to the target.

        Params:
            data: The data to send (must be serializable to JSON).
        """
        try:
            # Add the packet number to the data dictionary
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary to include packet_number.")
            data['packet_number'] = self.packet_number
            self.packet_number += 1

            # Serialize the data to a JSON string 
            json_data = json.dumps(data)

            # Send the JSON-encoded data as a UDP datagram
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
