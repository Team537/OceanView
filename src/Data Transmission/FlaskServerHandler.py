from flask import Flask, Response, jsonify, render_template  
import threading
import cv2
import psutil
import subprocess
import datetime

class FlaskServerHandler:
    def __init__(self):
        self.app = Flask(__name__)
        self.frame_lock = threading.Lock()  # Ensures thread-safe access to the frame
        self.current_frame = None  # Stores the latest frame

        # Add routes
        self.app.add_url_rule("/", view_func=self.index, methods=["GET"])
        self.app.errorhandler(404)(self.page_not_found)
        self.app.add_url_rule("/api/status", view_func=self.status, methods=["GET"])
        self.app.add_url_rule("/camera-stream", view_func=self.video_feed, methods=["GET"])

    def run(self):
        """Start sending data to the server."""
        self.app.run(host="0.0.0.0", port=5000)

    # --- Status Page Functionality --- #
    def index(self):
        return render_template("index.html")

    def page_not_found(self, e):
        return render_template('404.html'), 404

    def get_gpu_info(self):
        try:
            # Get GPU memory usage and clean the output
            gpu_mem_raw = subprocess.check_output(['vcgencmd', 'get_mem', 'gpu']).decode('utf-8').strip()
            gpu_mem = gpu_mem_raw.split('=')[1].replace('M', '')  # Get the memory in MB and remove 'M'

            # Get GPU temperature and clean the output
            gpu_temp_raw = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8').strip()
            gpu_temp = gpu_temp_raw.split('=')[1].replace("'", '')  # Remove the 'C'

            return {
                'gpu_mem': gpu_mem,  # In MB
                'gpu_temp': gpu_temp,  # In Celsius
            }
        except Exception as e:
            return {'error': str(e)}

    def get_cpu_temperature(self):
        try:
            # Get CPU temperature in Celsius
            temp = float(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
            return round(temp, 2)
        except Exception as e:
            return f"Error: {e}"

    def get_uptime(self):
        try:
            with open("/proc/uptime", "r") as f:
                uptime_seconds = float(f.readline().split()[0])
                # Convert uptime to hours, minutes, and seconds
                uptime = str(datetime.timedelta(seconds=uptime_seconds))
                return uptime
        except Exception as e:
            return f"Error: {e}"

    def get_network_usage(self):
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent / (1024 ** 2),  # MB
                'bytes_recv': net_io.bytes_recv / (1024 ** 2)   # MB
            }
        except Exception as e:
            return f"Error: {e}"

    def get_disk_health(self):
        try:
            disk_io = psutil.disk_io_counters()
            return {
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
        except Exception as e:
            return f"Error: {e}"

    def get_system_load(self):
        try:
            load = psutil.getloadavg()
            return {
                '1min': load[0],
                '5min': load[1],
                '15min': load[2]
            }
        except Exception as e:
            return f"Error: {e}"

    def get_swap_usage(self):
        try:
            swap = psutil.swap_memory()
            return {
                'used': swap.used / (1024 ** 3),  # GB
                'free': swap.free / (1024 ** 3),  # GB
                'percent': swap.percent
            }
        except Exception as e:
            return f"Error: {e}"

    def get_top_cpu_processes(self):
        try:
            # Get all processes and their CPU usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    # Append process info if it has valid CPU usage
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Skip processes that no longer exist or cannot be accessed
                    pass

            # Sort processes by CPU usage in descending order
            processes = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)

            # Get top 10 CPU-intensive processes
            top_processes = processes[:10]
            return top_processes
        except Exception as e:
            return {'error': str(e)}

    def terminal_output(self):
        try:
            with open("/home/camer/logs/output.log", 'r') as file: # NOTE: This will vary based on system, custom log file.

                # Read all lines from the file and get the last 25 lines
                lines = file.readlines()[-25:]

                # Join the lines with <br> for line breaks
                return "<br>".join([line.strip() for line in lines])
        except Exception as e:
            
            # Return the error message with <br> for line breaks
            return f"Error reading log file: {str(e)}".replace("\n", "<br>")
    
    def status(self):

        # Example data for testing
        data = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_used': psutil.virtual_memory().used / (1024 ** 3),
            'memory_total': psutil.virtual_memory().total / (1024 ** 3),
            'disk_used': psutil.disk_usage('/').used / (1024 ** 3),
            'disk_total': psutil.disk_usage('/').total / (1024 ** 3),
            'gpu_info': self.get_gpu_info(),
            'temperature': self.get_cpu_temperature(),
            'network_usage': self.get_network_usage(),
            'uptime': self.get_uptime(),
            'disk_health': self.get_disk_health(),
            'system_load': self.get_system_load(),
            'swap_usage': self.get_swap_usage(),
            'process_info': self.get_top_cpu_processes(),
            'terminal_output': self.terminal_output()
        }
        return jsonify(data)
    
    # --- Vision Processing Functionality --- #
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