import depthai as dai
import numpy as np


class DepthAIPipeline:
    """
    This class creates and manages the DepthAI pipeline. It sets up a unified
    Camera node for the RGB (color) stream (with undistortion enabled) and two
    MonoCamera nodes for the left/right stereo pair used in depth calculation.
    """

    def __init__(self):
        # The DepthAI pipeline instance.
        self.pipeline = None
        # The DepthAI device instance.
        self.device = None
        # Output queues for color and depth frames.
        self.video_queue = None
        self.depth_queue = None

    def create_pipeline(self):
        """
        Create and return a DepthAI pipeline that streams both undistorted color (RGB)
        and depth frames.

        Returns:
            dai.Pipeline: Configured pipeline object.
        """
        # Create a new pipeline instance.
        pipeline = dai.Pipeline()

        # ---------------------------
        # Create the unified Camera node for RGB output.
        # ---------------------------
        # Use the unified "Camera" node instead of the older ColorCamera node.
        cam = pipeline.create(dai.node.Camera)
        # Set the board socket to the RGB camera.
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        # Set the sensor resolution (this is the size of the sensor output).
        # In this example, we set it to 1080x720.
        cam.setSize(1080, 720)
        # Set the preview size (the size of the output image) to match sensor size.
        cam.setPreviewSize(1080, 720)
        # Enable on-device undistortion using the calibration mesh.
        cam.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
        # Set the output frame rate.
        cam.setFps(35)

        # ---------------------------
        # Create MonoCamera nodes for stereo depth calculation.
        # ---------------------------
        left = pipeline.createMonoCamera()
        right = pipeline.createMonoCamera()
        # Assign the left and right board sockets.
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # ---------------------------
        # Create and configure the StereoDepth node.
        # ---------------------------
        depth = pipeline.createStereoDepth()
        # Link the output of the mono cameras to the depth node inputs.
        left.out.link(depth.left)
        right.out.link(depth.right)
        # Set a preset for high accuracy depth calculation.
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Set the input resolution used for depth calculation.
        depth.setInputResolution(640, 480)
        # Set the output resolution for the depth map.
        depth.setOutputSize(640, 480)
        # Enable on-device distortion correction for the stereo pair.
        depth.enableDistortionCorrection(True)

        # Optional: Adjust post-processing filters for the depth map.
        config = depth.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 25
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        depth.initialConfig.set(config)

        # ---------------------------
        # Create XLinkOut nodes to send the frames to the host.
        # ---------------------------
        # For the color (RGB) stream.
        xout_video = pipeline.createXLinkOut()
        xout_video.setStreamName("video")
        # Link the preview output of the camera to the XLink output.
        cam.preview.link(xout_video.input)

        # For the depth stream.
        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        depth.depth.link(xout_depth.input)

        # Return the completed pipeline.
        return pipeline

    def start_pipeline(self):
        """
        Initialize the DepthAI pipeline and start streaming frames.
        """
        print("Starting DepthAI Pipeline")
        # Create the pipeline.
        self.pipeline = self.create_pipeline()
        # Open the device with the created pipeline.
        self.device = dai.Device(self.pipeline)
        # Start the pipeline on the device.
        self.device.startPipeline()
        # Create output queues to fetch frames.
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        # Prime the video queue by retrieving one frame.
        self.video_queue.get().getRaw()

    def stop_pipeline(self):
        """
        Stop the pipeline and clean up device resources.
        """
        self.device = None

    def get_color_camera_intrinsics(self):
        """
        Retrieve the raw calibration intrinsics for the RGB camera.
        Note: These are for the raw sensor image, not the undistorted output.

        Returns:
            numpy.ndarray: The camera intrinsic matrix.
        """
        return np.array(self.device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.RGB))

    def get_color_camera_distortion(self):
        """
        Retrieve the distortion coefficients for the RGB camera.

        Returns:
            numpy.ndarray: Distortion coefficients.
        """
        return np.array(self.device.readCalibration().getDistortionCoefficients(dai.CameraBoardSocket.RGB))

    def get_frame(self):
        """
        Get the most recent undistorted color (RGB) frame.

        Returns:
            numpy.ndarray: The RGB image frame.
        """
        return self.video_queue.get().getCvFrame()

    def get_depth_frame(self):
        """
        Get the most recent depth frame.

        Returns:
            numpy.ndarray: The depth image frame.
        """
        return self.depth_queue.get().getCvFrame()