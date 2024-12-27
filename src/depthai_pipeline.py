import depthai as dai

def create_pipeline():
    """
    Creates and returns a DepthAI pipeline that streams both RGB and Depth frames.
    
    Returns:
        pipeline (dai.Pipeline): The DepthAI pipeline
    """
    pipeline = dai.Pipeline()

    # Create a pipeline
    pipeline = dai.Pipeline()

    # Create camera nodes.
    camera = pipeline.createColorCamera()
    camera.setPreviewSize(640, 480)  # Resolution of the preview
    camera.setInterleaved(False)     # Non-interleaved format for OpenCV compatibility
    camera.setFps(35)                # Frames per second

    # Create Mono cameras for depth calculation
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Create depth node
    depth = pipeline.createStereoDepth()
    left.out.link(depth.left)
    right.out.link(depth.right)

    # Create XLink output for video and depth streams
    xout_video = pipeline.createXLinkOut()
    xout_video.setStreamName("video")
    camera.preview.link(xout_video.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    depth.depth.link(xout_depth.input)

    # Return the newly created pipeline
    return pipeline