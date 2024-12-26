
from pathlib import Path
from typing import Set
import datetime
import depthai
from typing import Any, ClassVar, overload

class AprilTag(depthai.Node):
    Properties: ClassVar[type[depthai.AprilTagProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setWaitForConfigInput(self, wait: bool) -> None:
        """setWaitForConfigInput(self: depthai.node.AprilTag, wait: bool) -> None

        Specify whether or not wait until configuration message arrives to inputConfig
        Input.

        Parameter ``wait``:
            True to wait for configuration message, false otherwise.
        """
    @property
    def initialConfig(self) -> depthai.AprilTagConfig: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputImage(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...
    @property
    def passthroughInputImage(self) -> depthai.Node.Output: ...

class Camera(depthai.Node):
    Properties: ClassVar[type[depthai.CameraProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getBoardSocket(self) -> depthai.CameraBoardSocket:
        """getBoardSocket(self: depthai.node.Camera) -> depthai.CameraBoardSocket

        Retrieves which board socket to use

        Returns:
            Board socket to use
        """
    def getCalibrationAlpha(self) -> float | None:
        """getCalibrationAlpha(self: depthai.node.Camera) -> Optional[float]

        Get calibration alpha parameter that determines FOV of undistorted frames
        """
    def getCamera(self) -> str:
        """getCamera(self: depthai.node.Camera) -> str

        Retrieves which camera to use by name

        Returns:
            Name of the camera to use
        """
    def getFps(self) -> float:
        """getFps(self: depthai.node.Camera) -> float

        Get rate at which camera should produce frames

        Returns:
            Rate in frames per second
        """
    def getHeight(self) -> int:
        """getHeight(self: depthai.node.Camera) -> int

        Get sensor resolution height
        """
    def getImageOrientation(self) -> depthai.CameraImageOrientation:
        """getImageOrientation(self: depthai.node.Camera) -> depthai.CameraImageOrientation

        Get camera image orientation
        """
    def getMeshSource(self) -> depthai.CameraProperties.WarpMeshSource:
        """getMeshSource(self: depthai.node.Camera) -> depthai.CameraProperties.WarpMeshSource

        Gets the source of the warp mesh
        """
    def getMeshStep(self) -> tuple[int, int]:
        """getMeshStep(self: depthai.node.Camera) -> tuple[int, int]

        Gets the distance between mesh points
        """
    def getPreviewHeight(self) -> int:
        """getPreviewHeight(self: depthai.node.Camera) -> int

        Get preview height
        """
    def getPreviewSize(self) -> tuple[int, int]:
        """getPreviewSize(self: depthai.node.Camera) -> tuple[int, int]

        Get preview size as tuple
        """
    def getPreviewWidth(self) -> int:
        """getPreviewWidth(self: depthai.node.Camera) -> int

        Get preview width
        """
    def getSize(self) -> tuple[int, int]:
        """getSize(self: depthai.node.Camera) -> tuple[int, int]

        Get sensor resolution as size
        """
    def getStillHeight(self) -> int:
        """getStillHeight(self: depthai.node.Camera) -> int

        Get still height
        """
    def getStillSize(self) -> tuple[int, int]:
        """getStillSize(self: depthai.node.Camera) -> tuple[int, int]

        Get still size as tuple
        """
    def getStillWidth(self) -> int:
        """getStillWidth(self: depthai.node.Camera) -> int

        Get still width
        """
    def getVideoHeight(self) -> int:
        """getVideoHeight(self: depthai.node.Camera) -> int

        Get video height
        """
    def getVideoSize(self) -> tuple[int, int]:
        """getVideoSize(self: depthai.node.Camera) -> tuple[int, int]

        Get video size as tuple
        """
    def getVideoWidth(self) -> int:
        """getVideoWidth(self: depthai.node.Camera) -> int

        Get video width
        """
    def getWidth(self) -> int:
        """getWidth(self: depthai.node.Camera) -> int

        Get sensor resolution width
        """
    def loadMeshData(self, warpMesh: buffer) -> None:
        """loadMeshData(self: depthai.node.Camera, warpMesh: buffer) -> None

        Specify mesh calibration data for undistortion See `loadMeshFiles` for the
        expected data format
        """
    def loadMeshFile(self, warpMesh: Path) -> None:
        """loadMeshFile(self: depthai.node.Camera, warpMesh: Path) -> None

        Specify local filesystem paths to the undistort mesh calibration files.

        When a mesh calibration is set, it overrides the camera intrinsics/extrinsics
        matrices. Overrides useHomographyRectification behavior. Mesh format: a sequence
        of (y,x) points as 'float' with coordinates from the input image to be mapped in
        the output. The mesh can be subsampled, configured by `setMeshStep`.

        With a 1280x800 resolution and the default (16,16) step, the required mesh size
        is:

        width: 1280 / 16 + 1 = 81

        height: 800 / 16 + 1 = 51
        """
    def setBoardSocket(self, boardSocket: depthai.CameraBoardSocket) -> None:
        """setBoardSocket(self: depthai.node.Camera, boardSocket: depthai.CameraBoardSocket) -> None

        Specify which board socket to use

        Parameter ``boardSocket``:
            Board socket to use
        """
    def setCalibrationAlpha(self, alpha: float) -> None:
        """setCalibrationAlpha(self: depthai.node.Camera, alpha: float) -> None

        Set calibration alpha parameter that determines FOV of undistorted frames
        """
    def setCamera(self, name: str) -> None:
        """setCamera(self: depthai.node.Camera, name: str) -> None

        Specify which camera to use by name

        Parameter ``name``:
            Name of the camera to use
        """
    def setFps(self, fps: float) -> None:
        """setFps(self: depthai.node.Camera, fps: float) -> None

        Set rate at which camera should produce frames

        Parameter ``fps``:
            Rate in frames per second
        """
    def setImageOrientation(self, imageOrientation: depthai.CameraImageOrientation) -> None:
        """setImageOrientation(self: depthai.node.Camera, imageOrientation: depthai.CameraImageOrientation) -> None

        Set camera image orientation
        """
    def setIsp3aFps(self, isp3aFps: int) -> None:
        """setIsp3aFps(self: depthai.node.Camera, isp3aFps: int) -> None

        Isp 3A rate (auto focus, auto exposure, auto white balance, camera controls
        etc.). Default (0) matches the camera FPS, meaning that 3A is running on each
        frame. Reducing the rate of 3A reduces the CPU usage on CSS, but also increases
        the convergence rate of 3A. Note that camera controls will be processed at this
        rate. E.g. if camera is running at 30 fps, and camera control is sent at every
        frame, but 3A fps is set to 15, the camera control messages will be processed at
        15 fps rate, which will lead to queueing.
        """
    def setMeshSource(self, source: depthai.CameraProperties.WarpMeshSource) -> None:
        """setMeshSource(self: depthai.node.Camera, source: depthai.CameraProperties.WarpMeshSource) -> None

        Set the source of the warp mesh or disable
        """
    def setMeshStep(self, width: int, height: int) -> None:
        """setMeshStep(self: depthai.node.Camera, width: int, height: int) -> None

        Set the distance between mesh points. Default: (32, 32)
        """
    @overload
    def setPreviewSize(self, width: int, height: int) -> None:
        """setPreviewSize(*args, **kwargs)
        Overloaded function.

        1. setPreviewSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set preview output size

        2. setPreviewSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set preview output size, as a tuple <width, height>
        """
    @overload
    def setPreviewSize(self, size: tuple[int, int]) -> None:
        """setPreviewSize(*args, **kwargs)
        Overloaded function.

        1. setPreviewSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set preview output size

        2. setPreviewSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set preview output size, as a tuple <width, height>
        """
    def setRawOutputPacked(self, packed: bool) -> None:
        """setRawOutputPacked(self: depthai.node.Camera, packed: bool) -> None

        Configures whether the camera `raw` frames are saved as MIPI-packed to memory.
        The packed format is more efficient, consuming less memory on device, and less
        data to send to host: RAW10: 4 pixels saved on 5 bytes, RAW12: 2 pixels saved on
        3 bytes. When packing is disabled (`false`), data is saved lsb-aligned, e.g. a
        RAW10 pixel will be stored as uint16, on bits 9..0: 0b0000'00pp'pppp'pppp.
        Default is auto: enabled for standard color/monochrome cameras where ISP can
        work with both packed/unpacked, but disabled for other cameras like ToF.
        """
    @overload
    def setSize(self, width: int, height: int) -> None:
        """setSize(*args, **kwargs)
        Overloaded function.

        1. setSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set desired resolution. Sets sensor size to best fit

        2. setSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set desired resolution. Sets sensor size to best fit
        """
    @overload
    def setSize(self, size: tuple[int, int]) -> None:
        """setSize(*args, **kwargs)
        Overloaded function.

        1. setSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set desired resolution. Sets sensor size to best fit

        2. setSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set desired resolution. Sets sensor size to best fit
        """
    @overload
    def setStillSize(self, width: int, height: int) -> None:
        """setStillSize(*args, **kwargs)
        Overloaded function.

        1. setStillSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set still output size

        2. setStillSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set still output size, as a tuple <width, height>
        """
    @overload
    def setStillSize(self, size: tuple[int, int]) -> None:
        """setStillSize(*args, **kwargs)
        Overloaded function.

        1. setStillSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set still output size

        2. setStillSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set still output size, as a tuple <width, height>
        """
    @overload
    def setVideoSize(self, width: int, height: int) -> None:
        """setVideoSize(*args, **kwargs)
        Overloaded function.

        1. setVideoSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set video output size

        2. setVideoSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set video output size, as a tuple <width, height>
        """
    @overload
    def setVideoSize(self, size: tuple[int, int]) -> None:
        """setVideoSize(*args, **kwargs)
        Overloaded function.

        1. setVideoSize(self: depthai.node.Camera, width: int, height: int) -> None

        Set video output size

        2. setVideoSize(self: depthai.node.Camera, size: tuple[int, int]) -> None

        Set video output size, as a tuple <width, height>
        """
    @property
    def frameEvent(self) -> depthai.Node.Output: ...
    @property
    def initialControl(self) -> depthai.CameraControl: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputControl(self) -> depthai.Node.Input: ...
    @property
    def isp(self) -> depthai.Node.Output: ...
    @property
    def preview(self) -> depthai.Node.Output: ...
    @property
    def raw(self) -> depthai.Node.Output: ...
    @property
    def still(self) -> depthai.Node.Output: ...
    @property
    def video(self) -> depthai.Node.Output: ...

class Cast(depthai.Node):
    Properties: ClassVar[type[depthai.CastProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setNumFramesPool(self, arg0: int) -> Cast:
        """setNumFramesPool(self: depthai.node.Cast, arg0: int) -> depthai.node.Cast

        Set number of frames in pool

        Parameter ``numFramesPool``:
            Number of frames in pool
        """
    def setOffset(self, arg0: float) -> Cast:
        """setOffset(self: depthai.node.Cast, arg0: float) -> depthai.node.Cast

        Set offset

        Parameter ``offset``:
            Offset
        """
    def setOutputFrameType(self, arg0: depthai.RawImgFrame.Type) -> Cast:
        """setOutputFrameType(self: depthai.node.Cast, arg0: depthai.RawImgFrame.Type) -> depthai.node.Cast

        Set output frame type

        Parameter ``outputType``:
            Output frame type
        """
    def setScale(self, arg0: float) -> Cast:
        """setScale(self: depthai.node.Cast, arg0: float) -> depthai.node.Cast

        Set scale

        Parameter ``scale``:
            Scale
        """
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def output(self) -> depthai.Node.Output: ...
    @property
    def passthroughInput(self) -> depthai.Node.Output: ...

class ColorCamera(depthai.Node):
    Properties: ClassVar[type[depthai.ColorCameraProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getBoardSocket(self) -> depthai.CameraBoardSocket:
        """getBoardSocket(self: depthai.node.ColorCamera) -> depthai.CameraBoardSocket

        Retrieves which board socket to use

        Returns:
            Board socket to use
        """
    def getCamId(self) -> int:
        """getCamId(self: depthai.node.ColorCamera) -> int"""
    def getCamera(self) -> str:
        """getCamera(self: depthai.node.ColorCamera) -> str

        Retrieves which camera to use by name

        Returns:
            Name of the camera to use
        """
    def getColorOrder(self) -> depthai.ColorCameraProperties.ColorOrder:
        """getColorOrder(self: depthai.node.ColorCamera) -> depthai.ColorCameraProperties.ColorOrder

        Get color order of preview output frames. RGB or BGR
        """
    def getFp16(self) -> bool:
        """getFp16(self: depthai.node.ColorCamera) -> bool

        Get fp16 (0..255) data of preview output frames
        """
    def getFps(self) -> float:
        """getFps(self: depthai.node.ColorCamera) -> float

        Get rate at which camera should produce frames

        Returns:
            Rate in frames per second
        """
    def getFrameEventFilter(self) -> list[depthai.FrameEvent]:
        """getFrameEventFilter(self: depthai.node.ColorCamera) -> list[depthai.FrameEvent]"""
    def getImageOrientation(self) -> depthai.CameraImageOrientation:
        """getImageOrientation(self: depthai.node.ColorCamera) -> depthai.CameraImageOrientation

        Get camera image orientation
        """
    def getInterleaved(self) -> bool:
        """getInterleaved(self: depthai.node.ColorCamera) -> bool

        Get planar or interleaved data of preview output frames
        """
    def getIspHeight(self) -> int:
        """getIspHeight(self: depthai.node.ColorCamera) -> int

        Get 'isp' output height
        """
    def getIspNumFramesPool(self) -> int:
        """getIspNumFramesPool(self: depthai.node.ColorCamera) -> int

        Get number of frames in isp pool
        """
    def getIspSize(self) -> tuple[int, int]:
        """getIspSize(self: depthai.node.ColorCamera) -> tuple[int, int]

        Get 'isp' output resolution as size, after scaling
        """
    def getIspWidth(self) -> int:
        """getIspWidth(self: depthai.node.ColorCamera) -> int

        Get 'isp' output width
        """
    def getPreviewHeight(self) -> int:
        """getPreviewHeight(self: depthai.node.ColorCamera) -> int

        Get preview height
        """
    def getPreviewKeepAspectRatio(self) -> bool:
        """getPreviewKeepAspectRatio(self: depthai.node.ColorCamera) -> bool

        See also:
            setPreviewKeepAspectRatio

        Returns:
            Preview keep aspect ratio option
        """
    def getPreviewNumFramesPool(self) -> int:
        """getPreviewNumFramesPool(self: depthai.node.ColorCamera) -> int

        Get number of frames in preview pool
        """
    def getPreviewSize(self) -> tuple[int, int]:
        """getPreviewSize(self: depthai.node.ColorCamera) -> tuple[int, int]

        Get preview size as tuple
        """
    def getPreviewWidth(self) -> int:
        """getPreviewWidth(self: depthai.node.ColorCamera) -> int

        Get preview width
        """
    def getRawNumFramesPool(self) -> int:
        """getRawNumFramesPool(self: depthai.node.ColorCamera) -> int

        Get number of frames in raw pool
        """
    def getResolution(self) -> depthai.ColorCameraProperties.SensorResolution:
        """getResolution(self: depthai.node.ColorCamera) -> depthai.ColorCameraProperties.SensorResolution

        Get sensor resolution
        """
    def getResolutionHeight(self) -> int:
        """getResolutionHeight(self: depthai.node.ColorCamera) -> int

        Get sensor resolution height
        """
    def getResolutionSize(self) -> tuple[int, int]:
        """getResolutionSize(self: depthai.node.ColorCamera) -> tuple[int, int]

        Get sensor resolution as size
        """
    def getResolutionWidth(self) -> int:
        """getResolutionWidth(self: depthai.node.ColorCamera) -> int

        Get sensor resolution width
        """
    def getSensorCrop(self) -> tuple[float, float]:
        """getSensorCrop(self: depthai.node.ColorCamera) -> tuple[float, float]

        Returns:
            Sensor top left crop coordinates
        """
    def getSensorCropX(self) -> float:
        """getSensorCropX(self: depthai.node.ColorCamera) -> float

        Get sensor top left x crop coordinate
        """
    def getSensorCropY(self) -> float:
        """getSensorCropY(self: depthai.node.ColorCamera) -> float

        Get sensor top left y crop coordinate
        """
    def getStillHeight(self) -> int:
        """getStillHeight(self: depthai.node.ColorCamera) -> int

        Get still height
        """
    def getStillNumFramesPool(self) -> int:
        """getStillNumFramesPool(self: depthai.node.ColorCamera) -> int

        Get number of frames in still pool
        """
    def getStillSize(self) -> tuple[int, int]:
        """getStillSize(self: depthai.node.ColorCamera) -> tuple[int, int]

        Get still size as tuple
        """
    def getStillWidth(self) -> int:
        """getStillWidth(self: depthai.node.ColorCamera) -> int

        Get still width
        """
    def getVideoHeight(self) -> int:
        """getVideoHeight(self: depthai.node.ColorCamera) -> int

        Get video height
        """
    def getVideoNumFramesPool(self) -> int:
        """getVideoNumFramesPool(self: depthai.node.ColorCamera) -> int

        Get number of frames in video pool
        """
    def getVideoSize(self) -> tuple[int, int]:
        """getVideoSize(self: depthai.node.ColorCamera) -> tuple[int, int]

        Get video size as tuple
        """
    def getVideoWidth(self) -> int:
        """getVideoWidth(self: depthai.node.ColorCamera) -> int

        Get video width
        """
    def getWaitForConfigInput(self) -> bool:
        """getWaitForConfigInput(self: depthai.node.ColorCamera) -> bool

        See also:
            setWaitForConfigInput

        Returns:
            True if wait for inputConfig message, false otherwise
        """
    def sensorCenterCrop(self) -> None:
        """sensorCenterCrop(self: depthai.node.ColorCamera) -> None

        Specify sensor center crop. Resolution size / video size
        """
    def setBoardSocket(self, boardSocket: depthai.CameraBoardSocket) -> None:
        """setBoardSocket(self: depthai.node.ColorCamera, boardSocket: depthai.CameraBoardSocket) -> None

        Specify which board socket to use

        Parameter ``boardSocket``:
            Board socket to use
        """
    def setCamId(self, arg0: int) -> None:
        """setCamId(self: depthai.node.ColorCamera, arg0: int) -> None"""
    def setCamera(self, name: str) -> None:
        """setCamera(self: depthai.node.ColorCamera, name: str) -> None

        Specify which camera to use by name

        Parameter ``name``:
            Name of the camera to use
        """
    def setColorOrder(self, colorOrder: depthai.ColorCameraProperties.ColorOrder) -> None:
        """setColorOrder(self: depthai.node.ColorCamera, colorOrder: depthai.ColorCameraProperties.ColorOrder) -> None

        Set color order of preview output images. RGB or BGR
        """
    def setFp16(self, fp16: bool) -> None:
        """setFp16(self: depthai.node.ColorCamera, fp16: bool) -> None

        Set fp16 (0..255) data type of preview output frames
        """
    def setFps(self, fps: float) -> None:
        """setFps(self: depthai.node.ColorCamera, fps: float) -> None

        Set rate at which camera should produce frames

        Parameter ``fps``:
            Rate in frames per second
        """
    def setFrameEventFilter(self, events: list[depthai.FrameEvent]) -> None:
        """setFrameEventFilter(self: depthai.node.ColorCamera, events: list[depthai.FrameEvent]) -> None"""
    def setImageOrientation(self, imageOrientation: depthai.CameraImageOrientation) -> None:
        """setImageOrientation(self: depthai.node.ColorCamera, imageOrientation: depthai.CameraImageOrientation) -> None

        Set camera image orientation
        """
    def setInterleaved(self, interleaved: bool) -> None:
        """setInterleaved(self: depthai.node.ColorCamera, interleaved: bool) -> None

        Set planar or interleaved data of preview output frames
        """
    def setIsp3aFps(self, isp3aFps: int) -> None:
        """setIsp3aFps(self: depthai.node.ColorCamera, isp3aFps: int) -> None

        Isp 3A rate (auto focus, auto exposure, auto white balance, camera controls
        etc.). Default (0) matches the camera FPS, meaning that 3A is running on each
        frame. Reducing the rate of 3A reduces the CPU usage on CSS, but also increases
        the convergence rate of 3A. Note that camera controls will be processed at this
        rate. E.g. if camera is running at 30 fps, and camera control is sent at every
        frame, but 3A fps is set to 15, the camera control messages will be processed at
        15 fps rate, which will lead to queueing.
        """
    def setIspNumFramesPool(self, arg0: int) -> None:
        """setIspNumFramesPool(self: depthai.node.ColorCamera, arg0: int) -> None

        Set number of frames in isp pool
        """
    @overload
    def setIspScale(self, numerator: int, denominator: int) -> None:
        """setIspScale(*args, **kwargs)
        Overloaded function.

        1. setIspScale(self: depthai.node.ColorCamera, numerator: int, denominator: int) -> None

        Set 'isp' output scaling (numerator/denominator), preserving the aspect ratio.
        The fraction numerator/denominator is simplified first to a irreducible form,
        then a set of hardware scaler constraints applies: max numerator = 16, max
        denominator = 63

        2. setIspScale(self: depthai.node.ColorCamera, scale: tuple[int, int]) -> None

        Set 'isp' output scaling, as a tuple <numerator, denominator>

        3. setIspScale(self: depthai.node.ColorCamera, horizNum: int, horizDenom: int, vertNum: int, vertDenom: int) -> None

        Set 'isp' output scaling, per each direction. If the horizontal scaling factor
        (horizNum/horizDen) is different than the vertical scaling factor
        (vertNum/vertDen), a distorted (stretched or squished) image is generated

        4. setIspScale(self: depthai.node.ColorCamera, horizScale: tuple[int, int], vertScale: tuple[int, int]) -> None

        Set 'isp' output scaling, per each direction, as <numerator, denominator> tuples
        """
    @overload
    def setIspScale(self, scale: tuple[int, int]) -> None:
        """setIspScale(*args, **kwargs)
        Overloaded function.

        1. setIspScale(self: depthai.node.ColorCamera, numerator: int, denominator: int) -> None

        Set 'isp' output scaling (numerator/denominator), preserving the aspect ratio.
        The fraction numerator/denominator is simplified first to a irreducible form,
        then a set of hardware scaler constraints applies: max numerator = 16, max
        denominator = 63

        2. setIspScale(self: depthai.node.ColorCamera, scale: tuple[int, int]) -> None

        Set 'isp' output scaling, as a tuple <numerator, denominator>

        3. setIspScale(self: depthai.node.ColorCamera, horizNum: int, horizDenom: int, vertNum: int, vertDenom: int) -> None

        Set 'isp' output scaling, per each direction. If the horizontal scaling factor
        (horizNum/horizDen) is different than the vertical scaling factor
        (vertNum/vertDen), a distorted (stretched or squished) image is generated

        4. setIspScale(self: depthai.node.ColorCamera, horizScale: tuple[int, int], vertScale: tuple[int, int]) -> None

        Set 'isp' output scaling, per each direction, as <numerator, denominator> tuples
        """
    @overload
    def setIspScale(self, horizNum: int, horizDenom: int, vertNum: int, vertDenom: int) -> None:
        """setIspScale(*args, **kwargs)
        Overloaded function.

        1. setIspScale(self: depthai.node.ColorCamera, numerator: int, denominator: int) -> None

        Set 'isp' output scaling (numerator/denominator), preserving the aspect ratio.
        The fraction numerator/denominator is simplified first to a irreducible form,
        then a set of hardware scaler constraints applies: max numerator = 16, max
        denominator = 63

        2. setIspScale(self: depthai.node.ColorCamera, scale: tuple[int, int]) -> None

        Set 'isp' output scaling, as a tuple <numerator, denominator>

        3. setIspScale(self: depthai.node.ColorCamera, horizNum: int, horizDenom: int, vertNum: int, vertDenom: int) -> None

        Set 'isp' output scaling, per each direction. If the horizontal scaling factor
        (horizNum/horizDen) is different than the vertical scaling factor
        (vertNum/vertDen), a distorted (stretched or squished) image is generated

        4. setIspScale(self: depthai.node.ColorCamera, horizScale: tuple[int, int], vertScale: tuple[int, int]) -> None

        Set 'isp' output scaling, per each direction, as <numerator, denominator> tuples
        """
    @overload
    def setIspScale(self, horizScale: tuple[int, int], vertScale: tuple[int, int]) -> None:
        """setIspScale(*args, **kwargs)
        Overloaded function.

        1. setIspScale(self: depthai.node.ColorCamera, numerator: int, denominator: int) -> None

        Set 'isp' output scaling (numerator/denominator), preserving the aspect ratio.
        The fraction numerator/denominator is simplified first to a irreducible form,
        then a set of hardware scaler constraints applies: max numerator = 16, max
        denominator = 63

        2. setIspScale(self: depthai.node.ColorCamera, scale: tuple[int, int]) -> None

        Set 'isp' output scaling, as a tuple <numerator, denominator>

        3. setIspScale(self: depthai.node.ColorCamera, horizNum: int, horizDenom: int, vertNum: int, vertDenom: int) -> None

        Set 'isp' output scaling, per each direction. If the horizontal scaling factor
        (horizNum/horizDen) is different than the vertical scaling factor
        (vertNum/vertDen), a distorted (stretched or squished) image is generated

        4. setIspScale(self: depthai.node.ColorCamera, horizScale: tuple[int, int], vertScale: tuple[int, int]) -> None

        Set 'isp' output scaling, per each direction, as <numerator, denominator> tuples
        """
    def setNumFramesPool(self, raw: int, isp: int, preview: int, video: int, still: int) -> None:
        """setNumFramesPool(self: depthai.node.ColorCamera, raw: int, isp: int, preview: int, video: int, still: int) -> None

        Set number of frames in all pools
        """
    def setPreviewKeepAspectRatio(self, keep: bool) -> None:
        """setPreviewKeepAspectRatio(self: depthai.node.ColorCamera, keep: bool) -> None

        Specifies whether preview output should preserve aspect ratio, after downscaling
        from video size or not.

        Parameter ``keep``:
            If true, a larger crop region will be considered to still be able to create
            the final image in the specified aspect ratio. Otherwise video size is
            resized to fit preview size
        """
    def setPreviewNumFramesPool(self, arg0: int) -> None:
        """setPreviewNumFramesPool(self: depthai.node.ColorCamera, arg0: int) -> None

        Set number of frames in preview pool
        """
    @overload
    def setPreviewSize(self, width: int, height: int) -> None:
        """setPreviewSize(*args, **kwargs)
        Overloaded function.

        1. setPreviewSize(self: depthai.node.ColorCamera, width: int, height: int) -> None

        Set preview output size

        2. setPreviewSize(self: depthai.node.ColorCamera, size: tuple[int, int]) -> None

        Set preview output size, as a tuple <width, height>
        """
    @overload
    def setPreviewSize(self, size: tuple[int, int]) -> None:
        """setPreviewSize(*args, **kwargs)
        Overloaded function.

        1. setPreviewSize(self: depthai.node.ColorCamera, width: int, height: int) -> None

        Set preview output size

        2. setPreviewSize(self: depthai.node.ColorCamera, size: tuple[int, int]) -> None

        Set preview output size, as a tuple <width, height>
        """
    def setRawNumFramesPool(self, arg0: int) -> None:
        """setRawNumFramesPool(self: depthai.node.ColorCamera, arg0: int) -> None

        Set number of frames in raw pool
        """
    def setRawOutputPacked(self, packed: bool) -> None:
        """setRawOutputPacked(self: depthai.node.ColorCamera, packed: bool) -> None

        Configures whether the camera `raw` frames are saved as MIPI-packed to memory.
        The packed format is more efficient, consuming less memory on device, and less
        data to send to host: RAW10: 4 pixels saved on 5 bytes, RAW12: 2 pixels saved on
        3 bytes. When packing is disabled (`false`), data is saved lsb-aligned, e.g. a
        RAW10 pixel will be stored as uint16, on bits 9..0: 0b0000'00pp'pppp'pppp.
        Default is auto: enabled for standard color/monochrome cameras where ISP can
        work with both packed/unpacked, but disabled for other cameras like ToF.
        """
    def setResolution(self, resolution: depthai.ColorCameraProperties.SensorResolution) -> None:
        """setResolution(self: depthai.node.ColorCamera, resolution: depthai.ColorCameraProperties.SensorResolution) -> None

        Set sensor resolution
        """
    def setSensorCrop(self, x: float, y: float) -> None:
        """setSensorCrop(self: depthai.node.ColorCamera, x: float, y: float) -> None

        Specifies the cropping that happens when converting ISP to video output. By
        default, video will be center cropped from the ISP output. Note that this
        doesn't actually do on-sensor cropping (and MIPI-stream only that region), but
        it does postprocessing on the ISP (on RVC).

        Parameter ``x``:
            Top left X coordinate

        Parameter ``y``:
            Top left Y coordinate
        """
    def setStillNumFramesPool(self, arg0: int) -> None:
        """setStillNumFramesPool(self: depthai.node.ColorCamera, arg0: int) -> None

        Set number of frames in preview pool
        """
    @overload
    def setStillSize(self, width: int, height: int) -> None:
        """setStillSize(*args, **kwargs)
        Overloaded function.

        1. setStillSize(self: depthai.node.ColorCamera, width: int, height: int) -> None

        Set still output size

        2. setStillSize(self: depthai.node.ColorCamera, size: tuple[int, int]) -> None

        Set still output size, as a tuple <width, height>
        """
    @overload
    def setStillSize(self, size: tuple[int, int]) -> None:
        """setStillSize(*args, **kwargs)
        Overloaded function.

        1. setStillSize(self: depthai.node.ColorCamera, width: int, height: int) -> None

        Set still output size

        2. setStillSize(self: depthai.node.ColorCamera, size: tuple[int, int]) -> None

        Set still output size, as a tuple <width, height>
        """
    def setVideoNumFramesPool(self, arg0: int) -> None:
        """setVideoNumFramesPool(self: depthai.node.ColorCamera, arg0: int) -> None

        Set number of frames in preview pool
        """
    @overload
    def setVideoSize(self, width: int, height: int) -> None:
        """setVideoSize(*args, **kwargs)
        Overloaded function.

        1. setVideoSize(self: depthai.node.ColorCamera, width: int, height: int) -> None

        Set video output size

        2. setVideoSize(self: depthai.node.ColorCamera, size: tuple[int, int]) -> None

        Set video output size, as a tuple <width, height>
        """
    @overload
    def setVideoSize(self, size: tuple[int, int]) -> None:
        """setVideoSize(*args, **kwargs)
        Overloaded function.

        1. setVideoSize(self: depthai.node.ColorCamera, width: int, height: int) -> None

        Set video output size

        2. setVideoSize(self: depthai.node.ColorCamera, size: tuple[int, int]) -> None

        Set video output size, as a tuple <width, height>
        """
    def setWaitForConfigInput(self, wait: bool) -> None:
        """setWaitForConfigInput(self: depthai.node.ColorCamera, wait: bool) -> None

        Specify to wait until inputConfig receives a configuration message, before
        sending out a frame.

        Parameter ``wait``:
            True to wait for inputConfig message, false otherwise
        """
    @property
    def frameEvent(self) -> depthai.Node.Output: ...
    @property
    def initialControl(self) -> depthai.CameraControl: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputControl(self) -> depthai.Node.Input: ...
    @property
    def isp(self) -> depthai.Node.Output: ...
    @property
    def preview(self) -> depthai.Node.Output: ...
    @property
    def raw(self) -> depthai.Node.Output: ...
    @property
    def still(self) -> depthai.Node.Output: ...
    @property
    def video(self) -> depthai.Node.Output: ...

class DetectionNetwork(NeuralNetwork):
    Properties: ClassVar[type[depthai.DetectionNetworkProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getConfidenceThreshold(self) -> float:
        """getConfidenceThreshold(self: depthai.node.DetectionNetwork) -> float

        Retrieves threshold at which to filter the rest of the detections.

        Returns:
            Detection confidence
        """
    def setConfidenceThreshold(self, thresh: float) -> None:
        """setConfidenceThreshold(self: depthai.node.DetectionNetwork, thresh: float) -> None

        Specifies confidence threshold at which to filter the rest of the detections.

        Parameter ``thresh``:
            Detection confidence must be greater than specified threshold to be added to
            the list
        """
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...
    @property
    def outNetwork(self) -> depthai.Node.Output: ...
    @property
    def passthrough(self) -> depthai.Node.Output: ...

class DetectionParser(depthai.Node):
    Properties: ClassVar[type[depthai.DetectionParserProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getAnchorMasks(self) -> dict[str, list[int]]:
        """getAnchorMasks(self: depthai.node.DetectionParser) -> dict[str, list[int]]

        Get anchor masks
        """
    def getAnchors(self) -> list[float]:
        """getAnchors(self: depthai.node.DetectionParser) -> list[float]

        Get anchors
        """
    def getConfidenceThreshold(self) -> float:
        """getConfidenceThreshold(self: depthai.node.DetectionParser) -> float

        Retrieves threshold at which to filter the rest of the detections.

        Returns:
            Detection confidence
        """
    def getCoordinateSize(self) -> int:
        """getCoordinateSize(self: depthai.node.DetectionParser) -> int

        Get coordianate size
        """
    def getIouThreshold(self) -> float:
        """getIouThreshold(self: depthai.node.DetectionParser) -> float

        Get Iou threshold
        """
    def getNNFamily(self) -> depthai.DetectionNetworkType:
        """getNNFamily(self: depthai.node.DetectionParser) -> depthai.DetectionNetworkType

        Gets NN Family to parse
        """
    def getNumClasses(self) -> int:
        """getNumClasses(self: depthai.node.DetectionParser) -> int

        Get num classes
        """
    def getNumFramesPool(self) -> int:
        """getNumFramesPool(self: depthai.node.DetectionParser) -> int

        Returns number of frames in pool
        """
    def setAnchorMasks(self, anchorMasks: dict[str, list[int]]) -> None:
        """setAnchorMasks(self: depthai.node.DetectionParser, anchorMasks: dict[str, list[int]]) -> None

        Set anchor masks
        """
    def setAnchors(self, anchors: list[float]) -> None:
        """setAnchors(self: depthai.node.DetectionParser, anchors: list[float]) -> None

        Set anchors
        """
    def setBlob(self, blob: depthai.OpenVINO.Blob) -> None:
        """setBlob(self: depthai.node.DetectionParser, blob: depthai.OpenVINO.Blob) -> None

        Retrieves some input tensor information from the blob

        Parameter ``blob``:
            OpenVINO blob to retrieve the information from
        """
    def setConfidenceThreshold(self, thresh: float) -> None:
        """setConfidenceThreshold(self: depthai.node.DetectionParser, thresh: float) -> None

        Specifies confidence threshold at which to filter the rest of the detections.

        Parameter ``thresh``:
            Detection confidence must be greater than specified threshold to be added to
            the list
        """
    def setCoordinateSize(self, coordinates: int) -> None:
        """setCoordinateSize(self: depthai.node.DetectionParser, coordinates: int) -> None

        Set coordianate size
        """
    def setIouThreshold(self, thresh: float) -> None:
        """setIouThreshold(self: depthai.node.DetectionParser, thresh: float) -> None

        Set Iou threshold
        """
    def setNNFamily(self, type: depthai.DetectionNetworkType) -> None:
        """setNNFamily(self: depthai.node.DetectionParser, type: depthai.DetectionNetworkType) -> None

        Sets NN Family to parse
        """
    def setNumClasses(self, numClasses: int) -> None:
        """setNumClasses(self: depthai.node.DetectionParser, numClasses: int) -> None

        Set num classes
        """
    def setNumFramesPool(self, numFramesPool: int) -> None:
        """setNumFramesPool(self: depthai.node.DetectionParser, numFramesPool: int) -> None

        Specify number of frames in pool.

        Parameter ``numFramesPool``:
            How many frames should the pool have
        """
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...

class EdgeDetector(depthai.Node):
    Properties: ClassVar[type[depthai.EdgeDetectorProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getWaitForConfigInput(self) -> bool:
        """getWaitForConfigInput(self: depthai.node.EdgeDetector) -> bool

        See also:
            setWaitForConfigInput

        Returns:
            True if wait for inputConfig message, false otherwise
        """
    def setMaxOutputFrameSize(self, arg0: int) -> None:
        """setMaxOutputFrameSize(self: depthai.node.EdgeDetector, arg0: int) -> None

        Specify maximum size of output image.

        Parameter ``maxFrameSize``:
            Maximum frame size in bytes
        """
    def setNumFramesPool(self, arg0: int) -> None:
        """setNumFramesPool(self: depthai.node.EdgeDetector, arg0: int) -> None

        Specify number of frames in pool.

        Parameter ``numFramesPool``:
            How many frames should the pool have
        """
    def setWaitForConfigInput(self, wait: bool) -> None:
        """setWaitForConfigInput(self: depthai.node.EdgeDetector, wait: bool) -> None

        Specify whether or not wait until configuration message arrives to inputConfig
        Input.

        Parameter ``wait``:
            True to wait for configuration message, false otherwise.
        """
    @property
    def initialConfig(self) -> depthai.EdgeDetectorConfig: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputImage(self) -> depthai.Node.Input: ...
    @property
    def outputImage(self) -> depthai.Node.Output: ...

class FeatureTracker(depthai.Node):
    Properties: ClassVar[type[depthai.FeatureTrackerProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getWaitForConfigInput(self) -> bool:
        """getWaitForConfigInput(self: depthai.node.FeatureTracker) -> bool

        See also:
            setWaitForConfigInput

        Returns:
            True if wait for inputConfig message, false otherwise
        """
    def setHardwareResources(self, numShaves: int, numMemorySlices: int) -> None:
        """setHardwareResources(self: depthai.node.FeatureTracker, numShaves: int, numMemorySlices: int) -> None

        Specify allocated hardware resources for feature tracking. 2 shaves/memory
        slices are required for optical flow, 1 for corner detection only.

        Parameter ``numShaves``:
            Number of shaves. Maximum 2.

        Parameter ``numMemorySlices``:
            Number of memory slices. Maximum 2.
        """
    def setWaitForConfigInput(self, wait: bool) -> None:
        """setWaitForConfigInput(self: depthai.node.FeatureTracker, wait: bool) -> None

        Specify whether or not wait until configuration message arrives to inputConfig
        Input.

        Parameter ``wait``:
            True to wait for configuration message, false otherwise.
        """
    @property
    def initialConfig(self) -> depthai.FeatureTrackerConfig: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputImage(self) -> depthai.Node.Input: ...
    @property
    def outputFeatures(self) -> depthai.Node.Output: ...
    @property
    def passthroughInputImage(self) -> depthai.Node.Output: ...

class IMU(depthai.Node):
    Properties: ClassVar[type[depthai.IMUProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def enableFirmwareUpdate(self, arg0: bool) -> None:
        """enableFirmwareUpdate(self: depthai.node.IMU, arg0: bool) -> None"""
    @overload
    def enableIMUSensor(self, sensorConfig: depthai.IMUSensorConfig) -> None:
        """enableIMUSensor(*args, **kwargs)
        Overloaded function.

        1. enableIMUSensor(self: depthai.node.IMU, sensorConfig: depthai.IMUSensorConfig) -> None

        Enable a new IMU sensor with explicit configuration

        2. enableIMUSensor(self: depthai.node.IMU, sensorConfigs: list[depthai.IMUSensorConfig]) -> None

        Enable a list of IMU sensors with explicit configuration

        3. enableIMUSensor(self: depthai.node.IMU, sensor: depthai.IMUSensor, reportRate: int) -> None

        Enable a new IMU sensor with default configuration

        4. enableIMUSensor(self: depthai.node.IMU, sensors: list[depthai.IMUSensor], reportRate: int) -> None

        Enable a list of IMU sensors with default configuration
        """
    @overload
    def enableIMUSensor(self, sensorConfigs: list[depthai.IMUSensorConfig]) -> None:
        """enableIMUSensor(*args, **kwargs)
        Overloaded function.

        1. enableIMUSensor(self: depthai.node.IMU, sensorConfig: depthai.IMUSensorConfig) -> None

        Enable a new IMU sensor with explicit configuration

        2. enableIMUSensor(self: depthai.node.IMU, sensorConfigs: list[depthai.IMUSensorConfig]) -> None

        Enable a list of IMU sensors with explicit configuration

        3. enableIMUSensor(self: depthai.node.IMU, sensor: depthai.IMUSensor, reportRate: int) -> None

        Enable a new IMU sensor with default configuration

        4. enableIMUSensor(self: depthai.node.IMU, sensors: list[depthai.IMUSensor], reportRate: int) -> None

        Enable a list of IMU sensors with default configuration
        """
    @overload
    def enableIMUSensor(self, sensor: depthai.IMUSensor, reportRate: int) -> None:
        """enableIMUSensor(*args, **kwargs)
        Overloaded function.

        1. enableIMUSensor(self: depthai.node.IMU, sensorConfig: depthai.IMUSensorConfig) -> None

        Enable a new IMU sensor with explicit configuration

        2. enableIMUSensor(self: depthai.node.IMU, sensorConfigs: list[depthai.IMUSensorConfig]) -> None

        Enable a list of IMU sensors with explicit configuration

        3. enableIMUSensor(self: depthai.node.IMU, sensor: depthai.IMUSensor, reportRate: int) -> None

        Enable a new IMU sensor with default configuration

        4. enableIMUSensor(self: depthai.node.IMU, sensors: list[depthai.IMUSensor], reportRate: int) -> None

        Enable a list of IMU sensors with default configuration
        """
    @overload
    def enableIMUSensor(self, sensors: list[depthai.IMUSensor], reportRate: int) -> None:
        """enableIMUSensor(*args, **kwargs)
        Overloaded function.

        1. enableIMUSensor(self: depthai.node.IMU, sensorConfig: depthai.IMUSensorConfig) -> None

        Enable a new IMU sensor with explicit configuration

        2. enableIMUSensor(self: depthai.node.IMU, sensorConfigs: list[depthai.IMUSensorConfig]) -> None

        Enable a list of IMU sensors with explicit configuration

        3. enableIMUSensor(self: depthai.node.IMU, sensor: depthai.IMUSensor, reportRate: int) -> None

        Enable a new IMU sensor with default configuration

        4. enableIMUSensor(self: depthai.node.IMU, sensors: list[depthai.IMUSensor], reportRate: int) -> None

        Enable a list of IMU sensors with default configuration
        """
    def getBatchReportThreshold(self) -> int:
        """getBatchReportThreshold(self: depthai.node.IMU) -> int

        Above this packet threshold data will be sent to host, if queue is not blocked
        """
    def getMaxBatchReports(self) -> int:
        """getMaxBatchReports(self: depthai.node.IMU) -> int

        Maximum number of IMU packets in a batch report
        """
    def setBatchReportThreshold(self, batchReportThreshold: int) -> None:
        """setBatchReportThreshold(self: depthai.node.IMU, batchReportThreshold: int) -> None

        Above this packet threshold data will be sent to host, if queue is not blocked
        """
    def setMaxBatchReports(self, maxBatchReports: int) -> None:
        """setMaxBatchReports(self: depthai.node.IMU, maxBatchReports: int) -> None

        Maximum number of IMU packets in a batch report
        """
    @property
    def out(self) -> depthai.Node.Output: ...

class ImageAlign(depthai.Node):
    Properties: ClassVar[type[depthai.ImageAlignProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setInterpolation(self, arg0: depthai.Interpolation) -> ImageAlign:
        """setInterpolation(self: depthai.node.ImageAlign, arg0: depthai.Interpolation) -> depthai.node.ImageAlign

        Specify interpolation method to use when resizing
        """
    def setNumFramesPool(self, arg0: int) -> ImageAlign:
        """setNumFramesPool(self: depthai.node.ImageAlign, arg0: int) -> depthai.node.ImageAlign

        Specify number of frames in the pool
        """
    def setNumShaves(self, arg0: int) -> ImageAlign:
        """setNumShaves(self: depthai.node.ImageAlign, arg0: int) -> depthai.node.ImageAlign

        Specify number of shaves to use for this node
        """
    def setOutKeepAspectRatio(self, arg0: bool) -> ImageAlign:
        """setOutKeepAspectRatio(self: depthai.node.ImageAlign, arg0: bool) -> depthai.node.ImageAlign

        Specify whether to keep aspect ratio when resizing
        """
    def setOutputSize(self, arg0: int, arg1: int) -> ImageAlign:
        """setOutputSize(self: depthai.node.ImageAlign, arg0: int, arg1: int) -> depthai.node.ImageAlign

        Specify the output size of the aligned image
        """
    @property
    def initialConfig(self) -> depthai.ImageAlignConfig: ...
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def inputAlignTo(self) -> depthai.Node.Input: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def outputAligned(self) -> depthai.Node.Output: ...
    @property
    def passthroughInput(self) -> depthai.Node.Output: ...

class ImageManip(depthai.Node):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getWaitForConfigInput(self) -> bool:
        """getWaitForConfigInput(self: depthai.node.ImageManip) -> bool

        See also:
            setWaitForConfigInput

        Returns:
            True if wait for inputConfig message, false otherwise
        """
    def setCenterCrop(self, arg0: float, arg1: float) -> None:
        """setCenterCrop(self: depthai.node.ImageManip, arg0: float, arg1: float) -> None"""
    def setCropRect(self, arg0: float, arg1: float, arg2: float, arg3: float) -> None:
        """setCropRect(self: depthai.node.ImageManip, arg0: float, arg1: float, arg2: float, arg3: float) -> None"""
    def setFrameType(self, arg0: depthai.RawImgFrame.Type) -> None:
        """setFrameType(self: depthai.node.ImageManip, arg0: depthai.RawImgFrame.Type) -> None"""
    def setHorizontalFlip(self, arg0: bool) -> None:
        """setHorizontalFlip(self: depthai.node.ImageManip, arg0: bool) -> None"""
    def setKeepAspectRatio(self, arg0: bool) -> None:
        """setKeepAspectRatio(self: depthai.node.ImageManip, arg0: bool) -> None"""
    def setMaxOutputFrameSize(self, arg0: int) -> None:
        """setMaxOutputFrameSize(self: depthai.node.ImageManip, arg0: int) -> None

        Specify maximum size of output image.

        Parameter ``maxFrameSize``:
            Maximum frame size in bytes
        """
    def setNumFramesPool(self, arg0: int) -> None:
        """setNumFramesPool(self: depthai.node.ImageManip, arg0: int) -> None

        Specify number of frames in pool.

        Parameter ``numFramesPool``:
            How many frames should the pool have
        """
    def setResize(self, arg0: int, arg1: int) -> None:
        """setResize(self: depthai.node.ImageManip, arg0: int, arg1: int) -> None"""
    def setResizeThumbnail(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: int) -> None:
        """setResizeThumbnail(self: depthai.node.ImageManip, arg0: int, arg1: int, arg2: int, arg3: int, arg4: int) -> None"""
    def setWaitForConfigInput(self, wait: bool) -> None:
        """setWaitForConfigInput(self: depthai.node.ImageManip, wait: bool) -> None

        Specify whether or not wait until configuration message arrives to inputConfig
        Input.

        Parameter ``wait``:
            True to wait for configuration message, false otherwise.
        """
    @overload
    def setWarpMesh(self, arg0: list[depthai.Point2f], arg1: int, arg2: int) -> None:
        """setWarpMesh(*args, **kwargs)
        Overloaded function.

        1. setWarpMesh(self: depthai.node.ImageManip, arg0: list[depthai.Point2f], arg1: int, arg2: int) -> None

        2. setWarpMesh(self: depthai.node.ImageManip, arg0: list[tuple[float, float]], arg1: int, arg2: int) -> None
        """
    @overload
    def setWarpMesh(self, arg0: list[tuple[float, float]], arg1: int, arg2: int) -> None:
        """setWarpMesh(*args, **kwargs)
        Overloaded function.

        1. setWarpMesh(self: depthai.node.ImageManip, arg0: list[depthai.Point2f], arg1: int, arg2: int) -> None

        2. setWarpMesh(self: depthai.node.ImageManip, arg0: list[tuple[float, float]], arg1: int, arg2: int) -> None
        """
    @property
    def initialConfig(self) -> depthai.ImageManipConfig: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputImage(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...

class MessageDemux(depthai.Node):
    Properties: ClassVar[type[depthai.MessageDemuxProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def outputs(self) -> depthai.Node.OutputMap: ...

class MobileNetDetectionNetwork(DetectionNetwork):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MobileNetSpatialDetectionNetwork(SpatialDetectionNetwork):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class MonoCamera(depthai.Node):
    Properties: ClassVar[type[depthai.MonoCameraProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getBoardSocket(self) -> depthai.CameraBoardSocket:
        """getBoardSocket(self: depthai.node.MonoCamera) -> depthai.CameraBoardSocket

        Retrieves which board socket to use

        Returns:
            Board socket to use
        """
    def getCamId(self) -> int:
        """getCamId(self: depthai.node.MonoCamera) -> int"""
    def getCamera(self) -> str:
        """getCamera(self: depthai.node.MonoCamera) -> str

        Retrieves which camera to use by name

        Returns:
            Name of the camera to use
        """
    def getFps(self) -> float:
        """getFps(self: depthai.node.MonoCamera) -> float

        Get rate at which camera should produce frames

        Returns:
            Rate in frames per second
        """
    def getFrameEventFilter(self) -> list[depthai.FrameEvent]:
        """getFrameEventFilter(self: depthai.node.MonoCamera) -> list[depthai.FrameEvent]"""
    def getImageOrientation(self) -> depthai.CameraImageOrientation:
        """getImageOrientation(self: depthai.node.MonoCamera) -> depthai.CameraImageOrientation

        Get camera image orientation
        """
    def getNumFramesPool(self) -> int:
        """getNumFramesPool(self: depthai.node.MonoCamera) -> int

        Get number of frames in main (ISP output) pool
        """
    def getRawNumFramesPool(self) -> int:
        """getRawNumFramesPool(self: depthai.node.MonoCamera) -> int

        Get number of frames in raw pool
        """
    def getResolution(self) -> depthai.MonoCameraProperties.SensorResolution:
        """getResolution(self: depthai.node.MonoCamera) -> depthai.MonoCameraProperties.SensorResolution

        Get sensor resolution
        """
    def getResolutionHeight(self) -> int:
        """getResolutionHeight(self: depthai.node.MonoCamera) -> int

        Get sensor resolution height
        """
    def getResolutionSize(self) -> tuple[int, int]:
        """getResolutionSize(self: depthai.node.MonoCamera) -> tuple[int, int]

        Get sensor resolution as size
        """
    def getResolutionWidth(self) -> int:
        """getResolutionWidth(self: depthai.node.MonoCamera) -> int

        Get sensor resolution width
        """
    def setBoardSocket(self, boardSocket: depthai.CameraBoardSocket) -> None:
        """setBoardSocket(self: depthai.node.MonoCamera, boardSocket: depthai.CameraBoardSocket) -> None

        Specify which board socket to use

        Parameter ``boardSocket``:
            Board socket to use
        """
    def setCamId(self, arg0: int) -> None:
        """setCamId(self: depthai.node.MonoCamera, arg0: int) -> None"""
    def setCamera(self, name: str) -> None:
        """setCamera(self: depthai.node.MonoCamera, name: str) -> None

        Specify which camera to use by name

        Parameter ``name``:
            Name of the camera to use
        """
    def setFps(self, fps: float) -> None:
        """setFps(self: depthai.node.MonoCamera, fps: float) -> None

        Set rate at which camera should produce frames

        Parameter ``fps``:
            Rate in frames per second
        """
    def setFrameEventFilter(self, events: list[depthai.FrameEvent]) -> None:
        """setFrameEventFilter(self: depthai.node.MonoCamera, events: list[depthai.FrameEvent]) -> None"""
    def setImageOrientation(self, imageOrientation: depthai.CameraImageOrientation) -> None:
        """setImageOrientation(self: depthai.node.MonoCamera, imageOrientation: depthai.CameraImageOrientation) -> None

        Set camera image orientation
        """
    def setIsp3aFps(self, isp3aFps: int) -> None:
        """setIsp3aFps(self: depthai.node.MonoCamera, isp3aFps: int) -> None

        Isp 3A rate (auto focus, auto exposure, auto white balance, camera controls
        etc.). Default (0) matches the camera FPS, meaning that 3A is running on each
        frame. Reducing the rate of 3A reduces the CPU usage on CSS, but also increases
        the convergence rate of 3A. Note that camera controls will be processed at this
        rate. E.g. if camera is running at 30 fps, and camera control is sent at every
        frame, but 3A fps is set to 15, the camera control messages will be processed at
        15 fps rate, which will lead to queueing.
        """
    def setNumFramesPool(self, arg0: int) -> None:
        """setNumFramesPool(self: depthai.node.MonoCamera, arg0: int) -> None

        Set number of frames in main (ISP output) pool
        """
    def setRawNumFramesPool(self, arg0: int) -> None:
        """setRawNumFramesPool(self: depthai.node.MonoCamera, arg0: int) -> None

        Set number of frames in raw pool
        """
    def setRawOutputPacked(self, packed: bool) -> None:
        """setRawOutputPacked(self: depthai.node.MonoCamera, packed: bool) -> None

        Configures whether the camera `raw` frames are saved as MIPI-packed to memory.
        The packed format is more efficient, consuming less memory on device, and less
        data to send to host: RAW10: 4 pixels saved on 5 bytes, RAW12: 2 pixels saved on
        3 bytes. When packing is disabled (`false`), data is saved lsb-aligned, e.g. a
        RAW10 pixel will be stored as uint16, on bits 9..0: 0b0000'00pp'pppp'pppp.
        Default is auto: enabled for standard color/monochrome cameras where ISP can
        work with both packed/unpacked, but disabled for other cameras like ToF.
        """
    def setResolution(self, resolution: depthai.MonoCameraProperties.SensorResolution) -> None:
        """setResolution(self: depthai.node.MonoCamera, resolution: depthai.MonoCameraProperties.SensorResolution) -> None

        Set sensor resolution
        """
    @property
    def frameEvent(self) -> depthai.Node.Output: ...
    @property
    def initialControl(self) -> depthai.CameraControl: ...
    @property
    def inputControl(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...
    @property
    def raw(self) -> depthai.Node.Output: ...

class NeuralNetwork(depthai.Node):
    Properties: ClassVar[type[depthai.NeuralNetworkProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getNumInferenceThreads(self) -> int:
        """getNumInferenceThreads(self: depthai.node.NeuralNetwork) -> int

        How many inference threads will be used to run the network

        Returns:
            Number of threads, 0, 1 or 2. Zero means AUTO
        """
    @overload
    def setBlob(self, blob: depthai.OpenVINO.Blob) -> None:
        """setBlob(*args, **kwargs)
        Overloaded function.

        1. setBlob(self: depthai.node.NeuralNetwork, blob: depthai.OpenVINO.Blob) -> None

        Load network blob into assets and use once pipeline is started.

        Parameter ``blob``:
            Network blob

        2. setBlob(self: depthai.node.NeuralNetwork, path: Path) -> None

        Same functionality as the setBlobPath(). Load network blob into assets and use
        once pipeline is started.

        Throws:
            Error if file doesn't exist or isn't a valid network blob.

        Parameter ``path``:
            Path to network blob
        """
    @overload
    def setBlob(self, path: Path) -> None:
        """setBlob(*args, **kwargs)
        Overloaded function.

        1. setBlob(self: depthai.node.NeuralNetwork, blob: depthai.OpenVINO.Blob) -> None

        Load network blob into assets and use once pipeline is started.

        Parameter ``blob``:
            Network blob

        2. setBlob(self: depthai.node.NeuralNetwork, path: Path) -> None

        Same functionality as the setBlobPath(). Load network blob into assets and use
        once pipeline is started.

        Throws:
            Error if file doesn't exist or isn't a valid network blob.

        Parameter ``path``:
            Path to network blob
        """
    def setBlobPath(self, path: Path) -> None:
        """setBlobPath(self: depthai.node.NeuralNetwork, path: Path) -> None

        Load network blob into assets and use once pipeline is started.

        Throws:
            Error if file doesn't exist or isn't a valid network blob.

        Parameter ``path``:
            Path to network blob
        """
    def setNumInferenceThreads(self, numThreads: int) -> None:
        """setNumInferenceThreads(self: depthai.node.NeuralNetwork, numThreads: int) -> None

        How many threads should the node use to run the network.

        Parameter ``numThreads``:
            Number of threads to dedicate to this node
        """
    def setNumNCEPerInferenceThread(self, numNCEPerThread: int) -> None:
        """setNumNCEPerInferenceThread(self: depthai.node.NeuralNetwork, numNCEPerThread: int) -> None

        How many Neural Compute Engines should a single thread use for inference

        Parameter ``numNCEPerThread``:
            Number of NCE per thread
        """
    def setNumPoolFrames(self, numFrames: int) -> None:
        """setNumPoolFrames(self: depthai.node.NeuralNetwork, numFrames: int) -> None

        Specifies how many frames will be available in the pool

        Parameter ``numFrames``:
            How many frames will pool have
        """
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def inputs(self) -> depthai.Node.InputMap: ...
    @property
    def out(self) -> depthai.Node.Output: ...
    @property
    def passthrough(self) -> depthai.Node.Output: ...
    @property
    def passthroughs(self) -> depthai.Node.OutputMap: ...

class ObjectTracker(depthai.Node):
    Properties: ClassVar[type[depthai.ObjectTrackerProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setDetectionLabelsToTrack(self, labels: list[int]) -> None:
        """setDetectionLabelsToTrack(self: depthai.node.ObjectTracker, labels: list[int]) -> None

        Specify detection labels to track.

        Parameter ``labels``:
            Detection labels to track. Default every label is tracked from image
            detection network output.
        """
    def setMaxObjectsToTrack(self, maxObjectsToTrack: int) -> None:
        """setMaxObjectsToTrack(self: depthai.node.ObjectTracker, maxObjectsToTrack: int) -> None

        Specify maximum number of object to track.

        Parameter ``maxObjectsToTrack``:
            Maximum number of object to track. Maximum 60 in case of SHORT_TERM_KCF,
            otherwise 1000.
        """
    def setTrackerIdAssignmentPolicy(self, type: depthai.TrackerIdAssignmentPolicy) -> None:
        """setTrackerIdAssignmentPolicy(self: depthai.node.ObjectTracker, type: depthai.TrackerIdAssignmentPolicy) -> None

        Specify tracker ID assignment policy.

        Parameter ``type``:
            Tracker ID assignment policy.
        """
    def setTrackerThreshold(self, threshold: float) -> None:
        """setTrackerThreshold(self: depthai.node.ObjectTracker, threshold: float) -> None

        Specify tracker threshold.

        Parameter ``threshold``:
            Above this threshold the detected objects will be tracked. Default 0, all
            image detections are tracked.
        """
    def setTrackerType(self, type: depthai.TrackerType) -> None:
        """setTrackerType(self: depthai.node.ObjectTracker, type: depthai.TrackerType) -> None

        Specify tracker type algorithm.

        Parameter ``type``:
            Tracker type.
        """
    def setTrackingPerClass(self, trackingPerClass: bool) -> None:
        """setTrackingPerClass(self: depthai.node.ObjectTracker, trackingPerClass: bool) -> None

        Whether tracker should take into consideration class label for tracking.
        """
    @property
    def inputDetectionFrame(self) -> depthai.Node.Input: ...
    @property
    def inputDetections(self) -> depthai.Node.Input: ...
    @property
    def inputTrackerFrame(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...
    @property
    def passthroughDetectionFrame(self) -> depthai.Node.Output: ...
    @property
    def passthroughDetections(self) -> depthai.Node.Output: ...
    @property
    def passthroughTrackerFrame(self) -> depthai.Node.Output: ...

class PointCloud(depthai.Node):
    Properties: ClassVar[type[depthai.PointCloudProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setNumFramesPool(self, arg0: int) -> None:
        """setNumFramesPool(self: depthai.node.PointCloud, arg0: int) -> None

        Specify number of frames in pool.

        Parameter ``numFramesPool``:
            How many frames should the pool have
        """
    @property
    def initialConfig(self) -> depthai.PointCloudConfig: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputDepth(self) -> depthai.Node.Input: ...
    @property
    def outputPointCloud(self) -> depthai.Node.Output: ...
    @property
    def passthroughDepth(self) -> depthai.Node.Output: ...

class SPIIn(depthai.Node):
    Properties: ClassVar[type[depthai.SPIInProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getBusId(self) -> int:
        """getBusId(self: depthai.node.SPIIn) -> int

        Get bus id
        """
    def getMaxDataSize(self) -> int:
        """getMaxDataSize(self: depthai.node.SPIIn) -> int

        Get maximum messages size in bytes
        """
    def getNumFrames(self) -> int:
        """getNumFrames(self: depthai.node.SPIIn) -> int

        Get number of frames in pool
        """
    def getStreamName(self) -> str:
        """getStreamName(self: depthai.node.SPIIn) -> str

        Get stream name
        """
    def setBusId(self, id: int) -> None:
        """setBusId(self: depthai.node.SPIIn, id: int) -> None

        Specifies SPI Bus number to use

        Parameter ``id``:
            SPI Bus id
        """
    def setMaxDataSize(self, maxDataSize: int) -> None:
        """setMaxDataSize(self: depthai.node.SPIIn, maxDataSize: int) -> None

        Set maximum message size it can receive

        Parameter ``maxDataSize``:
            Maximum size in bytes
        """
    def setNumFrames(self, numFrames: int) -> None:
        """setNumFrames(self: depthai.node.SPIIn, numFrames: int) -> None

        Set number of frames in pool for sending messages forward

        Parameter ``numFrames``:
            Maximum number of frames in pool
        """
    def setStreamName(self, name: str) -> None:
        """setStreamName(self: depthai.node.SPIIn, name: str) -> None

        Specifies stream name over which the node will receive data

        Parameter ``name``:
            Stream name
        """
    @property
    def out(self) -> depthai.Node.Output: ...

class SPIOut(depthai.Node):
    Properties: ClassVar[type[depthai.SPIOutProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setBusId(self, id: int) -> None:
        """setBusId(self: depthai.node.SPIOut, id: int) -> None

        Specifies SPI Bus number to use

        Parameter ``id``:
            SPI Bus id
        """
    def setStreamName(self, name: str) -> None:
        """setStreamName(self: depthai.node.SPIOut, name: str) -> None

        Specifies stream name over which the node will send data

        Parameter ``name``:
            Stream name
        """
    @property
    def input(self) -> depthai.Node.Input: ...

class Script(depthai.Node):
    Properties: ClassVar[type[depthai.ScriptProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getProcessor(self) -> depthai.ProcessorType:
        """getProcessor(self: depthai.node.Script) -> depthai.ProcessorType

        Get on which processor the script should run

        Returns:
            Processor type - Leon CSS or Leon MSS
        """
    def getScriptName(self) -> str:
        '''getScriptName(self: depthai.node.Script) -> str

        Get the script name in utf-8.

        When name set with setScript() or setScriptPath(), returns that name. When
        script loaded with setScriptPath() with name not provided, returns the utf-8
        string of that path. Otherwise, returns "<script>"

        Returns:
            std::string of script name in utf-8
        '''
    def setProcessor(self, arg0: depthai.ProcessorType) -> None:
        """setProcessor(self: depthai.node.Script, arg0: depthai.ProcessorType) -> None

        Set on which processor the script should run

        Parameter ``type``:
            Processor type - Leon CSS or Leon MSS
        """
    @overload
    def setScript(self, script: str, name: str = ...) -> None:
        """setScript(*args, **kwargs)
        Overloaded function.

        1. setScript(self: depthai.node.Script, script: str, name: str = '') -> None

        Sets script data to be interpreted

        Parameter ``script``:
            Script string to be interpreted

        Parameter ``name``:
            Optionally set a name of this script

        2. setScript(self: depthai.node.Script, data: list[int], name: str = '') -> None

        Sets script data to be interpreted

        Parameter ``data``:
            Binary data that represents the script to be interpreted

        Parameter ``name``:
            Optionally set a name of this script
        """
    @overload
    def setScript(self, data: list[int], name: str = ...) -> None:
        """setScript(*args, **kwargs)
        Overloaded function.

        1. setScript(self: depthai.node.Script, script: str, name: str = '') -> None

        Sets script data to be interpreted

        Parameter ``script``:
            Script string to be interpreted

        Parameter ``name``:
            Optionally set a name of this script

        2. setScript(self: depthai.node.Script, data: list[int], name: str = '') -> None

        Sets script data to be interpreted

        Parameter ``data``:
            Binary data that represents the script to be interpreted

        Parameter ``name``:
            Optionally set a name of this script
        """
    @overload
    def setScriptPath(self, arg0: Path, arg1: str) -> None:
        """setScriptPath(*args, **kwargs)
        Overloaded function.

        1. setScriptPath(self: depthai.node.Script, arg0: Path, arg1: str) -> None

        Specify local filesystem path to load the script

        Parameter ``path``:
            Filesystem path to load the script

        Parameter ``name``:
            Optionally set a name of this script, otherwise the name defaults to the
            path

        2. setScriptPath(self: depthai.node.Script, path: Path, name: str = '') -> None

        Specify local filesystem path to load the script

        Parameter ``path``:
            Filesystem path to load the script

        Parameter ``name``:
            Optionally set a name of this script, otherwise the name defaults to the
            path
        """
    @overload
    def setScriptPath(self, path: Path, name: str = ...) -> None:
        """setScriptPath(*args, **kwargs)
        Overloaded function.

        1. setScriptPath(self: depthai.node.Script, arg0: Path, arg1: str) -> None

        Specify local filesystem path to load the script

        Parameter ``path``:
            Filesystem path to load the script

        Parameter ``name``:
            Optionally set a name of this script, otherwise the name defaults to the
            path

        2. setScriptPath(self: depthai.node.Script, path: Path, name: str = '') -> None

        Specify local filesystem path to load the script

        Parameter ``path``:
            Filesystem path to load the script

        Parameter ``name``:
            Optionally set a name of this script, otherwise the name defaults to the
            path
        """
    @property
    def inputs(self) -> depthai.Node.InputMap: ...
    @property
    def outputs(self) -> depthai.Node.OutputMap: ...

class SpatialDetectionNetwork(DetectionNetwork):
    Properties: ClassVar[type[depthai.SpatialDetectionNetworkProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setBoundingBoxScaleFactor(self, scaleFactor: float) -> None:
        """setBoundingBoxScaleFactor(self: depthai.node.SpatialDetectionNetwork, scaleFactor: float) -> None

        Specifies scale factor for detected bounding boxes.

        Parameter ``scaleFactor``:
            Scale factor must be in the interval (0,1].
        """
    def setDepthLowerThreshold(self, lowerThreshold: int) -> None:
        """setDepthLowerThreshold(self: depthai.node.SpatialDetectionNetwork, lowerThreshold: int) -> None

        Specifies lower threshold in depth units (millimeter by default) for depth
        values which will used to calculate spatial data

        Parameter ``lowerThreshold``:
            LowerThreshold must be in the interval [0,upperThreshold] and less than
            upperThreshold.
        """
    def setDepthUpperThreshold(self, upperThreshold: int) -> None:
        """setDepthUpperThreshold(self: depthai.node.SpatialDetectionNetwork, upperThreshold: int) -> None

        Specifies upper threshold in depth units (millimeter by default) for depth
        values which will used to calculate spatial data

        Parameter ``upperThreshold``:
            UpperThreshold must be in the interval (lowerThreshold,65535].
        """
    def setSpatialCalculationAlgorithm(self, calculationAlgorithm: depthai.SpatialLocationCalculatorAlgorithm) -> None:
        """setSpatialCalculationAlgorithm(self: depthai.node.SpatialDetectionNetwork, calculationAlgorithm: depthai.SpatialLocationCalculatorAlgorithm) -> None

        Specifies spatial location calculator algorithm: Average/Min/Max

        Parameter ``calculationAlgorithm``:
            Calculation algorithm.
        """
    def setSpatialCalculationStepSize(self, stepSize: int) -> None:
        """setSpatialCalculationStepSize(self: depthai.node.SpatialDetectionNetwork, stepSize: int) -> None

        Specifies spatial location calculator step size for depth calculation. Step size
        1 means that every pixel is taken into calculation, size 2 means every second
        etc.

        Parameter ``stepSize``:
            Step size.
        """
    @property
    def boundingBoxMapping(self) -> depthai.Node.Output: ...
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def inputDepth(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...
    @property
    def passthrough(self) -> depthai.Node.Output: ...
    @property
    def passthroughDepth(self) -> depthai.Node.Output: ...
    @property
    def spatialLocationCalculatorOutput(self) -> depthai.Node.Output: ...

class SpatialLocationCalculator(depthai.Node):
    Properties: ClassVar[type[depthai.SpatialLocationCalculatorProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getWaitForConfigInput(self) -> bool:
        """getWaitForConfigInput(self: depthai.node.SpatialLocationCalculator) -> bool

        See also:
            setWaitForConfigInput

        Returns:
            True if wait for inputConfig message, false otherwise
        """
    def setWaitForConfigInput(self, wait: bool) -> None:
        """setWaitForConfigInput(self: depthai.node.SpatialLocationCalculator, wait: bool) -> None

        Specify whether or not wait until configuration message arrives to inputConfig
        Input.

        Parameter ``wait``:
            True to wait for configuration message, false otherwise.
        """
    @property
    def initialConfig(self) -> depthai.SpatialLocationCalculatorConfig: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def inputDepth(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...
    @property
    def passthroughDepth(self) -> depthai.Node.Output: ...

class StereoDepth(depthai.Node):
    class PresetMode:
        HIGH_ACCURACY: ClassVar[StereoDepth.PresetMode] = ...  # read-only
        HIGH_DENSITY: ClassVar[StereoDepth.PresetMode] = ...  # read-only
        __members__: ClassVar[dict] = ...  # read-only
        DEFAULT: ClassVar[StereoDepth.PresetMode] = ...
        FACE: ClassVar[StereoDepth.PresetMode] = ...
        HIGH_DETAIL: ClassVar[StereoDepth.PresetMode] = ...
        ROBOTICS: ClassVar[StereoDepth.PresetMode] = ...
        __entries: ClassVar[dict] = ...
        def __init__(self, value: int) -> None:
            """__init__(self: depthai.node.StereoDepth.PresetMode, value: int) -> None"""
        def __eq__(self, other: object) -> bool:
            """__eq__(self: object, other: object) -> bool"""
        def __hash__(self) -> int:
            """__hash__(self: object) -> int"""
        def __index__(self) -> int:
            """__index__(self: depthai.node.StereoDepth.PresetMode) -> int"""
        def __int__(self) -> int:
            """__int__(self: depthai.node.StereoDepth.PresetMode) -> int"""
        def __ne__(self, other: object) -> bool:
            """__ne__(self: object, other: object) -> bool"""
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...
    Properties: ClassVar[type[depthai.StereoDepthProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def enableDistortionCorrection(self, arg0: bool) -> None:
        """enableDistortionCorrection(self: depthai.node.StereoDepth, arg0: bool) -> None

        Equivalent to useHomographyRectification(!enableDistortionCorrection)
        """
    def getMaxDisparity(self) -> float:
        """getMaxDisparity(self: depthai.node.StereoDepth) -> float

        Useful for normalization of the disparity map.

        Returns:
            Maximum disparity value that the node can return
        """
    def loadMeshData(self, dataLeft: list[int], dataRight: list[int]) -> None:
        """loadMeshData(self: depthai.node.StereoDepth, dataLeft: list[int], dataRight: list[int]) -> None

        Specify mesh calibration data for 'left' and 'right' inputs, as vectors of
        bytes. Overrides useHomographyRectification behavior. See `loadMeshFiles` for
        the expected data format
        """
    def loadMeshFiles(self, pathLeft: Path, pathRight: Path) -> None:
        """loadMeshFiles(self: depthai.node.StereoDepth, pathLeft: Path, pathRight: Path) -> None

        Specify local filesystem paths to the mesh calibration files for 'left' and
        'right' inputs.

        When a mesh calibration is set, it overrides the camera intrinsics/extrinsics
        matrices. Overrides useHomographyRectification behavior. Mesh format: a sequence
        of (y,x) points as 'float' with coordinates from the input image to be mapped in
        the output. The mesh can be subsampled, configured by `setMeshStep`.

        With a 1280x800 resolution and the default (16,16) step, the required mesh size
        is:

        width: 1280 / 16 + 1 = 81

        height: 800 / 16 + 1 = 51
        """
    def setAlphaScaling(self, arg0: float) -> None:
        """setAlphaScaling(self: depthai.node.StereoDepth, arg0: float) -> None

        Free scaling parameter between 0 (when all the pixels in the undistorted image
        are valid) and 1 (when all the source image pixels are retained in the
        undistorted image). On some high distortion lenses, and/or due to rectification
        (image rotated) invalid areas may appear even with alpha=0, in these cases alpha
        < 0.0 helps removing invalid areas. See getOptimalNewCameraMatrix from opencv
        for more details.
        """
    def setBaseline(self, arg0: float) -> None:
        """setBaseline(self: depthai.node.StereoDepth, arg0: float) -> None

        Override baseline from calibration. Used only in disparity to depth conversion.
        Units are centimeters.
        """
    def setConfidenceThreshold(self, arg0: int) -> None:
        """setConfidenceThreshold(self: depthai.node.StereoDepth, arg0: int) -> None

        Confidence threshold for disparity calculation

        Parameter ``confThr``:
            Confidence threshold value 0..255
        """
    def setDefaultProfilePreset(self, arg0: StereoDepth.PresetMode) -> None:
        """setDefaultProfilePreset(self: depthai.node.StereoDepth, arg0: depthai.node.StereoDepth.PresetMode) -> None

        Sets a default preset based on specified option.

        Parameter ``mode``:
            Stereo depth preset mode
        """
    @overload
    def setDepthAlign(self, align: depthai.RawStereoDepthConfig.AlgorithmControl.DepthAlign) -> None:
        """setDepthAlign(*args, **kwargs)
        Overloaded function.

        1. setDepthAlign(self: depthai.node.StereoDepth, align: depthai.RawStereoDepthConfig.AlgorithmControl.DepthAlign) -> None

        Parameter ``align``:
            Set the disparity/depth alignment: centered (between the 'left' and 'right'
            inputs), or from the perspective of a rectified output stream

        2. setDepthAlign(self: depthai.node.StereoDepth, camera: depthai.CameraBoardSocket) -> None

        Parameter ``camera``:
            Set the camera from whose perspective the disparity/depth will be aligned
        """
    @overload
    def setDepthAlign(self, camera: depthai.CameraBoardSocket) -> None:
        """setDepthAlign(*args, **kwargs)
        Overloaded function.

        1. setDepthAlign(self: depthai.node.StereoDepth, align: depthai.RawStereoDepthConfig.AlgorithmControl.DepthAlign) -> None

        Parameter ``align``:
            Set the disparity/depth alignment: centered (between the 'left' and 'right'
            inputs), or from the perspective of a rectified output stream

        2. setDepthAlign(self: depthai.node.StereoDepth, camera: depthai.CameraBoardSocket) -> None

        Parameter ``camera``:
            Set the camera from whose perspective the disparity/depth will be aligned
        """
    def setDepthAlignmentUseSpecTranslation(self, arg0: bool) -> None:
        """setDepthAlignmentUseSpecTranslation(self: depthai.node.StereoDepth, arg0: bool) -> None

        Use baseline information for depth alignment from specs (design data) or from
        calibration. Default: true
        """
    def setDisparityToDepthUseSpecTranslation(self, arg0: bool) -> None:
        """setDisparityToDepthUseSpecTranslation(self: depthai.node.StereoDepth, arg0: bool) -> None

        Use baseline information for disparity to depth conversion from specs (design
        data) or from calibration. Default: true
        """
    def setEmptyCalibration(self) -> None:
        """setEmptyCalibration(self: depthai.node.StereoDepth) -> None

        Specify that a passthrough/dummy calibration should be used, when input frames
        are already rectified (e.g. sourced from recordings on the host)
        """
    def setExtendedDisparity(self, enable: bool) -> None:
        """setExtendedDisparity(self: depthai.node.StereoDepth, enable: bool) -> None

        Disparity range increased from 0-95 to 0-190, combined from full resolution and
        downscaled images.

        Suitable for short range objects. Currently incompatible with sub-pixel
        disparity
        """
    def setFocalLength(self, arg0: float) -> None:
        """setFocalLength(self: depthai.node.StereoDepth, arg0: float) -> None

        Override focal length from calibration. Used only in disparity to depth
        conversion. Units are pixels.
        """
    def setFocalLengthFromCalibration(self, arg0: bool) -> None:
        """setFocalLengthFromCalibration(self: depthai.node.StereoDepth, arg0: bool) -> None

        Whether to use focal length from calibration intrinsics or calculate based on
        calibration FOV. Default value is true.
        """
    @overload
    def setInputResolution(self, width: int, height: int) -> None:
        """setInputResolution(*args, **kwargs)
        Overloaded function.

        1. setInputResolution(self: depthai.node.StereoDepth, width: int, height: int) -> None

        Specify input resolution size

        Optional if MonoCamera exists, otherwise necessary

        2. setInputResolution(self: depthai.node.StereoDepth, resolution: tuple[int, int]) -> None

        Specify input resolution size

        Optional if MonoCamera exists, otherwise necessary
        """
    @overload
    def setInputResolution(self, resolution: tuple[int, int]) -> None:
        """setInputResolution(*args, **kwargs)
        Overloaded function.

        1. setInputResolution(self: depthai.node.StereoDepth, width: int, height: int) -> None

        Specify input resolution size

        Optional if MonoCamera exists, otherwise necessary

        2. setInputResolution(self: depthai.node.StereoDepth, resolution: tuple[int, int]) -> None

        Specify input resolution size

        Optional if MonoCamera exists, otherwise necessary
        """
    def setLeftRightCheck(self, enable: bool) -> None:
        """setLeftRightCheck(self: depthai.node.StereoDepth, enable: bool) -> None

        Computes and combines disparities in both L-R and R-L directions, and combine
        them.

        For better occlusion handling, discarding invalid disparity values
        """
    def setMedianFilter(self, arg0: depthai.MedianFilter) -> None:
        """setMedianFilter(self: depthai.node.StereoDepth, arg0: depthai.MedianFilter) -> None

        Parameter ``median``:
            Set kernel size for disparity/depth median filtering, or disable
        """
    def setMeshStep(self, width: int, height: int) -> None:
        """setMeshStep(self: depthai.node.StereoDepth, width: int, height: int) -> None

        Set the distance between mesh points. Default: (16, 16)
        """
    def setNumFramesPool(self, arg0: int) -> None:
        """setNumFramesPool(self: depthai.node.StereoDepth, arg0: int) -> None

        Specify number of frames in pool.

        Parameter ``numFramesPool``:
            How many frames should the pool have
        """
    def setOutputDepth(self, arg0: bool) -> None:
        """setOutputDepth(self: depthai.node.StereoDepth, arg0: bool) -> None"""
    def setOutputKeepAspectRatio(self, keep: bool) -> None:
        """setOutputKeepAspectRatio(self: depthai.node.StereoDepth, keep: bool) -> None

        Specifies whether the frames resized by `setOutputSize` should preserve aspect
        ratio, with potential cropping when enabled. Default `true`
        """
    def setOutputRectified(self, arg0: bool) -> None:
        """setOutputRectified(self: depthai.node.StereoDepth, arg0: bool) -> None"""
    def setOutputSize(self, width: int, height: int) -> None:
        """setOutputSize(self: depthai.node.StereoDepth, width: int, height: int) -> None

        Specify disparity/depth output resolution size, implemented by scaling.

        Currently only applicable when aligning to RGB camera
        """
    def setPostProcessingHardwareResources(self, arg0: int, arg1: int) -> None:
        """setPostProcessingHardwareResources(self: depthai.node.StereoDepth, arg0: int, arg1: int) -> None

        Specify allocated hardware resources for stereo depth. Suitable only to increase
        post processing runtime.

        Parameter ``numShaves``:
            Number of shaves.

        Parameter ``numMemorySlices``:
            Number of memory slices.
        """
    def setRectification(self, enable: bool) -> None:
        """setRectification(self: depthai.node.StereoDepth, enable: bool) -> None

        Rectify input images or not.
        """
    def setRectificationUseSpecTranslation(self, arg0: bool) -> None:
        """setRectificationUseSpecTranslation(self: depthai.node.StereoDepth, arg0: bool) -> None

        Obtain rectification matrices using spec translation (design data) or from
        calibration in calculations. Should be used only for debugging. Default: false
        """
    def setRectifyEdgeFillColor(self, color: int) -> None:
        """setRectifyEdgeFillColor(self: depthai.node.StereoDepth, color: int) -> None

        Fill color for missing data at frame edges

        Parameter ``color``:
            Grayscale 0..255, or -1 to replicate pixels
        """
    def setRectifyMirrorFrame(self, arg0: bool) -> None:
        """setRectifyMirrorFrame(self: depthai.node.StereoDepth, arg0: bool) -> None

        DEPRECATED function. It was removed, since rectified images are not flipped
        anymore. Mirror rectified frames, only when LR-check mode is disabled. Default
        `true`. The mirroring is required to have a normal non-mirrored disparity/depth
        output.

        A side effect of this option is disparity alignment to the perspective of left
        or right input: `false`: mapped to left and mirrored, `true`: mapped to right.
        With LR-check enabled, this option is ignored, none of the outputs are mirrored,
        and disparity is mapped to right.

        Parameter ``enable``:
            True for normal disparity/depth, otherwise mirrored
        """
    def setRuntimeModeSwitch(self, arg0: bool) -> None:
        """setRuntimeModeSwitch(self: depthai.node.StereoDepth, arg0: bool) -> None

        Enable runtime stereo mode switch, e.g. from standard to LR-check. Note: when
        enabled resources allocated for worst case to enable switching to any mode.
        """
    def setSubpixel(self, enable: bool) -> None:
        """setSubpixel(self: depthai.node.StereoDepth, enable: bool) -> None

        Computes disparity with sub-pixel interpolation (3 fractional bits by default).

        Suitable for long range. Currently incompatible with extended disparity
        """
    def setSubpixelFractionalBits(self, subpixelFractionalBits: int) -> None:
        """setSubpixelFractionalBits(self: depthai.node.StereoDepth, subpixelFractionalBits: int) -> None

        Number of fractional bits for subpixel mode. Default value: 3. Valid values:
        3,4,5. Defines the number of fractional disparities: 2^x. Median filter
        postprocessing is supported only for 3 fractional bits.
        """
    def useHomographyRectification(self, arg0: bool) -> None:
        """useHomographyRectification(self: depthai.node.StereoDepth, arg0: bool) -> None

        Use 3x3 homography matrix for stereo rectification instead of sparse mesh
        generated on device. Default behaviour is AUTO, for lenses with FOV over 85
        degrees sparse mesh is used, otherwise 3x3 homography. If custom mesh data is
        provided through loadMeshData or loadMeshFiles this option is ignored.

        Parameter ``useHomographyRectification``:
            true: 3x3 homography matrix generated from calibration data is used for
            stereo rectification, can't correct lens distortion. false: sparse mesh is
            generated on-device from calibration data with mesh step specified with
            setMeshStep (Default: (16, 16)), can correct lens distortion. Implementation
            for generating the mesh is same as opencv's initUndistortRectifyMap
            function. Only the first 8 distortion coefficients are used from calibration
            data.
        """
    @property
    def confidenceMap(self) -> depthai.Node.Output: ...
    @property
    def debugDispCostDump(self) -> depthai.Node.Output: ...
    @property
    def debugDispLrCheckIt1(self) -> depthai.Node.Output: ...
    @property
    def debugDispLrCheckIt2(self) -> depthai.Node.Output: ...
    @property
    def debugExtDispLrCheckIt1(self) -> depthai.Node.Output: ...
    @property
    def debugExtDispLrCheckIt2(self) -> depthai.Node.Output: ...
    @property
    def depth(self) -> depthai.Node.Output: ...
    @property
    def disparity(self) -> depthai.Node.Output: ...
    @property
    def initialConfig(self) -> depthai.StereoDepthConfig: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def left(self) -> depthai.Node.Input: ...
    @property
    def outConfig(self) -> depthai.Node.Output: ...
    @property
    def rectifiedLeft(self) -> depthai.Node.Output: ...
    @property
    def rectifiedRight(self) -> depthai.Node.Output: ...
    @property
    def right(self) -> depthai.Node.Input: ...
    @property
    def syncedLeft(self) -> depthai.Node.Output: ...
    @property
    def syncedRight(self) -> depthai.Node.Output: ...

class Sync(depthai.Node):
    Properties: ClassVar[type[depthai.SyncProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getSyncAttempts(self) -> int:
        """getSyncAttempts(self: depthai.node.Sync) -> int

        Gets the number of sync attempts
        """
    def getSyncThreshold(self) -> datetime.timedelta:
        """getSyncThreshold(self: depthai.node.Sync) -> datetime.timedelta

        Gets the maximal interval between messages in the group in milliseconds
        """
    def setSyncAttempts(self, maxDataSize: int) -> None:
        """setSyncAttempts(self: depthai.node.Sync, maxDataSize: int) -> None

        Set the number of attempts to get the specified max interval between messages in
        the group

        Parameter ``syncAttempts``:
            Number of attempts to get the specified max interval between messages in the
            group: - if syncAttempts = 0 then the node sends a message as soon at the
            group is filled - if syncAttempts > 0 then the node will make syncAttemts
            attempts to synchronize before sending out a message - if syncAttempts = -1
            (default) then the node will only send a message if successfully
            synchronized
        """
    def setSyncThreshold(self, syncThreshold: datetime.timedelta) -> None:
        """setSyncThreshold(self: depthai.node.Sync, syncThreshold: datetime.timedelta) -> None

        Set the maximal interval between messages in the group

        Parameter ``syncThreshold``:
            Maximal interval between messages in the group
        """
    @property
    def inputs(self) -> depthai.Node.InputMap: ...
    @property
    def out(self) -> depthai.Node.Output: ...

class SystemLogger(depthai.Node):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getRate(self) -> float:
        """getRate(self: depthai.node.SystemLogger) -> float

        Gets logging rate, at which messages will be sent out
        """
    def setRate(self, hz: float) -> None:
        """setRate(self: depthai.node.SystemLogger, hz: float) -> None

        Specify logging rate, at which messages will be sent out

        Parameter ``hz``:
            Sending rate in hertz (messages per second)
        """
    @property
    def out(self) -> depthai.Node.Output: ...

class ToF(depthai.Node):
    Properties: ClassVar[type[depthai.ToFProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setNumFramesPool(self, arg0: int) -> ToF:
        """setNumFramesPool(self: depthai.node.ToF, arg0: int) -> depthai.node.ToF

        Specify number of frames in output pool

        Parameter ``numFramesPool``:
            Number of frames in output pool
        """
    def setNumShaves(self, arg0: int) -> ToF:
        """setNumShaves(self: depthai.node.ToF, arg0: int) -> depthai.node.ToF

        Specify number of shaves reserved for ToF decoding.
        """
    @property
    def amplitude(self) -> depthai.Node.Output: ...
    @property
    def depth(self) -> depthai.Node.Output: ...
    @property
    def initialConfig(self) -> depthai.ToFConfig: ...
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def inputConfig(self) -> depthai.Node.Input: ...
    @property
    def intensity(self) -> depthai.Node.Output: ...
    @property
    def phase(self) -> depthai.Node.Output: ...

class UVC(depthai.Node):
    Properties: ClassVar[type[depthai.UVCProperties]] = ...
    def __init__(self, USBVideoClass) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def setGpiosOnInit(self, list: dict[int, int]) -> None:
        """setGpiosOnInit(self: depthai.node.UVC, list: dict[int, int]) -> None

        Set GPIO list <gpio_number, value> for GPIOs to set (on/off) at init
        """
    def setGpiosOnStreamOff(self, list: dict[int, int]) -> None:
        """setGpiosOnStreamOff(self: depthai.node.UVC, list: dict[int, int]) -> None

        Set GPIO list <gpio_number, value> for GPIOs to set when streaming is disabled
        """
    def setGpiosOnStreamOn(self, list: dict[int, int]) -> None:
        """setGpiosOnStreamOn(self: depthai.node.UVC, list: dict[int, int]) -> None

        Set GPIO list <gpio_number, value> for GPIOs to set when streaming is enabled
        """
    @property
    def input(self) -> depthai.Node.Input: ...

class VideoEncoder(depthai.Node):
    Properties: ClassVar[type[depthai.VideoEncoderProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getBitrate(self) -> int:
        """getBitrate(self: depthai.node.VideoEncoder) -> int

        Get bitrate in bps
        """
    def getBitrateKbps(self) -> int:
        """getBitrateKbps(self: depthai.node.VideoEncoder) -> int

        Get bitrate in kbps
        """
    def getFrameRate(self) -> float:
        """getFrameRate(self: depthai.node.VideoEncoder) -> float

        Get frame rate
        """
    def getHeight(self) -> int:
        """getHeight(self: depthai.node.VideoEncoder) -> int

        Get input height
        """
    def getKeyframeFrequency(self) -> int:
        """getKeyframeFrequency(self: depthai.node.VideoEncoder) -> int

        Get keyframe frequency
        """
    def getLossless(self) -> bool:
        """getLossless(self: depthai.node.VideoEncoder) -> bool

        Get lossless mode. Applies only when using [M]JPEG profile.
        """
    def getMaxOutputFrameSize(self) -> int:
        """getMaxOutputFrameSize(self: depthai.node.VideoEncoder) -> int"""
    def getNumBFrames(self) -> int:
        """getNumBFrames(self: depthai.node.VideoEncoder) -> int

        Get number of B frames
        """
    def getNumFramesPool(self) -> int:
        """getNumFramesPool(self: depthai.node.VideoEncoder) -> int

        Get number of frames in pool

        Returns:
            Number of pool frames
        """
    def getProfile(self) -> depthai.VideoEncoderProperties.Profile:
        """getProfile(self: depthai.node.VideoEncoder) -> depthai.VideoEncoderProperties.Profile

        Get profile
        """
    def getQuality(self) -> int:
        """getQuality(self: depthai.node.VideoEncoder) -> int

        Get quality
        """
    def getRateControlMode(self) -> depthai.VideoEncoderProperties.RateControlMode:
        """getRateControlMode(self: depthai.node.VideoEncoder) -> depthai.VideoEncoderProperties.RateControlMode

        Get rate control mode
        """
    def getSize(self) -> tuple[int, int]:
        """getSize(self: depthai.node.VideoEncoder) -> tuple[int, int]

        Get input size
        """
    def getWidth(self) -> int:
        """getWidth(self: depthai.node.VideoEncoder) -> int

        Get input width
        """
    def setBitrate(self, bitrate: int) -> None:
        """setBitrate(self: depthai.node.VideoEncoder, bitrate: int) -> None

        Set output bitrate in bps, for CBR rate control mode. 0 for auto (based on frame
        size and FPS). Applicable only to H264 and H265 profiles
        """
    def setBitrateKbps(self, bitrateKbps: int) -> None:
        """setBitrateKbps(self: depthai.node.VideoEncoder, bitrateKbps: int) -> None

        Set output bitrate in kbps, for CBR rate control mode. 0 for auto (based on
        frame size and FPS). Applicable only to H264 and H265 profiles
        """
    @overload
    def setDefaultProfilePreset(self, fps: float, profile: depthai.VideoEncoderProperties.Profile) -> None:
        """setDefaultProfilePreset(*args, **kwargs)
        Overloaded function.

        1. setDefaultProfilePreset(self: depthai.node.VideoEncoder, fps: float, profile: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified frame rate and profile

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile

        2. setDefaultProfilePreset(self: depthai.node.VideoEncoder, arg0: int, arg1: int, arg2: float, arg3: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified input size, frame rate and profile

        Parameter ``width``:
            Input frame width

        Parameter ``height``:
            Input frame height

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile

        3. setDefaultProfilePreset(self: depthai.node.VideoEncoder, arg0: tuple[int, int], arg1: float, arg2: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified input size, frame rate and profile

        Parameter ``size``:
            Input frame size

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile
        """
    @overload
    def setDefaultProfilePreset(self, arg0: int, arg1: int, arg2: float, arg3: depthai.VideoEncoderProperties.Profile) -> None:
        """setDefaultProfilePreset(*args, **kwargs)
        Overloaded function.

        1. setDefaultProfilePreset(self: depthai.node.VideoEncoder, fps: float, profile: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified frame rate and profile

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile

        2. setDefaultProfilePreset(self: depthai.node.VideoEncoder, arg0: int, arg1: int, arg2: float, arg3: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified input size, frame rate and profile

        Parameter ``width``:
            Input frame width

        Parameter ``height``:
            Input frame height

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile

        3. setDefaultProfilePreset(self: depthai.node.VideoEncoder, arg0: tuple[int, int], arg1: float, arg2: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified input size, frame rate and profile

        Parameter ``size``:
            Input frame size

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile
        """
    @overload
    def setDefaultProfilePreset(self, arg0: tuple[int, int], arg1: float, arg2: depthai.VideoEncoderProperties.Profile) -> None:
        """setDefaultProfilePreset(*args, **kwargs)
        Overloaded function.

        1. setDefaultProfilePreset(self: depthai.node.VideoEncoder, fps: float, profile: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified frame rate and profile

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile

        2. setDefaultProfilePreset(self: depthai.node.VideoEncoder, arg0: int, arg1: int, arg2: float, arg3: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified input size, frame rate and profile

        Parameter ``width``:
            Input frame width

        Parameter ``height``:
            Input frame height

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile

        3. setDefaultProfilePreset(self: depthai.node.VideoEncoder, arg0: tuple[int, int], arg1: float, arg2: depthai.VideoEncoderProperties.Profile) -> None

        Sets a default preset based on specified input size, frame rate and profile

        Parameter ``size``:
            Input frame size

        Parameter ``fps``:
            Frame rate in frames per second

        Parameter ``profile``:
            Encoding profile
        """
    def setFrameRate(self, frameRate: float) -> None:
        """setFrameRate(self: depthai.node.VideoEncoder, frameRate: float) -> None

        Sets expected frame rate

        Parameter ``frameRate``:
            Frame rate in frames per second
        """
    def setKeyframeFrequency(self, freq: int) -> None:
        """setKeyframeFrequency(self: depthai.node.VideoEncoder, freq: int) -> None

        Set keyframe frequency. Every Nth frame a keyframe is inserted.

        Applicable only to H264 and H265 profiles

        Examples:

        - 30 FPS video, keyframe frequency: 30. Every 1s a keyframe will be inserted

        - 60 FPS video, keyframe frequency: 180. Every 3s a keyframe will be inserted
        """
    def setLossless(self, arg0: bool) -> None:
        """setLossless(self: depthai.node.VideoEncoder, arg0: bool) -> None

        Set lossless mode. Applies only to [M]JPEG profile

        Parameter ``lossless``:
            True to enable lossless jpeg encoding, false otherwise
        """
    def setMaxOutputFrameSize(self, maxFrameSize: int) -> None:
        """setMaxOutputFrameSize(self: depthai.node.VideoEncoder, maxFrameSize: int) -> None

        Specifies maximum output encoded frame size
        """
    def setNumBFrames(self, numBFrames: int) -> None:
        """setNumBFrames(self: depthai.node.VideoEncoder, numBFrames: int) -> None

        Set number of B frames to be inserted. Applicable only to H264 and H265 profiles
        """
    def setNumFramesPool(self, frames: int) -> None:
        """setNumFramesPool(self: depthai.node.VideoEncoder, frames: int) -> None

        Set number of frames in pool

        Parameter ``frames``:
            Number of pool frames
        """
    @overload
    def setProfile(self, profile: depthai.VideoEncoderProperties.Profile) -> None:
        """setProfile(*args, **kwargs)
        Overloaded function.

        1. setProfile(self: depthai.node.VideoEncoder, profile: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile

        2. setProfile(self: depthai.node.VideoEncoder, arg0: tuple[int, int], arg1: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile

        3. setProfile(self: depthai.node.VideoEncoder, arg0: int, arg1: int, arg2: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile
        """
    @overload
    def setProfile(self, arg0: tuple[int, int], arg1: depthai.VideoEncoderProperties.Profile) -> None:
        """setProfile(*args, **kwargs)
        Overloaded function.

        1. setProfile(self: depthai.node.VideoEncoder, profile: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile

        2. setProfile(self: depthai.node.VideoEncoder, arg0: tuple[int, int], arg1: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile

        3. setProfile(self: depthai.node.VideoEncoder, arg0: int, arg1: int, arg2: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile
        """
    @overload
    def setProfile(self, arg0: int, arg1: int, arg2: depthai.VideoEncoderProperties.Profile) -> None:
        """setProfile(*args, **kwargs)
        Overloaded function.

        1. setProfile(self: depthai.node.VideoEncoder, profile: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile

        2. setProfile(self: depthai.node.VideoEncoder, arg0: tuple[int, int], arg1: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile

        3. setProfile(self: depthai.node.VideoEncoder, arg0: int, arg1: int, arg2: depthai.VideoEncoderProperties.Profile) -> None

        Set encoding profile
        """
    def setQuality(self, quality: int) -> None:
        """setQuality(self: depthai.node.VideoEncoder, quality: int) -> None

        Set quality for [M]JPEG profile

        Parameter ``quality``:
            Value between 0-100%. Approximates quality
        """
    def setRateControlMode(self, mode: depthai.VideoEncoderProperties.RateControlMode) -> None:
        """setRateControlMode(self: depthai.node.VideoEncoder, mode: depthai.VideoEncoderProperties.RateControlMode) -> None

        Set rate control mode Applicable only to H264 and H265 profiles
        """
    @property
    def bitstream(self) -> depthai.Node.Output: ...
    @property
    def input(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...

class Warp(depthai.Node):
    Properties: ClassVar[type[depthai.WarpProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getHwIds(self) -> list[int]:
        """getHwIds(self: depthai.node.Warp) -> list[int]

        Retrieve which hardware warp engines to use
        """
    def getInterpolation(self) -> depthai.Interpolation:
        """getInterpolation(self: depthai.node.Warp) -> depthai.Interpolation

        Retrieve which interpolation method to use
        """
    def setHwIds(self, arg0: list[int]) -> None:
        """setHwIds(self: depthai.node.Warp, arg0: list[int]) -> None

        Specify which hardware warp engines to use

        Parameter ``ids``:
            Which warp engines to use (0, 1, 2)
        """
    def setInterpolation(self, arg0: depthai.Interpolation) -> None:
        """setInterpolation(self: depthai.node.Warp, arg0: depthai.Interpolation) -> None

        Specify which interpolation method to use

        Parameter ``interpolation``:
            type of interpolation
        """
    def setMaxOutputFrameSize(self, arg0: int) -> None:
        """setMaxOutputFrameSize(self: depthai.node.Warp, arg0: int) -> None

        Specify maximum size of output image.

        Parameter ``maxFrameSize``:
            Maximum frame size in bytes
        """
    def setNumFramesPool(self, arg0: int) -> None:
        """setNumFramesPool(self: depthai.node.Warp, arg0: int) -> None

        Specify number of frames in pool.

        Parameter ``numFramesPool``:
            How many frames should the pool have
        """
    @overload
    def setOutputSize(self, arg0: int, arg1: int) -> None:
        """setOutputSize(*args, **kwargs)
        Overloaded function.

        1. setOutputSize(self: depthai.node.Warp, arg0: int, arg1: int) -> None

        Sets output frame size in pixels

        Parameter ``size``:
            width and height in pixels

        2. setOutputSize(self: depthai.node.Warp, arg0: tuple[int, int]) -> None
        """
    @overload
    def setOutputSize(self, arg0: tuple[int, int]) -> None:
        """setOutputSize(*args, **kwargs)
        Overloaded function.

        1. setOutputSize(self: depthai.node.Warp, arg0: int, arg1: int) -> None

        Sets output frame size in pixels

        Parameter ``size``:
            width and height in pixels

        2. setOutputSize(self: depthai.node.Warp, arg0: tuple[int, int]) -> None
        """
    @overload
    def setWarpMesh(self, arg0: list[depthai.Point2f], arg1: int, arg2: int) -> None:
        """setWarpMesh(*args, **kwargs)
        Overloaded function.

        1. setWarpMesh(self: depthai.node.Warp, arg0: list[depthai.Point2f], arg1: int, arg2: int) -> None

        2. setWarpMesh(self: depthai.node.Warp, arg0: list[tuple[float, float]], arg1: int, arg2: int) -> None
        """
    @overload
    def setWarpMesh(self, arg0: list[tuple[float, float]], arg1: int, arg2: int) -> None:
        """setWarpMesh(*args, **kwargs)
        Overloaded function.

        1. setWarpMesh(self: depthai.node.Warp, arg0: list[depthai.Point2f], arg1: int, arg2: int) -> None

        2. setWarpMesh(self: depthai.node.Warp, arg0: list[tuple[float, float]], arg1: int, arg2: int) -> None
        """
    @property
    def inputImage(self) -> depthai.Node.Input: ...
    @property
    def out(self) -> depthai.Node.Output: ...

class XLinkIn(depthai.Node):
    Properties: ClassVar[type[depthai.XLinkInProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getMaxDataSize(self) -> int:
        """getMaxDataSize(self: depthai.node.XLinkIn) -> int

        Get maximum messages size in bytes
        """
    def getNumFrames(self) -> int:
        """getNumFrames(self: depthai.node.XLinkIn) -> int

        Get number of frames in pool
        """
    def getStreamName(self) -> str:
        """getStreamName(self: depthai.node.XLinkIn) -> str

        Get stream name
        """
    def setMaxDataSize(self, maxDataSize: int) -> None:
        """setMaxDataSize(self: depthai.node.XLinkIn, maxDataSize: int) -> None

        Set maximum message size it can receive

        Parameter ``maxDataSize``:
            Maximum size in bytes
        """
    def setNumFrames(self, numFrames: int) -> None:
        """setNumFrames(self: depthai.node.XLinkIn, numFrames: int) -> None

        Set number of frames in pool for sending messages forward

        Parameter ``numFrames``:
            Maximum number of frames in pool
        """
    def setStreamName(self, streamName: str) -> None:
        """setStreamName(self: depthai.node.XLinkIn, streamName: str) -> None

        Specifies XLink stream name to use.

        The name should not start with double underscores '__', as those are reserved
        for internal use.

        Parameter ``name``:
            Stream name
        """
    @property
    def out(self) -> depthai.Node.Output: ...

class XLinkOut(depthai.Node):
    Properties: ClassVar[type[depthai.XLinkOutProperties]] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getFpsLimit(self) -> float:
        """getFpsLimit(self: depthai.node.XLinkOut) -> float

        Get rate limit in messages per second
        """
    def getMetadataOnly(self) -> bool:
        """getMetadataOnly(self: depthai.node.XLinkOut) -> bool

        Get whether to transfer only messages attributes and not buffer data
        """
    def getStreamName(self) -> str:
        """getStreamName(self: depthai.node.XLinkOut) -> str

        Get stream name
        """
    def setFpsLimit(self, fpsLimit: float) -> None:
        """setFpsLimit(self: depthai.node.XLinkOut, fpsLimit: float) -> None

        Specifies a message sending limit. It's approximated from specified rate.

        Parameter ``fps``:
            Approximate rate limit in messages per second
        """
    def setMetadataOnly(self, arg0: bool) -> None:
        """setMetadataOnly(self: depthai.node.XLinkOut, arg0: bool) -> None

        Specify whether to transfer only messages attributes and not buffer data
        """
    def setStreamName(self, streamName: str) -> None:
        """setStreamName(self: depthai.node.XLinkOut, streamName: str) -> None

        Specifies XLink stream name to use.

        The name should not start with double underscores '__', as those are reserved
        for internal use.

        Parameter ``name``:
            Stream name
        """
    @property
    def input(self) -> depthai.Node.Input: ...

class YoloDetectionNetwork(DetectionNetwork):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getAnchorMasks(self) -> dict[str, list[int]]:
        """getAnchorMasks(self: depthai.node.YoloDetectionNetwork) -> dict[str, list[int]]

        Get anchor masks
        """
    def getAnchors(self) -> list[float]:
        """getAnchors(self: depthai.node.YoloDetectionNetwork) -> list[float]

        Get anchors
        """
    def getCoordinateSize(self) -> int:
        """getCoordinateSize(self: depthai.node.YoloDetectionNetwork) -> int

        Get coordianate size
        """
    def getIouThreshold(self) -> float:
        """getIouThreshold(self: depthai.node.YoloDetectionNetwork) -> float

        Get Iou threshold
        """
    def getNumClasses(self) -> int:
        """getNumClasses(self: depthai.node.YoloDetectionNetwork) -> int

        Get num classes
        """
    def setAnchorMasks(self, anchorMasks: dict[str, list[int]]) -> None:
        """setAnchorMasks(self: depthai.node.YoloDetectionNetwork, anchorMasks: dict[str, list[int]]) -> None

        Set anchor masks
        """
    def setAnchors(self, anchors: list[float]) -> None:
        """setAnchors(self: depthai.node.YoloDetectionNetwork, anchors: list[float]) -> None

        Set anchors
        """
    def setCoordinateSize(self, coordinates: int) -> None:
        """setCoordinateSize(self: depthai.node.YoloDetectionNetwork, coordinates: int) -> None

        Set coordianate size
        """
    def setIouThreshold(self, thresh: float) -> None:
        """setIouThreshold(self: depthai.node.YoloDetectionNetwork, thresh: float) -> None

        Set Iou threshold
        """
    def setNumClasses(self, numClasses: int) -> None:
        """setNumClasses(self: depthai.node.YoloDetectionNetwork, numClasses: int) -> None

        Set num classes
        """

class YoloSpatialDetectionNetwork(SpatialDetectionNetwork):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getAnchorMasks(self) -> dict[str, list[int]]:
        """getAnchorMasks(self: depthai.node.YoloSpatialDetectionNetwork) -> dict[str, list[int]]

        Get anchor masks
        """
    def getAnchors(self) -> list[float]:
        """getAnchors(self: depthai.node.YoloSpatialDetectionNetwork) -> list[float]

        Get anchors
        """
    def getCoordinateSize(self) -> int:
        """getCoordinateSize(self: depthai.node.YoloSpatialDetectionNetwork) -> int

        Get coordianate size
        """
    def getIouThreshold(self) -> float:
        """getIouThreshold(self: depthai.node.YoloSpatialDetectionNetwork) -> float

        Get Iou threshold
        """
    def getNumClasses(self) -> int:
        """getNumClasses(self: depthai.node.YoloSpatialDetectionNetwork) -> int

        Get num classes
        """
    def setAnchorMasks(self, anchorMasks: dict[str, list[int]]) -> None:
        """setAnchorMasks(self: depthai.node.YoloSpatialDetectionNetwork, anchorMasks: dict[str, list[int]]) -> None

        Set anchor masks
        """
    def setAnchors(self, anchors: list[float]) -> None:
        """setAnchors(self: depthai.node.YoloSpatialDetectionNetwork, anchors: list[float]) -> None

        Set anchors
        """
    def setCoordinateSize(self, coordinates: int) -> None:
        """setCoordinateSize(self: depthai.node.YoloSpatialDetectionNetwork, coordinates: int) -> None

        Set coordianate size
        """
    def setIouThreshold(self, thresh: float) -> None:
        """setIouThreshold(self: depthai.node.YoloSpatialDetectionNetwork, thresh: float) -> None

        Set Iou threshold
        """
    def setNumClasses(self, numClasses: int) -> None:
        """setNumClasses(self: depthai.node.YoloSpatialDetectionNetwork, numClasses: int) -> None

        Set num classes
        """
