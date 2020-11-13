import airsim
from airsim import MultirotorClient, YawMode, DrivetrainType
from PIL import Image
import numpy as np

class myAirSimClient(MultirotorClient):
    """
    Client class for interfacing with airsim api.
    """
    def __init__(self):
        MultirotorClient.__init__(self)
        self.confirmConnection()
        self.enableApiControl(True)
        self.armDisarm(True)

        self.home_pos = self.getMultirotorState().kinematics_estimated.position
        self.home_orientation = self.getMultirotorState().kinematics_estimated.orientation
        self.home = np.array([self.home_pos.x_val, self.home_pos.y_val, self.home_pos.z_val])

        self.camera_name = "bottom_center_custom"

    def sim_reset(self):
        """
        Resets drone in simulation and takes off.
        """
        self.reset()
        self.enableApiControl(True)
        self.armDisarm(True)
        self.moveToZAsync(3, 1).join()

    def modifyVel(self, offset):
        """
        Modify velocity by specified offset.

        Args:
            offset (tuple(double, double, double, double)): x, y, z, yaw rate

        """
        vx, vy, vz, yaw_rate = offset
        yaw_mode = YawMode(is_rate=True, yaw_or_rate=yaw_rate)

        curr_vel = self.getMultirotorState().kinematics_estimated.linear_velocity
        vx += curr_vel.x_val
        vy += curr_vel.y_val
        vz += curr_vel.z_val
        
        self.moveByVelocityAsync(vx , vy, vz, 1.0, yaw_mode=yaw_mode).join()

    def getDepthImage(self):
        """
        Downsample depth image from camera.
        """
        responses = self.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d) # not sure what this is
        img2d = np.reshape(img1d, (responses[0].height, responses[1].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L'))
        
        return im_final
