import airsim
from airsim import MultirotorClient, YawMode, DrivetrainType
from PIL import Image
import numpy as np

class MyAirSimClient(MultirotorClient):
    """
    Client class for interfacing with airsim api.
    """
    def __init__(self):
        MultirotorClient.__init__(self)
        self.confirmConnection()
        self.enableApiControl(True)
        self.armDisarm(True)

        self.vehicle_name = "SimpleFlight"
        self.camera_name = "front_center_custom"
        
        self.home_pos = self.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        self.home_orientation = self.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.orientation
        self.home = np.array([self.home_pos.x_val, self.home_pos.y_val, self.home_pos.z_val])

    def sim_reset(self):
        """
        Resets drone in simulation and takes off.
        """
        self.reset()
        self.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.armDisarm(True, vehicle_name=self.vehicle_name)
        self.moveToZAsync(-2, 0.5, vehicle_name=self.vehicle_name).join()

    def modifyVel(self, offset):
        """
        Modify velocity by specified offset.

        Args:
            offset (tuple(double, double, double, double)): x, y, z, yaw rate

        """
        vx, vy, vz, yaw_rate = offset
        yaw_mode = YawMode(is_rate=True, yaw_or_rate=yaw_rate)

        curr_vel = self.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.linear_velocity
        vx += curr_vel.x_val
        vy += curr_vel.y_val
        vz += curr_vel.z_val
        
        self.moveByVelocityAsync(vx , vy, vz, 1.0, yaw_mode=yaw_mode, vehicle_name=self.vehicle_name).join()

    def getDepthImage(self):
        """
        Downsample depth image from camera.
        """
        responses = self.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d) # not sure what this is
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')).reshape(-1, 84, 84, 1)
        
        return im_final

    def getState(self):
        """
        Get the current state of the drone in AirSim.

        Returns:
            has_collided (bool): if the drone has collided with an object
            curr_pos (tuple): the x,y,z state
        """        
        curr_pos = self.getMultirotorState().kinematics_estimated.position
        curr_pos = np.array([curr_pos.x_val, curr_pos.y_val, curr_pos.z_val])
        has_collided = self.simGetCollisionInfo().has_collided

        return has_collided, curr_pos
