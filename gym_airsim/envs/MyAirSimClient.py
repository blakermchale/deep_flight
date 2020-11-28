import airsim
from airsim import MultirotorClient, YawMode, DrivetrainType
from PIL import Image
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

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

        self.curr_vel = np.zeros((1,3))

    def sim_reset(self):
        """
        Resets drone in simulation and takes off.
        """
        self.reset()
        self.curr_vel = np.zeros((1,3))
        self.simPause(False)
        self.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.armDisarm(True, vehicle_name=self.vehicle_name)
        self.moveToZAsync(-2, 0.5, vehicle_name=self.vehicle_name).join()
        self.simPause(True)

    def getDepthImage(self):
        """
        Downsample depth image from camera.
        """
        while True:
            responses = self.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)])
            img1d = np.array(responses[0].image_data_float, dtype=np.float)
            if responses[0].height != 0 and responses[0].width != 0:
                break
            print("Error getting image! Trying again!")
        img1d = 255/np.maximum(np.ones(img1d.size), img1d) # not sure what this is
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')).reshape(-1, 84, 84, 1)
        # im_final = np.array(image.resize((84, 84)).convert('L'))
        
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

    def modifyVel(self, vel_offset, yaw_rate):     
        """
        Modify velocity by specified offset and a given yaw rate.

        Args:
            vel_offset (double): forward velocity offset m/s
            yaw_rate (double): the yaw rate in deg/s
        """

        state = self.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated
        quat = state.orientation.to_numpy_array()
        self.curr_vel += vel_offset

        r = R.from_quat(quat)
        rpy = r.as_euler('XYZ', degrees=True)
        yaw = rpy[2]

        speed = np.linalg.norm(self.curr_vel[:2])
        vx = np.cos(yaw) * speed
        vy = np.sin(yaw) * speed

        yaw_mode = YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        cmd = self.moveByVelocityZAsync(vx, vy, -2, 1.0, yaw_mode=yaw_mode, vehicle_name=self.vehicle_name).join()
        
