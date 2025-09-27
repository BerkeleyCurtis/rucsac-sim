import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from mujoco.glfw import glfw
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mujoco_base import MuJoCoBase


class ControlUnicycle(MuJoCoBase):
    def __init__(self, xml_path, alpha=0.9995, dt=0.002):
        super().__init__(xml_path)
        self.actuator_type = "torque"
        self.gyro_data = np.zeros(3)
        self.accel_data = np.zeros(3)
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = 0.0
        self.alpha = alpha
        self.dt = dt
        self.angle = 0.0  # Initial angle estimate
        self.position = [0.0, 0.0, 0.0] # Initial position estimate

    def reset(self):
        # Set initial angle of pendulum
        self.data.qpos[0] = np.pi/2

        # Set camera configuration
        # self.cam.azimuth = 95.0
        # self.cam.distance = 5.0
        # self.cam.elevation = -5
        # self.cam.lookat = np.array([0.012768, -0.000000, 1.254336])

        # Set camera configuration for a side view of the unicycle
        self.cam.azimuth = 180.0  # Side view (180 degrees)
        self.cam.distance = 5.0
        self.cam.elevation = 0.0  # Level with the robot
        self.cam.lookat = np.array([0.0, 0.0, 1.2])  # Centered on the main body


        mj.set_mjcb_control(self.controller)

    def pos_update(self):

        self.position = [self.position[0] + 0.0 * self.dt, 
                        self.position[1] + (np.cos(self.angle) * self.accel_data[1] + np.sin(self.angle) * self.accel_data[2]) * self.dt, 
                        self.position[2] + (np.sin(self.angle) * self.accel_data[1] + np.cos(self.angle) * self.accel_data[2] - 9.81) * self.dt]
        
        # print(f"Position: {self.position}")
        return self.position
    
    def angle_update(self):
        """
        Updates the angle estimate using new accelerometer and gyroscope data.

        Args:
            accel_data (float): Accelerometer reading (e.g., in g's).
                                For roll, this would be Ay/g or similar;
                                for pitch, Ax/g.
            gyro_data (float): Gyroscope reading (e.g., in degrees/second or radians/second).
                                For roll, this would be Gx; for pitch, Gy.

        Returns:
            float: The updated estimated angle.
        """
        # Calculate angle from Accel
        angle_accel = np.atan2(self.accel_data[1], np.sqrt(self.accel_data[0]**2 + self.accel_data[2]**2))

        # Integrate gyroscope data to get angular change
        angle_gyro_change = self.gyro_data[0] * self.dt
        # print(f"Gyro angle: {np.rad2deg(angle_gyro_change)}, Accel angle: {np.rad2deg(angle_accel)}")

        # Apply the complementary filter equation
        self.angle = self.alpha * (self.angle + angle_gyro_change) + (1 - self.alpha) * angle_accel
        # print(f"Angle is: {np.rad2deg(self.angle)} degrees")
        return self.angle

    def balance_control(self, target_angle=0.0, 
                        kp=400.0, ki=0.0, kd=0.00001):
        """
        Computes control output for an inverted pendulum using IMU inputs.

        Parameters:
            accel_angle (float): Angle from accelerometer (degrees or radians).
            gyro_rate (float): Angular velocity from gyroscope (deg/s or rad/s).
            target_angle (float): Desired upright angle (default 0.0).
            kp, ki, kd (float): PID controller gains.
            dt (float): Time step (seconds).

        Returns:
            control (float): Control output (e.g., motor torque or force).
        """
        # angle = self.angle_update(accel_data, gyro_data)
        quat = self.data.qpos[3:7]
        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        # print(f"Ground truth angle: {euler[2]} degrees")
        # ======================= PID calculations ==============================
        ang_error = target_angle - self.angle
        self.integral += ang_error * self.dt
        derivative = (ang_error - self.prev_error) / self.dt
        self.prev_error = ang_error
        control = kp * ang_error + ki * self.integral + kd * derivative
        print(f"Control: {control}, Angle: {np.rad2deg(self.angle)}, Diff from truth: {np.rad2deg(self.angle)-euler[2]}")
        # ======================= Position control ==============================
        # pos_error = 0.0 - self.position[1]
        # kd_pos = 10.0
        # pos_control = kd_pos * pos_error ** 2 * np.sign(pos_error)
        # print(f"Error from position: {pos_error}, Position: {self.position[1]}, Pos control: {pos_control}")
        return control

    def controller(self, model, data):
        """
        This function implements a PD controller

        Since there are no gravity compensation,
        it will not be very accurate at tracking
        the set point. It will be accurate is
        gravity is turned off.
        """
        self.time = self.data.time
        self.dt = self.time - self.last_time
        self.last_time = self.time
        # print(f"dt: {self.dt}")
        # Print out the angle of the unicycle wheel (hinge joint "pin")
        wheel_angle = self.data.joint('pin').qpos
        print(f"Unicycle wheel angle (rad): {wheel_angle}")
        self.gyro_data = self.data.sensor('imu_gyro').data
        self.accel_data = self.data.sensor('imu_accel').data
        self.pos_update()
        angle = self.angle_update()
        control = self.balance_control()
        # control = 0.0
        if self.actuator_type == "torque":
            self.model.actuator_gainprm[0, 0] = 100
            self.data.ctrl[0] = control
        elif self.actuator_type == "servo":
            # kp = 0.1
            # self.model.actuator_gainprm[1, 0] = kp
            # self.model.actuator_biasprm[1, 1] = -kp
            # # self.data.ctrl[1] = -0.5
            # self.data.ctrl[1] = 0
            kv = 12.0
            self.model.actuator_gainprm[2, 0] = kv
            # self.model.actuator_biasprm[2, 2] = 0
            self.data.ctrl[2] = control

    def simulate(self):
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)
            
            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


def main():
    xml_path = "./xml/unicycle.xml"
    sim = ControlUnicycle(xml_path)
    sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()
