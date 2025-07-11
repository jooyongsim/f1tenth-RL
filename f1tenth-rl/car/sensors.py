import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import time
import math
import argparse
import subprocess
import numpy as np


class Sensors():
    def __init__(self, is_simulator=False, use_back_sensors=False):
        self.is_simulator = is_simulator
        self.use_back_sensors = use_back_sensors
        self.custom_lidar_callback = None
        self.lidar_data = None
        self.odometry = None
        if not is_simulator:
            odom_topic = "/vesc/odom"
        else:
            odom_topic = "/odom"
        self.lidar_subscriber = rospy.Subscriber("scan", LaserScan, self.lidar_callback)
        self.odom_subscriber = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)
        self.imu_subscriber = rospy.Subscriber("imu", Imu, self.imu_callback)

        if not is_simulator:
            if use_back_sensors:
                #Setup pin of Orbitty Carrier for NVIDIA Jetson TX2
                #check out http://connecttech.com/resource-center/kdb342-using-gpio-connect-tech-jetson-tx1-carriers/
                subprocess.run('echo 388 > /sys/class/gpio/export', shell=True)
                subprocess.run('echo 298 > /sys/class/gpio/export', shell=True)

    def add_lidar_callback(self, callback):
        self.custom_lidar_callback = callback

    def lidar_callback(self, lidar_data):
        self.lidar_data = lidar_data
        if self.custom_lidar_callback:
            self.custom_lidar_callback(lidar_data)

    def odometry_callback(self, odometry):
        self.odometry = odometry

    def imu_callback(self, imu):
        self.imu = imu

    def get_lidar_ranges(self):
        if not self.is_simulator:
            return self.lidar_data.ranges[:-1]
        else:
            return self.lidar_data.ranges

    def get_car_linear_velocity(self):
        if self.odometry is None or (self.odometry.twist.twist.linear.x == 0 and self.odometry.twist.twist.linear.x == 0):
            return 0
        return math.sqrt(self.odometry.twist.twist.linear.x ** 2 + self.odometry.twist.twist.linear.y ** 2)

    def get_car_angular_acceleration(self):
        return self.odometry.twist.twist.angular

    def get_car_linear_acceleration(self):
        return math.sqrt(self.imu.linear_acceleration.x ** 2 + self.imu.linear_acceleration.y ** 2)

    def get_car_orientation(self):
        q = self.imu.orientation
        return math.atan2(2.0*(q.w*q.z + q.x*q.y), q.w**2 + q.x**2 - q.y**2 - q.z**2)
    
    #car pose 추가
    def get_car_pose(self):
        """ 차량의 (x, y) 위치 및 yaw (방향)를 반환 """
        if self.odometry is None:
            return 0, 0, 0  # 기본값 반환 (데이터 없음)
    
        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y

        # Quaternion을 yaw로 변환
        q = self.odometry.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), q.w**2 + q.x**2 - q.y**2 - q.z**2)

        # print(f"[DEBUG] pose x: {x}, y: {y}, yaw: {yaw}")
        # if any(np.isnan(v) or np.isinf(v) for v in [x, y, yaw]):
        #    print("[WARNING] pose has NaN or Inf value!")

        # if abs(x) > 1000 or abs(y) > 1000:
        #    print("[WARNING] pose x/y too large:", x, y)
           
        # if abs(yaw) > 10:
        #     print("[WARNING] yaw value seems unrealistic:", yaw)

        return x, y, yaw

    def back_obstacle(self):
        if not self.is_simulator:
            if not self.use_back_sensors:
                return False
            #Read pin of Orbitty Carrier for NVIDIA Jetson TX2
            #check out http://connecttech.com/resource-center/kdb342-using-gpio-connect-tech-jetson-tx1-carriers/
            lx_value = subprocess.run('cat /sys/class/gpio/gpio298/value', shell=True, stdout=subprocess.PIPE)
            rx_value = subprocess.run('cat /sys/class/gpio/gpio388/value', shell=True, stdout=subprocess.PIPE)
            return (lx_value.stdout == b"0\n" or rx_value.stdout == b"0\n")
        else:
            back_lidar_ranges = self.lidar_data.ranges[:100] + self.lidar_data.ranges[-100:]
            return min(back_lidar_ranges) < 0.8

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
    args = parser.parse_args()

    rospy.init_node('sensors_test')
    sensor = Sensors(args.simulator)
    time.sleep(1)
    while True:
        print("######################################")
        print(sensor.lidar_data)
        print(sensor.odometry)
        print(sensor.get_car_linear_velocity())
        if not args.simulator:
            print(sensor.back_obstacle())
        time.sleep(5)