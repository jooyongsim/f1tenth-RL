import rospy
from nav_msgs.msg import Odometry
import time

start_time = None
last_lap_time = None

# 기준 좌표 (랩 시작점)
START_X, START_Y = 0.0, 0.0
LAP_RADIUS = 1.0  # 기준점 반경 1m 이내일 때만 랩 인정

def odom_callback(msg):
    global start_time, last_lap_time
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    distance = ((x - START_X)**2 + (y - START_Y)**2)**0.5

    if distance < LAP_RADIUS:
        now = time.time()
        if last_lap_time is None:
            print("[시작 지점 통과]")
            last_lap_time = now
        elif now - last_lap_time > 5:  # 최소 5초 이상 지난 후에만 다음 랩으로 인정
            lap_time = now - last_lap_time
            print(f"[랩 완료] 소요 시간: {lap_time:.2f}초")
            last_lap_time = now

rospy.init_node("lap_timer")
rospy.Subscriber("/odom", Odometry, odom_callback)
rospy.spin()
