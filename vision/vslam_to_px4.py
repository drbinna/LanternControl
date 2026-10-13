#!/usr/bin/env python3
"""Lantern vision bridge + watchdog.

Isaac VSLAM odometry (ROS, FLU)  ->  PX4 /fmu/in/vehicle_visual_odometry (FRD)

Watchdog: a bad vision sample is never sent to PX4. Checks on every sample:
  * VSLAM reports tracking failed (vo_state == 2)
  * position jumps further than the drone could physically move since the
    last sample (MAX_SPEED * dt + NOISE)
  * reported velocity is impossible (> MAX_SPEED * 1.5)
  * NaN values or a huge covariance

On a failure the bridge goes LOST and stops publishing. PX4 then times the
vision source out and runs its own position-loss failsafe (COM_POSCTL_NAVL),
which also puts a warning in QGroundControl.

To recover it needs RECOVER_SAMPLES consecutive good samples (~0.5 s), then
resumes with reset_counter + 1 so PX4 knows the position may have shifted
and re-anchors instead of flying toward the old origin.

Health is published as JSON on /lantern/vision_health for the dashboard.
"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from px4_msgs.msg import VehicleOdometry

# ---- tuning (indoor X650) -------------------------------------------------
MAX_SPEED = 4.0          # m/s - faster than we ever fly indoors
NOISE = 0.05             # m   - per-sample allowance for SLAM jitter
MAX_POS_VAR = 1.0        # m^2 - covariance above this = not trustworthy
RECOVER_SAMPLES = 45     # ~0.5 s of clean tracking before resuming
MIN_VAR = 1e-4
VEL_VAR = 0.01
LOG_EVERY = 900


class Bridge(Node):
    def __init__(self):
        super().__init__('vslam_to_px4')
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pub = self.create_publisher(
            VehicleOdometry, '/fmu/in/vehicle_visual_odometry', px4_qos)
        self.health_pub = self.create_publisher(
            String, '/lantern/vision_health', 10)
        self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry', self.cb, sub_qos)

        self.vo_state = None
        try:
            from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
            self.create_subscription(VisualSlamStatus, '/visual_slam/status',
                                     self.on_status, sub_qos)
        except ImportError:
            self.get_logger().warn('VSLAM status msg not found; '
                                   'using jump checks only')

        self.state = 'STARTING'      # STARTING / OK / LOST / RECOVERING
        self.reason = 'waiting for SLAM'
        self.prev = None             # (t_sample_s, [x, y, z]) last raw sample
        self.good_run = 0
        self.reset_counter = 0
        self.sent = 0
        self.rejected = 0
        self.losses = 0
        self.last_loss = ''
        self.last_msg = 0.0
        self.create_timer(0.2, self.publish_health)

    # ------------------------------------------------------------------
    def on_status(self, msg):
        self.vo_state = int(msg.vo_state)

    def set_state(self, state, reason):
        if state != self.state:
            # rclpy forbids one call site logging at different severities,
            # so each severity gets its own call.
            text = 'VISION %s: %s' % (state, reason)
            if state == 'LOST':
                self.losses += 1
                self.last_loss = reason
                self.get_logger().error(text)
            elif state == 'RECOVERING':
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)
        self.state, self.reason = state, reason

    def check(self, t, pos, vel, pvar):
        """Return None if the sample is plausible, else a reason string."""
        if any(math.isnan(v) or math.isinf(v) for v in pos + vel):
            return 'NaN in odometry'
        if self.vo_state == 2:
            return 'SLAM tracking failed'
        if max(pvar) > MAX_POS_VAR:
            return 'covariance too large'
        speed = math.sqrt(sum(v * v for v in vel))
        if speed > MAX_SPEED * 1.5:
            return 'impossible speed %.1f m/s' % speed
        if self.prev is not None:
            dt = t - self.prev[0]
            if dt <= 0:
                return 'stale timestamp'
            step = math.dist(pos, self.prev[1])
            limit = MAX_SPEED * min(dt, 0.5) + NOISE
            if step > limit:
                return 'jump %.2f m in %.0f ms' % (step, dt * 1000)
        return None

    def cb(self, msg):
        try:
            self._cb(msg)
        except Exception as e:  # a bug must not kill the bridge
            self.state, self.reason = 'LOST', 'bridge error: %s' % e
            self.good_run = 0
            self.get_logger().error('bridge error: %r' % e)

    def _cb(self, msg):
        self.last_msg = time.monotonic()
        st = msg.header.stamp
        t = st.sec + st.nanosec * 1e-9
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        cov = msg.pose.covariance

        # FLU -> FRD
        pos = [p.x, -p.y, -p.z]
        vel = [v.x, -v.y, -v.z]
        pvar = [max(cov[0], MIN_VAR), max(cov[7], MIN_VAR),
                max(cov[14], MIN_VAR)]

        bad = self.check(t, pos, vel, pvar)
        self.prev = (t, pos)

        if bad:
            self.rejected += 1
            self.good_run = 0
            if self.state != 'STARTING':
                self.set_state('LOST', bad)
            else:
                self.reason = bad
            return

        self.good_run += 1
        if self.state in ('LOST', 'RECOVERING', 'STARTING'):
            if self.good_run < RECOVER_SAMPLES:
                if self.state == 'LOST':
                    self.set_state('RECOVERING', 'tracking looks clean again')
                return
            if self.state != 'STARTING':
                self.reset_counter = (self.reset_counter + 1) % 256
            self.set_state('OK', 'tracking')

        out = VehicleOdometry()
        out.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        out.timestamp_sample = int(st.sec * 1_000_000 + st.nanosec / 1000)
        out.pose_frame = 2          # POSE_FRAME_FRD
        out.position = [float(x) for x in pos]
        out.q = [float(o.w), float(o.x), float(-o.y), float(-o.z)]
        out.velocity_frame = 3      # VELOCITY_FRAME_BODY_FRD
        out.velocity = [float(x) for x in vel]
        out.angular_velocity = [float('nan')] * 3
        out.position_variance = [float(x) for x in pvar]
        out.orientation_variance = [max(cov[21], MIN_VAR),
                                    max(cov[28], MIN_VAR),
                                    max(cov[35], MIN_VAR)]
        out.velocity_variance = [VEL_VAR] * 3
        out.reset_counter = self.reset_counter
        out.quality = 100
        self.pub.publish(out)

        self.sent += 1
        if self.sent % LOG_EVERY == 0:
            self.get_logger().info(
                'sent %d | rejected %d | losses %d | pos %.2f %.2f %.2f' %
                (self.sent, self.rejected, self.losses, *pos))

    def publish_health(self):
        state, reason = self.state, self.reason
        if self.last_msg and time.monotonic() - self.last_msg > 0.5:
            if state == 'OK':
                self.set_state('LOST', 'no SLAM output')
                self.good_run = 0
                self.prev = None
            state, reason = self.state, 'no SLAM output'
        msg = String()
        msg.data = json.dumps({
            'state': state, 'reason': reason, 'sent': self.sent,
            'rejected': self.rejected, 'losses': self.losses,
            'last_loss': self.last_loss,
            'reset_counter': self.reset_counter, 'vo_state': self.vo_state})
        self.health_pub.publish(msg)


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
