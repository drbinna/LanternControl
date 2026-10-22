#!/usr/bin/env python3
"""Lantern perception: object detection + tracking on the D455 colour camera.

  /camera/camera/color/image_raw ──► YOLO (TensorRT) ──► tracker ──► /lantern/detections

This is NOT part of the flight stack. It runs as its own service
(lantern-perception) at low priority, only reads the colour camera, and
never talks to PX4. If it crashes or is stopped, flying is unaffected; if
it is restarted, the SLAM origin is untouched.

Published on /lantern/detections (std_msgs/String, JSON) for every processed
frame, plus a heartbeat twice a second when there is nothing to process:

  {"state": "RUNNING", "reason": "", "stamp": 1729560000.12,
   "w": 640, "h": 360, "fps": 14.8, "infer_ms": 9.1, "model": "yolo11n.engine",
   "tracks": [{"id": 7, "label": "person", "conf": 0.88,
               "box": [x1, y1, x2, y2], "vel": [px/s, px/s], "age": 3.2}]}

state is one of LOADING, RUNNING, NO_MODEL, NO_CAMERA, ERROR.

Settings (environment, e.g. /etc/lantern/perception.env):
  LANTERN_MODEL     model file (default ~/lantern/models/yolo11n.engine,
                    falls back to the .pt next to it)
  LANTERN_CLASSES   comma-separated labels to report
                    (default person,bicycle,car,motorcycle,bus,truck)
  LANTERN_DETECT_FPS  max frames per second to process (default 15)
"""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import String

import tracker as trk

COLOR_TOPIC = '/camera/camera/color/image_raw'
OUT_TOPIC = '/lantern/detections'
MODELS = Path(os.environ.get('LANTERN_MODELS',
                             str(Path.home() / 'lantern' / 'models')))
MODEL = os.environ.get('LANTERN_MODEL', str(MODELS / 'yolo11n.engine'))
CLASSES = [c.strip() for c in os.environ.get(
    'LANTERN_CLASSES', 'person,bicycle,car,motorcycle,bus,truck').split(',')
    if c.strip()]
MAX_FPS = float(os.environ.get('LANTERN_DETECT_FPS', '15'))
IMGSZ = 640
NO_FRAMES_S = 2.0


def to_bgr(msg):
    """sensor_msgs/Image (colour) -> BGR numpy, or None."""
    a = np.frombuffer(msg.data, dtype=np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    if enc in ('rgb8', 'bgr8'):
        img = a.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        return np.ascontiguousarray(img[:, :, ::-1]) if enc == 'rgb8' else img
    if enc in ('yuv422', 'yuyv', 'yuv422_yuy2'):
        import cv2
        return cv2.cvtColor(a.reshape(h, step)[:, :w * 2].reshape(h, w, 2),
                            cv2.COLOR_YUV2BGR_YUY2)
    return None


class Detector:
    """YOLO via Ultralytics; TensorRT .engine when available."""

    def __init__(self, path):
        from ultralytics import YOLO      # imported here so a missing
        p = Path(path)                    # package is a clean NO_MODEL state
        if not p.exists() and p.suffix == '.engine' and \
                p.with_suffix('.pt').exists():
            p = p.with_suffix('.pt')
        if not p.exists():
            raise FileNotFoundError(
                '%s not found - run lantern-perception-setup' % path)
        self.name = p.name
        self.model = YOLO(str(p), task='detect')
        names = self.model.names
        self.names = dict(names) if isinstance(names, dict) else \
            dict(enumerate(names))
        wanted = {c.lower() for c in CLASSES}
        self.class_ids = [i for i, n in self.names.items()
                          if n.lower() in wanted] or None
        # first inference builds CUDA context / TensorRT runtime
        self.detect(np.zeros((360, 640, 3), np.uint8))

    def detect(self, img):
        r = self.model.predict(img, imgsz=IMGSZ, conf=trk.LOW,
                               classes=self.class_ids, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 6))
        return np.concatenate([b.xyxy.cpu().numpy(),
                               b.conf.cpu().numpy()[:, None],
                               b.cls.cpu().numpy()[:, None]], axis=1)


class Perception(Node):
    def __init__(self):
        super().__init__('lantern_perception')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(String, OUT_TOPIC, 10)
        self.create_subscription(Image, COLOR_TOPIC, self.on_image, qos)

        self.state, self.reason = 'LOADING', 'loading model'
        self.detector = None
        self.tracker = trk.Tracker()
        self.cond = threading.Condition()
        self.frame = None              # newest unprocessed frame only
        self.last_frame = 0.0
        self.last_pub = 0.0
        self.fps = 0.0
        self.infer_ms = 0.0
        self.create_timer(0.5, self.heartbeat)
        threading.Thread(target=self.worker, daemon=True).start()

    # ------------------------------------------------------------------ io
    def on_image(self, msg):
        self.last_frame = time.monotonic()
        with self.cond:
            self.frame = msg           # drop anything not yet processed
            self.cond.notify()

    def publish(self, tracks=(), w=0, h=0, stamp=0.0):
        self.last_pub = time.monotonic()
        out = String()
        out.data = json.dumps({
            'state': self.state, 'reason': self.reason, 'stamp': stamp,
            'w': w, 'h': h, 'fps': round(self.fps, 1),
            'infer_ms': round(self.infer_ms, 1),
            'model': self.detector.name if self.detector else '',
            'tracks': [t.as_dict(self.detector.names if self.detector
                                 else None) for t in tracks]})
        self.pub.publish(out)

    def heartbeat(self):
        if self.state == 'RUNNING' and self.last_frame and \
                time.monotonic() - self.last_frame > NO_FRAMES_S:
            self.set_state('NO_CAMERA', 'no colour frames')
            self.tracker.reset()
        if time.monotonic() - self.last_pub > 0.4:
            self.publish()

    def set_state(self, state, reason=''):
        if state != self.state:
            text = 'PERCEPTION %s %s' % (state, reason)
            if state in ('NO_MODEL', 'ERROR'):
                self.get_logger().error(text)
            elif state == 'NO_CAMERA':
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)
        self.state, self.reason = state, reason

    # -------------------------------------------------------------- worker
    def worker(self):
        try:
            self.detector = Detector(MODEL)
            self.get_logger().info('model %s, reporting %s' % (
                self.detector.name, ', '.join(CLASSES)))
            self.set_state('NO_CAMERA', 'waiting for colour frames')
        except ImportError:
            self.set_state('NO_MODEL', 'ultralytics not installed - '
                           'run lantern-perception-setup')
            return                     # heartbeat keeps reporting why
        except Exception as e:
            self.set_state('NO_MODEL', str(e)[:160])
            return

        min_dt = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0
        last_t = 0.0
        while rclpy.ok():
            with self.cond:
                self.cond.wait_for(lambda: self.frame is not None, timeout=1.0)
                msg, self.frame = self.frame, None
            if msg is None:
                continue
            now = time.monotonic()
            if now - last_t < min_dt:
                continue
            try:
                img = to_bgr(msg)
                if img is None:
                    self.set_state('ERROR', 'unsupported encoding %s'
                                   % msg.encoding)
                    continue
                t0 = time.monotonic()
                dets = self.detector.detect(img)
                self.infer_ms = 0.8 * self.infer_ms + \
                    0.2 * (time.monotonic() - t0) * 1000
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                tracks = self.tracker.step(dets, stamp)
                if last_t:
                    self.fps = 0.9 * self.fps + 0.1 / max(now - last_t, 1e-3)
                last_t = now
                self.set_state('RUNNING')
                self.publish(tracks, img.shape[1], img.shape[0], stamp)
            except Exception as e:     # never die on one bad frame
                self.set_state('ERROR', str(e)[:160])
                time.sleep(0.2)


def main():
    rclpy.init()
    node = Perception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
