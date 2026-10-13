#!/usr/bin/env python3
"""Lantern Control - web dashboard for the Lantern vision-positioning stack.

Serves a live dashboard on http://<jetson-ip>:8080 showing camera, SLAM,
PX4 link and EKF status, a top-down position trail, the infra camera feed,
and a few guarded controls (restart services - refused while armed).

Run:  python3 lantern_ui.py [--port 8080] [--demo]
"""
import argparse
import collections
import hmac
import json
import math
import os
import random
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIXHAWK_IP = os.environ.get('LANTERN_PIXHAWK_IP', '192.168.0.3')
CAMERA_USB = '8086:0b5c'
SERVICES = {'agent': 'microxrceagent', 'vision': 'lantern-vision'}
UI_KEY = os.environ.get('LANTERN_UI_KEY', '')
STALE_S = 1.5

NAV_STATES = {
    0: 'Manual', 1: 'Altitude', 2: 'Position', 3: 'Mission', 4: 'Hold',
    5: 'Return', 10: 'Acro', 12: 'Descend', 13: 'Terminate', 14: 'Offboard',
    15: 'Stabilized', 17: 'Takeoff', 18: 'Land', 19: 'Follow',
    20: 'Precision land', 21: 'Orbit',
}


class Rate:
    """Messages per second over a sliding window."""

    def __init__(self, window=2.0):
        self.window = window
        self.t = collections.deque()
        self.last = 0.0

    def tick(self):
        now = time.monotonic()
        self.last = now
        self.t.append(now)
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()

    def hz(self):
        now = time.monotonic()
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()
        return len(self.t) / self.window

    def fresh(self):
        return self.last and time.monotonic() - self.last < STALE_S


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.slam = Rate()
        self.bridge = Rate()
        self.lpos_rate = Rate()
        self.flags_rate = Rate()
        self.vo_state = None          # Isaac VSLAM status, if available
        self.vision_pos = None        # FRD from bridge
        self.flags = {}
        self.lpos = {}
        self.status = {}
        self.status_t = 0.0
        self.battery = {}
        self.battery_t = 0.0
        self.health = {'camera': None, 'pixhawk': None, 'agent': None,
                       'vision': None, 'temp_c': None, 'load': None,
                       'mem_pct': None}
        self.frame = None
        self.frame_id = 0
        self.viewers = 0
        self.video_available = False
        self.events = collections.deque(maxlen=40)
        self.snapshot = {}
        self.wd = {}
        self.wd_t = 0.0
        self.prev_checks = {}

    def event(self, text, level='info'):
        self.events.appendleft({'t': time.strftime('%H:%M:%S'),
                                'text': text, 'level': level})


S = State()


# --------------------------------------------------------------------- ROS
def start_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                           HistoryPolicy)
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image

    # BEST_EFFORT + VOLATILE subscribers match any publisher QoS.
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=5)

    try:
        import cv2
        import numpy as np
        S.video_available = True
    except ImportError:
        cv2 = np = None

    rclpy.init()
    node = Node('lantern_ui')

    def on_slam(msg):
        with S.lock:
            S.slam.tick()
    node.create_subscription(Odometry, '/visual_slam/tracking/odometry',
                             on_slam, qos)

    try:
        from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus

        def on_vs(msg):
            with S.lock:
                S.vo_state = int(msg.vo_state)
        node.create_subscription(VisualSlamStatus, '/visual_slam/status',
                                 on_vs, qos)
    except ImportError:
        pass

    from px4_msgs.msg import VehicleOdometry, VehicleLocalPosition
    from std_msgs.msg import String

    def on_wd(msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        with S.lock:
            # the loss counter catches losses shorter than the 4 Hz UI refresh
            prev = S.wd.get('losses')
            if prev is not None and d.get('losses', 0) > prev:
                S.event('VISION LOST · %s' % d.get('last_loss', '?'), 'bad')
            S.wd, S.wd_t = d, time.monotonic()
    node.create_subscription(String, '/lantern/vision_health', on_wd, 10)

    def on_bridge(msg):
        with S.lock:
            S.bridge.tick()
            S.vision_pos = [float(msg.position[0]), float(msg.position[1]),
                            float(msg.position[2])]
    node.create_subscription(VehicleOdometry,
                             '/fmu/in/vehicle_visual_odometry', on_bridge, qos)

    def on_lpos(msg):
        with S.lock:
            S.lpos_rate.tick()
            S.lpos = {'x': msg.x, 'y': msg.y, 'z': msg.z,
                      'vx': msg.vx, 'vy': msg.vy, 'vz': msg.vz,
                      'heading': msg.heading,
                      'xy_valid': bool(msg.xy_valid),
                      'z_valid': bool(msg.z_valid),
                      'eph': msg.eph, 'epv': msg.epv}
    node.create_subscription(VehicleLocalPosition,
                             '/fmu/out/vehicle_local_position', on_lpos, qos)

    try:
        from px4_msgs.msg import EstimatorStatusFlags

        def on_flags(msg):
            with S.lock:
                S.flags_rate.tick()
                S.flags = {k: bool(getattr(msg, k, False)) for k in (
                    'cs_ev_pos', 'cs_ev_vel', 'cs_ev_yaw', 'cs_ev_hgt',
                    'cs_tilt_align', 'cs_yaw_align', 'cs_in_air')}
        node.create_subscription(EstimatorStatusFlags,
                                 '/fmu/out/estimator_status_flags',
                                 on_flags, qos)
    except ImportError:
        pass

    try:
        from px4_msgs.msg import VehicleStatus

        def on_status(msg):
            with S.lock:
                S.status_t = time.monotonic()
                S.status = {'armed': int(msg.arming_state) == 2,
                            'nav_state': int(msg.nav_state)}
        node.create_subscription(VehicleStatus, '/fmu/out/vehicle_status',
                                 on_status, qos)
    except ImportError:
        pass

    try:
        from px4_msgs.msg import BatteryStatus

        def on_batt(msg):
            with S.lock:
                S.battery_t = time.monotonic()
                S.battery = {'v': float(msg.voltage_v),
                             'pct': float(msg.remaining) * 100.0}
        node.create_subscription(BatteryStatus, '/fmu/out/battery_status',
                                 on_batt, qos)
    except ImportError:
        pass

    # Camera feed: only subscribe while someone is watching (saves CPU).
    img = {'sub': None, 'last': 0.0}

    def on_image(msg):
        now = time.monotonic()
        if now - img['last'] < 0.1:          # ~10 fps to the browser
            return
        img['last'] = now
        try:
            a = np.frombuffer(msg.data, dtype=np.uint8)
            a = a.reshape(msg.height, msg.step)[:, :msg.width]
            ok, jpg = cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with S.lock:
                    S.frame = jpg.tobytes()
                    S.frame_id += 1
        except Exception:
            pass

    def manage_video():
        if not S.video_available:
            return
        with S.lock:
            want = S.viewers > 0
        if want and img['sub'] is None:
            img['sub'] = node.create_subscription(
                Image, '/camera/camera/infra1/image_rect_raw', on_image, qos)
        elif not want and img['sub'] is not None:
            node.destroy_subscription(img['sub'])
            img['sub'] = None
    node.create_timer(1.0, manage_video)

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


# ------------------------------------------------------------ system checks
def run(cmd, timeout=3):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except Exception:
        return None


def health_loop():
    while True:
        r = run(['lsusb', '-d', CAMERA_USB])
        camera = bool(r and r.returncode == 0 and r.stdout.strip())
        r = run(['ping', '-c', '1', '-W', '1', PIXHAWK_IP])
        pixhawk = bool(r and r.returncode == 0)
        r = run(['systemctl', 'is-active'] + list(SERVICES.values()))
        lines = (r.stdout.split() if r else []) + ['unknown'] * 2
        temps = []
        for z in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            try:
                temps.append(int(z.read_text()) / 1000.0)
            except Exception:
                pass
        try:
            load = float(Path('/proc/loadavg').read_text().split()[0])
        except Exception:
            load = None
        try:
            mi = dict(l.split(':', 1) for l in
                      Path('/proc/meminfo').read_text().splitlines())
            tot = int(mi['MemTotal'].split()[0])
            av = int(mi['MemAvailable'].split()[0])
            mem = 100.0 * (tot - av) / tot
        except Exception:
            mem = None
        with S.lock:
            S.health.update(camera=camera, pixhawk=pixhawk,
                            agent=lines[0] == 'active',
                            vision=lines[1] == 'active',
                            temp_c=max(temps) if temps else None,
                            load=load, mem_pct=mem)
        time.sleep(2.0)


# ----------------------------------------------------------- state builder
def build_snapshot():
    with S.lock:
        h = dict(S.health)
        slam_hz = S.slam.hz() if S.slam.fresh() else 0.0
        bridge_hz = S.bridge.hz() if S.bridge.fresh() else 0.0
        lpos = dict(S.lpos) if S.lpos_rate.fresh() else {}
        flags = dict(S.flags) if S.flags_rate.fresh() else {}
        status = dict(S.status) if time.monotonic() - S.status_t < 3 else {}
        batt = dict(S.battery) if time.monotonic() - S.battery_t < 5 else {}
        vpos = S.vision_pos if S.bridge.fresh() else None
        vo = S.vo_state
        wd = dict(S.wd) if time.monotonic() - S.wd_t < 2 else {}

    tracking_ok = slam_hz > 30 and vo not in (2,)
    fusing = flags.get('cs_ev_pos') if flags else None
    checks = [
        ('camera', 'Depth camera', h['camera'],
         'D455 on USB' if h['camera'] else 'D455 not detected'),
        ('pixhawk', 'Pixhawk link', h['pixhawk'],
         PIXHAWK_IP + (' reachable' if h['pixhawk'] else ' not answering')),
        ('agent', 'DDS agent', h['agent'],
         'running' if h['agent'] else 'service stopped'),
        ('vision', 'Vision service', h['vision'],
         'running' if h['vision'] else 'service stopped'),
        ('slam', 'Visual SLAM', tracking_ok,
         ('%.0f Hz' % slam_hz) if slam_hz else 'no output'),
        ('bridge', 'Vision → PX4', bridge_hz > 30,
         ('%.0f Hz' % bridge_hz) if bridge_hz else 'nothing sent'),
        ('watchdog', 'Vision watchdog', wd.get('state') == 'OK',
         ('OK · %d rejected' % wd.get('rejected', 0)) if wd.get('state') == 'OK'
         else (('%s · %s' % (wd['state'], wd.get('reason', ''))) if wd
               else 'not running')),
        ('fusing', 'PX4 using vision', fusing,
         'EKF fusing position' if fusing else
         ('not fused' if flags else 'no EKF data')),
        ('posvalid', 'Position valid', lpos.get('xy_valid'),
         ('±%.2f m' % lpos['eph']) if lpos.get('xy_valid') else
         ('invalid' if lpos else 'no data')),
    ]
    if fusing is None and lpos.get('xy_valid') and bridge_hz > 30:
        # flags topic not exported; infer from a valid vision-only estimate
        checks[7] = ('fusing', 'PX4 using vision', True, 'inferred (xy valid)')

    # event log on state transitions
    for key, label, ok, detail in checks:
        prev = S.prev_checks.get(key)
        if prev is not None and bool(ok) != prev:
            with S.lock:
                S.event('%s: %s' % (label, detail), 'ok' if ok else 'bad')
        S.prev_checks[key] = bool(ok)
    armed = status.get('armed')
    if S.prev_checks.get('_armed') is not None and \
            bool(armed) != S.prev_checks['_armed']:
        with S.lock:
            S.event('ARMED' if armed else 'Disarmed',
                    'warn' if armed else 'info')
    S.prev_checks['_armed'] = bool(armed)

    failing = [c for c in checks if not c[2]]
    wd_lost = wd.get('state') in ('LOST', 'RECOVERING')
    if wd_lost:
        verdict = ('FLYING · ' if armed else '') + 'VISION LOST · ' + \
            wd.get('reason', '')
        level = 'bad lost'
    elif armed:
        verdict = 'FLYING' if not failing else 'FLYING · ' + failing[0][1].upper() + ' LOST'
        level = 'fly' if not failing else 'bad'
    elif not failing:
        verdict, level = 'READY FOR POSITION MODE', 'ok'
    else:
        verdict, level = 'NOT READY · ' + failing[0][3], 'bad'

    with S.lock:
        events = list(S.events)
    return {
        't': time.time(),
        'verdict': verdict, 'level': level,
        'checks': [{'key': k, 'label': l, 'ok': bool(o), 'detail': d}
                   for k, l, o, d in checks],
        'slam_hz': round(slam_hz, 1), 'bridge_hz': round(bridge_hz, 1),
        'vo_state': vo, 'watchdog': wd,
        'lpos': lpos, 'vision_pos': vpos, 'flags': flags,
        'armed': bool(armed),
        'mode': NAV_STATES.get(status.get('nav_state'), '—') if status else '—',
        'battery': batt,
        'jetson': {k: h[k] for k in ('temp_c', 'load', 'mem_pct')},
        'video': S.video_available,
        'controls': bool(UI_KEY),
        'events': events,
    }


def snapshot_loop():
    while True:
        try:
            S.snapshot = build_snapshot()
        except Exception as e:  # never let the UI die on a bad message
            S.snapshot = {'verdict': 'UI ERROR: %s' % e, 'level': 'bad',
                          'checks': [], 'events': []}
        time.sleep(0.25)


# ------------------------------------------------------------------- demo
def demo_loop():
    """Synthetic data so the UI can be previewed without a drone."""
    S.video_available = False
    t0 = time.time()
    with S.lock:
        S.event('Demo mode: synthetic data', 'warn')
    while True:
        t = time.time() - t0
        x = 0.6 * math.sin(t / 4) + random.gauss(0, 0.01)
        y = 0.6 * math.sin(t / 2) + random.gauss(0, 0.01)
        with S.lock:
            for _ in range(9):
                S.slam.tick()
                S.bridge.tick()
            S.lpos_rate.tick()
            S.flags_rate.tick()
            S.vo_state = 1
            S.vision_pos = [x + 0.02, y - 0.01, -1.0]
            S.lpos = {'x': x, 'y': y, 'z': -1.0 + 0.05 * math.sin(t),
                      'vx': 0.15, 'vy': 0.3, 'vz': 0.0,
                      'heading': 0.3 * math.sin(t / 5), 'xy_valid': True,
                      'z_valid': True, 'eph': 0.015, 'epv': 0.02}
            S.flags = {'cs_ev_pos': True, 'cs_ev_vel': True,
                       'cs_ev_yaw': True, 'cs_ev_hgt': True,
                       'cs_tilt_align': True, 'cs_yaw_align': False,
                       'cs_in_air': True}
            S.status_t = time.monotonic()
            S.status = {'armed': False, 'nav_state': 2}
            S.battery_t = time.monotonic()
            S.battery = {'v': 15.6, 'pct': 82.0}
            S.wd_t = time.monotonic()
            S.wd = {'state': 'OK', 'reason': 'tracking', 'rejected': 3,
                    'losses': 0, 'sent': 1000}
            S.health.update(camera=True, pixhawk=True, agent=True,
                            vision=True, temp_c=48.5, load=2.1, mem_pct=41.0)
        time.sleep(0.1)


# ------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = 'LanternControl/1.0'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            return self._send(200, (HERE / 'index.html').read_bytes(),
                              'text/html; charset=utf-8')
        if path == '/api/state':
            return self._send(200, S.snapshot)
        if path == '/events':
            return self._sse()
        if path == '/video.mjpg':
            return self._mjpeg()
        self._send(404, {'error': 'not found'})

    def _sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            while True:
                self.wfile.write(b'data: ' + json.dumps(S.snapshot).encode()
                                 + b'\n\n')
                self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _mjpeg(self):
        if not S.video_available:
            return self._send(503, {'error': 'video unavailable'})
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        with S.lock:
            S.viewers += 1
        last = -1
        try:
            while True:
                with S.lock:
                    fid, frame = S.frame_id, S.frame
                if frame is not None and fid != last:
                    last = fid
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                     b'Content-Length: %d\r\n\r\n' % len(frame)
                                     + frame + b'\r\n')
                    self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with S.lock:
                S.viewers -= 1

    def do_POST(self):
        if self.path != '/api/action':
            return self._send(404, {'error': 'not found'})
        try:
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(min(n, 4096)) or b'{}')
        except Exception:
            return self._send(400, {'error': 'bad request'})
        action = req.get('action')
        targets = {'restart_vision': SERVICES['vision'],
                   'restart_agent': SERVICES['agent']}
        if action not in targets:
            return self._send(400, {'error': 'unknown action'})
        if not UI_KEY or not hmac.compare_digest(
                str(req.get('key', '')).encode(), UI_KEY.encode()):
            return self._send(403, {'error': 'wrong or missing access key'})
        if S.snapshot.get('armed'):
            return self._send(409, {'error': 'refused: drone is armed'})
        if ARGS.demo:
            with S.lock:
                S.event('Demo: would restart %s' % targets[action], 'warn')
            return self._send(200, {'ok': True})
        r = run(['sudo', '-n', '/usr/bin/systemctl', 'restart',
                 targets[action]], timeout=30)
        ok = bool(r and r.returncode == 0)
        with S.lock:
            S.event('Restarted %s' % targets[action] if ok else
                    'Restart of %s failed' % targets[action],
                    'warn' if ok else 'bad')
        if ok:
            return self._send(200, {'ok': True})
        return self._send(500, {'error': (r.stderr.strip() if r else
                                          'timeout') or 'failed'})


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--demo', action='store_true')
    ARGS = p.parse_args()

    if ARGS.demo:
        threading.Thread(target=demo_loop, daemon=True).start()
    else:
        start_ros()
        threading.Thread(target=health_loop, daemon=True).start()
        with S.lock:
            S.event('Lantern Control started')
    threading.Thread(target=snapshot_loop, daemon=True).start()

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True
    print('Lantern Control on http://%s:%d' % (ARGS.host, ARGS.port),
          flush=True)
    srv.serve_forever()


ARGS = None
if __name__ == '__main__':
    main()
