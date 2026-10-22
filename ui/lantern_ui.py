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
import re
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import media  # noqa: E402

PIXHAWK_IP = os.environ.get('LANTERN_PIXHAWK_IP', '192.168.0.3')
CAMERA_USB = '8086:0b5c'
SERVICES = {'agent': 'microxrceagent', 'vision': 'lantern-vision',
            'perception': 'lantern-perception'}
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
                       'vision': None, 'perception': None,
                       'temp_c': None, 'load': None,
                       'mem_pct': None}
        self.events = collections.deque(maxlen=40)
        self.snapshot = {}
        self.wd = {}
        self.wd_t = 0.0
        self.det = {}                 # latest /lantern/detections
        self.det_t = 0.0
        self.det_seen = {}            # track id -> label, for the event log
        self.det_ev_t = 0.0
        self.det_pending = []
        self.prev_checks = {}

    def event(self, text, level='info'):
        self.events.appendleft({'t': time.strftime('%H:%M:%S'),
                                'text': text, 'level': level})


S = State()

CAMS = {
    'color': media.Camera('color', 'Colour', '/camera/camera/color/image_raw',
                          view_fps=15),
    'ir': media.Camera('ir', 'Infrared', '/camera/camera/infra1/image_rect_raw',
                       view_fps=10),
}
REC = None   # media.Recorder, created in main()


def overlay_text():
    """Telemetry burned into recordings."""
    s = S.snapshot or {}
    lp = s.get('lpos') or {}
    h = ('H %.2fm' % -lp['z']) if 'z' in lp else 'H --'
    wd = (s.get('watchdog') or {}).get('state', '--')
    return '%s  %s  %s  %s  VISION %s' % (
        time.strftime('%Y-%m-%d %H:%M:%S'), s.get('mode', '--').upper(),
        'ARMED' if s.get('armed') else 'DISARMED', h, wd)


for _c in CAMS.values():
    _c.overlay = overlay_text


def fresh_detections():
    """Latest detections if recent enough to draw, else None."""
    with S.lock:
        if time.monotonic() - S.det_t < 0.5:
            return S.det
    return None


CAMS['color'].decorate = lambda img: media.draw_detections(
    img, fresh_detections())


def on_detections(d):
    """Store a /lantern/detections message and log new objects."""
    now = time.monotonic()
    with S.lock:
        prev = S.det.get('state') if S.det else None
        S.det, S.det_t = d, now
        st = d.get('state')
        if prev and st != prev:
            if st == 'RUNNING':
                S.event('Object detection running (%s)' % d.get('model', ''),
                        'ok')
            elif st in ('NO_MODEL', 'ERROR', 'NO_CAMERA'):
                S.event('Object detection %s · %s' % (
                    st.replace('_', ' ').lower(), d.get('reason', '')), 'bad')
        for t in d.get('tracks') or []:
            if t['id'] not in S.det_seen:
                S.det_seen[t['id']] = t.get('label', '?')
                S.det_pending.append('%s %d' % (t.get('label', '?'), t['id']))
        if len(S.det_seen) > 500:            # long sessions: forget old ids
            S.det_seen = dict(list(S.det_seen.items())[-200:])
        # at most one "new object" line every 3 s, so a crowd can't flood it
        if S.det_pending and now - S.det_ev_t > 3.0:
            items = S.det_pending
            S.det_pending, S.det_ev_t = [], now
            text = ', '.join(items[:4]) + (' +%d more' % (len(items) - 4)
                                           if len(items) > 4 else '')
            S.event('New object: ' + text, 'info')


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

    def on_det(msg):
        try:
            on_detections(json.loads(msg.data))
        except ValueError:
            pass
    node.create_subscription(String, '/lantern/detections', on_det, 10)

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

    # Cameras: subscribed only while someone is watching or recording.
    subs = {}

    def manage_video():
        if not media.available():
            return
        for cam in CAMS.values():
            want = cam.wanted()
            if want and cam.name not in subs:
                subs[cam.name] = node.create_subscription(
                    Image, cam.topic,
                    lambda m, c=cam: c.on_frame(lambda: media.to_array(m)), qos)
            elif not want and cam.name in subs:
                node.destroy_subscription(subs.pop(cam.name))
    node.create_timer(0.5, manage_video)

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
        lines = (r.stdout.split() if r else []) + ['unknown'] * 3
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
                            perception=lines[2] == 'active',
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
        det = dict(S.det) if time.monotonic() - S.det_t < 2 else {}

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

    tracks = det.get('tracks') or []
    counts = {}
    for t in tracks:
        counts[t.get('label', '?')] = counts.get(t.get('label', '?'), 0) + 1
    perception = {
        'state': det.get('state') or (
            'STOPPED' if h.get('perception') is False else 'OFF'),
        'reason': det.get('reason', '' if det else
                          'lantern-perception service not reporting'),
        'fps': det.get('fps', 0), 'infer_ms': det.get('infer_ms', 0),
        'model': det.get('model', ''), 'counts': counts,
        'tracks': [{k: t.get(k) for k in ('id', 'label', 'conf', 'age')}
                   for t in tracks],
    }

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
        'video': media.available(),
        'cams': {n: {'label': c.label, 'live': c.live()}
                 for n, c in CAMS.items()},
        'recording': REC.status() if REC else {},
        'perception': perception,
        'controls': bool(UI_KEY),
        'events': events,
    }


def snapshot_loop():
    while True:
        try:
            S.snapshot = build_snapshot()
            if REC:
                REC.tick()
        except Exception as e:  # never let the UI die on a bad message
            S.snapshot = {'verdict': 'UI ERROR: %s' % e, 'level': 'bad',
                          'checks': [], 'events': []}
        time.sleep(0.25)


# ------------------------------------------------------------------- demo
def demo_loop():
    """Synthetic data so the UI can be previewed without a drone."""
    t0 = time.time()
    threading.Thread(target=demo_cams, daemon=True).start()
    arm_cycle = os.environ.get('LANTERN_DEMO_ARMCYCLE')
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
            armed = bool(arm_cycle) and int(t // float(arm_cycle)) % 2 == 1
            S.status = {'armed': armed, 'nav_state': 2 if armed else 1}
            S.battery_t = time.monotonic()
            S.battery = {'v': 15.6, 'pct': 82.0}
            S.wd_t = time.monotonic()
            S.wd = {'state': 'OK', 'reason': 'tracking', 'rejected': 3,
                    'losses': 0, 'sent': 1000}
            S.health.update(camera=True, pixhawk=True, agent=True,
                            vision=True, temp_c=48.5, load=2.1, mem_pct=41.0)
        time.sleep(0.1)


def demo_cams():
    """Synthetic camera frames for demo mode."""
    if not media.available():
        return
    np, cv2 = media.np, media.cv2
    yy, xx = np.mgrid[0:360, 0:640]
    n = 0
    while True:
        n += 1
        t = time.time()
        col = np.zeros((360, 640, 3), np.uint8)
        col[..., 0] = (xx * 255 // 640).astype(np.uint8)
        col[..., 1] = ((yy + n * 3) % 360 * 255 // 360).astype(np.uint8)
        col[..., 2] = 120
        cx = int(320 + 200 * math.sin(t))
        cv2.circle(col, (cx, 180), 40, (36, 165, 245), -1)
        cv2.rectangle(col, (470, 230), (600, 320), (90, 90, 90), -1)
        if n % 2 == 0:
            on_detections({
                'state': 'RUNNING', 'reason': '', 'w': 640, 'h': 360,
                'fps': 15.0, 'infer_ms': 8.7, 'model': 'demo',
                'tracks': [
                    {'id': 1, 'label': 'person', 'conf': 0.91,
                     'box': [cx - 44, 132, cx + 44, 232],
                     'vel': [200 * math.cos(t), 0], 'age': round(t % 60, 1)},
                    {'id': 2, 'label': 'car', 'conf': 0.78,
                     'box': [466, 226, 604, 324], 'vel': [0, 0],
                     'age': round(t % 60, 1)}]})
        ir = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
        CAMS['color'].on_frame(lambda: col)
        if n % 2 == 0:
            CAMS['ir'].on_frame(lambda: ir)
        time.sleep(1 / 30)


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
            q = parse_qs(urlparse(self.path).query)
            return self._mjpeg(q.get('cam', ['ir'])[0])
        if path == '/api/recordings':
            return self._send(200, {'recordings': REC.list() if REC else [],
                                    'free_gb': round(REC.free_gb(), 1)
                                    if REC else None})
        if path.startswith('/media/'):
            return self._file(path[len('/media/'):])
        self._send(404, {'error': 'not found'})

    def _file(self, name):
        if not re.match(r'^\d{8}-\d{6}-[a-z]+\.(mp4|avi)$', name):
            return self._send(404, {'error': 'not found'})
        f = media.REC_DIR / name
        if not f.is_file():
            return self._send(404, {'error': 'not found'})
        size = f.stat().st_size
        start, end, code = 0, size - 1, 200
        m = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header('Content-Range', 'bytes */%d' % size)
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header('Content-Type', 'video/mp4' if name.endswith('.mp4')
                         else 'video/x-msvideo')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if code == 206:
            self.send_header('Content-Range',
                             'bytes %d-%d/%d' % (start, end, size))
        if 'download=1' in self.path:
            self.send_header('Content-Disposition',
                             'attachment; filename="lantern-%s"' % name)
        self.end_headers()
        try:
            with open(f, 'rb') as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

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

    def _mjpeg(self, cam_name):
        cam = CAMS.get(cam_name)
        if cam is None or not media.available():
            return self._send(503, {'error': 'video unavailable'})
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        with cam.lock:
            cam.viewers += 1
        last = -1
        try:
            while True:
                with cam.lock:
                    fid, frame = cam.jpeg_id, cam.jpeg
                if frame is not None and fid != last:
                    last = fid
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                     b'Content-Length: %d\r\n\r\n' % len(frame)
                                     + frame + b'\r\n')
                    self.wfile.flush()
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with cam.lock:
                cam.viewers -= 1

    def _key_ok(self, req):
        return bool(UI_KEY) and hmac.compare_digest(
            str(req.get('key', '')).encode(), UI_KEY.encode())

    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(min(n, 4096)) or b'{}')
        except Exception:
            return self._send(400, {'error': 'bad request'})
        path = self.path.split('?')[0]
        if path == '/api/record':
            return self._record(req)
        if path == '/api/recordings/delete':
            if not self._key_ok(req):
                return self._send(403, {'error': 'wrong or missing access key'})
            try:
                REC.delete(req.get('name'))
            except Exception as e:
                return self._send(400, {'error': str(e)})
            return self._send(200, {'ok': True})
        if path != '/api/action':
            return self._send(404, {'error': 'not found'})
        action = req.get('action')
        targets = {'restart_vision': SERVICES['vision'],
                   'restart_agent': SERVICES['agent'],
                   'restart_perception': SERVICES['perception']}
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


def _record(self, req):
    cam, action = req.get('cam'), req.get('action')
    try:
        if action == 'start':
            name = REC.start(cam)
        elif action == 'stop':
            name = REC.stop(cam)
        else:
            return self._send(400, {'error': 'unknown action'})
    except Exception as e:
        return self._send(400, {'error': str(e)})
    return self._send(200, {'ok': True, 'name': name})


Handler._record = _record


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
    global REC
    REC = media.Recorder(CAMS, lambda text, level='info': S.event(text, level))
    threading.Thread(target=snapshot_loop, daemon=True).start()

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True
    print('Lantern Control on http://%s:%d' % (ARGS.host, ARGS.port),
          flush=True)
    srv.serve_forever()


ARGS = None
if __name__ == '__main__':
    main()
