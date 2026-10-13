"""Camera streams and recordings for Lantern Control.

Cameras are only subscribed while someone is watching or recording, so an
idle dashboard costs the Jetson nothing.

Recording is manual only: it starts when the operator presses Record and
stops when they press Stop (or a safety limit is hit).

Recordings are written as MJPEG .avi (cheap and crash-tolerant) with a
telemetry overlay burned in. When a recording stops, ffmpeg converts it in
the background (lowest CPU priority) to an H.264 .mp4 that plays in any
browser, including Safari on iPhone.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # dashboard still works, just without video
    cv2 = np = None

DATA = Path(os.environ.get('LANTERN_DATA',
                           str(Path.home() / 'lantern' / 'data')))
REC_DIR = DATA / 'recordings'
REC_FPS = 15
MIN_FREE_GB = 1.0        # refuse to start below this
STOP_FREE_GB = 0.5       # stop a running recording below this
MAX_REC_S = 30 * 60      # one file is at most 30 minutes
NO_FRAMES_S = 10         # give up if a camera sends nothing for this long
NAME_RE = re.compile(r'^\d{8}-\d{6}-[a-z]+$')


def available():
    return cv2 is not None


def to_array(msg):
    """sensor_msgs/Image -> numpy (gray or BGR), or None if unsupported."""
    a = np.frombuffer(msg.data, dtype=np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    if enc in ('mono8', '8uc1'):
        return a.reshape(h, step)[:, :w]
    if enc in ('rgb8', 'bgr8', '8uc3'):
        img = a.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        return np.ascontiguousarray(img[:, :, ::-1]) if enc == 'rgb8' else img
    if enc in ('yuv422', 'yuyv', 'yuv422_yuy2'):
        return cv2.cvtColor(a.reshape(h, step)[:, :w * 2].reshape(h, w, 2),
                            cv2.COLOR_YUV2BGR_YUY2)
    return None


def draw_overlay(img, text):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()
    h, w = img.shape[:2]
    bar = max(18, h // 18)
    img[:bar] = (img[:bar] * 0.35).astype(img.dtype)
    cv2.putText(img, text, (6, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                bar / 40.0, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class Recording:
    def __init__(self, cam):
        self.cam = cam
        self.start = time.time()
        self.t0 = time.monotonic()
        self.name = '%s-%s' % (time.strftime('%Y%m%d-%H%M%S',
                                             time.localtime(self.start)), cam)
        self.avi = REC_DIR / (self.name + '.avi')
        self.writer = None
        self.frames = 0
        self.closed = False
        self.lock = threading.Lock()

    def write(self, img):
        with self.lock:
            if self.closed:
                return
            if self.writer is None:
                h, w = img.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.avi), cv2.VideoWriter_fourcc(*'MJPG'),
                    REC_FPS, (w, h))
            self.writer.write(img)
            self.frames += 1

    def close(self):
        with self.lock:
            self.closed = True
            if self.writer is not None:
                self.writer.release()
            return self.frames, time.monotonic() - self.t0

    def elapsed(self):
        return time.monotonic() - self.t0


class Camera:
    def __init__(self, name, label, topic, view_fps=10):
        self.name, self.label, self.topic = name, label, topic
        self.view_fps = view_fps
        self.lock = threading.Lock()
        self.jpeg = None
        self.jpeg_id = 0
        self.viewers = 0
        self.rec = None
        self.last_view = self.last_rec = self.last_frame = 0.0
        self.overlay = lambda: ''

    def wanted(self):
        return self.viewers > 0 or self.rec is not None

    def live(self):
        return time.monotonic() - self.last_frame < 2.0

    def on_frame(self, get_img):
        """Called for every incoming frame; converts only when needed."""
        now = time.monotonic()
        self.last_frame = now
        view = self.viewers > 0 and now - self.last_view >= 0.9 / self.view_fps
        rec = self.rec
        do_rec = rec is not None and now - self.last_rec >= 0.8 / REC_FPS
        if not (view or do_rec):
            return
        img = get_img()
        if img is None:
            return
        if view:
            self.last_view = now
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self.lock:
                    self.jpeg = jpg.tobytes()
                    self.jpeg_id += 1
        if do_rec:
            self.last_rec = now
            try:
                rec.write(draw_overlay(img, self.overlay()))
            except Exception:
                pass


class Recorder:
    def __init__(self, cams, event):
        self.cams = cams
        self.event = event
        self.lock = threading.Lock()
        REC_DIR.mkdir(parents=True, exist_ok=True)
        self.recover()

    # ---------------------------------------------------------------- meta
    def _meta_path(self, name):
        return REC_DIR / (name + '.json')

    def _write_meta(self, name, **fields):
        p = self._meta_path(name)
        try:
            meta = json.loads(p.read_text())
        except (OSError, ValueError):
            meta = {'name': name}
        meta.update(fields)
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(meta))
        tmp.replace(p)
        return meta

    def free_gb(self):
        return shutil.disk_usage(REC_DIR).free / 1e9

    # ------------------------------------------------------------ control
    def start(self, cam_name):
        cam = self.cams.get(cam_name)
        if cam is None:
            raise ValueError('unknown camera')
        if not available():
            raise RuntimeError('OpenCV not installed on the Jetson')
        with self.lock:
            if cam.rec is not None:
                return cam.rec.name
            if self.free_gb() < MIN_FREE_GB:
                raise RuntimeError('disk almost full (%.1f GB free)'
                                   % self.free_gb())
            rec = Recording(cam_name)
            self._write_meta(rec.name, cam=cam_name, label=cam.label,
                             start=rec.start, status='recording',
                             file=rec.avi.name, duration=0, frames=0)
            cam.rec = rec
        self.event('Recording %s camera' % cam.label.lower(), 'warn')
        return rec.name

    def stop(self, cam_name, why=''):
        cam = self.cams.get(cam_name)
        if cam is None:
            raise ValueError('unknown camera')
        with self.lock:
            rec, cam.rec = cam.rec, None
        if rec is None:
            return None
        frames, dur = rec.close()
        if frames == 0:
            for f in (rec.avi, self._meta_path(rec.name)):
                f.unlink(missing_ok=True)
            self.event('Recording discarded: no video from %s camera'
                       % cam.label.lower(), 'bad')
            return None
        self._write_meta(rec.name, duration=round(dur, 1), frames=frames,
                         status='processing')
        self.event('Saved %s recording (%d s)%s' % (
            cam.label.lower(), dur, (' · ' + why) if why else ''), 'ok')
        threading.Thread(target=self.transcode, args=(rec.name, frames, dur),
                         daemon=True).start()
        return rec.name

    def transcode(self, name, frames=0, dur=0):
        avi = REC_DIR / (name + '.avi')
        mp4 = REC_DIR / (name + '.mp4')
        if not avi.exists():
            return
        fps = frames / dur if frames and dur > 1 else REC_FPS
        fps = min(max(fps, 1.0), 60.0)
        ff = shutil.which('ffmpeg')
        if not ff:
            self._write_meta(name, status='ready', file=avi.name,
                             note='install ffmpeg for in-browser playback')
            return
        cmd = ['nice', '-n', '19', ff, '-y', '-loglevel', 'error',
               '-r', '%.3f' % fps, '-i', str(avi),
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=3600)
            ok = r.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0
        except Exception:
            ok = False
        if ok:
            avi.unlink(missing_ok=True)
            self._write_meta(name, status='ready', file=mp4.name, note='')
        else:
            mp4.unlink(missing_ok=True)
            self._write_meta(name, status='ready', file=avi.name,
                             note='MP4 conversion failed; download the AVI')

    def recover(self):
        """Finish recordings interrupted by a crash or power loss."""
        for p in REC_DIR.glob('*.json'):
            try:
                meta = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if meta.get('status') in ('recording', 'processing'):
                self._write_meta(meta['name'], status='processing',
                                 note='recovered after restart')
                threading.Thread(target=self.transcode,
                                 args=(meta['name'], meta.get('frames', 0),
                                       meta.get('duration', 0)),
                                 daemon=True).start()

    def delete(self, name):
        if not NAME_RE.match(name or ''):
            raise ValueError('bad name')
        for cam in self.cams.values():
            if cam.rec is not None and cam.rec.name == name:
                raise RuntimeError('stop the recording first')
        for ext in ('.json', '.avi', '.mp4', '.tmp'):
            (REC_DIR / (name + ext)).unlink(missing_ok=True)

    # --------------------------------------------------------- housekeeping
    def tick(self):
        """Called a few times a second: safety limits and dead cameras."""
        free = None
        for cam in self.cams.values():
            rec = cam.rec
            if rec is None:
                continue
            if free is None:
                free = self.free_gb()
            if rec.elapsed() > MAX_REC_S:
                # keep the operator's recording going in a fresh file
                self.stop(cam.name, '30 min limit, continuing in a new file')
                try:
                    self.start(cam.name)
                except Exception as e:
                    self.event('Recording stopped: %s' % e, 'bad')
            elif free < STOP_FREE_GB:
                self.stop(cam.name, 'disk full')
            elif rec.frames == 0 and rec.elapsed() > NO_FRAMES_S:
                self.stop(cam.name)

    def list(self):
        out = []
        for p in sorted(REC_DIR.glob('*.json'), reverse=True):
            try:
                meta = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            f = REC_DIR / meta.get('file', '')
            meta['size_mb'] = round(f.stat().st_size / 1e6, 1) \
                if f.is_file() else 0
            meta['url'] = '/media/' + meta.get('file', '')
            meta['playable'] = meta.get('file', '').endswith('.mp4')
            out.append(meta)
        return out

    def status(self):
        return {name: {'name': c.rec.name, 'elapsed': round(c.rec.elapsed()),
                       'frames': c.rec.frames}
                for name, c in self.cams.items() if c.rec is not None}
