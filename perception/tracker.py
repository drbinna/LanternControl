"""Multi-object tracker for Lantern perception (no ROS, no GPU, numpy only).

ByteTrack-style:
  * every track carries a constant-velocity Kalman filter on its box
    (centre x, centre y, width, height and their rates), so it keeps a
    sensible position through a missed frame or two
  * confident detections are matched to tracks first, then low-confidence
    detections get a second chance against the tracks still unmatched -
    this is what keeps an ID alive when a person is partly hidden
  * a new track needs MIN_HITS matches before it is reported ("confirmed"),
    so a one-frame false detection never shows up as an object
  * a confirmed track that stops being seen is kept for MAX_LOST_S seconds
    and re-attached if the object comes back, so the ID does not change

Detections are an (N, 6) array: x1, y1, x2, y2, confidence, class_id
(pixels). Matching is per class: a person is never re-labelled as a car.
"""
import math

import numpy as np

HIGH = 0.45          # confident detection
LOW = 0.15           # below this a detection is ignored entirely
NEW_TRACK = 0.55     # a new object must start at least this confident
MATCH_IOU = 0.25     # minimum box overlap to call it the same object
MIN_HITS = 3         # matches before a track is reported
MAX_LOST_S = 1.5     # how long an unseen confirmed track is kept


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and b (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)[:, None, :]
    b = np.asarray(b, float)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) -
                 np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) -
                 np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def greedy_match(iou, thresh):
    """Pair rows with columns, highest overlap first. -> [(row, col)]."""
    pairs = []
    if iou.size == 0:
        return pairs
    used_r, used_c = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None),
                                       iou.shape))[0]
    for r, c in order:
        if iou[r, c] < thresh:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((int(r), int(c)))
    return pairs


class Kalman:
    """Constant-velocity filter on [cx, cy, w, h, vcx, vcy, vw, vh]."""

    def __init__(self, box):
        cx, cy, w, h = self.to_cxcywh(box)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        s = max(w, h)
        self.P = np.diag([s, s, s, s, 10 * s, 10 * s, 10 * s, 10 * s]) ** 2 \
            * 0.01

    @staticmethod
    def to_cxcywh(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0))

    def box(self):
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def predict(self, dt):
        dt = min(max(dt, 0.0), 1.0)
        F = np.eye(8)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = dt
        s = max(self.x[2], self.x[3])
        q_pos = (0.05 * s) ** 2 * dt
        q_vel = (0.6 * s) ** 2 * dt
        Q = np.diag([q_pos] * 4 + [q_vel] * 4)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, box):
        z = np.array(self.to_cxcywh(box), float)
        H = np.zeros((4, 8))
        H[:, :4] = np.eye(4)
        s = max(self.x[2], self.x[3])
        R = np.eye(4) * (0.05 * s) ** 2
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P


class Track:
    def __init__(self, tid, det, t):
        self.id = tid
        self.cls = int(det[5])
        self.conf = float(det[4])
        self.kf = Kalman(det[:4])
        self.hits = 1
        self.first_seen = t
        self.last_seen = t
        self.confirmed = False

    def update(self, det, t):
        self.kf.update(det[:4])
        self.conf = float(det[4])
        self.hits += 1
        self.last_seen = t
        if self.hits >= MIN_HITS:
            self.confirmed = True

    def as_dict(self, names=None):
        x1, y1, x2, y2 = self.kf.box()
        return {
            'id': self.id,
            'cls': self.cls,
            'label': (names or {}).get(self.cls, str(self.cls)),
            'conf': round(self.conf, 2),
            'box': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            'vel': [round(float(self.kf.x[4]), 1),
                    round(float(self.kf.x[5]), 1)],
            'age': round(self.last_seen - self.first_seen, 2),
        }


class Tracker:
    def __init__(self, high=HIGH, low=LOW, new_track=NEW_TRACK,
                 match_iou=MATCH_IOU, max_lost_s=MAX_LOST_S):
        self.high, self.low, self.new_track = high, low, new_track
        self.match_iou, self.max_lost_s = match_iou, max_lost_s
        self.tracks = []
        self.next_id = 1
        self.t = None

    def reset(self):
        self.tracks, self.t = [], None

    def _match(self, tracks, dets):
        """Per-class greedy IoU matching. -> pairs of (track_idx, det_idx)."""
        pairs = []
        if not tracks or len(dets) == 0:
            return pairs
        for c in set(int(d[5]) for d in dets):
            ti = [i for i, tr in enumerate(tracks) if tr.cls == c]
            di = [j for j, d in enumerate(dets) if int(d[5]) == c]
            if not ti or not di:
                continue
            iou = iou_matrix([tracks[i].kf.box() for i in ti],
                             [dets[j][:4] for j in di])
            pairs += [(ti[r], di[k]) for r, k in
                      greedy_match(iou, self.match_iou)]
        return pairs

    def step(self, dets, t):
        """Feed one frame of detections at time t (seconds).

        Returns the confirmed tracks that were seen in this frame.
        """
        dets = np.asarray(dets, float).reshape(-1, 6)
        dets = dets[np.isfinite(dets).all(axis=1)]
        dets = dets[(dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])]
        dt = 0.0 if self.t is None else t - self.t
        if dt < 0:            # clock went backwards (bag replay, restart)
            self.reset()
            dt = 0.0
        self.t = t
        for tr in self.tracks:
            tr.kf.predict(dt)

        high = dets[dets[:, 4] >= self.high]
        low = dets[(dets[:, 4] >= self.low) & (dets[:, 4] < self.high)]

        # 1) confident detections against every track
        seen = set()                     # indices into self.tracks
        used_high = set()                # indices into high
        for ti, di in self._match(self.tracks, high):
            self.tracks[ti].update(high[di], t)
            seen.add(ti)
            used_high.add(di)

        # 2) weak detections get a second chance with confirmed tracks that
        #    are still unmatched (partly hidden objects)
        rest = [i for i, tr in enumerate(self.tracks)
                if i not in seen and tr.confirmed]
        for ri, di in self._match([self.tracks[i] for i in rest], low):
            self.tracks[rest[ri]].update(low[di], t)
            seen.add(rest[ri])

        # 3) drop dead tracks: an unconfirmed track that missed a frame,
        #    or a confirmed one unseen for too long
        self.tracks = [tr for i, tr in enumerate(self.tracks)
                       if i in seen or
                       (tr.confirmed and t - tr.last_seen <= self.max_lost_s)]

        # 4) unmatched confident detections start new (tentative) tracks
        for di, d in enumerate(high):
            if di not in used_high and d[4] >= self.new_track:
                self.tracks.append(Track(self.next_id, d, t))
                self.next_id += 1

        return [tr for tr in self.tracks
                if tr.confirmed and tr.last_seen == t]
