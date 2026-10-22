#!/usr/bin/env python3
"""Offline tracker tests (no ROS, no GPU): python3 tests/test_tracker.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'perception'))
import tracker as T  # noqa: E402

PERSON, CAR = 0, 2
FPS = 15.0


def det(x, y, w=40, h=100, conf=0.9, cls=PERSON):
    return [x, y, x + w, y + h, conf, cls]


def run(frames, trk=None):
    trk = trk or T.Tracker()
    out = []
    for i, dets in enumerate(frames):
        out.append(trk.step(dets, i / FPS))
    return trk, out


def test_walking_person_keeps_one_id():
    frames = [[det(100 + 6 * i, 120)] for i in range(60)]
    _, out = run(frames)
    ids = {tr.id for f in out for tr in f}
    assert ids == {1}, ids
    # not reported until confirmed (MIN_HITS frames)
    assert all(len(f) == 0 for f in out[:T.MIN_HITS - 1])
    assert all(len(f) == 1 for f in out[T.MIN_HITS - 1:])
    vx = out[-1][0].kf.x[4]
    assert 60 < vx < 120, vx   # moving ~90 px/s


def test_single_false_detection_never_reported():
    frames = [[] for _ in range(10)]
    frames[3] = [det(300, 50)]
    _, out = run(frames)
    assert all(len(f) == 0 for f in out)


def test_id_survives_short_gap():
    frames = [[det(100 + 5 * i, 120)] for i in range(20)]
    frames += [[] for _ in range(8)]                  # ~0.5 s hidden
    frames += [[det(100 + 5 * i, 120)] for i in range(28, 45)]
    _, out = run(frames)
    ids = {tr.id for f in out for tr in f}
    assert ids == {1}, ids


def test_id_dropped_after_long_gap():
    frames = [[det(100, 120)] for _ in range(10)]
    frames += [[] for _ in range(int(FPS * T.MAX_LOST_S) + 5)]
    frames += [[det(100, 120)] for _ in range(10)]
    _, out = run(frames)
    ids = sorted({tr.id for f in out for tr in f})
    assert ids == [1, 2], ids


def test_low_confidence_keeps_occluded_track():
    frames = [[det(200, 100)] for _ in range(10)]
    frames += [[det(200, 100, conf=0.25)] for _ in range(10)]   # half hidden
    _, out = run(frames)
    assert all(len(f) == 1 and f[0].id == 1 for f in out[T.MIN_HITS - 1:])


def test_low_confidence_never_starts_track():
    frames = [[det(200, 100, conf=0.3)] for _ in range(20)]
    _, out = run(frames)
    assert all(len(f) == 0 for f in out)


def test_classes_never_swap():
    # a person and a car overlapping heavily keep their own IDs and labels
    frames = [[det(100 + 3 * i, 100, cls=PERSON),
               det(105 + 3 * i, 105, cls=CAR)] for i in range(30)]
    _, out = run(frames)
    last = {tr.cls: tr.id for tr in out[-1]}
    assert set(last) == {PERSON, CAR}
    for f in out[T.MIN_HITS:]:
        assert {tr.cls: tr.id for tr in f} == last


def test_two_people_crossing():
    frames = []
    for i in range(40):
        frames.append([det(50 + 8 * i, 100), det(370 - 8 * i, 100)])
    _, out = run(frames)
    ids = {tr.id for f in out for tr in f}
    assert len(ids) <= 3, ids   # may swap at the crossing, must not explode
    assert len(out[-1]) == 2


def test_garbage_is_ignored():
    frames = [[[np.nan, 0, 10, 10, 0.9, 0], [50, 50, 40, 40, 0.9, 0]]
              for _ in range(10)]
    _, out = run(frames)
    assert all(len(f) == 0 for f in out)


def test_clock_going_backwards_resets():
    trk = T.Tracker()
    for i in range(10):
        trk.step([det(100, 100)], i / FPS)
    trk.step([det(100, 100)], 0.0)
    assert len(trk.tracks) == 1 and not trk.tracks[0].confirmed


def test_as_dict():
    trk, out = run([[det(10, 20)] for _ in range(5)])
    d = out[-1][0].as_dict({0: 'person'})
    assert d['label'] == 'person' and d['id'] == 1 and len(d['box']) == 4


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print('PASS', t.__name__)
        except AssertionError as e:
            failed += 1
            print('FAIL', t.__name__, e)
    print('%d/%d passed' % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
