# Lantern Control

Onboard software for Lantern's indoor X650. A RealSense D455 tracks the
drone's position with visual SLAM on a Jetson Orin NX. That position feeds
PX4, so Position mode works with no GPS and no compass. A web dashboard on
the Jetson shows the whole chain live.

```
D455 (IR stereo, 90 Hz) ──► Isaac ROS Visual SLAM ──► bridge + watchdog ──► uXRCE-DDS ──► PX4 EKF2
                                                            │                 (Ethernet)
                                                            └──► /lantern/vision_health ──► dashboard :8080
```

## Hardware

| Part | Detail |
|---|---|
| Frame | Holybro X650 |
| Flight controller | Pixhawk 6X, PX4 1.15.4 |
| Companion | Jetson Orin NX 16GB on the Holybro Pixhawk Jetson Baseboard, JetPack 6.2 |
| Camera | Intel RealSense D455 (front, IR stereo, emitter off) |
| Radio | RadioMaster TX16S |

## Repo layout

| Path | What it is |
|---|---|
| `vision/lantern.launch.py` | Starts the camera, the VSLAM node and the bridge (the bridge restarts itself if it stops) |
| `vision/vslam_to_px4.py` | Converts SLAM odometry to PX4 axes (FLU→FRD) and runs the **vision watchdog** |
| `ui/` | Lantern Control dashboard: `lantern_ui.py` (server), `media.py` (camera streams and recordings), `index.html` |
| `systemd/` | Boot services: `microxrceagent`, `lantern-vision`, `lantern-ui` |
| `tools/lantern-status` | Terminal health check |
| `install.sh` | Installs or updates everything on the Jetson |
| `tests/test_watchdog.py` | Offline watchdog test (no ROS needed): `python3 tests/test_watchdog.py` |

## Install / update on the Jetson

First time:
```bash
git clone https://github.com/drbinna/LanternControl.git ~/LanternControl
cd ~/LanternControl
./install.sh --check     # shows what would change; changes nothing
./install.sh
```

Every update after that:
```bash
cd ~/LanternControl && git pull && ./install.sh
```

The installer:
- backs up every file it replaces to `~/lantern/backups/<date>/`
- only restarts the services whose files changed
- **refuses to run while the drone is armed**

## Dashboard

Open `http://<jetson-ip>:8080` from any browser on the same network. It shows:
- the pre-flight chain, with a READY / NOT READY / VISION LOST banner
- a top-down position trail (PX4 estimate vs raw vision)
- the IR camera feed
- battery and Jetson stats
- an event log
- restart buttons, which are locked while armed and need the access key (`sudo cat /etc/lantern/ui.env`)

### Cameras and recordings

The dashboard has three screens: **Flight**, **Cameras** and **Recordings**.

- **Cameras** shows the D455 colour camera (640×360, 15 fps) next to the infrared camera that the tracking uses. Each camera is only streamed while someone is watching or recording it.
- **Record** saves a video with time, mode, armed state, height and vision state written into the top of the picture.
- Recording is **manual only**: nothing is recorded unless you press Record, and it saves when you press Stop.
- Recordings are stored in `~/lantern/data/recordings/`. Each file is limited to 30 minutes, recording stops if the disk gets low, and a recording cut off by a crash or power loss is finished on the next boot.
- Each recording is saved as MJPEG first, then converted to H.264 MP4 in the background so it plays in any browser. Deleting a recording needs the access key.
- To turn the colour camera off: `ros2 launch ... color:=false`.

## Vision watchdog

A bad vision sample never reaches PX4. A sample is rejected when:
- the position jumps further than `MAX_SPEED · dt + 5 cm` (MAX_SPEED = 4 m/s)
- the reported speed is impossible
- SLAM says it lost tracking
- the values are NaN or the covariance is too large

On a rejection the bridge stops publishing. PX4 times out the vision source and runs its position-loss failsafe.

Publishing resumes after about 0.5 s of clean tracking, with `reset_counter` incremented so PX4 takes the new position instead of flying back to the old one.

## PX4 parameters

| Param | Value | Why |
|---|---|---|
| `EKF2_EV_CTRL` | 15 | fuse vision position, velocity and yaw |
| `EKF2_HGT_REF` | 3 | height from vision |
| `EKF2_EV_DELAY` | 0 | |
| `SYS_HAS_MAG` / `EKF2_MAG_TYPE` | 0 / 0 | no compass indoors |
| `UXRCE_DDS_CFG` | Ethernet | |
| `UXRCE_DDS_AG_IP` | -1062731775 | agent at 192.168.0.1 |
| `UXRCE_DDS_PRT` | 8888 | |
| `COM_POSCTL_NAVL` | 0 | fall back to Altitude when position is lost |
| `MPC_VEL_MANUAL` | 3.0 (PX4 1.15 minimum; QGC shows 6.7 mph) | indoor speed cap, stays under the watchdog's 4 m/s |
| `EKF2_EV_POS_X/Y/Z` | TBD | camera offset from the Pixhawk, in metres |

## Network

| Link | Addresses |
|---|---|
| Jetson ↔ Pixhawk (Ethernet) | Jetson `enP8p1s0` = 192.168.0.1/24 (nmcli connection `px4-link`); Pixhawk = 192.168.0.3 |
| Jetson on Wi-Fi | reachable as `ubuntu.local` |

## Flying safely

- Check the dashboard shows **READY FOR POSITION MODE** before arming.
- Take off in Altitude mode, then switch to Position.
- Keep Altitude mode on a switch, so you can drop out of Position mode instantly.
- Keep people out of the camera's view, and point it at a textured, static scene.
