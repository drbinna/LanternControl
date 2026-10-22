#!/bin/bash
# Lantern Control installer for the Jetson.
#   ./install.sh          install / update everything
#   ./install.sh --check  show what would change, change nothing
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
APP=${LANTERN_APP:-/home/lantern/lantern}
SYSD=${LANTERN_SYSD:-/etc/systemd/system}
CHECK=0; [ "$1" = "--check" ] && CHECK=1

# repo file -> installed location
MAP=(
  "vision/lantern.launch.py:$APP/lantern.launch.py"
  "vision/vslam_to_px4.py:$APP/vslam_to_px4.py"
  "ui/lantern_ui.py:$APP/ui/lantern_ui.py"
  "ui/index.html:$APP/ui/index.html"
  "ui/media.py:$APP/ui/media.py"
  "perception/lantern_detect.py:$APP/perception/lantern_detect.py"
  "perception/tracker.py:$APP/perception/tracker.py"
  "systemd/microxrceagent.service:$SYSD/microxrceagent.service"
  "systemd/lantern-vision.service:$SYSD/lantern-vision.service"
  "systemd/lantern-ui.service:$SYSD/lantern-ui.service"
  "systemd/lantern-perception.service:$SYSD/lantern-perception.service"
  "tools/lantern-status:/usr/local/bin/lantern-status"
  "tools/lantern-perception-setup:/usr/local/bin/lantern-perception-setup"
)

if [ $CHECK = 1 ]; then
  echo "Comparing repo with what is installed (nothing will change):"
  for m in "${MAP[@]}"; do
    src="$REPO/${m%%:*}"; dst="${m##*:}"
    if [ ! -f "$dst" ]; then echo "  NEW      $dst"
    elif cmp -s "$src" "$dst"; then echo "  same     $dst"
    else echo "  CHANGED  $dst"; diff -u "$dst" "$src" | sed 's/^/           /' | head -60
    fi
  done
  exit 0
fi

# Never restart the flight stack under a flying drone.
if curl -s --max-time 2 localhost:8080/api/state | grep -q '"armed": true'; then
  echo "Drone is ARMED - refusing to install. Disarm first."; exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP="$APP/backups/$STAMP"
mkdir -p "$BACKUP" "$APP/ui" "$APP/perception" "$APP/models"
echo "==> Backing up current files to $BACKUP"

CHANGED=""
for m in "${MAP[@]}"; do
  src="$REPO/${m%%:*}"; dst="${m##*:}"
  [ -f "$dst" ] && cmp -s "$src" "$dst" && continue
  [ -f "$dst" ] && sudo cp "$dst" "$BACKUP/$(basename "$dst")"
  case "$dst" in
    $APP/*) cp "$src" "$dst" ;;
    *)      sudo install -m 755 "$src" "$dst" ;;
  esac
  [ "${dst##*.}" = "service" ] && sudo chmod 644 "$dst"
  echo "   updated $dst"
  CHANGED="$CHANGED $dst"
done
python3 -m py_compile "$APP/vslam_to_px4.py" "$APP/lantern.launch.py" "$APP/ui/lantern_ui.py" "$APP/ui/media.py" \
  "$APP/perception/lantern_detect.py" "$APP/perception/tracker.py"
mkdir -p "$APP/data/recordings"

echo "==> Video tools"
if ! command -v ffmpeg >/dev/null; then
  sudo apt-get install -y ffmpeg >/dev/null 2>&1 && echo "   installed ffmpeg" \
    || echo "   WARNING: could not install ffmpeg (no internet?) - recordings will be AVI downloads only"
else
  echo "   ffmpeg present"
fi

echo "==> Dashboard access key"
sudo mkdir -p /etc/lantern
if ! sudo grep -q '^LANTERN_UI_KEY=' /etc/lantern/ui.env 2>/dev/null; then
  KEY=$(openssl rand -hex 4)
  echo "LANTERN_UI_KEY=$KEY" | sudo tee /etc/lantern/ui.env >/dev/null
  echo "   new key (shown once): $KEY"
else
  echo "   existing key kept (sudo cat /etc/lantern/ui.env)"
fi
sudo chmod 600 /etc/lantern/ui.env
echo 'lantern ALL=(root) NOPASSWD: /usr/bin/systemctl restart lantern-vision, /usr/bin/systemctl restart microxrceagent, /usr/bin/systemctl restart lantern-perception' \
  | sudo tee /etc/sudoers.d/lantern-ui >/dev/null
sudo chmod 440 /etc/sudoers.d/lantern-ui
sudo visudo -cf /etc/sudoers.d/lantern-ui >/dev/null

echo "==> Services"
sudo systemctl daemon-reload
sudo systemctl enable microxrceagent lantern-vision lantern-ui lantern-perception >/dev/null 2>&1
# Only restart what actually changed (restarting vision resets the SLAM origin).
echo "$CHANGED" | grep -q microxrceagent && sudo systemctl restart microxrceagent && echo "   restarted microxrceagent"
echo "$CHANGED" | grep -qE "lantern.launch|vslam_to_px4|lantern-vision" && sudo systemctl restart lantern-vision && echo "   restarted lantern-vision"
echo "$CHANGED" | grep -qE "/ui/|lantern-ui" && sudo systemctl restart lantern-ui && echo "   restarted lantern-ui"
# detection is separate from flight: restarting it never touches SLAM
echo "$CHANGED" | grep -qE "/perception/|lantern-perception" && sudo systemctl restart lantern-perception && echo "   restarted lantern-perception"
[ -z "$CHANGED" ] && echo "   nothing changed"

echo "==> Object detection"
if ls "$APP"/models/*.engine "$APP"/models/*.pt >/dev/null 2>&1; then
  echo "   model present: $(cd "$APP/models" && ls *.engine *.pt 2>/dev/null | tr '\n' ' ')"
else
  echo "   no model yet - run once (needs internet, takes a few minutes): lantern-perception-setup"
fi

IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
echo
echo "Done ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo local)). Dashboard: http://${IP:-<jetson-ip>}:8080"
echo "Health check: lantern-status"
