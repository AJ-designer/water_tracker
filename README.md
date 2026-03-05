# Water Tracker

Look, nobody drinks enough water. We all know it, we just don't do anything about it. You sit down at your desk, get stuck into work, and before you know it it's 5pm and you've had half a glass since breakfast. There are apps that remind you to drink, sure — but they rely on you actually logging it yourself, and nobody does that consistently either.

That's what sparked this project. I wanted to build something that just *watches* — sits in the background, uses your webcam, and automatically detects every time you take a sip. No tapping a button, no logging anything manually. You drink, it counts. Simple.

The idea is that if this were a proper app, it could run quietly in the background on your laptop or phone, build up a picture of your daily hydration habits over time, and give you genuinely useful nudges — not just "drink water!" every hour, but "hey, you haven't had anything in 3 hours" or "you're only 40% of the way to your goal today." Real data, not guesswork.

This is the Python prototype that proves the concept works. The computer vision side is done — it detects drinking motions in real time using pose estimation, calibrates to your individual style, and logs every sip with a timestamp. What it doesn't have yet is the app layer: a proper UI, a mobile version, history graphs, notifications. That's the next step, and I've laid out how I'd approach it below.

---

## How It Works

It uses your webcam and **MediaPipe** (Google's pose detection library) to track your wrist positions frame by frame. When it sees your wrist rise above a calibrated threshold with enough upward movement — the motion you make every time you lift a drink — it registers a sip and adds 15ml to your running total.

```
Webcam → MediaPipe Pose Detection → Wrist Tracking → Motion Analysis → Sip Detection → Volume Log
```

There's a short calibration step at the start so it can learn *your* drinking motion specifically, which keeps false positives low.

---

## Features

- **Real-time pose detection** — processes webcam feed at 30+ FPS using MediaPipe
- **Personal calibration** — adapts to your individual drinking style and environment
- **Automatic sip detection** — no manual logging required
- **Volume tracking** — each detected sip adds 15ml to your running total
- **Noise filtering** — smoothing and dead-zone filtering prevent hand tremors triggering false counts
- **Live overlay** — status, total volume, and sip count shown directly on the camera feed
- **Auto-save on exit** — full session log saved to JSON with timestamps
- **Session summary** — total sips and volume (ml and litres) printed at the end of every session

---

## Getting Started

### Requirements

- Python 3.7+
- A webcam

### Install

```bash
# Clone the repo
git clone https://github.com/your-username/water-tracker.git
cd water-tracker

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install opencv-python mediapipe numpy
```

### Run

```bash
python water_tracking/water_tracking.py
```

1. A webcam window opens
2. Press `c` — do a few natural drinking motions over 10 seconds to calibrate
3. Press `s` to start tracking
4. Drink normally — sips are counted automatically
5. Press `q` or `ESC` to quit — your session is saved

---

## Controls

- `c` — Start 10-second calibration
- `s` — Toggle tracking on / off
- `r` — Reset session data
- `h` — Show help
- `q` / `ESC` — Quit and save

---

## On-Screen Status

- **Yellow / CALIBRATING...** — calibration is running
- **Green / TRACKING — 250ml** — actively tracking, current total shown
- **Red / PAUSED** — tracking is stopped

---

## Output

Session data is saved to `water_tracking_data.json` on exit:

```json
[
  {
    "timestamp": 1709385600.123,
    "datetime": "2024-03-02T10:00:00.123000",
    "volume_ml": 15.0,
    "sip_count": 1
  },
  {
    "timestamp": 1709385720.456,
    "datetime": "2024-03-02T10:02:00.456000",
    "volume_ml": 30.0,
    "sip_count": 2
  }
]
```

---

## Project Structure

```
water-tracker/
├── water_tracking/
│   └── water_tracking.py     # Main application
├── water_tracking_data.json  # Auto-generated on exit
└── README.md
```

### Classes

- **WaterTrackerApp** — main controller, runs the loop and ties everything together
- **WristTracker** — tracks wrist coordinates, applies smoothing, calculates movement
- **CalibrationManager** — runs calibration and sets personalised detection thresholds
- **WaterConsumptionTracker** — detects sips, accumulates volume, saves events
- **DrinkingEvent** — data class for a single sip event

---

## Turning This Into a Real App

This Python script is the working core of the idea. To turn it into something people could actually download and use day-to-day, here are the two most realistic paths:

---

### Option 1 — React (Web / Desktop App)

**Best for:** A desktop or browser-based version. Quickest way to add a proper UI around the existing Python backend.

Keep the Python as a backend API and build a React frontend on top of it.

```
React Frontend  ←→  FastAPI (Python)  ←→  water_tracking.py
```

**Steps:**

1. **Wrap the Python in a REST API** using [FastAPI](https://fastapi.tiangolo.com/)

```python
# example endpoint
@app.get("/session/summary")
def get_summary():
    return {"total_ml": tracker.get_daily_total(), "sips": tracker.total_sips}
```

2. **Build a React dashboard** that hits those endpoints — show a daily progress bar, a history chart (Recharts works well for this), and a live sip counter

3. **Package it as a desktop app** with [Electron](https://www.electronjs.org/) so it can run in the system tray and access the webcam natively

**Pros:** Fast to prototype, easy to build nice-looking dashboards, tons of charting libraries available

**Cons:** Webcam + real-time CV in a browser is tricky — Electron makes it easier but adds complexity

---

### Option 2 — Flutter (Mobile / Cross-platform App)

**Best for:** A proper mobile app (iOS and Android) — the more ambitious but more useful end goal.

Run the Python detection logic on a backend server and call it from a Flutter mobile app. Or, longer term, port the detection to TensorFlow Lite and run it on-device.

```
Flutter App  ←→  Python API (hosted)  ←→  MediaPipe Detection
```

**Steps:**

1. **Host the Python backend** (e.g. on a cheap VPS or Railway.app)
2. **Build the Flutter app** — Flutter's `camera` package gives you webcam/phone camera access, you stream frames to your Python API for processing
3. **Add the app layer** — daily goal, progress ring, history, push notifications ("you haven't had water in 2 hours")

**Pros:** One codebase for iOS, Android, and desktop; great for building a polished consumer-facing product

**Cons:** More work upfront; streaming video frames to an API has latency — you'd eventually want on-device ML

---

### Which Should You Pick?

If you just want something working quickly with a nice UI, **start with React + FastAPI**. The Python backend stays almost as-is, you just layer a frontend over it — lower effort, faster results.

If the goal is a mobile app that people can actually download from the App Store or Google Play, **Flutter is the better long-term bet**. More work upfront, but one codebase covers iOS, Android, and desktop.

Either way, the hard part — the detection engine — is already done.

---

## Troubleshooting

**No sips being detected**
- Re-calibrate (`c`) in the same lighting you'll be tracking in
- Make sure your upper body is fully visible in the frame
- Try making your drinking motion more deliberate during calibration

**Too many false positives**
- Increase `dead_zone` in `WristTracker` to filter out more minor movements
- Avoid large arm gestures while tracking is active

**Camera won't open**
- Make sure no other app is using the webcam
- Try `cv2.VideoCapture(1)` instead of `(0)` for a secondary camera

**Performance feels slow**
- Add `self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` after the `VideoCapture` line to reduce resolution

---

## License

MIT
