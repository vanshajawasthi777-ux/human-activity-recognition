"""MotionSense desktop dashboard for live hand gesture and pose activity."""

import base64
from collections import Counter, deque
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np


CAMERA_INDEX = 0
NUM_HANDS = 2
MIN_GESTURE_SCORE = 0.55
MIN_LANDMARK_VISIBILITY = 0.40
PANEL_WIDTH = 310

GESTURE_NAMES = {
    "Closed_Fist": "Closed Fist",
    "Open_Palm": "Open Palm",
    "Pointing_Up": "Pointing Up",
    "Thumb_Up": "Thumbs Up",
    "Thumb_Down": "Thumbs Down",
    "Victory": "Victory",
    "ILoveYou": "I Love You",
}

# MediaPipe PoseLandmarker indexes (the Tasks API uses the 33-point BlazePose layout).
POSE = {"left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28}

BG = (20, 24, 32)
PANEL = (29, 34, 44)
WHITE = (238, 242, 247)
MUTED = (153, 165, 181)
TEAL = (100, 220, 190)
CYAN = (255, 190, 90)
ORANGE = (60, 170, 255)
PINK = (220, 120, 245)

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision
HAND_CONNECTIONS = Vision.HandLandmarksConnections.HAND_CONNECTIONS
POSE_CONNECTIONS = Vision.PoseLandmarksConnections.POSE_LANDMARKS


def find_model(name):
    path = Path(__file__).resolve().parents[1] / "models" / name
    if not path.is_file():
        raise FileNotFoundError(f"Required model was not found: {path}")
    return path


def put_text(image, text, pos, scale=0.50, color=WHITE, thickness=1):
    cv2.putText(image, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (8, 11, 16), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def landmark_point(landmark, width, height):
    x, y = int(landmark.x * width), int(landmark.y * height)
    if -2 <= x < width + 2 and -2 <= y < height + 2:
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))
    return None


def gesture_for(result, index):
    try:
        category = result.gestures[index][0]
        name = GESTURE_NAMES.get(category.category_name, "Unknown")
        score = float(category.score)
        return (name if name != "Unknown" and score >= MIN_GESTURE_SCORE else "Unknown"), score
    except (IndexError, AttributeError, TypeError):
        return "Unknown", 0.0


def hand_side(result, index):
    try:
        return result.handedness[index][0].category_name
    except (IndexError, AttributeError, TypeError):
        return f"Hand {index + 1}"


class GestureSmoother:
    """Short per-side vote window to stabilize labels without adding much lag."""

    def __init__(self, size=5):
        self.history = {}
        self.size = size

    def update(self, side, label, score):
        history = self.history.setdefault(side, deque(maxlen=self.size))
        history.append((label, score))
        votes = Counter(name for name, _ in history if name != "Unknown")
        if not votes:
            return "Unknown", score if label == "Unknown" else 0.0
        chosen, count = votes.most_common(1)[0]
        support = [confidence for name, confidence in history if name == chosen]
        smoothed_score = sum(support) / len(support)
        if count < 2:
            return "Unknown", smoothed_score
        return chosen, smoothed_score


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [landmark_point(point, w, h) for point in landmarks]
    for edge in HAND_CONNECTIONS:
        a, b = points[edge.start], points[edge.end]
        if a is not None and b is not None:
            cv2.line(frame, a, b, (12, 18, 24), 5, cv2.LINE_AA)
            cv2.line(frame, a, b, CYAN, 2, cv2.LINE_AA)
    fingertips = {4, 8, 12, 16, 20}
    joints = {0, 5, 9, 13, 17}
    for i, point in enumerate(points):
        if point is None:
            continue
        radius, color = (5, PINK) if i in fingertips else ((4, TEAL) if i in joints else (3, WHITE))
        cv2.circle(frame, point, radius + 2, (12, 18, 24), -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
    valid = [point for point in points if point is not None]
    if valid:
        return (min(p[0] for p in valid), min(p[1] for p in valid),
                max(p[0] for p in valid), max(p[1] for p in valid))
    return None


def pose_visibility(point):
    return float(getattr(point, "visibility", 1.0))


def visible(point):
    return pose_visibility(point) >= MIN_LANDMARK_VISIBILITY


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [landmark_point(p, w, h) if visible(p) else None for p in landmarks]
    for edge in POSE_CONNECTIONS:
        a, b = pts[edge.start], pts[edge.end]
        if a is not None and b is not None:
            cv2.line(frame, a, b, (12, 18, 24), 6, cv2.LINE_AA)
            cv2.line(frame, a, b, ORANGE, 3, cv2.LINE_AA)
    key_joints = set(POSE.values())
    for i, point in enumerate(pts):
        if point is not None:
            color, radius = (TEAL, 5) if i in key_joints else (WHITE, 3)
            cv2.circle(frame, point, radius + 2, (12, 18, 24), -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def angle_degrees(a, b, c):
    va = np.array([a.x - b.x, a.y - b.y], dtype=np.float32)
    vc = np.array([c.x - b.x, c.y - b.y], dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vc))
    if denom < 1e-6:
        return None
    cosine = float(np.clip(np.dot(va, vc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class ActivityClassifier:
    """Conservative standing/sitting rules based on visible Pose landmarks.

    These geometric rules are not a trained or calibrated activity model.
    Ambiguous and incomplete poses stay unknown instead of receiving a guessed label.
    """

    def __init__(self):
        self.labels = deque(maxlen=5)

    @staticmethod
    def pair(landmarks, first, second):
        a, b = landmarks[POSE[first]], landmarks[POSE[second]]
        if visible(a) and visible(b):
            return (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
        return None

    def classify(self, landmarks):
        shoulder = self.pair(landmarks, "left_shoulder", "right_shoulder")
        hip = self.pair(landmarks, "left_hip", "right_hip")
        knee = self.pair(landmarks, "left_knee", "right_knee")
        knee_angles = []
        for side in ("left", "right"):
            hip_p, knee_p, ankle_p = (landmarks[POSE[f"{side}_{joint}"]]
                                      for joint in ("hip", "knee", "ankle"))
            if all(visible(p) for p in (hip_p, knee_p, ankle_p)):
                value = angle_degrees(hip_p, knee_p, ankle_p)
                if value is not None:
                    knee_angles.append(value)

        torso_upright = (shoulder is not None and hip is not None
                         and shoulder[1] < hip[1] - 0.08)
        if not torso_upright or len(knee_angles) < 2:
            label, support = "Unknown / insufficient evidence", 0.0
        else:
            straight = sum(angle >= 155 for angle in knee_angles)
            bent = sum(angle <= 130 for angle in knee_angles)
            if bent == 2 and knee is not None and hip[1] < knee[1] - 0.03:
                label, support = "Sitting", bent / 2.0
            elif straight == 2:
                label, support = "Standing", straight / 2.0
            else:
                label, support = "Unknown / insufficient evidence", 0.0

        self.labels.append((label, support))
        counts = Counter(name for name, _ in self.labels)
        chosen, count = counts.most_common(1)[0]
        support_values = [score for name, score in self.labels if name == chosen]
        if count < 2:
            return "Unknown / insufficient evidence", 0.0
        return chosen, sum(support_values) / len(support_values)

def panel_text(canvas, x, y, text, color=WHITE, scale=0.48):
    text = str(text)
    max_width = max(40, canvas.shape[1] - x - 18)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while scale > 0.30 and cv2.getTextSize(text, font, scale, 1)[0][0] > max_width:
        scale -= 0.02
    put_text(canvas, text, (x, y), scale, color)


def draw_hand_label(frame, bounds, side, gesture, score):
    """Place a legible hand tag in nearby free space, away from landmarks."""
    h, w = frame.shape[:2]
    text = f"{side}: {gesture}  {score * 100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    pad_x, pad_y = 8, 6
    box_w, box_h = text_w + pad_x * 2, text_h + baseline + pad_y * 2
    left, top, right, bottom = bounds

    candidates = [
        (max(4, min(w - box_w - 4, left)), top - box_h - 9),
        (max(4, min(w - box_w - 4, left)), bottom + 9),
        (right + 9, max(4, min(h - box_h - 4, top))),
        (left - box_w - 9, max(4, min(h - box_h - 4, top))),
    ]
    x, y = next(((x, y) for x, y in candidates
                 if 4 <= x <= w - box_w - 4 and 4 <= y <= h - box_h - 4),
                (max(4, min(w - box_w - 4, left)), 4))
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (18, 24, 32), -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), TEAL, 1, cv2.LINE_AA)
    put_text(frame, text, (x + pad_x, y + pad_y + text_h), scale, WHITE)


def draw_dashboard(frame, fps, hands, activity, activity_score, body_detected,
                   resolution, events):
    h, w = frame.shape[:2]
    compact = h < 700
    output = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
    output[:, :w] = frame
    output[:, w:] = PANEL
    x = w + 20
    right = w + PANEL_WIDTH - 18

    panel_text(output, x, 29, "MOTIONSENSE", TEAL, 0.64 if compact else 0.72)
    panel_text(output, x, 50 if compact else 60,
               "LIVE GESTURE + BODY TRACKING", MUTED, 0.34)
    cv2.line(output, (x, 63 if compact else 76),
             (right, 63 if compact else 76), (67, 76, 91), 1, cv2.LINE_AA)

    if compact:
        panel_text(output, x, 87, "CAMERA", MUTED, 0.36)
        panel_text(output, x + 90, 87, "LIVE", TEAL, 0.40)
        panel_text(output, x, 111, f"{resolution[0]} x {resolution[1]}    {fps:4.1f} FPS", WHITE, 0.42)
        panel_text(output, x, 139, "BODY TRACKING", MUTED, 0.36)
        panel_text(output, x + 146, 139, "ON" if body_detected else "NO PERSON",
                   TEAL if body_detected else MUTED, 0.39)
        panel_text(output, x, 169, "POSTURE", MUTED, 0.36)
        panel_text(output, x, 193, activity, WHITE, 0.40)
        support = (f"Landmark support  {activity_score * 100:.0f}%"
                   if body_detected and activity_score > 0 else "Insufficient landmark evidence")
        panel_text(output, x, 215, support, MUTED, 0.34)
        cv2.line(output, (x, 230), (right, 230), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, 253, f"HAND GESTURES   {len(hands)} / {NUM_HANDS}", MUTED, 0.36)
        if not hands:
            panel_text(output, x, 278, "Show one or both hands", MUTED, 0.37)
        for index, (side, gesture, score) in enumerate(hands[:2]):
            panel_text(output, x, 279 + index * 24,
                       f"{side}   |   {gesture}   |   {score * 100:.0f}%", WHITE, 0.37)
        cv2.line(output, (x, 333), (right, 333), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, 356, "RECENT EVENTS", MUTED, 0.36)
        if events:
            for index, (when, label) in enumerate(list(events)[:3]):
                panel_text(output, x, 379 + index * 22, f"{when}  {label}", WHITE, 0.33)
        else:
            panel_text(output, x, 381, "Waiting for a detected gesture or pose", MUTED, 0.32)
        panel_text(output, x, h - 10, "Use Quit button   |   posture is heuristic", MUTED, 0.29)
    else:
        panel_text(output, x, 108, "CAMERA", MUTED, 0.39)
        panel_text(output, x, 137, "LIVE", TEAL, 0.48)
        panel_text(output, x, 166, f"Resolution  {resolution[0]} x {resolution[1]}")
        panel_text(output, x, 194, f"Processing  {fps:4.1f} FPS")
        cv2.line(output, (x, 211), (right, 211), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, 241, "BODY TRACKING", MUTED, 0.39)
        panel_text(output, x, 270, "TRACKING" if body_detected else "NO PERSON",
                   TEAL if body_detected else MUTED)
        panel_text(output, x, 305, "POSTURE", MUTED, 0.39)
        panel_text(output, x, 334, activity, WHITE, 0.47)
        support = (f"Landmark support  {activity_score * 100:.0f}%"
                   if body_detected and activity_score > 0 else "Insufficient landmark evidence")
        panel_text(output, x, 360, support, MUTED, 0.36)
        cv2.line(output, (x, 380), (right, 380), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, 410, f"HAND GESTURES  {len(hands)} / {NUM_HANDS}", MUTED, 0.39)
        y = 444
        if not hands:
            panel_text(output, x, y, "Show one or both hands", MUTED, 0.42)
        for side, gesture, score in hands[:2]:
            panel_text(output, x, y, f"{side}   |   {gesture}", WHITE, 0.40)
            panel_text(output, x, y + 24, f"Model score  {score * 100:.0f}%", MUTED, 0.36)
            y += 57
        event_y = max(y + 8, 565)
        cv2.line(output, (x, event_y - 18), (right, event_y - 18), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, event_y + 5, "RECENT EVENTS", MUTED, 0.38)
        if events:
            for index, (when, label) in enumerate(list(events)[:4]):
                panel_text(output, x, event_y + 31 + index * 24, f"{when}  {label}", WHITE, 0.34)
        else:
            panel_text(output, x, event_y + 31, "Waiting for detections", MUTED, 0.35)
        panel_text(output, x, h - 18, "Use Quit button   |   posture is heuristic", MUTED, 0.32)
    return output


class MotionSenseApp:
    """Tkinter controls around the existing OpenCV and MediaPipe pipeline."""

    UI_BG = "#141820"
    UI_PANEL = "#1d232d"
    UI_TEXT = "#eef2f7"
    UI_MUTED = "#99a5b5"
    UI_TEAL = "#64dcbe"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MotionSense | Human Activity and Gesture Recognition")
        self.root.configure(bg=self.UI_BG)
        self.root.minsize(760, 560)
        self.root.geometry("1020x680")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.cap = None
        self.recognizer = None
        self.pose_landmarker = None
        self.running = False
        self.frames = 0
        self.start_time = time.perf_counter()
        self.fps = 0.0
        self.last_tick = self.start_time
        self.last_ts = -1
        self.gesture_smoother = GestureSmoother()
        self.activity_classifier = ActivityClassifier()
        self.events = deque(maxlen=4)
        self.previous_events = {}
        self.photo = None

        self._build_ui()
        self.root.after(15, self._update_frame)

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=self.UI_BG)
        style.configure("Panel.TFrame", background=self.UI_PANEL)
        style.configure("TLabel", background=self.UI_BG, foreground=self.UI_TEXT,
                        font=("Segoe UI", 10))
        style.configure("Brand.TLabel", background=self.UI_PANEL, foreground=self.UI_TEAL,
                        font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background=self.UI_PANEL, foreground=self.UI_MUTED,
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=self.UI_PANEL, foreground=self.UI_MUTED,
                        font=("Segoe UI", 10, "bold"))
        style.configure("Start.TButton", background="#167d73", foreground="#ffffff",
                        padding=(14, 9), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Start.TButton", background=[("active", "#219688"), ("disabled", "#34434a")])
        style.configure("Stop.TButton", background="#303846", foreground=self.UI_TEXT,
                        padding=(14, 9), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Stop.TButton", background=[("active", "#414c5d"), ("disabled", "#282e38")])
        style.configure("Quit.TButton", background="#7d3944", foreground="#ffffff",
                        padding=(14, 9), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Quit.TButton", background=[("active", "#a34654")])

        toolbar = ttk.Frame(self.root, style="Panel.TFrame", padding=(20, 14))
        toolbar.pack(fill="x", padx=14, pady=(14, 8))
        brand = ttk.Frame(toolbar, style="Panel.TFrame")
        brand.pack(side="left", fill="y")
        ttk.Label(brand, text="MOTIONSENSE", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand, text="LIVE GESTURE + BODY TRACKING", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(toolbar, style="Panel.TFrame")
        controls.pack(side="right", anchor="center")
        self.start_button = ttk.Button(controls, text="Start Camera", style="Start.TButton",
                                       command=self.start_camera)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop Camera", style="Stop.TButton",
                                      command=self.stop_camera, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Quit", style="Quit.TButton",
                   command=self.quit).pack(side="left")

        status_bar = ttk.Frame(self.root, style="TFrame")
        status_bar.pack(fill="x", padx=20, pady=(0, 8))
        self.status_label = ttk.Label(status_bar, text="Camera stopped - press Start Camera",
                                      style="Status.TLabel")
        self.status_label.pack(side="left")
        self.view_label = tk.Label(self.root, bg="#0d1117", fg=self.UI_MUTED,
                                   text="Camera is stopped\n\nSelect Start Camera to begin",
                                   font=("Segoe UI", 14), justify="center", bd=0,
                                   highlightthickness=1, highlightbackground="#303846")
        self.view_label.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        ttk.Label(self.root,
                  text="Gesture scores come from the model. Posture is a conservative landmark rule.",
                  foreground=self.UI_MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(0, 12))

    def _set_status(self, message, color=None):
        self.status_label.configure(text=message,
                                    foreground=color or self.UI_MUTED)

    def _ensure_models(self):
        if self.recognizer is None:
            gesture_path = find_model("gesture_recognizer.task")
            self.recognizer = Vision.GestureRecognizer.create_from_options(
                Vision.GestureRecognizerOptions(
                    base_options=BaseOptions(model_asset_buffer=gesture_path.read_bytes()),
                    running_mode=Vision.RunningMode.VIDEO,
                    num_hands=NUM_HANDS,
                )
            )
        if self.pose_landmarker is None:
            pose_path = find_model("pose_landmarker_lite.task")
            self.pose_landmarker = Vision.PoseLandmarker.create_from_options(
                Vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_buffer=pose_path.read_bytes()),
                    running_mode=Vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )

    def start_camera(self):
        if self.running:
            return
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera index {CAMERA_INDEX}. Check camera access and close other camera apps."
                )
            self._ensure_models()
            self.running = True
            self.frames = 0
            self.fps = 0.0
            self.last_tick = time.perf_counter()
            self.gesture_smoother = GestureSmoother()
            self.activity_classifier = ActivityClassifier()
            self.previous_events.clear()
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self._set_status("Camera live - show your hands or stand in view", self.UI_TEAL)
            print(f"Gesture model: {find_model('gesture_recognizer.task')}")
            print(f"Pose model: {find_model('pose_landmarker_lite.task')}")
        except Exception as error:
            self._release_camera()
            self._set_status(f"Could not start camera: {error}", "#ff9b9b")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def stop_camera(self):
        was_running = self.running
        self.running = False
        self._release_camera()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if was_running:
            self._set_status("Camera stopped - press Start Camera to resume")
            self.view_label.configure(image="", text="Camera is stopped\n\nSelect Start Camera to resume")
            self.photo = None

    def _release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = timestamp
        gesture_result = self.recognizer.recognize_for_video(image, timestamp)
        pose_result = self.pose_landmarker.detect_for_video(image, timestamp)

        details = []
        for index, landmarks in enumerate(gesture_result.hand_landmarks):
            raw_name, raw_score = gesture_for(gesture_result, index)
            side = hand_side(gesture_result, index)
            name, score = self.gesture_smoother.update(side, raw_name, raw_score)
            bounds = draw_hand(frame, landmarks)
            details.append((side, name, score))
            if bounds:
                draw_hand_label(frame, bounds, side, name, score)

        body_detected = bool(pose_result.pose_landmarks)
        if body_detected:
            pose = pose_result.pose_landmarks[0]
            draw_pose(frame, pose)
            activity, activity_score = self.activity_classifier.classify(pose)
        else:
            activity, activity_score = "No person detected", 0.0
            self.activity_classifier.labels.clear()

        current_events = {"posture": activity if body_detected else "No person"}
        current_events.update({f"hand:{side}": gesture for side, gesture, _ in details})
        for key, label in current_events.items():
            if (label not in ("Unknown", "Unknown / insufficient evidence", "No person detected")
                    and self.previous_events.get(key) != label):
                kind = "Pose" if key == "posture" else key.split(":", 1)[1]
                self.events.appendleft((time.strftime("%H:%M:%S"), f"{kind} | {label}"))
        self.previous_events = current_events

        activity_text = f"POSTURE  {activity.upper()}"
        (activity_w, activity_h), activity_base = cv2.getTextSize(
            activity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (12, 12),
                      (activity_w + 30, activity_h + activity_base + 25),
                      (18, 24, 32), -1)
        cv2.rectangle(frame, (12, 12),
                      (activity_w + 30, activity_h + activity_base + 25),
                      CYAN, 1, cv2.LINE_AA)
        put_text(frame, activity_text, (21, 12 + activity_h + 8), 0.52, CYAN)

        now = time.perf_counter()
        dt = now - self.last_tick
        self.last_tick = now
        if dt > 0:
            instantaneous = 1.0 / dt
            self.fps = instantaneous if self.fps == 0.0 else 0.9 * self.fps + 0.1 * instantaneous
        self.frames += 1
        return draw_dashboard(frame, self.fps, details, activity, activity_score,
                              body_detected, (width, height), self.events)

    def _update_frame(self):
        if self.running and self.cap is not None:
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera returned no frame")
                dashboard = self._process_frame(frame)
                max_width = 1240
                if dashboard.shape[1] > max_width:
                    ratio = max_width / dashboard.shape[1]
                    dashboard = cv2.resize(
                        dashboard,
                        (max_width, max(1, int(dashboard.shape[0] * ratio))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(".png", dashboard)
                if not ok:
                    raise RuntimeError("Could not render the camera frame")
                image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
                self.photo = tk.PhotoImage(master=self.root, data=image_data)
                if self.frames == 1:
                    window_width = min(1320, max(1000, dashboard.shape[1] + 40))
                    window_height = min(900, max(650, dashboard.shape[0] + 150))
                    self.root.geometry(f"{window_width}x{window_height}")
                self.view_label.configure(image=self.photo, text="")
            except Exception as error:
                self.stop_camera()
                self._set_status(f"Camera stopped: {error}", "#ff9b9b")
        self.root.after(15, self._update_frame)

    def quit(self):
        self.running = False
        self._release_camera()
        if self.recognizer is not None:
            self.recognizer.close()
            self.recognizer = None
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()
            self.pose_landmarker = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = MotionSenseApp()
    app.run()

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, OSError) as error:
        print(f"MotionSense could not start: {error}", file=sys.stderr)
        raise SystemExit(1)
