"""Human Sense AI live hand gesture and pose analytics dashboard."""

import base64
import queue
from collections import Counter, deque
from pathlib import Path
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np


CAMERA_INDEX = 0
NUM_HANDS = 2
MIN_GESTURE_SCORE = 0.55
MIN_LANDMARK_VISIBILITY = 0.40
PANEL_WIDTH = 360

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

POSE_TAGS = (("L SH", 11), ("R SH", 12), ("L EL", 13), ("R EL", 14),
             ("L WR", 15), ("R WR", 16), ("L HIP", 23), ("R HIP", 24),
             ("L KN", 25), ("R KN", 26), ("L ANK", 27), ("R ANK", 28))
ANGLE_TRIPLETS = {
    "elbow_left": (11, 13, 15), "elbow_right": (12, 14, 16),
    "knee_left": (23, 25, 27), "knee_right": (24, 26, 28),
}

BG = (27, 20, 16)
PANEL = (43, 32, 25)
WHITE = (247, 242, 238)
MUTED = (181, 165, 153)
TEAL = (190, 220, 100)
CYAN = (255, 190, 90)
ORANGE = (60, 170, 255)
PINK = (245, 120, 220)

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
    for label, index in POSE_TAGS:
        point = pts[index]
        if point is not None:
            x = min(w - 42, max(2, point[0] + 7))
            y = min(h - 4, max(12, point[1] - 7))
            put_text(frame, label, (x, y), 0.28, TEAL, 1)


def angle_degrees(a, b, c, image_width=1.0, image_height=1.0):
    """Measure projected image-space angles from normalized Pose landmarks."""
    va = np.array([(a.x - b.x) * image_width,
                   (a.y - b.y) * image_height], dtype=np.float32)
    vc = np.array([(c.x - b.x) * image_width,
                   (c.y - b.y) * image_height], dtype=np.float32)
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

    def classify(self, landmarks, image_width=1.0, image_height=1.0):
        shoulder = self.pair(landmarks, "left_shoulder", "right_shoulder")
        hip = self.pair(landmarks, "left_hip", "right_hip")
        knee = self.pair(landmarks, "left_knee", "right_knee")
        knee_angles = []
        for side in ("left", "right"):
            hip_p, knee_p, ankle_p = (landmarks[POSE[f"{side}_{joint}"]]
                                      for joint in ("hip", "knee", "ankle"))
            if all(visible(p) for p in (hip_p, knee_p, ankle_p)):
                value = angle_degrees(hip_p, knee_p, ankle_p, image_width, image_height)
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

def pose_metrics(landmarks, image_width=1.0, image_height=1.0):
    """Return measured joint angles and visibility for the tracked major joints."""
    result = {}
    for name, indexes in ANGLE_TRIPLETS.items():
        points = [landmarks[index] for index in indexes]
        value = (angle_degrees(*points, image_width, image_height)
                 if all(visible(point) for point in points) else None)
        result[name] = value

    key_points = [landmarks[index] for _, index in POSE_TAGS]
    visibility = [min(1.0, max(0.0, pose_visibility(point))) for point in key_points]
    result["visible_count"] = sum(score >= MIN_LANDMARK_VISIBILITY for score in visibility)
    result["mean_visibility"] = sum(visibility) / len(visibility) if visibility else 0.0
    return result


class FpsMeter:
    """Smoothed frame-rate estimate based on elapsed processing time."""

    def __init__(self, smoothing=0.15):
        self.smoothing = smoothing
        self.last_time = None
        self.fps = 0.0

    def tick(self, now):
        if self.last_time is not None and now > self.last_time:
            instantaneous = 1.0 / (now - self.last_time)
            self.fps = (instantaneous if self.fps == 0.0 else
                        (1.0 - self.smoothing) * self.fps + self.smoothing * instantaneous)
        self.last_time = now
        return self.fps


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
                   resolution, events, pose_data):
    h, w = frame.shape[:2]
    compact = h < 720
    output = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
    output[:, :w] = frame
    output[:, w:] = PANEL
    x = w + 20
    right = w + PANEL_WIDTH - 18

    def y(value):
        return int(value * min(1.0, h / 480.0 if compact else h / 720.0))

    def s(value):
        factor = min(1.0, h / 480.0 if compact else h / 720.0)
        return max(0.25, value * factor)

    panel_text(output, x, y(29 if compact else 36), "HUMAN SENSE AI", TEAL, s(0.68 if compact else 0.78))
    panel_text(output, x, y(49 if compact else 63),
               "REAL-TIME HUMAN ACTIVITY & POSE ANALYTICS", MUTED, s(0.32 if compact else 0.38))
    cv2.line(output, (x, y(60 if compact else 79)),
             (right, y(60 if compact else 79)), (67, 76, 91), 1, cv2.LINE_AA)

    elbow_l = pose_data.get("elbow_left")
    elbow_r = pose_data.get("elbow_right")
    knee_l = pose_data.get("knee_left")
    knee_r = pose_data.get("knee_right")

    def format_angle(value):
        return f"{value:.0f} deg" if value is not None else "--"

    angle_line_elbow = f"Elbows  L {format_angle(elbow_l)}   R {format_angle(elbow_r)}"
    angle_line_knee = f"Knees   L {format_angle(knee_l)}   R {format_angle(knee_r)}"
    support = ("Evidence: torso + both leg chains"
               if body_detected and activity_score > 0 else "Insufficient landmark evidence")
    body_state = "TRACKING" if body_detected else "NO PERSON"
    visibility = (f"Key joints  {pose_data.get('visible_count', 0)}/12  |  Mean visibility "
                  f"{pose_data.get('mean_visibility', 0.0) * 100:.0f}%")

    if compact:
        panel_text(output, x, y(82), "CAMERA", MUTED, s(0.34))
        panel_text(output, x + 78, y(82), "LIVE", TEAL, s(0.40))
        panel_text(output, x, y(104),
                   f"{resolution[0]} x {resolution[1]}    {fps:4.1f} FPS", WHITE, s(0.38))
        panel_text(output, x, y(131), f"POSE TRACKING  {body_state}",
                   TEAL if body_detected else MUTED, s(0.36))
        panel_text(output, x, y(154), "POSTURE ESTIMATE", MUTED, s(0.34))
        panel_text(output, x, y(177), activity, WHITE, s(0.40))
        panel_text(output, x, y(197), support, MUTED, s(0.32))
        cv2.line(output, (x, y(207)), (right, y(207)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, y(229), "POSE ANALYTICS", MUTED, s(0.35))
        panel_text(output, x, y(251), angle_line_elbow, WHITE, s(0.34))
        panel_text(output, x, y(273), angle_line_knee, WHITE, s(0.34))
        panel_text(output, x, y(295), visibility, MUTED, s(0.31))
        cv2.line(output, (x, y(306)), (right, y(306)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, y(327), f"GESTURES  {len(hands)} / {NUM_HANDS}", MUTED, s(0.35))
        if not hands:
            panel_text(output, x, y(350), "No supported gesture detected", MUTED, s(0.33))
        for index, (side, gesture, score) in enumerate(hands[:2]):
            panel_text(output, x, y(350 + index * 22),
                       f"{side}  |  {gesture}  |  {score * 100:.0f}%", WHITE, s(0.34))
        cv2.line(output, (x, y(386)), (right, y(386)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, y(407), "RECENT EVENTS", MUTED, s(0.35))
        if events:
            for index, (when, label) in enumerate(list(events)[:3]):
                panel_text(output, x, y(429 + index * 20), f"{when}  {label}", WHITE, s(0.30))
        else:
            panel_text(output, x, y(430), "Waiting for detections", MUTED, s(0.32))
    else:
        panel_text(output, x, y(112), "CAMERA", MUTED, 0.40)
        panel_text(output, x + 90, y(112), "LIVE", TEAL, 0.48)
        panel_text(output, x, y(145), f"{resolution[0]} x {resolution[1]}   |   {fps:4.1f} FPS")
        panel_text(output, x, y(180), f"POSE TRACKING  {body_state}",
                   TEAL if body_detected else MUTED)
        panel_text(output, x, y(218), "POSTURE ESTIMATE", MUTED, 0.40)
        panel_text(output, x, y(248), activity, WHITE, 0.50)
        panel_text(output, x, y(276), support, MUTED, 0.38)
        cv2.line(output, (x, y(294)), (right, y(294)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, y(326), "POSE ANALYTICS", MUTED, 0.40)
        panel_text(output, x, y(356), angle_line_elbow, WHITE, 0.38)
        panel_text(output, x, y(383), angle_line_knee, WHITE, 0.38)
        panel_text(output, x, y(410), visibility, MUTED, 0.35)
        cv2.line(output, (x, y(428)), (right, y(428)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, y(460), f"GESTURES  {len(hands)} / {NUM_HANDS}", MUTED, 0.40)
        if not hands:
            panel_text(output, x, y(490), "No supported gesture detected", MUTED, 0.38)
        for index, (side, gesture, score) in enumerate(hands[:2]):
            panel_text(output, x, y(490 + index * 27),
                       f"{side}  |  {gesture}  |  model score {score * 100:.0f}%", WHITE, 0.38)
        cv2.line(output, (x, y(548)), (right, y(548)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, y(580), "RECENT EVENTS", MUTED, 0.40)
        if events:
            for index, (when, label) in enumerate(list(events)[:4]):
                panel_text(output, x, y(610 + index * 25), f"{when}  {label}", WHITE, 0.34)
        else:
            panel_text(output, x, y(612), "Waiting for detections", MUTED, 0.36)
    return output

class MotionSenseApp:
    """Responsive Tkinter controls around one persistent CV worker thread."""

    UI_BG = "#10141b"
    UI_PANEL = "#1a202a"
    UI_TEXT = "#eef2f7"
    UI_MUTED = "#99a5b5"
    UI_TEAL = "#64dcbe"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Human Sense AI | Real-Time Human Activity and Pose Analytics")
        self.root.configure(bg=self.UI_BG)
        self.root.geometry("1160x800")
        self.root.minsize(960, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.command_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=1)
        self.events = deque(maxlen=4)
        self.worker_thread = None
        self.ui_state = "stopped"
        self.has_started = False
        self.closing = False
        self.latest_frame = None
        self.photo = None
        self.resize_job = None
        self.sized_to_camera = False

        self._build_ui()
        self.worker_thread = threading.Thread(
            target=self._camera_worker,
            name="HumanSenseCameraWorker",
            daemon=True,
        )
        self.worker_thread.start()
        self.root.after(33, self._poll_worker)

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
                        font=("Segoe UI", 17, "bold"))
        style.configure("Sub.TLabel", background=self.UI_PANEL, foreground=self.UI_MUTED,
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=self.UI_BG, foreground=self.UI_MUTED,
                        font=("Segoe UI", 10))
        style.configure("Live.TLabel", background=self.UI_BG, foreground=self.UI_TEAL,
                        font=("Segoe UI", 10, "bold"))
        style.configure("Starting.TLabel", background=self.UI_BG, foreground="#f2c879",
                        font=("Segoe UI", 10, "bold"))
        style.configure("Stopped.TLabel", background=self.UI_BG, foreground=self.UI_MUTED,
                        font=("Segoe UI", 10, "bold"))
        style.configure("Start.TButton", background="#167d73", foreground="#ffffff",
                        padding=(12, 9), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Start.TButton", background=[("active", "#219688"), ("disabled", "#34434a")])
        style.configure("Resume.TButton", background="#244f77", foreground="#ffffff",
                        padding=(12, 9), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Resume.TButton", background=[("active", "#326d9f"), ("disabled", "#303943")])
        style.configure("Stop.TButton", background="#303846", foreground=self.UI_TEXT,
                        padding=(12, 9), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Stop.TButton", background=[("active", "#414c5d"), ("disabled", "#282e38")])
        style.configure("Quit.TButton", background="#7d3944", foreground="#ffffff",
                        padding=(12, 9), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Quit.TButton", background=[("active", "#a34654")])

        toolbar = ttk.Frame(self.root, style="Panel.TFrame", padding=(20, 14))
        toolbar.pack(fill="x", padx=14, pady=(14, 8))
        brand = ttk.Frame(toolbar, style="Panel.TFrame")
        brand.pack(side="left", fill="y")
        ttk.Label(brand, text="HUMAN SENSE AI", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand, text="REAL-TIME HUMAN ACTIVITY & POSE ANALYTICS",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(toolbar, style="Panel.TFrame")
        controls.pack(side="right", anchor="center")
        self.start_button = ttk.Button(controls, text="Start Camera", style="Start.TButton",
                                       command=self.start_camera)
        self.start_button.pack(side="left", padx=(0, 7))
        self.resume_button = ttk.Button(controls, text="Resume Camera", style="Resume.TButton",
                                        command=self.resume_camera, state="disabled")
        self.resume_button.pack(side="left", padx=(0, 7))
        self.stop_button = ttk.Button(controls, text="Stop Camera", style="Stop.TButton",
                                      command=self.stop_camera, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 7))
        self.quit_button = ttk.Button(controls, text="Quit Application", style="Quit.TButton",
                                      command=self.quit)
        self.quit_button.pack(side="left")

        status_bar = ttk.Frame(self.root, style="TFrame")
        status_bar.pack(fill="x", padx=20, pady=(0, 8))
        self.live_badge = ttk.Label(status_bar, text="STOPPED", style="Stopped.TLabel")
        self.live_badge.pack(side="left", padx=(0, 12))
        self.status_label = ttk.Label(status_bar, text="Camera stopped - press Start Camera",
                                      style="Status.TLabel")
        self.status_label.pack(side="left")

        self.view_label = tk.Label(self.root, bg="#0b0f15", fg=self.UI_MUTED,
                                   text="CAMERA STOPPED\n\nSelect Start Camera to begin",
                                   font=("Segoe UI", 14), justify="center", bd=0,
                                   highlightthickness=1, highlightbackground="#303846")
        self.view_label.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        self.view_label.bind("<Configure>", self._on_view_resize)
        ttk.Label(
            self.root,
            text="Gesture scores are model outputs. Posture is a landmark heuristic, not a probability; angles are image-space only.",
            foreground=self.UI_MUTED, font=("Segoe UI", 9), wraplength=1100,
        ).pack(anchor="w", padx=20, pady=(0, 12))

    def _set_status(self, message, state=None):
        status_color = "#ff9b9b" if state == "error" else self.UI_MUTED
        self.status_label.configure(text=message, foreground=status_color)
        if state == "live":
            self.live_badge.configure(text="LIVE", style="Live.TLabel")
        elif state == "starting":
            self.live_badge.configure(text="STARTING", style="Starting.TLabel")
        elif state == "stopping":
            self.live_badge.configure(text="STOPPING", style="Starting.TLabel")
        else:
            self.live_badge.configure(text="STOPPED", style="Stopped.TLabel")

    def _set_controls(self):
        if self.closing:
            for button in (self.start_button, self.resume_button, self.stop_button):
                button.configure(state="disabled")
            self.quit_button.configure(state="disabled")
        elif self.ui_state in ("starting", "running", "stopping"):
            self.start_button.configure(state="disabled")
            self.resume_button.configure(state="disabled")
            self.stop_button.configure(state="normal" if self.ui_state != "stopping" else "disabled")
        elif self.has_started:
            self.start_button.configure(state="disabled")
            self.resume_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.quit_button.configure(state="normal")
        else:
            self.start_button.configure(state="normal")
            self.resume_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.quit_button.configure(state="normal")

    def start_camera(self):
        if self.closing or self.has_started or self.ui_state != "stopped":
            return
        self._request_camera_start()

    def resume_camera(self):
        if self.closing or not self.has_started or self.ui_state != "stopped":
            return
        self._request_camera_start()

    def _request_camera_start(self):
        self.ui_state = "starting"
        self._set_controls()
        self._set_status("Opening camera and preparing models...", "starting")
        self.command_queue.put("start")

    def stop_camera(self):
        if self.ui_state not in ("starting", "running"):
            return
        self.ui_state = "stopping"
        self._set_controls()
        self._set_status("Stopping camera and releasing capture...", "stopping")
        self.command_queue.put("stop")

    def _record_event(self, label):
        self.events.appendleft((time.strftime("%H:%M:%S"), label))

    def _load_models(self):
        gesture_path = find_model("gesture_recognizer.task")
        pose_path = find_model("pose_landmarker_lite.task")
        recognizer = None
        pose_landmarker = None
        try:
            recognizer = Vision.GestureRecognizer.create_from_options(
                Vision.GestureRecognizerOptions(
                    base_options=BaseOptions(model_asset_buffer=gesture_path.read_bytes()),
                    running_mode=Vision.RunningMode.VIDEO,
                    num_hands=NUM_HANDS,
                )
            )
            pose_landmarker = Vision.PoseLandmarker.create_from_options(
                Vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_buffer=pose_path.read_bytes()),
                    running_mode=Vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
        except Exception:
            if recognizer is not None:
                recognizer.close()
            if pose_landmarker is not None:
                pose_landmarker.close()
            raise
        print(f"Gesture model: {gesture_path}")
        print(f"Pose model: {pose_path}")
        return recognizer, pose_landmarker

    def _publish_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    @staticmethod
    def _release_capture(capture):
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


    def _camera_worker(self):
        recognizer = None
        pose_landmarker = None
        cap = None
        active = False
        while True:
            if not active:
                command = self.command_queue.get()
                if command == "quit":
                    break
                if command != "start":
                    continue
                try:
                    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        raise RuntimeError(
                            f"Could not open camera index {CAMERA_INDEX}. Check access and close other camera apps."
                        )
                    if recognizer is None or pose_landmarker is None:
                        recognizer, pose_landmarker = self._load_models()
                    self._record_event("Camera started")
                    self.status_queue.put(("started", None))
                    self.gesture_smoother = GestureSmoother()
                    self.activity_classifier = ActivityClassifier()
                    self.fps_meter = FpsMeter()
                    self.previous_gestures = {}
                    self.previous_posture = None
                    self.previous_pose_detected = None
                    active = True
                except Exception as error:
                    if cap is not None:
                        self._release_capture(cap)
                        cap = None
                    self._record_event(f"Camera error: {error}")
                    self.status_queue.put(("stopped", str(error)))
                continue

            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                command = None
            if command == "quit":
                break
            if command == "stop":
                if cap is not None:
                    self._release_capture(cap)
                    cap = None
                active = False
                self._record_event("Camera stopped")
                self.status_queue.put(("stopped", None))
                continue
            if command == "start":
                continue

            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera returned no frame")
                dashboard = self._process_frame(frame, recognizer, pose_landmarker)
                self._publish_frame(dashboard)
            except Exception as error:
                if cap is not None:
                    self._release_capture(cap)
                    cap = None
                active = False
                self._record_event(f"Camera error: {error}")
                self.status_queue.put(("stopped", str(error)))

        if active:
            self._record_event("Camera stopped")
        self._release_capture(cap)
        for resource in (recognizer, pose_landmarker):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    self.status_queue.put(("warning", str(error)))
        self.status_queue.put(("closed", None))

    def _process_frame(self, frame, recognizer, pose_landmarker):
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(int(time.monotonic() * 1000), getattr(self, "last_ts", -1) + 1)
        self.last_ts = timestamp
        gesture_result = recognizer.recognize_for_video(image, timestamp)
        pose_result = pose_landmarker.detect_for_video(image, timestamp)

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
            activity, activity_score = self.activity_classifier.classify(pose, width, height)
            metrics = pose_metrics(pose, width, height)
        else:
            activity, activity_score = "No person detected", 0.0
            self.activity_classifier.labels.clear()
            metrics = {key: None for key in ANGLE_TRIPLETS}
            metrics.update({"visible_count": 0, "mean_visibility": 0.0})

        if body_detected and self.previous_pose_detected is not True:
            self._record_event("Pose tracking restored")
        elif not body_detected and self.previous_pose_detected is True:
            self._record_event("Pose tracking lost")
        self.previous_pose_detected = body_detected

        if activity != self.previous_posture and (
                activity in ("Standing", "Sitting") or self.previous_posture in ("Standing", "Sitting")):
            self._record_event(f"Posture: {activity}")
        self.previous_posture = activity

        current_gestures = {side: gesture for side, gesture, _ in details
                            if gesture not in ("Unknown", "Unknown / insufficient evidence")}
        for side, gesture in current_gestures.items():
            if self.previous_gestures.get(side) != gesture:
                self._record_event(f"Gesture {side}: {gesture}")
        for side in self.previous_gestures.keys() - current_gestures.keys():
            self._record_event(f"Gesture tracking lost: {side}")
        self.previous_gestures = current_gestures

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

        fps = self.fps_meter.tick(time.perf_counter())
        return draw_dashboard(frame, fps, details, activity, activity_score,
                              body_detected, (width, height), self.events, metrics)

    def _on_view_resize(self, event):
        self.view_label.configure(wraplength=max(400, event.width - 40))
        if self.latest_frame is not None and self.resize_job is None:
            self.resize_job = self.root.after(80, self._render_latest_frame)

    def _render_latest_frame(self):
        self.resize_job = None
        if self.latest_frame is None or self.closing:
            return
        frame = self.latest_frame
        source_h, source_w = frame.shape[:2]
        target_w = max(1, self.view_label.winfo_width() - 8)
        target_h = max(1, self.view_label.winfo_height() - 8)
        ratio = min(1.0, target_w / source_w, target_h / source_h)
        if ratio < 0.995:
            frame = cv2.resize(frame, (max(1, int(source_w * ratio)),
                                       max(1, int(source_h * ratio))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            return
        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.photo = tk.PhotoImage(master=self.root, data=image_data)
        self.view_label.configure(image=self.photo, text="")
        if not self.sized_to_camera:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            window_w = min(1440, max(1000, source_w + 40), screen_w - 60)
            window_h = min(980, max(650, source_h + 210), screen_h - 80)
            self.root.geometry(f"{max(760, window_w)}x{max(560, window_h)}")
            self.sized_to_camera = True

    def _show_stopped_view(self, error=None):
        self.latest_frame = None
        self.photo = None
        title = "CAMERA STOPPED"
        if error:
            title = f"CAMERA ERROR\n\n{error}"
        recent = "\n".join(f"{when}  {label}" for when, label in list(self.events)[:4])
        text = f"{title}\n\n" + (recent or "Select Start Camera or Resume Camera")
        self.view_label.configure(image="", text=text)

    def _poll_worker(self):
        stopped_error = None
        closed = False
        while True:
            try:
                kind, payload = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "started":
                self.has_started = True
                self.ui_state = "running"
                self._set_controls()
                self._set_status("Camera live - gesture and pose tracking active", "live")
                self.sized_to_camera = False
            elif kind == "stopped":
                stopped_error = payload
                self.ui_state = "stopped"
            elif kind == "closed":
                closed = True

        if self.ui_state == "running":
            newest = None
            while True:
                try:
                    newest = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            if newest is not None:
                self.latest_frame = newest
                self._render_latest_frame()
        else:
            while True:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

        if stopped_error is not None or self.ui_state == "stopped":
            self._set_controls()
            self._show_stopped_view(stopped_error)
            if stopped_error:
                self._set_status(f"Camera unavailable: {stopped_error}", "error")
            else:
                self._set_status("Camera stopped - use Resume Camera to continue")
        if closed and self.closing:
            self.root.destroy()
            return
        if not self.closing:
            self.root.after(33, self._poll_worker)

    def quit(self):
        if self.closing:
            return
        self.closing = True
        self.ui_state = "closing"
        self._set_controls()
        self._set_status("Closing camera and models...", "stopping")
        self.command_queue.put("quit")

    def run(self):
        self.root.mainloop()

def main():
    app = MotionSenseApp()
    app.run()

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, OSError, tk.TclError) as error:
        print(f"Human Sense AI could not start: {error}", file=sys.stderr)
        raise SystemExit(1)
