"""MotionSense live hand gesture and pose activity baseline.

Run from any working directory with ``python src/gesture_demo.py`` (from the
project root). Activity labels are heuristic estimates, not a trained model.
"""

from collections import Counter, deque
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np


CAMERA_INDEX = 0
NUM_HANDS = 2
MIN_GESTURE_SCORE = 0.55
MIN_LANDMARK_VISIBILITY = 0.40
WINDOW_NAME = "MotionSense | Human Activity & Gesture Recognition"
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
    """Transparent landmark rules plus short-window ankle motion estimate.

    These scores describe rule support, not calibrated probabilities. The
    camera view, occlusion, and a short motion window can confuse sitting with
    standing or ordinary body movement with walking; a labelled dataset and a
    trained temporal classifier are needed for dependable activity recognition.
    """

    def __init__(self):
        self.ankle_history = deque(maxlen=8)
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
        ankle = self.pair(landmarks, "left_ankle", "right_ankle")
        wrist_points = [landmarks[POSE[key]] for key in ("left_wrist", "right_wrist")]
        shoulder_points = [landmarks[POSE[key]] for key in ("left_shoulder", "right_shoulder")]
        raised = [w for w, s in zip(wrist_points, shoulder_points)
                  if visible(w) and visible(s) and w.y < s.y - 0.06]
        knee_angles = []
        for side in ("left", "right"):
            hip_p, knee_p, ankle_p = (landmarks[POSE[f"{side}_{joint}"]]
                                      for joint in ("hip", "knee", "ankle"))
            if all(visible(p) for p in (hip_p, knee_p, ankle_p)):
                value = angle_degrees(hip_p, knee_p, ankle_p)
                if value is not None:
                    knee_angles.append(value)

        if ankle is not None:
            self.ankle_history.append(ankle)
        moving = False
        if len(self.ankle_history) >= 6:
            xs = [p[0] for p in self.ankle_history]
            ys = [p[1] for p in self.ankle_history]
            moving = max(max(xs) - min(xs), max(ys) - min(ys)) > 0.035

        if len(raised) == 2:
            label, confidence = "Arms raised", 0.82
        elif moving and len(knee_angles) >= 1:
            label, confidence = "Walking", 0.62
        elif len(knee_angles) >= 1 and knee is not None and hip is not None:
            bent = sum(angle < 145 for angle in knee_angles)
            hips_low = hip[1] >= knee[1] - 0.04
            if bent and hips_low:
                label, confidence = "Sitting", 0.68
            elif all(angle >= 150 for angle in knee_angles) and shoulder is not None and hip is not None:
                label, confidence = "Standing", 0.65
            else:
                label, confidence = "Unknown / insufficient evidence", 0.0
        else:
            label, confidence = "Unknown / insufficient evidence", 0.0

        self.labels.append((label, confidence))
        counts = Counter(name for name, _ in self.labels)
        chosen, count = counts.most_common(1)[0]
        confs = [score for name, score in self.labels if name == chosen]
        if count < 2:
            return "Unknown / insufficient evidence", 0.0
        return chosen, sum(confs) / len(confs)


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


def draw_dashboard(frame, fps, hands, activity, activity_score, body_detected, resolution):
    h, w = frame.shape[:2]
    compact = h < 560
    fit = min(1.0, h / 720.0)
    output = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
    output[:, :w] = frame
    output[:, w:] = PANEL
    x = w + 20
    right = w + PANEL_WIDTH - 18

    if compact:
        # A shorter, single-row hand list preserves readable line spacing at 480p.
        panel_text(output, x, 31, "MOTIONSENSE", TEAL, 0.64)
        panel_text(output, x, 53, "HUMAN ACTIVITY + GESTURES", MUTED, 0.37)
        cv2.line(output, (x, 69), (right, 69), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, 94, "CAMERA", MUTED, 0.39)
        panel_text(output, x, 119, "ACTIVE", TEAL, 0.45)
        panel_text(output, x, 145, f"Resolution  {resolution[0]} x {resolution[1]}", WHITE, 0.43)
        panel_text(output, x, 170, f"Processing  {fps:4.1f} FPS", WHITE, 0.43)
        cv2.line(output, (x, 185), (right, 185), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, 210, "BODY TRACKING", MUTED, 0.39)
        panel_text(output, x, 236, "TRACKING" if body_detected else "NO PERSON",
                   TEAL if body_detected else MUTED, 0.45)
        panel_text(output, x, 274, "CURRENT ACTIVITY", MUTED, 0.39)
        panel_text(output, x, 301, activity, WHITE, 0.43)
        confidence = (f"Heuristic confidence  {activity_score * 100:.0f}%"
                      if body_detected and activity_score > 0 else "Confidence not available")
        panel_text(output, x, 327, confidence, MUTED, 0.38)
        cv2.line(output, (x, 344), (right, 344), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, 371, f"HANDS  {len(hands)} / {NUM_HANDS}", MUTED, 0.39)
        if not hands:
            panel_text(output, x, 402, "Show one or both hands", MUTED, 0.42)
        for index, (side, gesture, score) in enumerate(hands[:2]):
            y = 402 + index * 31
            panel_text(output, x, y, f"{side.upper()}  |  {gesture}  |  {score * 100:.0f}%",
                       WHITE, 0.39)
        panel_text(output, x, h - 14, "Q / ESC QUIT  |  Activity labels are heuristic", MUTED, 0.30)
    else:
        panel_text(output, x, 36, "MOTIONSENSE", TEAL, 0.72)
        panel_text(output, x, 60, "HUMAN ACTIVITY + GESTURES", MUTED, 0.37)
        cv2.line(output, (x, 76), (right, 76), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, int(105 * fit), "CAMERA", MUTED, 0.40)
        panel_text(output, x, int(132 * fit), "ACTIVE", TEAL, 0.49)
        panel_text(output, x, int(161 * fit), f"Resolution  {resolution[0]} x {resolution[1]}")
        panel_text(output, x, int(190 * fit), f"Processing  {fps:4.1f} FPS")
        cv2.line(output, (x, int(207 * fit)), (right, int(207 * fit)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, int(237 * fit), "BODY TRACKING", MUTED, 0.40)
        panel_text(output, x, int(266 * fit), "TRACKING" if body_detected else "NO PERSON",
                   TEAL if body_detected else MUTED)
        panel_text(output, x, int(300 * fit), "CURRENT ACTIVITY", MUTED, 0.40)
        panel_text(output, x, int(329 * fit), activity, WHITE, 0.49)
        confidence = (f"Heuristic confidence  {activity_score * 100:.0f}%"
                      if body_detected and activity_score > 0 else "Confidence not available")
        panel_text(output, x, int(356 * fit), confidence, MUTED, 0.39)
        cv2.line(output, (x, int(376 * fit)), (right, int(376 * fit)), (67, 76, 91), 1, cv2.LINE_AA)
        panel_text(output, x, int(405 * fit), f"HANDS  {len(hands)} / {NUM_HANDS}", MUTED, 0.40)
        y = int(438 * fit)
        if not hands:
            panel_text(output, x, y, "Show one or both hands", MUTED, 0.42)
        for side, gesture, score in hands[:2]:
            panel_text(output, x, y, side.upper(), TEAL, 0.45)
            panel_text(output, x, y + 24, gesture, WHITE, 0.46)
            panel_text(output, x, y + 47, f"Confidence  {score * 100:.0f}%", MUTED, 0.39)
            y += int(82 * fit)
        panel_text(output, x, h - 54, "Q / ESC  Quit", MUTED, 0.43)
        panel_text(output, x, h - 30, "Activity labels are heuristic estimates.", MUTED, 0.33)
    return output


def main():
    gesture_path = find_model("gesture_recognizer.task")
    pose_path = find_model("pose_landmarker_lite.task")
    print(f"Gesture model: {gesture_path}")
    print(f"Pose model: {pose_path}")

    gesture_options = Vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_buffer=gesture_path.read_bytes()),
        running_mode=Vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )
    pose_options = Vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=pose_path.read_bytes()),
        running_mode=Vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = None
    recognizer = None
    pose_landmarker = None
    frames = 0
    start = time.perf_counter()
    fps = 0.0
    last_tick = start
    last_ts = -1
    gesture_smoother = GestureSmoother()
    activity_classifier = ActivityClassifier()

    try:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Check camera access and close other camera apps.")
        recognizer = Vision.GestureRecognizer.create_from_options(gesture_options)
        pose_landmarker = Vision.PoseLandmarker.create_from_options(pose_options)
        print("MotionSense ready. Press Q or Esc in the video window to quit.")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera returned no frame; stopping.")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = timestamp
            gesture_result = recognizer.recognize_for_video(image, timestamp)
            pose_result = pose_landmarker.detect_for_video(image, timestamp)

            details = []
            for i, landmarks in enumerate(gesture_result.hand_landmarks):
                raw_name, raw_score = gesture_for(gesture_result, i)
                side = hand_side(gesture_result, i)
                name, score = gesture_smoother.update(side, raw_name, raw_score)
                bounds = draw_hand(frame, landmarks)
                details.append((side, name, score))
                if bounds:
                    draw_hand_label(frame, bounds, side, name, score)

            body_detected = bool(pose_result.pose_landmarks)
            if body_detected:
                pose = pose_result.pose_landmarks[0]
                draw_pose(frame, pose)
                activity, activity_score = activity_classifier.classify(pose)
            else:
                activity, activity_score = "No person detected", 0.0
                activity_classifier.labels.clear()
                activity_classifier.ankle_history.clear()

            activity_text = f"ACTIVITY  {activity.upper()}"
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
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                instantaneous = 1.0 / dt
                fps = instantaneous if fps == 0.0 else 0.9 * fps + 0.1 * instantaneous
            frames += 1
            dashboard = draw_dashboard(frame, fps, details, activity,
                                       activity_score, body_detected, (width, height))
            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if recognizer is not None:
            recognizer.close()
        if pose_landmarker is not None:
            pose_landmarker.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    if frames and elapsed > 0:
        print(f"Processed {frames} frames; average FPS {frames / elapsed:.1f}.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, OSError) as error:
        print(f"MotionSense could not start: {error}", file=sys.stderr)
        raise SystemExit(1)
