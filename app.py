"""
Cheetah Mesh — Video to Pose Landmarks (Flask / Render.com)
=============================================================
نفس منطق app.py الأصلي (MediaPipe Pose)، بس Flask عادي بدل Gradio —
لأن Render مش بيفهم بروتوكول Gradio، وبروتوكول REST عادي أبسط وأسهل
نستدعيه من HTTPRequest في جودوت أصلاً.

نقطتين:
  GET  /               → صفحة اختبار بسيطة فيها فورم رفع فيديو (تقدر
                          تجرب منها من متصفح الموبايل مباشرة من غير
                          جودوت خالص).
  POST /process_video  → بياخد ملف فيديو (حقل اسمه "video") ويرجّع
                          JSON فيه fps + landmarks لكل فريم.
"""

import os
import tempfile

import cv2
import mediapipe as mp
from flask import Flask, jsonify, request

app = Flask(__name__)
mp_pose = mp.solutions.pose

LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# ⚠️ Render's free tier gives only 0.1 CPU (جزء من كور واحد ضعيف جدًا)
# — أضعف بكتير من CPU tier اللي كنا هنستخدمه على Hugging Face. قللت
# المدة القصوى وخفّفت دقة الموديل عشان نضمن المعالجة تخلص من غير
# ما تضرب مهلة gunicorn. زوّد الأرقام دي براحتك لو جربت ولقيت هامش.
MAX_SECONDS = 8.0
MAX_SIDE_PX = 400
MODEL_COMPLEXITY = 0  # 0 = أخف وأسرع موديل عند MediaPipe (0/1/2)


def _resize_keep_aspect(frame, max_side: int):
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def process_video_file(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "cannot_open_video", "message": "Could not read the uploaded video."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    max_frames = int(MAX_SECONDS * fps)

    frames_out = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened() and frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _resize_keep_aspect(frame, MAX_SIDE_PX)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            landmarks = {}
            if results.pose_world_landmarks:
                for name, lm in zip(LANDMARK_NAMES, results.pose_world_landmarks.landmark):
                    landmarks[name] = {
                        "x": round(lm.x, 5),
                        "y": round(lm.y, 5),
                        "z": round(lm.z, 5),
                        "visibility": round(lm.visibility, 3),
                    }
            frames_out.append(landmarks)
            frame_idx += 1

    cap.release()

    if not frames_out:
        return {"error": "no_frames", "message": "No frames could be read from the video."}

    detected_count = sum(1 for f in frames_out if f)
    if detected_count == 0:
        return {
            "error": "no_pose_detected",
            "message": "No person/pose was detected in this video.",
            "fps": fps,
            "frame_count": len(frames_out),
        }

    return {
        "fps": fps,
        "frame_count": len(frames_out),
        "detected_frame_count": detected_count,
        "truncated": frame_idx >= max_frames,
        "frames": frames_out,
    }


@app.route("/", methods=["GET"])
def index():
    return f"""
    <!doctype html>
    <html>
    <head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
    <body style="font-family:sans-serif;max-width:480px;margin:40px auto;padding:0 16px;">
        <h3>🐆 Cheetah Mesh — Pose Server</h3>
        <p>Server is running ✓ (max {MAX_SECONDS:.0f}s per video)</p>
        <form action="/process_video" method="post" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*" required>
            <br><br>
            <button type="submit">Process video</button>
        </form>
        <p style="color:#888;font-size:13px;">
            Submitting shows the raw JSON result below — that's expected,
            this page is for testing only.
        </p>
    </body>
    </html>
    """


@app.route("/process_video", methods=["POST"])
def process_video_endpoint():
    if "video" not in request.files or request.files["video"].filename == "":
        return jsonify({"error": "no_video", "message": "No video file received."}), 400

    file = request.files["video"]
    suffix = os.path.splitext(file.filename)[1] or ".mp4"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        result = process_video_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    status_code = 200 if "error" not in result else 422
    return jsonify(result), status_code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
