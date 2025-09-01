# street_scanner.py
# Real-time video scanner that alerts when a target class (default: "fig_tree") is detected.
# Works with your trained YOLOv8 weights or a generic model (e.g., yolov8n.pt for a pipeline sanity check).

import argparse
import os
import time
import csv
from collections import deque

import cv2
import pyttsx3
from ultralytics import YOLO

# --- PyTorch 2.6 compatibility shim (safe-load allowlist for Ultralytics classes) ---
try:
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    add_safe_globals([DetectionModel])
except Exception:
    # If torch < 2.6 or Ultralytics already patched, this is harmless.
    pass
# ------------------------------------------------------------------------------------

WINDOW_TITLE = "Fig Street Scanner"

def init_tts(enable: bool):
    if not enable:
        return None
    try:
        eng = pyttsx3.init()
        eng.setProperty("rate", 175)
        eng.setProperty("volume", 1.0)
        return eng
    except Exception as e:
        print("[WARN] TTS init failed:", e)
        return None

def say(engine, text: str):
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[WARN] TTS speak failed:", e)

def save_crop(frame, xyxy, out_dir, basename):
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    crop = frame[y1:y2, x1:x2]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{basename}.jpg")
    cv2.imwrite(path, crop)
    return path

def parse_args():
    ap = argparse.ArgumentParser(
        description="Real-time YOLO street scanner that speaks/alerts when a target class is detected."
    )
    ap.add_argument("--model", default="runs/detect/train/weights/best.pt",
                    help="Path to YOLO .pt weights. If missing, will try 'yolov8n.pt'.")
    ap.add_argument("--cam", type=int, default=0, help="Camera index (0 = default).")
    ap.add_argument("--conf", type=float, default=0.55, help="Confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    ap.add_argument("--class_name", default="fig_tree",
                    help="Target class name to alert on (e.g., 'fig_tree' for custom model, 'person' for yolov8n).")
    ap.add_argument("--mute", action="store_true", help="Disable TTS voice alerts.")
    ap.add_argument("--cooldown", type=float, default=6.0, help="Min seconds between alerts.")
    ap.add_argument("--min_frames", type=int, default=10, help="Frames required between alerts.")
    ap.add_argument("--save_snaps", action="store_true", help="Save detection crops to disk.")
    ap.add_argument("--snap_dir", default="detections/snaps", help="Folder for saved crops (with --save_snaps).")
    ap.add_argument("--log_csv", default="", help="Path to CSV to append detections (timestamp, class, conf, bbox, path).")
    ap.add_argument("--width", type=int, default=640, help="Requested capture width.")
    ap.add_argument("--height", type=int, default=480, help="Requested capture height.")
    ap.add_argument("--show_fps", action="store_true", help="Overlay FPS on the video.")
    return ap.parse_args()

def main():
    args = parse_args()

    weights = args.model
    if not os.path.exists(weights):
        print(f"[WARN] Model not found at: {weights}")
        print("[INFO] Falling back to pretrained 'yolov8n.pt' for pipeline sanity check.")
        print("[TIP ] Set --class_name person when testing with yolov8n.pt.")
        weights = "yolov8n.pt"

    # Load YOLO
    model = YOLO(weights)
    device_str = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    model.to(device_str)
    print("[INFO] Using device:", device_str)

    # Map class names
    class_names = model.names  # dict: id -> name
    # Try to find the class id(s) that match the target name (case-insensitive)
    target_name = args.class_name.strip().lower()
    target_ids = [cid for cid, nm in class_names.items() if str(nm).lower() == target_name]
    if not target_ids:
        print(f"[WARN] '{args.class_name}' not found in model classes: {list(class_names.values())}")
        print("       Alerts will only trigger when detections are named exactly as --class_name.")
        print("       For yolov8n.pt testing, try: --class_name person")

    # TTS
    tts_engine = init_tts(enable=(not args.mute))

    # Open camera (DirectShow backend helps on Windows)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}. Try a different --cam (e.g., 1).")
    if args.width > 0:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Prepare window
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 540)

    # Alert debouncing
    last_alert_time = 0.0
    recent_alert_frames = deque(maxlen=max(1, args.min_frames))

    # CSV logging
    csv_file, csv_writer = None, None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        csv_file = open(args.log_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if os.path.getsize(args.log_csv) == 0:
            csv_writer.writerow(["timestamp", "class", "conf", "x1", "y1", "x2", "y2", "snap_path"])

    prev_time = time.time()
    fps_smoothed = None

    print("Press Q or Esc to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Empty frame from camera.")
                break

            # Inference
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                device=model.device
            )

            alert_needed = False
            snap_to_save = None  # (frame, bbox) to save if needed

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    name = class_names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # draw bbox
                    label = f"{name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    is_target = (name.lower() == target_name)
                    if is_target and conf >= args.conf:
                        alert_needed = True
                        if args.save_snaps:
                            snap_to_save = (frame.copy(), (x1, y1, x2, y2))
                        if csv_writer is not None:
                            csv_writer.writerow([int(time.time()), name, f"{conf:.4f}", x1, y1, x2, y2, ""])

            # Debounce alerts
            now = time.time()
            if alert_needed:
                recent_alert_frames.append(now)
                quiet_frames = (len(recent_alert_frames) < args.min_frames)
                cooldown_over = (now - last_alert_time) >= args.cooldown
                if quiet_frames and cooldown_over:
                    last_alert_time = now
                    # TTS or on-screen flash
                    if tts_engine is not None:
                        say(tts_engine, f"{args.class_name.replace('_',' ')} detected.")
                    else:
                        cv2.putText(frame, f"{args.class_name.upper()} DETECTED!", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    # Save crop if requested
                    if snap_to_save is not None:
                        snap_frame, bbox = snap_to_save
                        ts = int(time.time() * 1000)
                        path = save_crop(snap_frame, bbox, args.snap_dir, f"det_{ts}")
                        if csv_writer is not None:
                            # backfill last row with path (optional)
                            pass

            # FPS overlay
            if args.show_fps:
                cur = time.time()
                fps = 1.0 / max(1e-6, (cur - prev_time))
                prev_time = cur
                fps_smoothed = fps if fps_smoothed is None else (0.9 * fps_smoothed + 0.1 * fps)
                cv2.putText(frame, f"FPS: {fps_smoothed:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow(WINDOW_TITLE, frame)

            # Keys & window close handling
            key = cv2.waitKey(20)  # 20–30ms is safer on Windows than 1ms
            if key != -1:
                key = key & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 27=Esc
                    break
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

if __name__ == "__main__":
    main()
