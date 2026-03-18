import os
import cv2
from ultralytics import YOLO
from collections import defaultdict
from moviepy.editor import VideoFileClip

# === CONFIGURATION ===
model_path = "C:/Users/NOMAN TRADERS/Downloads/PlexorTask/detector.pt"
input_path = "C:/Users/NOMAN TRADERS/Downloads/PlexorTask/TestVideos/combined_video.mp4"

# === Create output directory ===
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "Output")
os.makedirs(output_folder, exist_ok=True)

# === Load YOLO model ===
print("[INFO] Loading YOLO model...")
try:
    model = YOLO(model_path)
    print("[SUCCESS] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    exit(1)


def process_video(input_video_path):
    print(f"\n[INFO] Starting processing: {input_video_path}")

    if not os.path.isfile(input_video_path):
        print(f"[ERROR] File not found: {input_video_path}")
        return

    filename = os.path.basename(input_video_path)
    name, ext = os.path.splitext(filename)

    temp_video_path = os.path.join(output_folder, f"{name}_temp_no_audio.mp4")
    final_video_path = os.path.join(output_folder, f"{name}_out.mp4")
    summary_path = os.path.join(output_folder, f"{name}_summary.txt")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video properties: {fps} FPS, {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_count = 0
    processed = 0

    # Track total phone usage time per ID
    phone_time = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Run inference with tracking
            results = model.track(frame, persist=True)

            annotated = frame.copy()

            if len(results) > 0:
                boxes = results[0].boxes
                if boxes.id is not None:  # If YOLO assigned IDs
                    for box, obj_id in zip(boxes, boxes.id.int().tolist()):
                        # Increment time (in seconds) this phone was detected
                        phone_time[obj_id] += 1 / fps

                        conf = float(box.conf[0]) if box.conf is not None else 0
                        usage_time = phone_time[obj_id]

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        # Draw bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # Custom combined label
                        label = f"ID:{obj_id} | Counter:{usage_time:.1f}s | Phone {conf:.2f}"

                        # Font settings (increase size here)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.2  # bigger font
                        thickness = 3     # bolder text

                        # Draw background for text
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (255, 0, 0), -1)

                        # Put text (white on blue)
                        cv2.putText(
                            annotated,
                            label,
                            (x1, y1 - 5),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA
                        )

            out.write(annotated)
            processed += 1

        except Exception as e:
            print(f"[ERROR] Error processing frame {frame_count}: {e}")

        frame_count += 1
        if frame_count % int(fps) == 0:  # Print every second
            print(f"  → Processed {frame_count} frames...")

    cap.release()
    out.release()

    print(f"[DONE] Processed {processed} frames.")
    print(f"[SAVED] Temporary video saved (no audio): {temp_video_path}")

    # === Merge audio back using MoviePy ===
    print("[INFO] Merging audio from original video...")
    original_clip = VideoFileClip(input_video_path)
    processed_clip = VideoFileClip(temp_video_path)
    final_clip = processed_clip.set_audio(original_clip.audio)
    final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")

    # Remove temp video
    os.remove(temp_video_path)

    print(f"[SAVED] Final output (with audio): {final_video_path}\n")

    # === Save summary ===
    with open(summary_path, "w") as f:
        f.write("Phone usage summary:\n")
        for obj_id, seconds in phone_time.items():
            f.write(f"Phone #{obj_id}: {seconds:.2f} seconds\n")

    print(f"[SAVED] Summary saved to: {summary_path}")


# === Main logic ===
abs_input_path = os.path.abspath(input_path)
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

if os.path.isfile(abs_input_path) and abs_input_path.lower().endswith(video_extensions):
    process_video(abs_input_path)

elif os.path.isdir(abs_input_path):
    video_files = [f for f in os.listdir(abs_input_path) if f.lower().endswith(video_extensions)]
    if not video_files:
        print("[WARNING] No supported video files found in the folder.")
    else:
        for file in video_files:
            process_video(os.path.join(abs_input_path, file))

else:
    print(f"[ERROR] Invalid input path: {abs_input_path}")
