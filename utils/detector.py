import uuid

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def intersect(A, B, C, D):
    """Check if line AB intersects line CD"""

    def ccw(X, Y, Z):
        return (Z[1] - X[1]) * (Y[0] - X[0]) > (Y[1] - X[1]) * (Z[0] - X[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def process_video(video_path, line_start, line_end, return_video=False):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    previous_positions = {}
    next_id = 0
    count = 0

    output_path = f"assets/videos/output_{uuid.uuid4().hex}.mp4"
    writer = None

    # if video for download, create a video writer
    if return_video:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    # iter through video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        boxes = results.boxes
        new_tracker = {}

        for box in boxes:
            cls = int(box.cls[0])
            if cls != 2:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Nearest previous object or new one
            matched_id = None
            for obj_id, (px, py) in previous_positions.items():
                if abs(cx - px) < 40 and abs(cy - py) < 40:
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = next_id
                next_id += 1

            new_tracker[matched_id] = (cx, cy)

            if return_video:
                # draw a rectangle around the car
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Line crossing check
            if matched_id in previous_positions:
                if intersect(
                    previous_positions[matched_id], (cx, cy), line_start, line_end
                ):
                    count += 1
                    del previous_positions[matched_id]

        previous_positions = new_tracker

        if return_video:
            # Drawing line
            cv2.line(frame, line_start, line_end, (100, 0, 200), 1)
            cv2.putText(
                frame,
                f"Count: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    return {"count": count, "output_path": output_path if return_video else None}
