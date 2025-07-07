import os, cv2

def video_to_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        cv2.imwrite(os.path.join(output_folder, f"frame_{idx:05d}.png"), frame)
        idx += 1
    cap.release()

def frames_to_video(frames_folder, output_path, fps=24):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
    if not frame_files:
        return

    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for file in frame_files:
        frame = cv2.imread(os.path.join(frames_folder, file))
        out.write(frame)
    out.release()
