import fire

from utils import *

def detect_faces_in_video(input_video_path, output_path):
    """
    Detects faces in a video and outputs a new video with the number of people in each frame displayed.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    mtcnn, resnet = load_models(device)

    # Read the video
    cap = cv2.VideoCapture(input_video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    list_embeddings = []
    original_frames = []
    list_counts = []
    list_total_counts = []

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count = process_frame(frame, mtcnn, resnet, list_embeddings, device)

        list_total_counts.append(cluster_embeddings(list_embeddings))
        original_frames.append(frame)
        list_counts.append(count)

    # Release the video capture
    cap.release()

    # Write to video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 2

    # Write processed frames to output video
    for frame, count, total_count in zip(original_frames, list_counts, list_total_counts):
        # Display the number of people and total number of people in the frame at top right corner
        cv2.putText(frame, "Faces detected: " + str(count), (900, 100), font, fontScale, fontColor, thickness, lineType)
        cv2.putText(frame, "Total faces: " + str(total_count), (900, 150), font, fontScale, fontColor, thickness, lineType)
        out.write(frame)

    # Release the video writer
    out.release()

if __name__ == "__main__":
    fire.Fire(detect_faces_in_video)
    