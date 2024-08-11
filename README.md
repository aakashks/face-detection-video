# Face Detection System

This project uses deep learning models to detect and analyze faces from video files. It leverages the `facenet_pytorch` package to detect faces and extract their embeddings, and then uses the DBSCAN algorithm from `sklearn` for clustering embeddings to differentiate individuals.


The primary output is a new video that displays two key pieces of information for each frame: the number of distinct faces currently visible, and the cumulative number of unique individuals recognized throughout the video. This functionality is particularly useful in applications such as surveillance, where monitoring the number and identity of individuals in a scene is crucial, or in event management, where the flow and number of participants need to be tracked over time. Additionally, it can be employed in marketing analysis to gauge crowd engagement or in smart environments to enhance security and personalization features.

## Requirements

To run this project, you need Python 3.9 and the packages listed in `requirements.txt`.

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

The main functionality is provided by the `detect_faces_in_video` function, which processes a video file to detect faces and outputs a new video indicating the number of faces detected per frame and the total number of distinct faces detected.

To use this function, run the script from the command line by providing the input video path and the output video path:

```bash
python main.py --input_video_path 'path_to_input_video.mp4' --output_path 'path_to_output_video.avi'
```

## Output

The script outputs a video with annotated information on the detected faces:
- Number of faces detected in the current frame.
- Total number of unique faces detected across all frames.
