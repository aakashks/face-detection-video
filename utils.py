import cv2
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.cluster import DBSCAN

def load_models(device):
    """
    Loads the MTCNN and InceptionResnetV1 models.
    """
    mtcnn = MTCNN(keep_all=True, device=device, post_process=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet

def process_frame(frame, mtcnn, resnet, list_embeddings, device):
    """
    Processes a single frame to detect faces and extract embeddings.
    """
    frame_extracted = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces_tensor = mtcnn(frame_extracted)
    count = len(faces_tensor) if faces_tensor is not None else 0
    if faces_tensor is not None:
        embeddings = resnet(faces_tensor.to(device)).detach().cpu()
        list_embeddings.append(embeddings)

    return count

def cluster_embeddings(list_embeddings):
    """
    Clusters the embeddings using DBSCAN.
    """
    if list_embeddings:
        embeddings = torch.cat(list_embeddings, dim=0)
        db = DBSCAN(eps=0.6, min_samples=2).fit(embeddings.view(embeddings.shape[0], -1).numpy())
        labels = db.labels_
        unique_faces = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        unique_faces = 0
    return unique_faces
