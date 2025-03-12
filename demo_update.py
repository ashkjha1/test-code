import os
import io
import faiss
import numpy as np
import torch
import torchaudio
import gc
import speechbrain as sb
import multiprocessing
from speechbrain.pretrained import EncoderClassifier #This line is removed in the update
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.features import Fbank
from speechbrain.processing.normalization import MeanVarianceNorm
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import yt_dlp
import logging

# Configure logging to a file
log_file = "my_application.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='a'
)

# FAISS Vector Database
embedding_size = 192
index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_size))
db_audio_map = {}

# Load Model Once
model_path = "./ecapa_tdnn"
os.makedirs(model_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ECAPA-TDNN model directly
model = ECAPA_TDNN(
    input_size=80,  # Number of features (Fbank)
    lin_neurons=192,
    activation=torch.nn.ReLU,
).to(device)

# Load pretrained weights
ckpt_path = os.path.join(model_path, "save", "CKPT+2023-04-03+12-42-32+00") #Adjust the CKPT path to what is downloaded.
if os.path.exists(ckpt_path):
    ckpt = torch.load(os.path.join(ckpt_path, "model.ckpt"), map_location=device)
    model.load_state_dict(ckpt["model"])
else:
    # Download the pretrained model if it doesn't exist.
    sb.utils.checkpoints.ckpt_utils.download_ckpt(
        "speechbrain/spkrec-ecapa-voxceleb",
        os.path.join(model_path, "save", "CKPT+2023-04-03+12-42-32+00"), #Adjust the CKPT path to what is downloaded.
        os.path.join(model_path, "save", "CKPT+2023-04-03+12-42-32+00", "model.ckpt"),
    )
    ckpt = torch.load(os.path.join(model_path, "save", "CKPT+2023-04-03+12-42-32+00", "model.ckpt"), map_location=device)
    model.load_state_dict(ckpt["model"])

model.eval()

# Feature Extraction
feature_extraction = Fbank(n_mels=80).to(device)
mean_var_norm = MeanVarianceNorm(norm_type="global").to(device)

def download_audio(url):
    # ... (rest of the download_audio function remains the same)

def download_all(urls):
    # ... (rest of the download_all function remains the same)

def extract_audio_embedding(audio_buffer, url):
    """Extract audio embedding from audio buffer."""
    if audio_buffer is None:
        return None
    try:
        audio = AudioSegment.from_file(audio_buffer).set_frame_rate(16000).set_channels(1)
        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_wav:
            audio.export(temp_wav.name, format="wav")
            signal, fs = torchaudio.load(temp_wav.name)
        del audio, audio_buffer
        gc.collect()

        with torch.no_grad():
            feats = feature_extraction(signal.to(device))
            feats = mean_var_norm(feats, torch.tensor([signal.shape[-1]]).to(device))
            embeddings = model(feats)
            embedding = embeddings.squeeze().detach().cpu().numpy()

        del signal
        gc.collect()
        logging.info(f"Extracted embedding from: {url}")
        print(f"Extracted embedding from: {url}") # Terminal print
        return embedding
    except Exception as e:
        logging.error(f"Error extracting embedding from {url}: {e}")
        print(f"Error extracting embedding from {url}: {e}") # Terminal Print
        return None

# ... (rest of your code remains the same)
