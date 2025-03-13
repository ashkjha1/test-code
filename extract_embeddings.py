from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio
import torch
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/ecapa-voxceleb", run_opts={"device": str(device)})

def extract_audio_embedding(audio_buffer, url):
    """Extract audio embedding from audio buffer."""
    if audio_buffer is None:
        return None
    try:
        waveform, sample_rate = torchaudio.load(audio_buffer)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        embedding = model.encode_batch(waveform)
        embedding = embedding.squeeze().detach().cpu().numpy()
        # logging.info(f"Extracted embedding from: {url}")
        print(f"Extracted embedding from: {url}") # Terminal print
        return embedding
    except Exception as e:
        # logging.error(f"Error extracting embedding from {url}: {e}")
        print(f"Error extracting embedding from {url}: {e}") # Terminal Print
        return None
