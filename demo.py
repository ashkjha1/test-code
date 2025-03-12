import os
import io
import faiss
import numpy as np
import torch
import torchaudio
import gc
import speechbrain as sb
import multiprocessing
from speechbrain.pretrained import EncoderClassifier
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
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir=model_path, run_opts={"device": str(device)}
)

def download_audio(url):
    """Download and return YouTube audio stream as buffer using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '-',  # Output to stdout
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            audio_url = info_dict['url']
            logging.info(f"Downloading {url} from {audio_url}")
            print(f"[DEBUG] Downloading URL: {audio_url}") # Debugging print

            process = subprocess.Popen(
                ['yt-dlp', '-f', 'bestaudio/best', '-o', '-', url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logging.error(f"yt-dlp error for {url}: {stderr.decode()}")
                print(f"[DEBUG] yt-dlp Error: {stderr.decode()}") # Debugging print
                return None
            print(f"[DEBUG] Download successful, byte size: {len(stdout)}") # Debugging print
            return io.BytesIO(stdout)
    except Exception as e:
        logging.error(f"Error downloading audio from {url}: {e}")
        print(f"[DEBUG] Exception during download: {e}") # Debugging print
        return None

def download_all(urls):
    """Download all audio files."""
    return [download_audio(url) for url in urls]

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
        with torch.amp.autocast(device_type=device.type):
            embeddings = model.encode_batch(signal)
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

def add_embedding_to_db(embedding, reference):
    """Add speaker embedding to FAISS DB."""
    global db_audio_map
    embedding = np.array(embedding).astype('float32').reshape(1, -1)
    index_id = index.ntotal
    index.add_with_ids(embedding, np.array([index_id]))
    db_audio_map[index_id] = reference
    del embedding
    gc.collect()
    logging.info(f"Added embedding for: {reference}")
    print(f"Added embedding for: {reference}") # Terminal print

def match_speakers(embedding, threshold=0.5):
    """Find matching speakers from the vector DB."""
    embedding = np.array(embedding).astype('float32').reshape(1, -1)
    if index.ntotal == 0:
        logging.info("No embeddings in the database.")
        print("No embeddings in the database.") # Terminal print
        return []
    D, I = index.search(embedding, k=min(3, index.ntotal))
    matches = [(db_audio_map[i], D[0][idx]) for idx, i in enumerate(I[0]) if i != -1 and D[0][idx] < threshold]
    del embedding
    gc.collect()
    logging.info(f"Matches found: {matches}")
    print(f"Matches found: {matches}") # Terminal print
    return matches

def process_youtube_audio(url, audio_buffer):
    """Process YouTube audio, extract embedding, and store in DB."""
    try:
        embedding = extract_audio_embedding(audio_buffer, url)
        if embedding is not None:
            add_embedding_to_db(embedding, url)
            logging.info(f"Processed and stored embedding for: {url}")
            print(f"Processed and stored embedding for: {url}") #Terminal print
        else:
            logging.warning(f"Skipped processing {url} due to download or embedding error.")
            print(f"Skipped processing {url} due to download or embedding error.") # Terminal print
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        print(f"Error processing {url}: {e}") # Terminal print

def process_input_clip(url):
    """Process a new input clip, match it with the database, and prompt for addition."""
    audio_buffer = download_audio(url)
    embedding = extract_audio_embedding(audio_buffer, url)
    if embedding is not None:
        matches = match_speakers(embedding)
        logging.info(f"Matches found: {matches}")
        print(f"Matches found: {matches}") # Terminal print
        if matches:
            choice = input("Add this clip to the database? (y/n): ")
            if choice.lower() == 'y':
                add_embedding_to_db(embedding, url)
        del embedding
        gc.collect()
    else:
        logging.warning(f"Could not process input clip: {url}")
        print(f"Could not process input clip: {url}") # Terminal print

def process_url_multiprocessing(url, audio_buffer):
    """Wrapper function for multiprocessing."""
    process_youtube_audio(url, audio_buffer)

def parallel_process(urls, audio_buffers):
    """Process multiple YouTube URLs in parallel using multiprocessing."""
    with multiprocessing.get_context("spawn").Pool(processes=min(4, len(urls))) as pool:
        pool.starmap(process_url_multiprocessing, zip(urls, audio_buffers))

def download_all_youtube_clips(urls, save_dir="downloaded_clips"):
    """Download all YouTube clips for manual review."""
    os.makedirs(save_dir, exist_ok=True)
    audio_buffers = download_all(urls)
    for url, audio_buffer in zip(urls, audio_buffers):
        if audio_buffer:
            video_id = url.split("=")[-1]
            save_path = os.path.join(save_dir, f"{video_id}.wav")
            audio = AudioSegment.from_file(audio_buffer).set_frame_rate(16000).set_channels(1)
            audio.export(save_path, format="wav")
            del audio, audio_buffer
            gc.collect()
        else:
            logging.warning(f"Could not download {url}")
            print(f"Could not download {url}") # Terminal Print


if __name__ == "__main__":
    youtube_urls = [
        "https://www.youtube.com/watch?v=oLC2M8ybhL0&ab_channel=%F0%9D%90%8F%F0%9D%90%AC%F0%9D%90%B2%F0%9D%90%9C%F0%9D%90%A1%F0%9D%90%A2%F0%9D%90%9C%F0%9D%90%84%F0%9D%90%9E%F0%9D%90%AC%F0%9D%90%A1",
        "https://www.youtube.com/shorts/8l8udF7Vs74",
        "https://www.youtube.com/shorts/IP0LSmb3tdI",
        "https://www.youtube.com/shorts/Bz8oOHg0Rrs",
        "https://www.youtube.com/watch?v=jW_ybtzjq7o&ab_channel=GadgetByte",
        "https://www.youtube.com/watch?v=WL4RWNptU_Y&ab_channel=Mrwhosetheboss",
        "https://www.youtube.com/shorts/1NXfkAJolF8",
    ]
    logging.info("Downloading all audio clips...")
    print("[DEBUG] Downloading all audio clips...")
    audio_buffers = download_all(youtube_urls)

    logging.info("Processing YouTube audio and storing embeddings using multiprocessing...")
    print("[DEBUG] Processing YouTube audio and storing embeddings using multiprocessing...")
    parallel_process(youtube_urls, audio_buffers)

    logging.info("Downloading all clips for review...")
    download_all_youtube_clips(youtube_urls)

    # new_input_url = "https://www.youtube.com/shorts/VXcKhP-vMZc"
    # logging.info("Processing new input clip and matching...")
    # process_input_clip(new_input_url)
