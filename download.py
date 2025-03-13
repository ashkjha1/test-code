import yt_dlp
import subprocess
import gc
import io

def download_audio(url):
    """Download and return YouTube audio stream as buffer using yt-dlp."""
    ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'wav',
            'outtmpl': '-',  # Output to stdout
        }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            audio_url = info_dict['url']
            id = info_dict['id']
            # logging.info(f"Downloading {url} from {audio_url}")
            print(f"[DEBUG] Downloading URL: {audio_url}") # Debugging print

            process = subprocess.Popen(
                ['yt-dlp', '-f', 'bestaudio/best', '-o', '-', url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                # logging.error(f"yt-dlp error for {url}: {stderr.decode()}")
                print(f"[DEBUG] yt-dlp Error: {stderr.decode()}") # Debugging print
                return None
            print(f"[DEBUG] Download successful, byte size: {len(stdout)}") # Debugging print
            return io.BytesIO(stdout)
    except Exception as e:
        # logging.error(f"Error downloading audio from {url}: {e}")
        print(f"[DEBUG] Exception during download: {e}") # Debugging print
        return None
