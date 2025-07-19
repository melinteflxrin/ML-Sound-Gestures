import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth


# load environment variables from .env
load_dotenv()


CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

scope = 'user-modify-playback-state user-read-playback-state'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=scope
))


def sound_gesture(gesture):
    try:
        if gesture == "double_clap":
            sp.next_track()
        elif gesture == "snap":
            current_playback = sp.current_playback()
            if current_playback and current_playback['is_playing']:
                sp.pause_playback()
            else:
                sp.start_playback()
    except Exception:
        pass


if __name__ == "__main__":
    """test Spotify connection"""
    try:
        sp.next_track()
        print("Successfully skipped track.")
    except Exception as e:
        print(f"Spotify connection failed: {e}")
        print("check your Spotify API credentials and make sure Spotify is running.")