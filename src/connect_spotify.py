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
            print("Double clap detected: Skipped to the next track.")
        elif gesture == "snap":
            sp.pause_playback()
            print("Snap detected: Paused playback.")
        else:
            print(f"Unknown gesture: {gesture}")
    except Exception as e:
        print(f"Spotify command failed: {e}\nMake sure Spotify is open and playing on an active device.")


# for testing
# sound_gesture("double_clap")