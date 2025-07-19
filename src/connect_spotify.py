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
            # Check current playback state to decide pause or resume
            current_playback = sp.current_playback()
            if current_playback and current_playback['is_playing']:
                sp.pause_playback()
                print("Snap detected: Paused playback.")
            else:
                # Try to resume playback
                sp.start_playback()
                print("Snap detected: Resumed playback.")
        else:
            print(f"Unknown gesture: {gesture}")
    except Exception as e:
        error_msg = str(e).lower()
        if "no active device" in error_msg or "device not found" in error_msg:
            print("No active Spotify device found. Please:")
            print("1. Open Spotify on your computer or phone")
            print("2. Start playing a song")
            print("3. Try the gesture again")
        elif "premium required" in error_msg:
            print("Spotify Premium is required for playback control.")
        elif "forbidden" in error_msg:
            print("Permission denied. Check your Spotify app permissions.")
        else:
            print(f"Spotify command failed: {e}")
            print("Make sure Spotify is open and playing on an active device.")


# for testing
# sound_gesture("double_clap")