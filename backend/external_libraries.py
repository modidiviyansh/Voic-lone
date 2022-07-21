import sys
import os
import string
import time
import argparse
import json
import numpy as np
import IPython
from IPython.display import Audio
import torch
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *
from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment
from google.colab import files
import librosa
import ffmpeg_normalize
import pynormalize

