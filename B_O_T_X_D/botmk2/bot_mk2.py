from librosa.util.utils import frame
from .ml import snufflupugus, CNN, CLASSES
from .frame_generator import frame_visualizer, generate_spectrogram, segment_spectrogram
import cv2

DEBUG = True
DEBUG_PATH = '307.png'

def main():
    # Just a little debug condition at the top :)
    if DEBUG:
        test = cv2.imread('307.png', 1)
        frame_visualizer(test, snufflupugus(test), CLASSES, save=True, fignum=111111)
        return
    
    # Lets start with making the spectrogram from the audio
    spec = generate_spectrogram('long_spec_wav.wav')
    # Now let us segment that bad boi ;)
    segments = segment_spectrogram(spec)
    # Finally we iterate, predict and plot
    i = 0
    for seg in segments:
        res = snufflupugus(seg)
        frame_visualizer(seg, res, CLASSES, save=True, fignum=i)
        i = i + 1
    return

if __name__ == '__main__':
    main()