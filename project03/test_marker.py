import pyaudio
import wave
import numpy as np
import os

CHUNK = 1024


def play_audio(filename):
    wf = wave.open(filename, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != b'':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()


def mark():
    while True:
        response = input("Is this debate drastic? \nYes: enter 'p' \nNo: enter 'n'. ")
        if response == 'p':
            return 1
        else:
            if response == 'n':
                return 0


if __name__ == '__main__':
    nums = 100
    idx = np.arange(nums)
    result = np.ones(nums) * (-1)
    ensure = input('Enter [y] to continue: ')
    if ensure == 'y':
        if os.path.exists("test_result.npy"):
            result = np.load("test_result.npy")
        left = np.argwhere(result == -1)
        idx = idx[left].reshape(-1)
        if idx is not None:
            for i in idx:
                print("Current Index: {}. Audio starts now.".format(i))
                play_audio("./audio/{}_audio.wav".format(i))
                print("Audio stops now.")
                result[i] = mark()
                np.save("test_result.npy", result)
            print("Annotation ends. ")
        else:
            print("You've finished all the annotation. ")
    else:
        if os.path.exists("test_result.npy"):
            result = np.load("test_result.npy")
            print(result)
        pass
