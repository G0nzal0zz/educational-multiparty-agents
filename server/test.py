import whisper

whisperSTT = whisper.load_model("tiny")


if __name__ == "__main__":
    t = whisperSTT.transcribe("harvard.wav")
    print("Test: ", t)
