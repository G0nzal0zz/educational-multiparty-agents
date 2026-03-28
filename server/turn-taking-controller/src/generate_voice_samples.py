import os

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

OUTPUT_DIR = "./voices/samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ChatterboxTurboTTS.from_pretrained(device="cuda")

sample_phrases = [
    "You've brought up something quite significant here, and I appreciate your engagement. Thank you for participating actively. Regarding your question, I will try to answer it as simply as possible since it is a hard topic to explain clearly.",
    "I understand your doubt and don't worry, I think I didn't explain myself properly. Let me try to clarify this in a simpler way so it makes more sense to you.",
    "That's a really good question which lies at the very root of this topic. Let me try to set it up in a way that you can easily understand. And if you think I misunderstood your question at any point, please don't hesitate to stop me.",
    "This is a very interesting question and it isn't easy to come up with an answer right away. I will do my best to answer you in the clearest way possible, so please bear with me for a moment.",
]

for i, text in enumerate(sample_phrases):
    print(f"Generating sample {i + 1}/{len(sample_phrases)}: '{text}'")
    wav = model.generate(text)
    filename = os.path.join(OUTPUT_DIR, f"sample_{i + 1:02d}.wav")
    ta.save(filename, wav, model.sr)
    print(f"Saved: {filename}")

print("\nAll sample audio files generated successfully!")
