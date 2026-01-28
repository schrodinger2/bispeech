import joblib
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


SAMPLE_RATE = 16000
DURATION = 15  # seconds

print("Speak now...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()
print("Recording finished")
i = 0
src = f"audio/input{i}.wav"
while(DURATION>i+3):
    src = f"audio/input{i}.wav"
    write(src, SAMPLE_RATE, audio[i* SAMPLE_RATE:(i+3) *SAMPLE_RATE])
    i += 2

src = f"audio/input{i}.wav"
write(src, SAMPLE_RATE,audio[i* SAMPLE_RATE:-1])


clf = joblib.load("lid_model.joblib")




for j in range(0,i,2):
    pred = clf.predict(f"audio/input{j}.wav")

    print(pred)

