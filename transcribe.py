import whisper


model = whisper.load_model("base")
transcription = model.transcribe("Audio/participant1_mono.wav")
text_only = transcription["text"]

# Write the text to a file
with open("transcription_p1.txt", "w") as f:
    f.write(text_only)