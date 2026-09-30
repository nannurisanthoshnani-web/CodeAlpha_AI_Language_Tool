from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from music21 import stream, note, chord, tempo


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_NOTES = 150

TEMPERATURE = 0.9


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

print("=" * 60)
print("AI MUSIC GENERATION")
print("=" * 60)
print()

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_DIR / "music_lstm.keras"
)

print("Model loaded successfully.")

print()


with open(
    MODEL_DIR / "mapping.json",
    "r",
    encoding="utf-8"
) as file:

    mapping = json.load(file)


with open(
    MODEL_DIR / "seed.json",
    "r",
    encoding="utf-8"
) as file:

    seed = json.load(file)


token_to_int = mapping["token_to_int"]

int_to_token = {
    int(key): value
    for key, value in mapping["int_to_token"].items()
}

sequence_length = mapping["sequence_length"]


# ============================================================
# GENERATE MUSIC TOKENS
# ============================================================

sequence = list(seed)

generated_tokens = []


print("Generating music...")
print()


for index in range(NUMBER_OF_NOTES):

    current_sequence = sequence[
        -sequence_length:
    ]

    encoded_sequence = [
        token_to_int[token]
        for token in current_sequence
    ]

    input_data = np.array(
        [encoded_sequence],
        dtype=np.int32
    )

    prediction = model.predict(
        input_data,
        verbose=0
    )[0]

    # Add small value to avoid log(0)
    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )

    prediction = np.log(
        prediction + 1e-8
    )

    prediction = prediction / TEMPERATURE

    probabilities = np.exp(
        prediction - np.max(prediction)
    )

    probabilities = (
        probabilities /
        probabilities.sum()
    )

    next_index = np.random.choice(
        len(probabilities),
        p=probabilities
    )

    next_token = int_to_token[next_index]

    sequence.append(next_token)

    generated_tokens.append(next_token)

    if (index + 1) % 25 == 0:

        print(
            f"Generated {index + 1}/"
            f"{NUMBER_OF_NOTES} notes"
        )


# ============================================================
# CONVERT TOKENS TO MUSIC21 OBJECTS
# ============================================================

print()
print("Converting generated music to MIDI...")

score = stream.Stream()

# Set tempo
score.insert(
    0,
    tempo.MetronomeMark(
        number=100
    )
)


for token in generated_tokens:

    try:

        music_data, duration = token.rsplit(
            "|",
            1
        )

        duration = float(duration)


        # ----------------------------------------------------
        # REST
        # ----------------------------------------------------

        if music_data == "REST":

            element = note.Rest()

            element.duration.quarterLength = duration

            score.append(element)


        # ----------------------------------------------------
        # CHORD
        # ----------------------------------------------------

        elif music_data.startswith("CHORD:"):

            chord_data = music_data[
                len("CHORD:"):
            ]

            pitches = chord_data.split(".")

            element = chord.Chord(
                pitches
            )

            element.duration.quarterLength = duration

            score.append(element)


        # ----------------------------------------------------
        # SINGLE NOTE
        # ----------------------------------------------------

        else:

            element = note.Note(
                music_data
            )

            element.duration.quarterLength = duration

            score.append(element)


    except Exception as error:

        print(
            f"Skipped token: {token}"
        )

        continue


# ============================================================
# SAVE MIDI
# ============================================================

output_file = (
    OUTPUT_DIR /
    "generated_music.mid"
)


score.write(
    "midi",
    fp=output_file
)


print()
print("=" * 60)
print("MUSIC GENERATION COMPLETE!")
print("=" * 60)
print()
print(
    f"MIDI file saved to:"
)
print(output_file)
print()