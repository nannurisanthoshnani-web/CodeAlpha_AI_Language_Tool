from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from music21 import converter, note, chord


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "midi"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 50

EPOCHS = 20

BATCH_SIZE = 64


# ============================================================
# CONVERT DURATION
# ============================================================

def quantize_duration(duration):

    allowed_durations = [
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0
    ]

    duration = float(duration)

    return min(
        allowed_durations,
        key=lambda x: abs(x - duration)
    )


# ============================================================
# CONVERT MUSIC ELEMENT TO TOKEN
# ============================================================

def element_to_token(element):

    duration = quantize_duration(
        element.duration.quarterLength
    )

    # Rest
    if isinstance(element, note.Rest):

        return f"REST|{duration}"

    # Single note
    if isinstance(element, note.Note):

        return f"{element.pitch.nameWithOctave}|{duration}"

    # Chord
    if isinstance(element, chord.Chord):

        pitches = ".".join(
            pitch.nameWithOctave
            for pitch in element.pitches
        )

        return f"CHORD:{pitches}|{duration}"

    return None


# ============================================================
# LOAD MIDI FILES
# ============================================================

def load_music_tokens():

    midi_files = list(DATA_DIR.glob("*.mid"))

    if not midi_files:

        raise FileNotFoundError(
            "No MIDI files found. "
            "Run collect_data.py first."
        )

    print()
    print("=" * 60)
    print("LOADING MIDI DATA")
    print("=" * 60)
    print()

    all_tokens = []

    for index, midi_file in enumerate(midi_files):

        try:

            print(
                f"[{index + 1}/{len(midi_files)}] "
                f"Processing {midi_file.name}"
            )

            score = converter.parse(midi_file)

            elements = score.flatten().notesAndRests

            file_tokens = []

            for element in elements:

                token = element_to_token(element)

                if token:

                    file_tokens.append(token)

            all_tokens.extend(file_tokens)

            print(
                f"    Tokens extracted: "
                f"{len(file_tokens)}"
            )

        except Exception as error:

            print(
                f"    Skipped because of error: "
                f"{error}"
            )

    print()
    print("=" * 60)
    print(f"TOTAL TOKENS: {len(all_tokens)}")
    print("=" * 60)
    print()

    return all_tokens


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

def prepare_sequences(tokens):

    vocabulary = sorted(set(tokens))

    print(f"Vocabulary size: {len(vocabulary)}")

    token_to_int = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    int_to_token = {
        index: token
        for token, index in token_to_int.items()
    }

    encoded = [
        token_to_int[token]
        for token in tokens
    ]

    X = []

    y = []

    for index in range(
        len(encoded) - SEQUENCE_LENGTH
    ):

        sequence = encoded[
            index:index + SEQUENCE_LENGTH
        ]

        target = encoded[
            index + SEQUENCE_LENGTH
        ]

        X.append(sequence)

        y.append(target)

    X = np.array(
        X,
        dtype=np.int32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    print(f"Training sequences: {len(X)}")

    return (
        X,
        y,
        token_to_int,
        int_to_token
    )


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_model(vocabulary_size):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(SEQUENCE_LENGTH,)
        ),

        tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=128
        ),

        tf.keras.layers.LSTM(
            128,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(
            128
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            vocabulary_size,
            activation="softmax"
        )
    ])

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print()
    print("=" * 60)
    print("AI MUSIC GENERATION - LSTM TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load music
    # --------------------------------------------------------

    tokens = load_music_tokens()

    if len(tokens) <= SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough music tokens for training."
        )

    # --------------------------------------------------------
    # Prepare sequences
    # --------------------------------------------------------

    (
        X,
        y,
        token_to_int,
        int_to_token
    ) = prepare_sequences(tokens)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print("Building LSTM model...")
    print()

    model = build_model(
        len(token_to_int)
    )

    model.summary()

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True
    )

    history = model.fit(

        X,

        y,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        validation_split=0.1,

        shuffle=True,

        callbacks=[
            early_stopping
        ]
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = MODEL_DIR / "music_lstm.keras"

    model.save(model_path)

    print()
    print(f"Model saved to:")
    print(model_path)

    # --------------------------------------------------------
    # Save token mapping
    # --------------------------------------------------------

    mapping = {

        "token_to_int": token_to_int,

        "int_to_token": {
            str(key): value
            for key, value in int_to_token.items()
        },

        "sequence_length": SEQUENCE_LENGTH
    }

    mapping_path = MODEL_DIR / "mapping.json"

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            mapping,
            file,
            indent=2
        )

    print()
    print(f"Mapping saved to:")
    print(mapping_path)

    # --------------------------------------------------------
    # Save seed sequence
    # --------------------------------------------------------

    seed = tokens[:SEQUENCE_LENGTH]

    seed_path = MODEL_DIR / "seed.json"

    with open(
        seed_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            seed,
            file,
            indent=2
        )

    print()
    print(f"Seed saved to:")
    print(seed_path)

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print()

    print("Created files:")

    print("1. music_lstm.keras")

    print("2. mapping.json")

    print("3. seed.json")

    print()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()