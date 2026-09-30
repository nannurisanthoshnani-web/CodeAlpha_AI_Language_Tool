from pathlib import Path
import json

import numpy as np
import streamlit as st
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
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = MODEL_DIR / "music_lstm.keras"

    return tf.keras.models.load_model(model_path)


# ============================================================
# LOAD TOKEN MAPPING
# ============================================================

@st.cache_data
def load_music_data():

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


    return (
        token_to_int,
        int_to_token,
        sequence_length,
        seed
    )


# ============================================================
# GENERATE MUSIC
# ============================================================

def generate_music(
    model,
    token_to_int,
    int_to_token,
    sequence_length,
    seed,
    number_of_notes,
    temperature
):

    sequence = list(seed)

    generated_tokens = []


    for _ in range(number_of_notes):

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


        prediction = np.asarray(
            prediction,
            dtype=np.float64
        )


        prediction = np.log(
            prediction + 1e-8
        )


        prediction = (
            prediction / temperature
        )


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


    return generated_tokens


# ============================================================
# CREATE MIDI FILE
# ============================================================

def create_midi(tokens, output_path):

    score = stream.Stream()


    score.insert(
        0,
        tempo.MetronomeMark(
            number=100
        )
    )


    for token in tokens:

        try:

            music_data, duration = token.rsplit(
                "|",
                1
            )


            duration = float(duration)


            # REST
            if music_data == "REST":

                element = note.Rest()


            # CHORD
            elif music_data.startswith("CHORD:"):

                chord_data = music_data[
                    len("CHORD:"):
                ]

                pitches = chord_data.split(".")

                element = chord.Chord(
                    pitches
                )


            # NOTE
            else:

                element = note.Note(
                    music_data
                )


            element.duration.quarterLength = (
                duration
            )


            score.append(element)


        except Exception:

            continue


    score.write(
        "midi",
        fp=output_path
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🎵 AI Music Generator")

st.write(
    "Generate original musical sequences "
    "using an LSTM neural network trained "
    "on classical MIDI music."
)


st.divider()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("🎛️ Music Settings")


number_of_notes = st.sidebar.slider(
    "Number of notes",
    min_value=25,
    max_value=300,
    value=150,
    step=25
)


temperature = st.sidebar.slider(
    "Creativity",
    min_value=0.3,
    max_value=1.5,
    value=0.9,
    step=0.1
)


st.sidebar.info(
    "Lower creativity produces more predictable "
    "music. Higher creativity produces more variation."
)


# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button(
    "🎼 Generate Music",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "AI is composing your music..."
        ):

            # Load trained model
            model = load_model()


            # Load mappings
            (
                token_to_int,
                int_to_token,
                sequence_length,
                seed
            ) = load_music_data()


            # Generate music
            generated_tokens = generate_music(
                model,
                token_to_int,
                int_to_token,
                sequence_length,
                seed,
                number_of_notes,
                temperature
            )


            # Output file
            output_path = (
                OUTPUT_DIR /
                "generated_music.mid"
            )


            # Convert to MIDI
            create_midi(
                generated_tokens,
                output_path
            )


            # Read MIDI file
            midi_data = output_path.read_bytes()


        st.success(
            "🎉 Music generated successfully!"
        )


        st.subheader("🎵 Your AI-generated music")


        st.write(
            f"Generated {len(generated_tokens)} "
            "musical events."
        )


        # Download button
        st.download_button(
            label="⬇️ Download MIDI",
            data=midi_data,
            file_name="generated_music.mid",
            mime="audio/midi",
            use_container_width=True
        )


        # Show generated tokens
        with st.expander(
            "View generated music sequence"
        ):

            st.code(
                " → ".join(
                    generated_tokens[:50]
                )
            )


    except Exception as error:

        st.error(
            f"Music generation failed: {error}"
        )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader("About this project")

st.write(
    """
    This project uses a Long Short-Term Memory (LSTM)
    neural network to learn patterns from classical
    MIDI music.

    The trained model predicts the next musical event
    based on previously generated events.

    The generated sequence is converted back into a
    MIDI file that can be downloaded and played using
    a MIDI-compatible application.
    """
)


st.caption(
    "CodeAlpha Internship — Task 3: "
    "Music Generation with AI"
)