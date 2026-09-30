from pathlib import Path

from music21 import corpus, converter


# Get the folder where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# MIDI output folder
OUTPUT_DIR = BASE_DIR / "data" / "midi"

# Create the folder if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("AI MUSIC GENERATION - DATA COLLECTION")
print("=" * 60)

# Get Bach works from the music21 corpus
pieces = corpus.getComposer("bach")

print(f"Found {len(pieces)} Bach works.")
print()


saved = 0

# Start with the first 40 works
for index, piece_path in enumerate(pieces[:40]):

    try:
        print(f"Processing {index + 1}: {piece_path}")

        # Read the musical work
        score = converter.parse(piece_path)

        # Create output filename
        output_file = OUTPUT_DIR / f"bach_{index:03d}.mid"

        # Save as MIDI
        score.write("midi", fp=output_file)

        print(f"Saved: {output_file.name}")
        print()

        saved += 1

    except Exception as error:

        print(f"Could not process {piece_path}")
        print(f"Error: {error}")
        print()


print("=" * 60)
print(f"DATA COLLECTION COMPLETE")
print(f"MIDI files created: {saved}")
print(f"Location: {OUTPUT_DIR}")
print("=" * 60)