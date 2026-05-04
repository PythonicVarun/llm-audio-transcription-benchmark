import os
import random
import argparse
from pydub import AudioSegment

def add_mixed_noises(main_audio_path, noise_folder, output_path, noise_volume_decrease_db=15):
    try:
        main_audio = AudioSegment.from_file(main_audio_path)
    except Exception as e:
        print(f"Error loading main audio: {e}")
        return

    valid_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    noise_files = [
        f for f in os.listdir(noise_folder)
        if f.lower().endswith(valid_extensions)
    ]

    if not noise_files:
        print(f"Error: No valid audio files found in '{noise_folder}'.")
        return

    print(f"Found {len(noise_files)} noise files. Loading into memory...")

    loaded_noises = []
    for filename in noise_files:
        try:
            path = os.path.join(noise_folder, filename)
            audio = AudioSegment.from_file(path)
            audio = audio - noise_volume_decrease_db
            loaded_noises.append(audio)
        except Exception as e:
            print(f"Skipping '{filename}' due to error: {e}")

    if not loaded_noises:
        print("Error: Could not load any background noise files.")
        return

    print("Generating mixed background track...")
    combined_noise = AudioSegment.empty()
    target_length = len(main_audio)

    while len(combined_noise) < target_length:
        selected_noise = random.choice(loaded_noises)
        combined_noise += selected_noise

    combined_noise = combined_noise[:target_length]

    print("Overlaying tracks...")
    final_audio = main_audio.overlay(combined_noise)

    export_format = output_path.split('.')[-1].lower()
    if export_format not in ['mp3', 'wav', 'ogg', 'flac']:
        export_format = 'wav'

    print(f"Exporting to {output_path}...")
    final_audio.export(output_path, format=export_format)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add mixed background noise to an audio file.")

    parser.add_argument("-i", "--input", required=True, help="Path to the main audio file.")
    parser.add_argument("-n", "--noise_dir", required=True, help="Directory containing background noise files.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output audio file.")
    parser.add_argument("-v", "--vol_decrease", type=int, default=15, help="Decibels to drop noise volume (default: 15).")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Fatal: Input file '{args.input}' does not exist.")
        exit(1)

    if not os.path.isdir(args.noise_dir):
        print(f"Fatal: Noise directory '{args.noise_dir}' does not exist.")
        exit(1)

    add_mixed_noises(args.input, args.noise_dir, args.output, noise_volume_decrease_db=args.vol_decrease)
