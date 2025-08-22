


import pretty_midi
import os
midi_dir = "trainingData/midi"
midi_files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith(".mid")]

for path in midi_files:
    pm = pretty_midi.PrettyMIDI(path)
    print(f"File: {path}")
    print("Initial tempo changes (time, bpm):", [(t, bpm) for t, bpm in zip(pm.get_tempo_changes()[0], pm.get_tempo_changes()[1])])
    print("Time signatures:", [(ts.time, ts.numerator, ts.denominator) for ts in pm.time_signature_changes])
    print("Key signatures:", [(ks.time, ks.key_number) for ks in pm.key_signature_changes])
    has_drum = any(instr.is_drum for instr in pm.instruments)
    print("Has drum track:", has_drum)
