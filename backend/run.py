from external_libraries import *
from model_paths import *



def compute_spec(ref_file):
  y, sr = librosa.load(ref_file, sr=ap.sample_rate)
  spec = ap.spectrogram(y)
  spec = torch.FloatTensor(spec).unsqueeze(0)
  return spec

print("Select speaker reference audios files:")
reference_files  = "one.wav"
# for ref in reference_files:
pynormalize.process_files([reference_files], target_dbfs=-27)
reference_files = './NORMALIZED/{}'.format(reference_files)
# reference_files = files.upload()
# reference_files = list(reference_files.keys())
# subprocess.run(args)
# for sample in reference_files:
#     !ffmpeg-normalize $sample -nt rms -t=-27 -o $sample -ar 16000 -f


reference_emb = SE_speaker_manager.compute_d_vector_from_clip(reference_files)
model.length_scale = 1  # scaler for the duration predictor. The larger it is, the slower the speech.
model.inference_noise_scale = 0.3 # defines the noise variance applied to the random z vector at inference.
model.inference_noise_scale_dp = 0.3 # defines the noise variance applied to the duration predictor z vector at inference.
text = "I am very pleased to inform that, this voice is good for a demo. Isn't it?"

model.language_manager.language_id_mapping
language_id = 0


# output
print(" > text: {}".format(text))
wav, alignment, _, _ = synthesis(
                    model,
                    text,
                    C,
                    "cuda" in str(next(model.parameters()).device),
                    ap,
                    speaker_id=None,
                    d_vector=reference_emb,
                    style_wav=None,
                    language_id=language_id,
                    enable_eos_bos_chars=C.enable_eos_bos_chars,
                    use_griffin_lim=True,
                    do_trim_silence=False,
                ).values()
print("Generated Audio")
IPython.display.display(Audio(wav, rate=ap.sample_rate))
file_name = text.replace(" ", "_")
file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
out_path = os.path.join(OUT_PATH, file_name)
print(" > Saving output to {}".format(out_path))
ap.save_wav(wav, out_path)
