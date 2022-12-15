#@title Generate new sounds from recording
import playsound
from imports_definitions import args, plot_and_hear, torch, gc, log_audio_to_wandb, rearrange, load_to_device, torchaudio
from model import model_fn, device, model_name, custom_ckpt_path, wget, os
from sampler_options import resample, sample, sampler_type
from enter_audio import file_path, recording_file_path, record_audio
from audio_diffusion.utils import Stereo, PadCrop
from scipy.io.wavfile import write
import datetime

x = datetime.datetime.now()
#NOTE: enter appropriate filename and output directory for your use case
file_name = 'generatefromaudio'
output_dir = '/home/xdoestech/harmonai/audio_out'
#@markdown Total number of steps (100 is a good start, more steps trades off speed for quality)
steps = 100#@param {type:"number"}

#@markdown How much (0-1) to re-noise the original sample. Adding more noise (a higher number) means a bigger change to the input audio
noise_level = 0.3#@param {type:"number"}

#@markdown Multiplier on the default sample length from the model, allows for longer audio clips at the expense of VRAM
sample_length_mult = 2#@param {type:"number"}

#@markdown How many variations to create
batch_size = 4 #@param {type:"number"}

#@markdown Check the box below to save your generated audio to [Weights & Biases](https://www.wandb.ai/site)
save_own_generations_to_wandb = True #@param {type: "boolean"}

#@markdown Check the box below to skip this section when running all cells
skip_for_run_all = False #@param {type: "boolean"}

effective_length = args.sample_size * sample_length_mult

if not skip_for_run_all:
  torch.cuda.empty_cache()
  gc.collect()

  augs = torch.nn.Sequential(
    PadCrop(effective_length, randomize=True),
    Stereo()
  )

  fp = recording_file_path if record_audio else file_path

  audio_sample = load_to_device(fp, args.sample_rate)

  audio_sample = augs(audio_sample).unsqueeze(0).repeat([batch_size, 1, 1])

  print("Initial audio sample")
  plot_and_hear(audio_sample[0], args.sample_rate)

  # def resample(model_fn, audio, batch_size, effective_length, steps=100, sampler_type="v-iplms", noise_level = 1.0):

  generated = resample(model_fn, audio_sample, batch_size, effective_length, steps, sampler_type, noise_level=noise_level)

  print("Regenerated audio samples")
  plot_and_hear(rearrange(generated, 'b d n -> d (b n)'), args.sample_rate)

  for ix, gen_sample in enumerate(generated):
    print(f'sample #{ix + 1}')
    # display(ipd.Audio(gen_sample.cpu(), rate=args.sample_rate))
    #NOTE: added this to save outputs as .wav files
    audio = gen_sample.cpu().numpy()
    # audio = audio.astype('int16')
    audio_path = os.path.join(
        output_dir, "{}_synthesis_{}_{}.wav".format(file_name, ix, x.strftime("%f")))
    write(audio_path, args.sample_rate, audio.T)
    print(audio_path)

  # If Weights & Biases logging enabled, save generations
  if save_own_generations_to_wandb:
    # Check if logged in to wandb
    try:
      import netrc
      netrc.netrc().hosts['api.wandb.ai']

      log_audio_to_wandb(generated, model_name, custom_ckpt_path, steps, batch_size, 
        args.sample_rate, args.sample_size, file_path=fp, original_sample=audio_sample[0].cpu().numpy(),
        noise_level=noise_level, gen_type='own_file')
    except:
      print("Not logged in to Weights & Biases, please tick the `save_to_wandb` box at the top of this notebook and run that cell again to log in to Weights & Biases first")

else:
  print("Skipping section, uncheck 'skip_for_run_all' to enable")