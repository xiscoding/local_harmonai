import playsound
from imports_definitions import args, plot_and_hear, torch, gc, log_audio_to_wandb, rearrange, load_to_device, torchaudio, x
from model import model_fn, device, model_name, custom_ckpt_path, wget, os
from sampler_options import resample, sample, sampler_type
import torch, torchaudio
from audio_diffusion.utils import Stereo, PadCrop
from scipy.io.wavfile import write
import datetime
import numpy as np
from sampler_options import reverse_sample

# Interpolation code taken and modified from CRASH
def compute_interpolation_in_latent(latent1, latent2, lambd):
    '''
    Implementation of Spherical Linear Interpolation: https://en.wikipedia.org/wiki/Slerp
    latent1: tensor of shape (2, n)
    latent2: tensor of shape (2, n)
    lambd: list of floats between 0 and 1 representing the parameter t of the Slerp
    '''
    device = latent1.device
    lambd = torch.tensor(lambd)

    assert(latent1.shape[0] == latent2.shape[0])

    # get the number of channels
    nc = latent1.shape[0]
    interps = []
    for channel in range(nc):
    
      cos_omega = latent1[channel]@latent2[channel] / \
          (torch.linalg.norm(latent1[channel])*torch.linalg.norm(latent2[channel]))
      omega = torch.arccos(cos_omega).item()

      a = torch.sin((1-lambd)*omega) / np.sin(omega)
      b = torch.sin(lambd*omega) / np.sin(omega)
      a = a.unsqueeze(1).to(device)
      b = b.unsqueeze(1).to(device)
      interps.append(a * latent1[channel] + b * latent2[channel])
    return rearrange(torch.cat(interps), "(c b) n -> b c n", c=nc) 

#@markdown Enter the paths to two audio files to interpolate between (.wav or .flac)
source_audio_path = "/home/xdoestech/Desktop/rapwavs/bigxthaplug_safehouse.wav" #@param{type:"string"}
target_audio_path = "/home/xdoestech/Desktop/rapwavs/pesopeso_relentless.wav" #@param{type:"string"}
#NOTE: enter appropriate filename and output directory for your use case
file_name = 'interpolate_bigxthaplug'
output_dir = '/home/xdoestech/harmonai/audio_out/interpolate'
#@markdown Total number of steps (100 is a good start, can go lower for more speed/less quality)
steps = 100#@param {type:"number"}

#@markdown Number of interpolated samples
n_interps = 12 #@param {type:"number"}

#@markdown Multiplier on the default sample length from the model, allows for longer audio clips at the expense of VRAM
sample_length_mult = 1#@param {type:"number"}

#@markdown Check the box below to skip this section when running all cells
skip_for_run_all = False #@param {type: "boolean"}

effective_length = args.sample_size * sample_length_mult

if not skip_for_run_all:

  augs = torch.nn.Sequential(
    PadCrop(effective_length, randomize=True),
    Stereo()
  )

  if source_audio_path == "":
    print("No file path provided for the source audio, please upload a file")
    # uploaded = files.upload()
    source_audio_path = input("File Path: ")

  audio_sample_1 = load_to_device(source_audio_path, args.sample_rate)

  print("Source audio sample loaded")

  if target_audio_path == "":
    print("No file path provided for the target audio, please upload a file")
    # uploaded = files.upload()
    target_audio_path = input("File Path: ")

  audio_sample_2 = load_to_device(target_audio_path, args.sample_rate)

  print("Target audio sample loaded")

  audio_samples = augs(audio_sample_1).unsqueeze(0).repeat([2, 1, 1])
  audio_samples[1] = augs(audio_sample_2)

  print("Initial audio samples")
  plot_and_hear(audio_samples[0], args.sample_rate)
  plot_and_hear(audio_samples[1], args.sample_rate)

  reversed = reverse_sample(model_fn, audio_samples, steps)

  latent_series = compute_interpolation_in_latent(reversed[0], reversed[1], [k/n_interps for k in range(n_interps + 2)])

  generated = sample(model_fn, latent_series, steps) 
  
  #sampling.iplms_sample(, latent_series, step_list.flip(0)[:-1], {})

  # Put the demos together
  generated_all = rearrange(generated, 'b d n -> d (b n)')

  print("Full interpolation")
  plot_and_hear(generated_all, args.sample_rate)
  for ix, gen_sample in enumerate(generated):
    print(f'sample #{ix + 1}')
    # display(ipd.Audio(gen_sample.cpu(), rate=args.sample_rate))
    audio = gen_sample.cpu().numpy()
    # audio = audio.astype('int16')
    audio_path = os.path.join(
        output_dir, "{}_synthesis_{}_{}.wav".format(file_name, ix, x.strftime("%f")))
    write(audio_path, args.sample_rate, audio.T)
    print(audio_path)
else:
  print("Skipping section, uncheck 'skip_for_run_all' to enable") 
