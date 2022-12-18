import playsound
from imports_definitions import args, plot_and_hear, torch, gc, log_audio_to_wandb, rearrange, x
from model import model_fn, device, model_name, custom_ckpt_path, os
from sampler_options import resample, sample, sampler_type
from scipy.io.wavfile import write
import datetime

#@markdown How many audio clips to create
batch_size =  4#@param {type:"number"}

#@markdown Number of steps (100 is a good start, more steps trades off speed for quality)
steps = 100 #@param {type:"number"}

#@markdown Multiplier on the default sample length from the model, allows for longer audio clips at the expense of VRAM
sample_length_mult = 2#@param {type:"number"}

#@markdown Check the box below to save your generated audio to [Weights & Biases](https://www.wandb.ai/site)
save_new_generations_to_wandb = True #@param {type: "boolean"}

#@markdown Check the box below to skip this section when running all cells
skip_for_run_all = False #@param {type: "boolean"}

effective_length = sample_length_mult * args.sample_size

if not skip_for_run_all:
  torch.cuda.empty_cache()
  gc.collect()

  # Generate random noise to sample from
  noise = torch.randn([batch_size, 2, effective_length]).to(device)

  generated = sample(model_fn, noise, steps, sampler_type)

  # Hard-clip the generated audio
  generated = generated.clamp(-1, 1)

  # Put the demos together
  generated_all = rearrange(generated, 'b d n -> d (b n)')
  #NOTE: enter appropriate filename and output directory for your use case
  file_name = 'generateraw'
  output_dir = '/home/xdoestech/harmonai/audio_out/generate_raw'
  print("All samples")
  plot_and_hear(generated_all, args.sample_rate)
  for ix, gen_sample in enumerate(generated):
    print(f'sample #{ix + 1}')
    # display(ipd.Audio(gen_sample.cpu(), rate=args.sample_rate))
    # playsound.playsound(gen_sample.cpu())
    #NOTE: added this code to save outputs as .wav files
    audio = gen_sample.cpu().numpy()
    # audio = audio.astype('int16')
    audio_path = os.path.join(
        output_dir, "{}_synthesis_{}_{}.wav".format(file_name, ix, x.strftime("%f")))
    write(audio_path, args.sample_rate, audio.T)
    print(audio_path)
  # If Weights & Biases logging enabled, save generations
  if save_new_generations_to_wandb:
    # Check if logged in to wandb
    try:
      import netrc
      netrc.netrc().hosts['api.wandb.ai']

      log_audio_to_wandb(generated, model_name, custom_ckpt_path, steps, batch_size, 
      args.sample_rate, args.sample_size, generated_all=generated_all)
    except:
      print("Not logged in to Weights & Biases, please tick the `save_to_wandb` box at the top of this notebook and run that cell again to log in to Weights & Biases first")

else:
  print("Skipping section, uncheck 'skip_for_run_all' to enable")
