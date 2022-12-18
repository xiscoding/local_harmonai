import imports_definitions as im_def
import k_diffusion as K
#@title Sampler options
sampler_type = "v-iplms" #@param ["v-iplms", "k-heun", "k-lms", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive"]

#@markdown ---
#@markdown **K-diffusion settings (advanced)**
sigma_min = 0.0001 #@param {type: "number"}
sigma_max = 1 #@param {type: "number"}
rho=7. #@param {type: "number"}
#@markdown k-dpm-adaptive settings
rtol = 0.01 #@param {type: "number"}
atol = 0.01 #@param {type: "number"}

device = im_def.torch.device('cuda' if im_def.torch.cuda.is_available() else 'cpu')


def sample(model_fn, noise, steps=100, sampler_type="v-iplms", noise_level = 1.0):
  #Check for k-diffusion
  if sampler_type.startswith('k-'):
    denoiser = K.external.VDenoiser(model_fn)
    sigmas = K.sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

  if sampler_type == "v-iplms":
    t = im_def.torch.linspace(1, 0, steps + 1, device=device)[:-1]
    step_list = im_def.get_crash_schedule(t)

    return im_def.sampling.iplms_sample(model_fn, noise, step_list, {})

  elif sampler_type == "k-heun":
    return K.sampling.sample_heun(denoiser, noise, sigmas, disable=False)
  elif sampler_type == "k-lms":
    return K.sampling.sample_lms(denoiser, noise, sigmas, disable=False)
  elif sampler_type == "k-dpm-2":
    return K.sampling.sample_dpm_2(denoiser, noise, sigmas, disable=False)
  elif sampler_type == "k-dpm-fast":
    return K.sampling.sample_dpm_fast(denoiser, noise, sigma_min, sigma_max, steps, disable=False)
  elif sampler_type == "k-dpm-adaptive":
    return K.sampling.sample_dpm_adaptive(denoiser, noise, sigma_min, sigma_max, rtol=rtol, atol=atol, disable=False)

#NOTE: added batch size and effective length
def resample(model_fn, audio, batch_size, effective_length, steps=100, sampler_type="v-iplms", noise_level = 1.0):
  #Noise the input
  if sampler_type == "v-iplms":
    t = im_def.torch.linspace(0, 1, steps + 1, device=device)
    step_list = im_def.get_crash_schedule(t)
    step_list = step_list[step_list < noise_level]

    alpha, sigma = im_def.t_to_alpha_sigma(step_list[-1])
    noised = im_def.torch.randn([batch_size, 2, effective_length], device='cuda')
    noised = audio * alpha + noised * sigma

  elif sampler_type.startswith("k-"):
    denoiser = K.external.VDenoiser(model_fn)
    noised = audio + im_def.torch.randn_like(audio) * noise_level
    sigmas = K.sampling.get_sigmas_karras(steps, sigma_min, noise_level, rho, device=device)

  # Denoise
  if sampler_type == "v-iplms":
    return im_def.sampling.iplms_sample(model_fn, noised, step_list.flip(0)[:-1], {})

  elif sampler_type == "k-heun":
    return K.sampling.sample_heun(denoiser, noised, sigmas, disable=False)

  elif sampler_type == "k-lms":
    return K.sampling.sample_lms(denoiser, noised, sigmas, disable=False)

  elif sampler_type == "k-dpm-2":
    return K.sampling.sample_dpm_2(denoiser, noised, sigmas, s_noise=0., disable=False)

  elif sampler_type == "k-dpm-fast":
    return K.sampling.sample_dpm_fast(denoiser, noised, sigma_min, noise_level, steps, disable=False)

  elif sampler_type == "k-dpm-adaptive":
    return K.sampling.sample_dpm_adaptive(denoiser, noised, sigma_min, noise_level, rtol=rtol, atol=atol, disable=False)

def reverse_sample(model_fn, audio, steps=100, sampler_type="v-iplms", noise_level = 1.0):
  
  if sampler_type == "v-iplms":
    t = im_def.torch.linspace(0, 1, steps + 1, device=device)
    step_list = im_def.get_crash_schedule(t)

    return im_def.sampling.iplms_sample(model_fn, audio, step_list, {}, is_reverse=True)

  elif sampler_type.startswith("k-"):
    denoiser = K.external.VDenoiser(model_fn)
    sigmas = K.sampling.get_sigmas_karras(steps, sigma_min, noise_level, rho, device=device)

  # Denoise
  if sampler_type == "k-heun":
    return K.sampling.sample_heun(denoiser, audio, sigmas.flip(0)[:-1], disable=False)
  elif sampler_type == "k-lms":
    return K.sampling.sample_lms(denoiser, audio, sigmas.flip(0)[:-1], disable=False)
  elif sampler_type == "k-dpm-2":
    return K.sampling.sample_dpm_2(denoiser, audio, sigmas.flip(0)[:-1], s_noise=0., disable=False)
  elif sampler_type == "k-dpm-fast":
    return K.sampling.sample_dpm_fast(denoiser, audio, noise_level, sigma_min, steps, disable=False)

  elif sampler_type == "k-dpm-adaptive":
    return K.sampling.sample_dpm_adaptive(denoiser, audio, noise_level, sigma_min, rtol=rtol, atol=atol, disable=False)



