from urllib.parse import urlparse
import hashlib
import k_diffusion as K
import os, signal, sys
import gc
from imports_definitions import args, torch, DiffusionUncond
from preparesys import model_path
import wget

#@title Create the model
model_name = "custom" #@param ["glitch-440k", "jmann-small-190k", "jmann-large-580k", "maestro-150k", "unlocked-250k", "honk-140k", "custom"]

#@markdown ###Custom options

#@markdown If you have a custom fine-tuned model, choose "custom" above and enter a path to the model checkpoint here

#@markdown These options will not affect non-custom models
custom_ckpt_path = '/home/xdoestech/harmonai/checkpoints/unlocked-uncond-250k.ckpt'#@param {type: 'string'}

custom_sample_rate = 16000 #@param {type: 'number'}
custom_sample_size = 65536 #@param {type: 'number'}

models_map = {

    "glitch-440k": {'downloaded': False,
                         'sha': "48caefdcbb7b15e1a0b3d08587446936302535de74b0e05e0d61beba865ba00a", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/gwf-440k.ckpt"],
                         'sample_rate': 48000,
                         'sample_size': 65536
                         },
    "jmann-small-190k": {'downloaded': False,
                         'sha': "1e2a23a54e960b80227303d0495247a744fa1296652148da18a4da17c3784e9b", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/jmann-small-190k.ckpt"],
                         'sample_rate': 48000,
                         'sample_size': 65536
                         },
    "jmann-large-580k": {'downloaded': False,
                         'sha': "6b32b5ff1c666c4719da96a12fd15188fa875d6f79f8dd8e07b4d54676afa096", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/jmann-large-580k.ckpt"],
                         'sample_rate': 48000,
                         'sample_size': 131072
                         },
    "maestro-150k": {'downloaded': False,
                         'sha': "49d9abcae642e47c2082cec0b2dce95a45dc6e961805b6500204e27122d09485", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/maestro-uncond-150k.ckpt"],
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
    "unlocked-250k": {'downloaded': False,
                         'sha': "af337c8416732216eeb52db31dcc0d49a8d48e2b3ecaa524cb854c36b5a3503a", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/unlocked-uncond-250k.ckpt"],
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
    "honk-140k": {'downloaded': False,
                         'sha': "a66847844659d287f55b7adbe090224d55aeafdd4c2b3e1e1c6a02992cb6e792", 
                         'uri_list': ["https://model-server.zqevans2.workers.dev/honk-140k.ckpt"],
                         'sample_rate': 16000,
                         'sample_size': 65536
                         },
}

#@markdown If you're having issues with model downloads, check this to compare the SHA:
check_model_SHA = False #@param{type:"boolean"}

def get_model_filename(diffusion_model_name):
    model_uri = models_map[diffusion_model_name]['uri_list'][0]
    model_filename = os.path.basename(urlparse(model_uri).path)
    return model_filename

def download_model(diffusion_model_name, uri_index=0):
    if diffusion_model_name != 'custom':
        model_filename = get_model_filename(diffusion_model_name)
        model_local_path = os.path.join(model_path, model_filename) #from prepare_folders colab section
        if os.path.exists(model_local_path) and check_model_SHA:
            print(f'Checking {diffusion_model_name} File')
            with open(model_local_path, "rb") as f:
                bytes = f.read() 
                hash = hashlib.sha256(bytes).hexdigest()
                print(f'SHA: {hash}')
            if hash == models_map[diffusion_model_name]['sha']:
                print(f'{diffusion_model_name} SHA matches')
                models_map[diffusion_model_name]['downloaded'] = True
            else:
                print(f"{diffusion_model_name} SHA doesn't match. Will redownload it.")
        elif os.path.exists(model_local_path) and not check_model_SHA or models_map[diffusion_model_name]['downloaded']:
            print(f'{diffusion_model_name} already downloaded. If the file is corrupt, enable check_model_SHA.')
            models_map[diffusion_model_name]['downloaded'] = True

        if not models_map[diffusion_model_name]['downloaded']:
            for model_uri in models_map[diffusion_model_name]['uri_list']:
                wget(model_uri, model_local_path) #NOTE: add .download
                with open(model_local_path, "rb") as f:
                  bytes = f.read() 
                  hash = hashlib.sha256(bytes).hexdigest()
                  print(f'SHA: {hash}')
                if os.path.exists(model_local_path):
                    models_map[diffusion_model_name]['downloaded'] = True
                    return
                else:
                    print(f'{diffusion_model_name} model download from {model_uri} failed. Will try any fallback uri.')
            print(f'{diffusion_model_name} download failed.')

if model_name == "custom":
  ckpt_path = custom_ckpt_path
  args.sample_size = custom_sample_size
  args.sample_rate = custom_sample_rate
else:
  model_info = models_map[model_name]
  download_model(model_name)
  ckpt_path = f'{model_path}/{get_model_filename(model_name)}'
  args.sample_size = model_info["sample_size"]
  args.sample_rate = model_info["sample_rate"]

print("Creating the model...")
model = DiffusionUncond(args)
model.load_state_dict(torch.load(ckpt_path)["state_dict"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.requires_grad_(False).to(device)
print("Model created")

# # Remove non-EMA
del model.diffusion

model_fn = model.diffusion_ema