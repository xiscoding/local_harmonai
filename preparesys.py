import os
root_path =os.getcwd()

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

initDirPath = f'{root_path}/init_audio'
createPath(initDirPath)
outDirPath = f'{root_path}/audio_out'
createPath(outDirPath)

# if is_colab:
#     if google_drive and not save_models_to_google_drive or not google_drive:
#         model_path = '/content/models'
#         createPath(model_path)
#     if google_drive and save_models_to_google_drive:
#         model_path = f'{ai_root}/models'
#         createPath(model_path)
# else:
model_path = f'{root_path}/models'
createPath(model_path)

# libraries = f'{root_path}/libraries'
# createPath(libraries)

#@markdown Check the box below to save your generated audio to [Weights & Biases](https://wandb.ai/site)
save_to_wandb = True #@param {type: "boolean"}

if save_to_wandb:
    print("\nInstalling wandb...")
    os.system("pip install -qqq wandb --upgrade")
    import wandb
    # Check if logged in to wandb
    try:
      import netrc
      netrc.netrc().hosts['api.wandb.ai']
      wandb.login()
    except:
      print("\nPlease log in to Weights & Biases...")
      print("1. Sign up for a free wandb account here: https://www.wandb.ai/site")
      print("2. Enter your wandb API key, from https://wandb.ai/authorize, in the field below to log in: \n")
      wandb.login()