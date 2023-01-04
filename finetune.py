#@markdown Name for the finetune project, used as the W&B project name, as well as the directory for the saved checkpoints
NAME="ENTER NAME FOR WANB" #@param {type:"string"}

#/home/xdoestech/harmonai/audio_out/finetune/44khz/run4_precision16/demo_00229001_synthesis_364_802714.wav
#@markdown Path to the directory of audio data to use for fine-tuning
TRAINING_DIR="ENTER PATH" #@param {type:"string"}

#@markdown Path to the checkpoint to fine-tune
CKPT_PATH="ENTER PATH" #@param {type:"string"}

#@markdown Directory path for saving the fine-tuned outputs
OUTPUT_DIR="ENTER PATH " #@param {type:"string"}

#@markdown Number of training steps between demos
DEMO_EVERY=250 #@param {type:"number"}

#@markdown Number of training steps between saving model checkpoints
CHECKPOINT_EVERY=500 #@param {type:"number"}

####FILE SIZE CALCULATION
# ((bitspersample(16-bit or 24-bit)*
# samplespersec(44.1KHz-48KHz)*
# no.of channels*
# duration(no.of sec the music played)

#@markdown Sample rate to train at
# glitch: 48k
# jmann: 48k
# maestro: 16k
# unlocked: 16k
# honk: 16k
SAMPLE_RATE=48100 #@param {type:"number"}

#@markdown Number of audio samples per training sample
SAMPLE_SIZE=(8192*16)#@param {type:"number"}
#NOTE must be a multiple of 8192
##8192 * 8 = 65536
##8192 * 4 = 32768
##8192 * 3 = 24576
##8192 * 2 = 16384

#@markdown If true, the audio samples provided will be randomly cropped to SAMPLE_SIZE samples
#@markdown
#@markdown Turn off if you want to ensure the training data always starts at the beginning of the audio files (good for things like drum one-shots)
RANDOM_CROP=True #@param {type:"boolean"}

#@markdown Batch size to fine-tune (make it as high as it can go for your GPU)
BATCH_SIZE=1 #@param {type:"number"}

#@markdown Accumulate gradients over n batches, useful for training on one GPU. 
#@markdown
#@markdown Effective batch size is BATCH_SIZE * ACCUM_BATCHES.
#@markdown
#@markdown Also increases the time between demos and saved checkpoints
ACCUM_BATCHES=1 #@param {type:"number"}

#XDTEDIT_1422
#cuda device number to use
DEVICE_NUM = 0
random_crop_str = f"--random-crop True" if RANDOM_CROP else ""

# Escape spaces in paths
CKPT_PATH = CKPT_PATH.replace(f" ", f"\ ")
OUTPUT_DIR = f"{OUTPUT_DIR}/{NAME}".replace(f" ", f"\ ")

import os
os.system("cd ~/harmonai")

os.system(f"python3 sample-generator/train_uncond.py --ckpt-path {CKPT_PATH}\
                                                          --name {NAME}\
                                                          --training-dir {TRAINING_DIR}\
                                                          --sample-size {SAMPLE_SIZE}\
                                                          --accum-batches {ACCUM_BATCHES}\
                                                          --sample-rate {SAMPLE_RATE}\
                                                          --batch-size {BATCH_SIZE}\
                                                          --demo-every {DEMO_EVERY}\
                                                          --checkpoint-every {CHECKPOINT_EVERY}\
                                                          --num-workers 8\
                                                          --num-gpus 1\
                                                          {random_crop_str}\
                                                          --device_num = {DEVICE_NUM}\
                                                          --save-path {OUTPUT_DIR}")


# samplesize:65536
# numworkers:4
# batchsize:2
# accumbatches:4
## vram: 10415MiB

# samplesize:65536-8192
# numworkers:4
# batchsize:2
# accumbatches:4
## vram: 9943MiB - 11649

# samplesize:65536-8192*2
# numworkers:4
# batchsize:2
# accumbatches:4
## vram: 

#NOTE: ValueError: num_samples should be a positive integer value, but got num_samples=0
    #WRONG TRAINING PATH