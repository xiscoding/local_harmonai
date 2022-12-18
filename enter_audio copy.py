#@title Record audio or enter a filepath to a prerecorded audio file
import torch
import torchaudio
from typing import Iterable, Tuple
import gradio as gr
from imports_definitions import np
import datetime
x = datetime.datetime.now()
Audio = Tuple[int, np.ndarray]

#@markdown Check the box below to create an audio recording interface below
record_audio = True #@param {type: "boolean"}

#@markdown If you left "record_audio" blank, enter a path to an audio file you want to alter, or leave blank to upload a file (.wav or .flac).
file_path = "/home/xdoestech/harmonai/init_audio/bigxthaplug_safehouse.wav" #@param{type:"string"}

#@markdown Number of audio recordings to combine into one clip. Only applies if the "record_audio" box is checked.
n_audio_recordings = 1 #@param{type:"number"}

# this is a global variable to be filled in by the generate_from_audio callback
recording_file_path = ""


def combine_audio(*audio_iterable: Iterable[Audio]) -> Audio:
    """Combines an iterable of audio signals into one."""
    max_len = max([x.shape for _, x in audio_iterable])
    combined_audio = np.zeros(max_len, dtype=np.int32)
    for _, a in audio_iterable:
        combined_audio[:a.shape[0]] = combined_audio[:a.shape[0]] * .5 + a * .5
    return combined_audio


def generate_from_audio(file_path: str, *audio_iterable: Iterable[Audio]):
    sample_rate = audio_iterable[0][0]
    combined_audio = combine_audio(*audio_iterable)
    tensor = torch.from_numpy(
        np.concatenate(
            [
                combined_audio.reshape(1, -1),
                combined_audio.reshape(1, -1)
            ],
            axis=0,
        )
    )
    global recording_file_path
    recording_file_path = file_path
    torchaudio.save(
        file_path,
        tensor,
        sample_rate=sample_rate,
        format="wav"
    )
    return (sample_rate, combined_audio), file_path

if record_audio:
    recording_interface = gr.Interface(
        fn=generate_from_audio,
        inputs=[
            gr.Textbox(
                f'recordings/recording{x.strftime("%f")}.wav',
                label="save recording to filepath",
            ),
            *[
                gr.Audio(source="microphone", label=f"audio clip {i}")
                for i in range(1, n_audio_recordings + 1)
            ]
        ],
        outputs=[
            gr.Audio(label="combined output audio"),
            gr.File(label="output file"),
        ],
        allow_flagging="never",
    )

    recording_interface.launch()
elif file_path == "":
    print("No file path provided, please upload a file")
    # uploaded = files.upload()
    # file_path = list(uploaded.keys())[0]
    file_path = input("File Path: ")

if not record_audio:
    print(f"Using file_path: {file_path} to regenerate new sounds.")