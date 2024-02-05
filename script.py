#!/usr/bin/env python

import re
import os
import playsound
from openai import OpenAI
import argparse
import pydub
from tqdm import tqdm
import subprocess

from elevenlabs import set_api_key
from elevenlabs import generate, stream, save
from elevenlabs import Voice, VoiceSettings

def parse_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--script", type=str, default=None, help="The location of the script to generate/read")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="The prompt to generate the scene")
    parser.add_argument("-f", "--file_only", action="store_true", help="Whether the audio playback should be skipped")
    parser.add_argument("--gptversion", type=int, default=4, help="The version of GPT to use")
    return parser.parse_args()

def create_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_el_voice(VID: str, stability: float = 0.5, similarity: float = 0.75, style: float = 0.5):
    return Voice(
        voice_id=VID,
        settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            style=style,
            use_speaker_boost=True
        )
    )

def generate_el_stream(voice: Voice, text: str):
    return generate_el_audio_data(voice, text, True)

def generate_el_audio_data(voice: Voice, text: str, stream: bool = False):
    """Generate audio using ElevenLabs

    Args:
        voice (Voice): The voice model to use
        text (str): The text to read
        stream (bool, optional): Whether the content should be streamed or not. Defaults to False.

    Returns:
        bytes | Iterator[bytes]: The complete audio data or a data iterator for streams
    """    
    return generate(
        text=text,
        voice=voice,
        #model="eleven_turbo_v2",
        model="eleven_multilingual_v2",
        stream=stream
    )

def generate_dectalk_audio(voice: int, dialog: str, file: str):
    """Generate an audio file using DecTalk

    Args:
        voice (int): Which preset voice to use (0-9)
        dialog (str): The text to read out
        file (str): The location of the audio file
    """ 
    text = re.sub("[^A-Za-z0-9 -.,!?()]", "", dialog)
    folder = os.path.dirname(file)
    create_folder(folder)
    subprocess.run(["./dectalk/say","-e","1","-s",str(voice),"-a",text,"-fo",file])

def generate_oai_audio(voice: str, dialog: str):
    """Generates audio data from OpenAI

    Args:
        voice (str): The voice model name
        dialog (str): The text to read

    Returns:
        HttpxBinaryResponseContent: The audio data
    """
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in voices: raise Exception(f"Voice {voice} is not an existing model. Pick one of: {voices}")
    
    return client.audio.speech.create(
        model = "tts-1",
        voice = voice,
        input=dialog
    )

def save_list(l: list, file: str):
    """Save a list to a file where each element is a separate line

    Args:
        l (list): The list to save
        file (str): The location of the file
    """    
    with open(file, "w") as f:
        f.writelines(f"{line}\n" for line in l)

args = parse_args()

# Fill out full file path for script name only entries
if args.script:
    if "script.txt" not in args.script:
        args.script = f"generated/{args.script}/script.txt"

DEFAULT_VOICE = {
    "voice": {
        #"source": "EL",
        #"voice": get_voice("onwK4e9ZLuTAKqWW03F9", style=0) # Daniel
        "source": "DT",
        "voice": 2,
    }
}

characters = {
    "FERGUS": {
        "name": "Fergus",
        "description": "a posh, British, university student who likes to belittle people despite him being a midwit",
        "voice": {
            "source": "OAI",
            "voice": "fable",
        }
    },
    "DAN": {
        "name": "Daniel",
        "description": "a creepy man who only speaks in provocotive innuendos",
        "voice": {
            "source": "EL",
            "voice": get_el_voice("onwK4e9ZLuTAKqWW03F9", style=0)
        }
    },
    "MIMI": {
        "name": "Mimi",
        "description": "an overly positive girl that everyone hates because she sounds like a tiktok influencer. She always shouts in CAPITAL LETTERS and an exclamation point!",
        "voice": {
            "source": "EL",
            "voice": get_el_voice("zrHiDhphv9ZnVXBqCLjz", style=0)
        }
    },
    "HAWK": {
        "name": "Steven Hawking",
        "description": "a super intelligent, wheelchair-bound British man",
        "voice": {
            "source": "DT",
            "voice": 0
        }
    },
    "SANDRA": {
        "name": "Sandra",
        "description": "a weird girl that enjoys cereal to a disturbing degree",
        "voice": {
            #"source": "DT",
            #"voice": 1
            "source": "OAI",
            "voice": "nova"
        }
    },
}

sfx = {
    "KEYBOARD_TYPING",
    "GNOME",
    "METAL_PIPE_CRASH",
    "DOOR_CLOSE",
    "DOOR_OPEN",
    "GOOFY_CAR_HORN",
    "BAD_TO_THE_BONE_FUNNY",
    "FBI_OPEN_UP",
    "AMOGUS_MUSIC_BASS_BOOSTED",
    "AUTOMATIC_GUN_FIRE",
    "BIG_GLASS_SMASH",
    "GLASS_SHATTER",
    "DOOR_SMASH_BREAK",
    "MAGIC",
}

pattern_dialog = re.compile(r"\[(.+)\]:\s?\[?(.+)\]?")
pattern_sfx = re.compile(r"\[?SFX\]?:\s?\[?(.+)\]?")

with open('openai.key', 'r') as file:
    oai_key = file.read().rstrip()

with open('elevenlabs.key', 'r') as file:
    el_key = file.read().rstrip()

# set_api_key(el_key)

client = OpenAI(
    # This is the default and can be omitted
    api_key=oai_key,
)

def generate_character_strlist():
    global characters
    prompt = ""
    for i, (key, details) in enumerate(characters.items(), 1):
        prompt += f"{i}. {details["name"]} ({key}) {details["description"]}\n"
    return prompt

def generate_sfx_strlist():
    global sfx
    return ", ".join(sfx)

def strip_stage_directions(dialog: str) -> str:
    """Remove the left-most parenthesis from the dialog.
    This should remove most stage direction that makes it through.

    Args:
        dialog (str): The generated dialog

    Returns:
        str: The dialog without stage direction
    """    
    result = re.sub(r"\(.*\)", "", dialog)
    return re.sub(r"\*.*\*", "", result)

def generate_voice_audio(voice: dict, dialog: str, audio_file_name: str):
    """Generate & play voice audio

    Args:
        voice (dict): The voice data
        dialog (str): The text to read
        audio_file_name (str): The location of the file to save to
    """    
    if voice["source"] == "EL": # Generate using elevenlabs
        data = generate_el_audio_data(voice["voice"], dialog)
        save(data, audio_file_name)
    elif voice["source"] == "DT": # Generate using DecTalk
        generate_dectalk_audio(voice["voice"], dialog, audio_file_name)
    elif voice["source"] == "OAI": # Generate using OpenAI
        data = generate_oai_audio(voice["voice"], dialog)
        data.write_to_file(audio_file_name)
    
    if not args.file_only: playsound.playsound(audio_file_name)
        
def soundfx(sound: str):
    sound = re.sub(r"[^A-Z_]", "", sound)
    audio_file_name = f"sfx/{sound}.mp3"
    if not os.path.exists(audio_file_name): return # return if file not present

    if not args.file_only: playsound.playsound(audio_file_name)
    # Add the audio for contat
    audio_file_list.append(audio_file_name)

def export_complete_audio(audio_file_list: list, audio_file_name: str):
    """Concat all audio clips into a single clip and export to a file

    Args:
        audio_file_list (list): The list of audio file locations in order of play
        file (str): The location of the file to save the audio to
    """    
    combined_audio = pydub.AudioSegment.empty()

    for audio in tqdm(audio_file_list, total=len(audio_file_list), desc="Audio Compile", unit=" files"):
        file = pydub.AudioSegment.from_file(audio)
        combined_audio += file

    combined_audio.export(audio_file_name, format="mp3")

def generate_script(prompt: str, version: int = 4):
    character_prompt = generate_character_strlist()
    sfx_prompt = generate_sfx_strlist()
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You will write scripts involving some or all of the following characters along with their CHARACTER_NAME variables in brackets:\n{character_prompt}. You will write character dialogue based on the user's prompt. Expand upon the user's prompt by generating a more detailed scenario. The first line of the output will be a title for the script. Do NOT end the story on a happy note unless the user has explicitly asked for it. Do NOT include any stage direction. Every line of dialog for a character must contain speech, and NOT sounds.",
            }, # , however, you can also write a detailed description of the visuals that can be passed into an image generator so the audience can see what you're thinking of
            {
                "role": "system",
                "content": f"You may also use any of the following sound effects [{sfx_prompt}] by starting the line with the keyword 'SFX' without brackets. The name of the sound MUST match the ones found in the list. ONLY use sound effects where the story calls for them.",
            },
            {
                "role": "system",
                "content": "Character dialogue will be on a new line and formatted as such: [CHARACTER_NAME]:[CHARACTER_DIALOGUE] where the brackets are kept intact. The output will be parsed by another program so the format must remain the same.",
            }, # \nVisual descriptions will begin with the keyword 'VISUAL'. 
            {
                "role": "user",
                "content": prompt or "make up a random story",
            }
        ],
        model="gpt-4-turbo-preview" if version == 4 else "gpt-3.5-turbo",
    )

# Generate a new script or open an existing one
if args.script is None:
    user_prompt = args.prompt
    if args.prompt is None:
        user_prompt = input("Prompt: ")
    output = generate_script(user_prompt, args.gptversion)
    generated_output = output.choices[0].message.content
    generated_output = generated_output.split("\n")
else:
    with open(args.script, "r") as f:
        generated_output = [line.strip() for line in f]
    

title = generated_output[0].replace("Title: ", "")
folder_name = title

# Create save location & save script, prompt
if args.script is None:
    create_folder(f"generated/{folder_name}")
    save_list(generated_output, f"generated/{folder_name}/script.txt")
    save_list([user_prompt], f"generated/{folder_name}/prompt.txt")

# Remove title
generated_output = generated_output[1:]

print("="*35)
print(title)
print("="*35)

#print(generated_output)

audio_count = 0

audio_file_list = []

for line in generated_output:
    match_dialog = pattern_dialog.match(line)
    match_sfx = pattern_sfx.match(line)
    print(f"{audio_count:03} - {line}")
    if match_dialog:
        character, dialog = match_dialog.groups()
        dialog = strip_stage_directions(dialog) # Remove stage direction if present

        if character == "SFX":
            soundfx(dialog)
            continue

        voice = characters.get(character, DEFAULT_VOICE)["voice"]

        audio_file_name = f"generated/{folder_name}/{audio_count:03} - {character}"

        # Increase count
        audio_count += 1

        # Select correct file type
        if voice["source"] in ["EL", "OAI"]: # ElevenLabs
            audio_file_name += ".mp3"
        elif voice["source"] == "DT": # DecTalk
            audio_file_name += ".wav"

        # Add the audio for contat
        audio_file_list.append(audio_file_name)
        
        # Play / Generate
        if os.path.exists(audio_file_name): # Play existing audio file
            if not args.file_only: playsound.playsound(audio_file_name)
        else:
            generate_voice_audio(voice, dialog, audio_file_name)
        
    elif match_sfx:
        sound = match_sfx.groups()[0].upper()
        soundfx(sound)

export_name = f"generated/{folder_name}/{re.sub("[^A-Za-z0-9 ]", "", title)}.mp3"
export_complete_audio(audio_file_list, export_name)