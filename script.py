#!/usr/bin/env python

import re
import os
import playsound
from openai import OpenAI
import argparse
import pydub
from tqdm import tqdm

from elevenlabs import set_api_key
from elevenlabs import generate, stream, save
from elevenlabs import Voice, VoiceSettings

def parse_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--script", type=str, default=None, help="The location of the script to generate/read")
    parser.add_argument("-f", "--file_only", action="store_true", help="Whether the audio playback should be skipped")
    return parser.parse_args()

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_voice(VID: str, stability: float = 0.5, similarity: float = 0.75, style: float = 0.5):
    return Voice(
        voice_id=VID,
        settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            style=style,
            use_speaker_boost=True
        )
    )

def generate_stream(voice: Voice, text: str):
    return generate_audio_data(voice, text, True)

def generate_audio_data(voice: Voice, text: str, stream: bool = False):
    return generate(
        text=text,
        voice=voice,
        #model="eleven_turbo_v2",
        model="eleven_multilingual_v2",
        stream=stream
    )

def save_list(l: list, file: str):
    with open(file, "w") as f:
        f.writelines(f"{line}\n" for line in l)

args = parse_args()

DEFAULT_VOICE = {"voice":get_voice("onwK4e9ZLuTAKqWW03F9", style=0)} # Daniel

characters = {
    "MIMI": {
        "name": "Mimi",
        "description": "an overly positive girl that everyone hates because she sounds like a tiktok influencer. She always shouts in CAPITAL LETTERS and an exclamation point!",
        "voice": get_voice("zrHiDhphv9ZnVXBqCLjz", style=0)
    }
}

sfx = {
    "KEYBOARD_TYPING",
    "GNOME",
    "METAL_PIPE_CRASH",
    "DOOR_CLOSE",
    "GOOFY_CAR_HORN",
    "BAD_TO_THE_BONE_FUNNY",
    "FBI_OPEN_UP",
    "AMOGUS",
    "AUTOMATIC_GUN_FIRE",
    "BIG_GLASS_SMASH",
    "GLASS_SHATTER",
    "DOOR_SMASH_BREAK",
}

pattern_dialog = re.compile(r"\[(.+)\]:\s?\[?(.+)\]?")
pattern_sfx = re.compile(r"\[?SFX\]?:\s?\[?(.+)\]?")

with open('openai.key', 'r') as file:
    oai_key = file.read().rstrip()

with open('elevenlabs.key', 'r') as file:
    el_key = file.read().rstrip()

set_api_key(el_key)

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

def generate_script(prompt: str, version: int = 4):
    character_prompt = generate_character_strlist()
    sfx_prompt = generate_sfx_strlist()
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You will write scripts involving some or all of the following characters along with their CHARACTER_NAME variables in brackets:\n{character_prompt}. You will write character dialogue based on the user's prompt. Expand upon the user's prompt by generating a more detailed scenario. The first line of the output will be a title for the script. Do NOT end the story on a happy note unless the user has explicitly asked for it. Do NOT include any stage direction.",
            }, # , however, you can also write a detailed description of the visuals that can be passed into an image generator so the audience can see what you're thinking of
            {
                "role": "system",
                "content": f"You may also use any of the following sound effects [{sfx_prompt}] by starting the line with the keyword 'SFX' without brackets. ONLY use sound effects where the story calls for them.",
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
    user_prompt = input("Prompt: ")
    output = generate_script(user_prompt)
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

def soundfx(sound: str):
    sound = re.sub(r"[^A-Z_]", "", sound)
    try:
        audio_file_name = f"sfx/{sound}.mp3"
        if not args.file_only: playsound.playsound(audio_file_name)
        # Add the audio for contat
        audio_file_list.append(audio_file_name)
    except:
        pass

for line in generated_output:
    match_dialog = pattern_dialog.match(line)
    match_sfx = pattern_sfx.match(line)
    print(f"{audio_count:03} - {line}")
    if match_dialog:
        character, dialog = match_dialog.groups()

        if character == "SFX":
            soundfx(dialog)
            continue

        voice = characters.get(character, DEFAULT_VOICE)["voice"]

        audio_file_name = f"generated/{folder_name}/{audio_count:03} - {character}.mp3"

        # Play existing audio file
        if os.path.exists(audio_file_name):
            if not args.file_only: playsound.playsound(audio_file_name)
        else:
            data = stream(generate_stream(voice, dialog))
            save(data, audio_file_name)

        # Add the audio for contat
        audio_file_list.append(audio_file_name)

        audio_count += 1
    elif match_sfx:
        sound = match_sfx.groups()[0].upper()
        soundfx(sound)


# Export complete audio file
combined_audio = pydub.AudioSegment.empty()

for audio in tqdm(audio_file_list, total=len(audio_file_list), desc="Audio Compile", unit=" files"):
    file = pydub.AudioSegment.from_mp3(audio)
    combined_audio += file

combined_audio.export(f"generated/{folder_name}/{title}.mp3", format="mp3")