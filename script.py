#!/usr/bin/env python

import re
import os
import subprocess
import glob
import argparse
from typing import Iterator, Union

from openai.types.chat import ChatCompletionChunk
from openai._streaming import Stream
import playsound
import pandas
from openai import OpenAI
from ollama import chat as ollamachat
from ollama import ChatResponse
import pydub
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import requests
# import gptrim, nltk # prompt compression
# import multiprocessing

from elevenlabs.client import ElevenLabs
from elevenlabs import save
from elevenlabs import Voice, VoiceSettings

# This requires a custom API
openvoice_url = "http://localhost:8000/generate"

TEXT_MODELS = {
    "gpt-4.1": {
        "use_alt_sys_prompt": False,
        "max_tokens": 32768,
        "provider": "OPENAI",
    },
    "gpt-4.1-mini": {
        "use_alt_sys_prompt": False,
        "max_tokens": 32768,
        "provider": "OPENAI",
    },
    "gpt-4.1-nano": {
        "use_alt_sys_prompt": False,
        "max_tokens": 32768,
        "provider": "OPENAI",
    },
    "gpt-4o": {
        "use_alt_sys_prompt": False,
        "max_tokens": 16384,
        "provider": "OPENAI",
    },
    "gpt-4o-mini": {
        "use_alt_sys_prompt": False,
        "max_tokens": 16384,
        "provider": "OPENAI",
    },
    "o4-mini": {
        "use_alt_sys_prompt": True,
        "max_tokens": None, # technically 100,000 tokens
        "provider": "OPENAI",
    },
    "gemma3:27b": {
        "use_alt_sys_prompt": False,
        "max_tokens": 16384,
        "provider": "OLLAMA",
    },
    "llama3.2": {
        "use_alt_sys_prompt": False,
        "max_tokens": 16384,
        "provider": "OLLAMA",
    },
    "llama2-uncensored": {
        "use_alt_sys_prompt": False,
        "max_tokens": 16384,
        "provider": "OLLAMA",
    },
    "deepseek-r1:7b": {
        "use_alt_sys_prompt": False,
        "max_tokens": 16384,
        "provider": "OLLAMA",
    },
    "deepseek-r1:14b": {
        "use_alt_sys_prompt": False,
        "max_tokens": 16384,
        "provider": "OLLAMA",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--script", type=str, default=None, help="The location of the script to generate/read")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="The prompt to generate the scene")
    parser.add_argument("--noplayback", action="store_true", help="Whether the audio playback should be skipped")
    parser.add_argument("--scriptonly", action="store_true", help="Whether the audio generation should be skipped")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", choices=list(TEXT_MODELS.keys()), help="The LLM to use")
    parser.add_argument("-c", "--compression", action="store_true", help="Enable instruction prompt compression")
    parser.add_argument("-i", "--dallever", type=int, default=3, choices=[2,3], help="Which version of DALL-E to use to generate visuals")
    parser.add_argument("--imagelimit", type=int, default=0, help="The limit of how many images to draw per script")
    parser.add_argument("--passes", type=int, default=1, help="How many times to prompt GPT (more passes makes a longer story)")
    parser.add_argument("-w", "--workers", type=int, default=2, help="How many worker threads to use when generating audio samples")
    parser.add_argument("--characters", type=str, nargs="+", required=True, help="A list of the characters you wish to use. Use LIST to see available characters.")
    parser.add_argument("--debug", action="store_true", help="Whether to print debug messages.")
    return parser.parse_args()

def debug(message: str):
    if args.debug:
        print(f"DEBUG: {message}")

def create_folder(folder_path: str):
    os.makedirs(folder_path, exist_ok=True)

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
    return el_client.generate(
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
    # TODO: Verify security of calling a command using direct text
    text = re.sub("[^A-Za-z0-9 -.,!?()]", "", dialog)
    folder = os.path.dirname(file)
    os.makedirs(folder, exist_ok=True)
    subprocess.run(["./dectalk/say","-e","1","-s",str(voice),"-a",text,"-fo",file], check=True)

def generate_oai_audio(voice: str, dialog: str):
    """Generates audio data from OpenAI

    Args:
        voice (str): The voice model name
        dialog (str): The text to read

    Returns:
        HttpxBinaryResponseContent: The audio data
    """
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "sage", "ash"]
    if voice not in voices:
        raise ValueError(f"Voice {voice} is not an existing model. Pick one of: {voices}")

    return client.audio.speech.create(
        model = "tts-1",
        voice = voice,
        input=dialog
    )

def generate_ov_audio(voice: str, dialog: str, file: str):
    response = requests.post(openvoice_url, json={
        "voice": voice,
        "text": dialog
    })
    if response.status_code == 200:
        with open(file, mode="wb") as f:
            f.write(response.content)
    else:
        print(f"Got {response.status_code} - Failed to generate: {voice} - {dialog}")

def trim_thinking(text: str) -> str:
    """Trims the thought process from the start of the text"""
    # if the text starts with <think>, then it should substring everything after </think>
    if text.startswith("<think>"):
        return text[(text.find("</think>")+len("</think>")):].strip()
    return text

def save_list(l: list, file: str):
    """Save a list to a file where each element is a separate line

    Args:
        l (list): The list to save
        file (str): The location of the file
    """
    with open(file, "w", encoding="utf-8") as f:
        f.writelines(f"{line}\n" for line in l)

def get_output_folder():
    return f"generated/{folder_name}"

def get_voice_audio_file_name(audio_count: int, character: str) -> str:
    audio_file_name = f"{get_output_folder()}/{audio_count:03} - {character}"
    voice = active_characters.get(character, DEFAULT_VOICE)["voice"]
    # Select correct file type
    if voice["source"] in ["EL", "OAI"]: # ElevenLabs
        audio_file_name += ".mp3"
    elif voice["source"] in ["DT", "OV"]: # DecTalk
        audio_file_name += ".wav"
    return audio_file_name

def process_line_to_generate(line: dict):
    """Generates the media file required for the line of the script

    Args:
        line (dict): The script line data including character, dialog, and file path for audio
    """
    file_name = line["file"]
    content = line["content"]
    character = line["character"]

    if len(content) < 1:
        raise ValueError(f"Content is blank for ``{file_name}``!")

    prompt = "A realistic image of "
    prompt += content

    if character == "[VISUAL]":
        if not os.path.exists(file_name):
            generate_and_save_image(prompt, file_name, args.dallever)
    elif character != "[SFX]": # do not attempt to generate sound effect
        voice = active_characters.get(character, DEFAULT_VOICE)["voice"]
        # Generate if file does not already exist
        if not os.path.exists(file_name):
            try:
                generate_voice_audio(voice, content, file_name)
            except Exception as e:
                print("Error generating audio:", e)

def _sfx_data(sfx_name: str) -> dict | None:
    output = None
    audio_file_name = soundfx(sfx_name)
    if audio_file_name:
        output = {
            "character": "[SFX]",
            "content": sfx_name,
            "file": audio_file_name
        }
    return output

def preprocess_output_lines(lines: list[str]) -> dict:
    """Read script lines and format each line into a dict for further processing

    Args:
        lines (str): The script lines to read through

    Returns:
        dict: The data for further processing
    """
    output = {}
    line_count = 0 # audio line count
    voice_count = 0 # voice line count
    img_count = 0 # image line count
    for line in tqdm(lines, total=len(lines), unit=" lines"):
        # Find matches
        match_dialog = pattern_dialog.match(line)
        match_sfx = pattern_sfx.match(line)
        match_img = pattern_img.match(line)

        if match_sfx: # Add sound effect
            sound = match_sfx.groups()[0].upper()
            data = _sfx_data(sound)
            if data:
                output[line_count] = data
                line_count += 1
        elif match_img: # Add image
            if img_count >= args.imagelimit:
                continue
            img_prompt = match_img.groups()[0]
            img_prompt = img_prompt.replace("]","")
            output[line_count] = {
                "character": "[VISUAL]",
                "content": img_prompt,
                "file": f"{get_output_folder()}/{img_count:02d}.jpg"
            }
            line_count += 1
            img_count += 1
        elif match_dialog: # Add dialog
            # TODO: Match case where dialog has notes either side in brackets
            character, dialog = match_dialog.groups()
            dialog = strip_stage_directions(dialog) # Remove stage direction if present
            dialog = dialog.strip() # strip whitespace
            character = character.upper()

            audio_file_name = get_voice_audio_file_name(voice_count, character)

            output[line_count] = {
                "character": character,
                "content": dialog,
                "file": audio_file_name
            }

            line_count += 1
            voice_count += 1

    return output

def print_standout(text: str):
    print("="*35)
    print(text)
    print("="*35)

def perform_script(script: dict):
    print_standout(title)
    for audio_count, data in sorted(script.items()):
        line = data.get("content", data.get("img_path"))
        # Get character
        character = data["character"]
        char_name = character
        # Resolve actual character name
        if character not in ["[SFX]", "[VISUAL]"]:
            char = active_characters.get(character, DEFAULT_VOICE)
            if char != DEFAULT_VOICE["name"]:
                char_name = char["name"]

        audio_file_name = data["file"]
        print(f"{audio_count:03} - {char_name}: {line}")
        if os.path.exists(audio_file_name): # Play existing audio file
            playsound.playsound(audio_file_name)

def generate_character_strlist():
    prompt = ""
    for i, (key, details) in enumerate(active_characters.items(), 1):
        prompt += f"{i}. {details["name"]} ({key}) {details["description"]}\n"
    return prompt

def generate_sfx_strlist():
    return ",".join(sfx)

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
    """Generate voice audio

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
    elif voice["source"] == "OV":
        generate_ov_audio(voice["voice"], dialog, audio_file_name)
        
def soundfx(sound: str) -> str | None:
    """Locate a sound effect

    Args:
        sound (str): The name of the sound to play

    Returns:
        str | None: The file path, or None if the file was not found
    """
    sound = re.sub(r"[^A-Z_]", "", sound)
    audio_file_name = f"sfx/{sound}.mp3"
    if not os.path.exists(audio_file_name):
        return None # return if file not present
    return audio_file_name

def compress_prompt(text: str) -> str:
    """Compress a text prompt
    Also prints out a summary of how many tokens it should now be

    Args:
        text (str): The text to compress

    Returns:
        str: The compressed text
    """
    trimmed = gptrim.trim(text)
    t=nltk.word_tokenize(text)
    tr=nltk.word_tokenize(trimmed)
    print("Compressed from", len(t), "->", len(tr))
    return trimmed

def export_complete_audio(script: dict, folder_name: str, title: str):
    """Concat all audio clips into a single clip and export to a file

    Args:
        audio_file_list (list): The list of audio file locations in order of play
        file (str): The location of the file to save the audio to
    """
    audio_file_list = [data["file"] for _,data in sorted(script.items()) if ".jpg" not in data["file"]]
    # print("\n".join(audio_file_list))
    export_name = f"generated/{folder_name}/{re.sub('[^A-Za-z0-9 ]', '', title)}.mp3"
    combined_audio = pydub.AudioSegment.empty()

    for audio in tqdm(audio_file_list, total=len(audio_file_list), desc="Audio Compile", unit=" files"):
        file = pydub.AudioSegment.from_file(audio)
        combined_audio += file

    combined_audio.export(export_name, format="mp3")

def generate_image(prompt: str, model: int = 2) -> str:
    """Generates an image using DALL-E and returns the image URL

    Args:
        prompt (str): The description of image to generate
        model (int, optional): The DALL-E model to use. Defaults to 2.

    Returns:
        str: The URL of the generated image
    """
    if model not in [2,3]: model = 2
    response = client.images.generate(
        model=f"dall-e-{model}",
        prompt=prompt,
        size="256x256" if model==2 else "1024x1024",
        n=1
    )
    # revised_prompt = response.data[0].revised_prompt
    return response.data[0].url

def download_image(url: str, img_path: str):
    """Download an image to a location

    Args:
        url (str): The image URL
        img_path (str): The path to save the image to
    """
    img_data = requests.get(url).content
    with open(img_path, "wb") as f:
        f.write(img_data)

def generate_and_save_image(prompt: str, img_path: str, model: int = 2):
    url = generate_image(prompt, model)
    download_image(url, img_path)

def dialog_regex_pattern() -> str:
    """A pattern to match characters encapulated in square brackets, or a known character name that isn't in square brackets

    Returns:
        str: The regex pattern
    """
    chars = "|".join(active_characters.keys())
    return fr"(?:(?:\[(.+)(?:,.*)?\])|({chars})):\s+?\[?(.*)\]?"

def construct_gpt_prompt(
    init_prompt: str,
    compress: bool = False,
    use_images: bool = True,
    use_alt_sys: bool = False,
    ) -> list:
    character_prompt = generate_character_strlist()
    sfx_prompt = generate_sfx_strlist()

    instruction_text = f"""
        You will write a long script with narrative flow.
        Include some or all of the following characters along with their CHARACTER_NAME variables in brackets:\n{character_prompt}.
        The characters' dialogue should sound like what the character would actually say.
        The dialogue should sound natural.
        If the character shares a name with a character from pop culture, assume that's what's intended.
        You will write character dialogue based on the user's prompt.
        Expand upon the user's prompt by generating a more detailed scenario.
        ONLY the first line of the output will be the title for the script.
        Do NOT end the story on a happy note unless the user has explicitly asked for it.
        Do NOT include any stage direction or action.
        If you receive a script, continue it where it leaves off.
    """
    # , however, you can also write a detailed description of the visuals that can be passed into an image generator so the audience can see what you're thinking of

    format_text = "Each character's dialogue will be on a new line and formatted as a key value pair on the same line and deliminated by a colon: [${CHARACTER_NAME}]:${CHARACTER_DIALOGUE}. The [CHARACTER_NAME] MUST be encapulated by square brackets and be an EXACT match to one previously listed."

    sfx_text = f"You may also use sound effects by using a key value pair delimited by a colon: SFX:${{SOUND_NAME}}. 'SFX' is a keyword whereas SOUND_NAME is replaced with an EXACT match to any from this list: [{sfx_prompt}]. ONLY use sound effects where the story calls for them."

    img_text = "You may also generate an image during the script by using a key value pair delimited by a colon: VISUAL:${IMG_PROMPT}. 'VISUAL' is a keyword, whereas IMG_PROMPT is to be replaced by a detailed but PG13 description of the scene. Each IMG_PROMPT is STATELESS and MUST describe the scene without relying on other context, meaning any characters used in the IMG_PROMPT MUST have a VISUAL description INSTEAD of their name. The visual is NOT a replacement for dialog - it MUST serve ONLY as an additional aid."

    if compress:
        # Disabled instruction compression for now until stability to verified
        # instruction_text = compress_prompt(instruction_text)
        sfx_text = compress_prompt(sfx_text)
        format_text = compress_prompt(format_text)
        # if use_images: img_text = compress_prompt(img_text)

    messages = []

    # models like o1 cannot use system roles.
    sys_role = "user" if use_alt_sys else "system"

    messages.append({ "role": sys_role, "content": instruction_text, })
    messages.append({ "role": sys_role, "content": format_text, })
    if len(sfx) > 0: # Only include sounds if the sound list is not empty
        messages.append({ "role": sys_role, "content": sfx_text, })
    if use_images:
        messages.append({ "role": sys_role, "content": img_text, })
    messages.append({ "role": "user", "content": init_prompt or "make up a random story", })

    return messages

def generate_script(
        messages: list,
        model: str = "gpt-4o-mini"
    ) -> Union[Stream[ChatCompletionChunk], Iterator[ChatResponse]]:
    provider = TEXT_MODELS[model]["provider"]
    max_tokens = TEXT_MODELS[model]["max_tokens"]

    if provider == "OPENAI":
        return client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            max_completion_tokens=max_tokens,
        )
    elif provider == "OLLAMA":
        return ollamachat(
            model=model,
            messages=messages,
            stream=True,
        )
    else:
        raise ValueError(f"Unknown model provider: {model} ({provider})")


def get_character_files() -> list[str]:
    folder_path = os.path.dirname(__file__)
    pattern = os.path.join(folder_path, 'characters*.csv')
    return glob.glob(pattern)

def get_character_dict(char_files: list[str]) -> dict:
    frames = []
    for path in char_files:
        df = pandas.read_csv(path)
        frames.append(df)
    return parse_character_file(pandas.concat(frames, ignore_index=True))

def parse_character_file(df: pandas.DataFrame) -> dict:
    result = {}
    # Fill blank spaces with default values
    df.fillna({"voice_style": 0.1}, inplace=True)
    df.fillna({"voice_similarity": 0.75}, inplace=True)
    df.fillna({"voice_stability": 0.5}, inplace=True)

    for _, row in df.iterrows():
        voice_id = str(row["voice_id"])
        voice_source = str(row["voice_source"])
        # ElevenLabs requires different processing
        if voice_source == "EL":
            style = float(row["voice_style"])
            similarity = float(row["voice_similarity"])
            stability = float(row["voice_stability"])
            voice_id = get_el_voice(voice_id, stability, similarity, style)

        # store character
        result[row["character_identifier"]] = {
            "name": row["displayed_name"],
            "description": row["description"],
            "voice": {
                "source": voice_source,
                "voice": voice_id,
            }
        }
    return result

def get_active_characters(characters: dict, selected: list) -> dict:
    result = {}
    for char, value in characters.items():
        if char in selected:
            result[char] = value
            selected.remove(char)

    if len(selected) > 0:
        print("\nFollowing characters do not exist: [", ", ".join(selected), "]")

    return result

args = parse_args()

# Fill out full file path for script name only entries
if args.script:
    if "script.txt" not in args.script:
        args.script = f"generated/{args.script}/script.txt"

DEFAULT_VOICE = {
    "name": "DEFAULT",
    "voice": {
        #"source": "EL",
        #"voice": get_voice("onwK4e9ZLuTAKqWW03F9", style=0) # Daniel
        "source": "DT",
        "voice": 2,
    }
}

all_characters = get_character_dict(get_character_files())

if args.characters[0] == "LIST":
    print("\nUseable characters: [")
    for char, value in all_characters.items():
        print(f"\t{char} - {value['description']}")
    print("]")
    exit(0)

active_characters = get_active_characters(all_characters, args.characters)

if len(active_characters) == 0:
    raise ValueError("No characters have been selected!")
else:
    print("\nUsing following characters: [", ", ".join(active_characters), "]")

sfx = [
    # "DOOR_CLOSE",
    # "DOOR_OPEN",
    # "DOOR_SMASH_BREAK",
    # "AUTOMATIC_GUN_FIRE",
    "PISTOL_SINGLE_FIRE",
    # "BIG_GLASS_SMASH",
    "GLASS_SHATTER",
    # "MAGIC_SPARKLES",
    # "GOOFY_CAR_HORN",
    # "BAD_TO_THE_BONE_FUNNY",
    # "FBI_OPEN_UP",
    # "AMOGUS_MUSIC_BASS_BOOSTED",
    # "FURIOUS_KEYBOARD_TYPING",
    "GNOME",
    "METAL_PIPE_CRASH",
    # "SCREAM_GET_OUT",
    # "SPINNING_CAT",
    # "WHAT_THE_HELL",
]

pattern_dialog = re.compile(r"\[?(.+?)\]?:\s*\[?(.*)\]?")
pattern_sfx = re.compile(r"\[?SFX\]?:\s?\[?(.+)\]?")
pattern_img = re.compile(r"\[?VISUAL\]?:\s?\[?(.+)\]?")

with open('openai.key', 'r', encoding='utf-8') as file:
    oai_key = file.read().rstrip()

with open('elevenlabs.key', 'r', encoding='utf-8') as file:
    el_key = file.read().rstrip()


el_client = ElevenLabs(
    api_key=el_key
)

client = OpenAI(
    # This is the default and can be omitted
    api_key=oai_key,
)

# Generate a new script or open an existing one
if args.script is None:
    user_prompt = args.prompt
    if args.prompt is None:
        user_prompt = input("Prompt: ")

    print("Generating script...")

    model_in_use = TEXT_MODELS[args.model]

    # models like o1 cannot use the system prompt
    use_alt_sys_prompt = model_in_use["use_alt_sys_prompt"]

    messages = construct_gpt_prompt(
        init_prompt=user_prompt,
        compress=args.compression,
        use_images=args.imagelimit > 0,
        use_alt_sys=use_alt_sys_prompt,
    )

    generated_output = ""

    for _ in tqdm(range(args.passes), desc="passes", unit=" pass"):
        output_stream = generate_script(messages, args.model)
        go = ""
        pbar = tqdm(unit=" chunks", desc=f"Generating using {args.model}")
        for chunk in output_stream:

            if model_in_use["provider"] == "OPENAI":
                content = chunk.choices[0].delta.content
            elif model_in_use["provider"] == "OLLAMA":
                content = chunk.message.content
            else:
                continue

            if content is not None:
                go += content
                pbar.set_postfix_str(f"generated {len(go.split("\n"))} lines")
                pbar.update()
        pbar.close()
        messages.append({ "role": "assistant", "content": go, })
        messages.append({ "role": "user", "content": "continue the script", }) # TODO: Further test the continuation prompt. Add compression?
        generated_output += f"{go}\n"
    # generated_output = output.choices[0].message.content
    debug("====TEXT OUTPUT====\n")
    debug(generated_output)
    debug("\n===================\n")
    generated_output = trim_thinking(generated_output) # Remove thought process
    generated_output = generated_output.split("\n")
else:
    with open(args.script, "r", encoding="utf-8") as f:
        generated_output = [line.strip() for line in f]


title = generated_output[0].split(":")[-1].strip()
title = re.sub(r"[^A-Za-z0-9 \-:]", "", title)
folder_name = title

print_standout(title)

# Create save location & save script, prompt
if args.script is None:
    script_folder = os.path.join("generated", folder_name)
    debug("Saving script")
    os.makedirs(script_folder, exist_ok=True)
    save_list(generated_output, os.path.join(script_folder, "script.txt"))
    save_list([user_prompt], os.path.join(script_folder, "prompt.txt"))

# Stop before audio generation
if args.scriptonly:
    exit(0)

# Remove title
generated_output = generated_output[1:]

print("Processing script...")
script = preprocess_output_lines(generated_output)

print("Generating media...")
# Generate media in parallel for speed
process_map(process_line_to_generate, script.values(), max_workers=args.workers, unit=" files")

print("Exporting compiled file...")
export_complete_audio(script, folder_name, title)

if not args.noplayback:
    print("Performing...")
    perform_script(script)
