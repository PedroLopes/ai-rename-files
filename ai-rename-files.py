#!/usr/bin/env python3

#---------------------------
# Credits 
# This code was written by Pedro Lopes
#
## The .txt, .docx and .PDF searches were written from scratch. 
## The image side (searches jpegs and pushed them to a VLM) was adapted of https://raw.githubusercontent.com/Tedfulk/ollama-rename-img/refs/heads/main/ollama_rename_img/main.py
# Differences to original
## 1. Support for different models and custom prompts
##    1.1. You can specify the name of the model you want to use with your ollama: -m <name_of_model>
##    1.2. You can append extra information to the promp by adding: -p <your_extra_prompt>
##    1.3. Supports passing photo metadata to prompt by adding: -e or -et (suppports GPS but requires exiftool)
##    1.4. Supports resetting the conversation with AI model (by default, bypass with -k)
##    1.5. Supports passing the directory name to the prompt using -d
##    1.6. Supports passing the file's timestamp (from OS) to the prompt using -t
## 2. Dependencies
##    2.1. This version does not require the 'poetry' program to manage depedencies
##    2.2. Dependencies can be managed/install via the 'pip' program using the 'requirements.txt' file
## 3. Minor
##    3.1. File count disregards non jpeg/jpg files better
##    TODO: add parallelism if needed
## More license information at Github repository
#---------------------------

# User settings (e.g., alter the prompt, etc)

## Change prompt
##   Note that {number} (of words) will be replaced later with --number (-n) argument if supplied
image_original_prompt = ("Describe the image in {number} simple keywords, never use more than {number} words. ")
text_original_prompt = ("""Describe the contents of this text file in {number} simple keywords, never use more than {number} words. Now let me paste a preview of the file contents: 
---------------------------""")

## Prompt format
##   Note that changing the prompt_output_format below is likely to cause errors (or requires code change)
prompt_output_format = ("Output in JSON format. "
                        "Use the following schema: { keywords: List[str] }.")

## Reset prompt
reset_prompt = "Reset conversation context."

## Select which image metadata to pass to prompt (note: more will contribute to a longer prompt)
##   Note: see EXIF or exiftool for all the available labels
metadata_filter = ["Date/Time Original", "Flash", "Make", "Camera Model Name", "Orientation", "GPS Position"]

## If you are using exiftool (--metadata) as the external program to parse metadata, indicate its path
exifToolPath = 'exiftool' #change here if this is not alias by the shell as exiftool

## Allowable extensions
valid_extensions_processed_as_text = {".pdf", ".docx", ".txt", ".md"}
valid_extensions_processed_as_image = {".jpeg", ".jpg"}

# ------------------------ program starts here ------------------
# -------------
# built-in imports
# -------------
import argparse
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import List
from datetime import datetime

# -------------
# additional critical imports
# -------------
import ollama
from pydantic import BaseModel, Field
from tqdm import tqdm
#PIL, geopy, pandas, etc are imported dynamically when you call with --metadata-python (-mp) to obtain image metadata
#subprocess will be loaded dynamically when you call with --metadata to call exiftool

# -----------------------------
# Renamer helper
# -----------------------------

def keywords_to_string_with_delimiter(keywords: list, args: list) -> str:
    if args.delimiter not in ["_", "-", " "]: #this is late to do this check, weird
        raise ValueError("Delimiter must be underscore '_', dash '-', or space ' '")
    cleaned_keywords = [] 
    for keyword in keywords:
        if keyword.find(" "):
            new = ''.join(word[0].upper() + word[1:].lower() for word in keyword.split())
        cleaned_keywords.append(new)
    return args.delimiter.join(cleaned_keywords[:args.number])

# -----------------------------
# Logging
# -----------------------------

logger = logging.getLogger(__name__)

def configure_logging(verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(levelname)s: %(message)s",
    )

# -----------------------------
# AI interaction
# -----------------------------

def prompt_setup(prompt: str, args: list) -> str:
    # perform substitution of --number (-n) argument into {number}
    prompt = re.sub(r'{number}', str(args.number), prompt)
    if args.prompt: 
        prompt = args.prompt + " " + prompt #when --prompt (-p) is present, prepend a new prompt
    elif args.override:
        prompt = args.override #when --override (-o) is present, replace the entire prompt
    return prompt

def prompt_closure_format(image_path: Path, args: list) -> str:
    prompt = ""
    if args.directory_name:
        prompt += "Additionally, consider also that this image is saved in a directory named " + str(image_path.parent) + ". "
    if args.timestamp: #passes timestamp info to prompt
        logger.info("Using timestamp")
        modification_datetime = datetime.fromtimestamp(os.path.getmtime(image_path))
        formatted_date = modification_datetime.strftime('%Y-%m-%d')
        prompt += "Also, consider that this image was created at " + str(formatted_date) + ". " 
    # add the format to the prompt
    prompt += prompt_output_format
    return prompt

def generate_text_keywords(text_path: Path, args: list) -> dict:
    global text_original_prompt

    with text_path.open("r") as text_file:
        content = text_file.read(args.readsize)
 
    prompt = prompt_setup(text_original_prompt, args)
    prompt = f"{content} " + """ (end of file preview) 
    -------------------------------------- 
    """
    prompt += prompt_closure_format(text_path, args)
   
    # Display the prompt to users
    logger.info(f"Using prompt: {prompt}")
 
    return ollama.chat(
        model=args.model_text,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

def generate_image_keywords(image_path: Path, args: list) -> dict:
    global image_original_prompt

    with image_path.open("rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
 
    prompt = prompt_setup(image_original_prompt, args)

    # ---------- metadata from images ----------- 
    if args.metadata or args.metadata_python: #if user requested metadata added to prompt in either way (-e or -et) 
        metadata_text = []
        location = ""
        if args.metadata: # metadata will be parsed externally using exiftool (--metadata)
            logger.info("Exiftool mode for metadata")
            import subprocess #subprocess dynamicaly loaded only for exiftool
            # modified from a snippet by Vaibhav K, from stackoverflow
            infoDict = {} #Creating the dict to get the metadata tags
            process = subprocess.Popen([exifToolPath,image_path],stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True) 
            for tag in process.stdout:
                line = tag.strip().split(':')
                if line[0].strip() in metadata_filter: 
                    compare = line[0].strip()
                    if compare == "GPS Position":
                        import pandas as pd # we are doing this dynamically as it currently is only needed for GPS
                        from lat_lon_parser import parse #only loaded for GPS
                        gps = line[-1].strip()
                        a = gps.split(",")
                        coord1 = parse(a[0])
                        coord2 = parse(a[1])
                        logger.info(f"Coordinate 1: {coord1}")
                        logger.info(f"Coordinate 2: {coord2}")
                        from geopy.geocoders import Nominatim #only loaded for GPS
                        geolocator = Nominatim(user_agent="Image tagger")
                        location = geolocator.reverse(f"{coord1}, {coord2}")
                        logger.info(location.address)
                        location = location.address
                    else: 
                        logger.info("added metadata" + str(line[0].strip()) + ':' + str(line[-1].strip()) +  "; ")
                        metadata_text.append(str(line[0].strip()) + ':' + str(line[-1].strip()) +  "; ")
        else: # metadata will be parsed internally using python libraries, which will be now loaded
            from PIL import Image
            from PIL.ExifTags import TAGS
            with Image.open(image_path) as img:
                exifdata = img.getexif()
                for tag_id in exifdata:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)
                    if isinstance(data, bytes):
                        data = data.decode()
                    metadata_text.append(f"{tag}: {data};") 
        if metadata_text:
            prompt += "You might find clues in the image's metadata (which is listed next using a colon separated list): " + str(metadata_text) + ". "          
            if location: 
                prompt += "Also, this image was taken in the following location, use this for clues as well: " + location + " ."
        else: 
            logger.info(f"{image_path}: metadata was empty")

    prompt += prompt_closure_format(image_path, args)
   
    # Display the prompt to users
    logger.info(f"Using prompt: {prompt}")
 
    return ollama.chat(
        model=args.model_image,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [base64_string],
            }
        ],
    )


# ------------------------------
# Convert non-text files to text
# ------------------------------



def convertDOCX(file: Path, args: list) -> Path:
    from docx import Document
    if not file.exists():
        raise FileNotFoundError(file)

    if file.suffix.lower() != ".docx":
        raise ValueError("Input file must be a .docx")

    doc = Document(str(file))

    extracted_text = []
    total_chars = 0

    for para in doc.paragraphs:
        if total_chars >= args.readsize:
            break

        text = para.text + "\n"
        remaining = args.readsize - total_chars
        extracted_text.append(text[:remaining])
        total_chars += len(text)

    output_path = file.with_suffix(".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(extracted_text))

    return output_path

def convertPDF(file: Path, args: list) -> Path:
    from PyPDF2 import PdfReader
    if not file.exists():
        raise FileNotFoundError(file)

    if file.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF")

    reader = PdfReader(str(file))

    extracted_text = []
    total_chars = 0

    for page in reader.pages:
        if total_chars >= args.readsize:
            break

        text = page.extract_text() or ""
        remaining = args.readsize - total_chars
        extracted_text.append(text[:remaining])
        total_chars += len(text)

    output_path = file.with_suffix(".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(extracted_text))

    return output_path


# ------------------------------


# -----------------------------
# Process images
# -----------------------------

def process_file(file: Path, args: list):

    try:
        if file.name.lower().endswith('.jpeg') or file.name.lower().endswith('.jpg'):
            response = generate_image_keywords(file, args)
        elif file.name.lower().endswith('.txt') or file.name.lower().endswith('.md'):
            response = generate_text_keywords(file, args)
        elif file.name.lower().endswith('.pdf'):
            new_file = convertPDF(file, args)
            response = generate_text_keywords(new_file, args)
            new_file.unlink(missing_ok=True)
        elif file.name.lower().endswith('.docx'):
            new_file = convertDOCX(file, args)
            response = generate_text_keywords(new_file, args)
            new_file.unlink(missing_ok=True)
        else: 
            # print("NOT a allowable file")
            return 
        logger.info(f"Response: {response}")
        content = (
            response["message"]["content"]
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
         
        keywords = json.loads(content)
        values=[]
        for key, value in keywords.items():
            values.append(value)
            #print(key, value)
        new_name = keywords_to_string_with_delimiter(values[0], args)

        #if timestamp is to be prefixed
        if args.prefix_timestamp or args.postfix_timestamp: #prefix the timestamp as YYYY-MM-DD
            modification_datetime = datetime.fromtimestamp(os.path.getmtime(file))
            formatted_date = modification_datetime.strftime(f'%Y{args.delimiter}%m{args.delimiter}%d')
            #new_name = str(formatted_date) + str(new_name)
            new_name = (formatted_date + args.delimiter + new_name) if args.prefix_timestamp else (new_name + args.delimiter + formatted_date)

        if args.prefix:
            new_name = args.prefix.join(args.prefix.split()) + args.delimiter + new_name
        if args.postfix:
            new_name = new_name + args.delimiter + args.postfix.join(args.postfix.split())

        # adding back the directory path        
        new_path = args.directory / f"{new_name}{file.suffix}"

        # renaming the file for real
        file.rename(new_path)
        logger.info(f"Renamed {file.name} → {new_path.name}")

    except Exception as e:
        logger.error(f"Error processing {file}: {e}")

# -----------------------------
# CLI arguments
# -----------------------------

def main():
    global valid_extensions_processed_as_text
    global valid_extensions_processed_as_image
    parser = argparse.ArgumentParser(
        description="Rename files based on their content using AI-generated keywords."
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the files",
    )

    parser.add_argument(
        "--delimiter",
        "-d",
        default="-",
        choices=["_", "-", " "],
        help="Delimiter for keywords in filename (default: -)",
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Add extra requests to the existing prompt (see code for existing prompt)",
    )

    parser.add_argument(
        "--override",
        "-o",
        type=str,
        help="Override the entire prompt (i.e., no other prompt but the text passed as argument)",
    )

    parser.add_argument(
        "--model-image",
        "-vlm",
        type=str,
        default="llava-phi3",
        help="Specify name of model that you want to use with ollama (default = llava-phi3)",
    )

    parser.add_argument(
        "--model-text",
        "-llm",
        type=str,
        default="qwen2.5-coder:7b",
        help="Specify name of model that you want to use with ollama (default = qwen2.5-coder:7b)",
    )


    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=3,
        help="Specify the number of keywords to be generated when describing and image",
    )

    parser.add_argument(
        "--readsize",
        "-r",
        type=int,
        default=100,
        help="Specify the number of characters to read from textual files (e.g., txt, pdf, etc)—longer sizes will create longer prompts",
    )



    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    parser.add_argument(
        "--directory-name",
        "-dir",
        action="store_true",
        default=False,
        help="Passes the directory name to the prompt, can be helpful to rename contextually using the directory name as a clue",
    )

    parser.add_argument(
        "--prefix",
        "-pre",
        type=str,
        default="",
        help="Adds a prefix to every renamed file, e.g., --prefix Japan, will add Japan followed by the delimiter (or default delimiter if not provided) before every final filename. (Tip: if you want to prefix a timestamp you can use --prefix-timestamp or --prefix $(date +%%d-%%m-%%Y) to format as you wish)",
    )

    parser.add_argument(
        "--prefix-timestamp",
        "-pretime",
        action="store_true",
        default=False,
        help="Adds a prefix timestamp (YYYY-MM-DD) to every renamed file. (Tip: if you want a custom date format that differs from this one, you can use  --prefix $(date +%%d-%%m-%%Y) to format as you wish)"
    )

    parser.add_argument(
        "--postfix",
        "-post",
        type=str,
        default="",
        help="Adds a postfix to every renamed file, e.g., --postfix Japan, will add Japan followed by the delimiter (or default delimiter if not provided) after every final filename. (Tip: if you want to postfix a timestamp you can use ==postfix-timestamp instead or --postfix $(date +%%d-%%m-%%Y) to format as you wish)",
    )

    parser.add_argument(
        "--postfix-timestamp",
        "-posttime",
        action="store_true",
        default=False,
        help="Adds a postfix (i.e., appends at the end) timestamp (YYYY-MM-DD) to the end of every renamed file. (Tip: if you want a custom date format that differs from this one, you can use  --postfix $(date +%%d-%%m-%%Y) to format as you wish)"
    )

    parser.add_argument(
        "--timestamp",
        "-t",
        action="store_true",
        help="Enable parsing the file's timestamp (created date) to the prompt for clues. (Tip: if you want to prefix or postfix a timestamp, you can use --prefix-timestamp or --postfix-timestamp instead).")

    parser.add_argument(
        "--metadata",
        "-mt",
        action="store_true",
        help="Enable parsing the metadata of an image file and passing it to the prompt. By default this will require exiftool to be installed, if you want to parse metadata with python libraries instead (e.g., exif, geopy and others are used to decode author name, camera model, GPS, etc) then you need to use --metadata-python",
    )

    parser.add_argument(
        "--metadata-python",
        "-mp",
        action="store_true",
        help="Enable parsing the metadata of an image file and passing it to the prompt. With this option it will use python libraries (e.g., exif, geopy and others are used to decode author name, camera model, GPS, etc). If you want to use exiftool instead, then you need to use --metadata",
    )

    parser.add_argument(
        "--keep",
        "-k",
        action="store_true",
        default=False,
        help="Do not reset the AI-model's conversation (i.e., continues the previous session if ollama still has it running). By default this is always reset.",
    )

    args = parser.parse_args()
    logger.info("Invoked with arguments: " + str(args))
    configure_logging(args.verbose)

    if args.override and args.prompt:
        logger.error("ERROR: either use --override (-o) for a brand new prompt or --prompt (-p) to append to the default prompt, both simultaneously is nonsensical")
        return

    if not args.directory.exists():
        raise FileNotFoundError(f"Directory not found: {args.directory}")

    if not args.keep:
        logger.info("Requesting a reset of the conversation with AI mode.")
        ollama.chat(
            model=args.model_image,
            messages=[
                {
                    "role": "system",
                    "content": reset_prompt,
                }
            ],
        )
        ollama.chat(
            model=args.model_text,
            messages=[
                {
                    "role": "system",
                    "content": reset_prompt,
                }
            ],
        )


    #files = list(args.directory.iterdir())
    extensions = valid_extensions_processed_as_text | valid_extensions_processed_as_image
    files = [
        f for f in args.directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    for file in tqdm(files, desc="Processing files", unit="file"):
        logger.info("Processing: " + file.name) #TODO add name to print
        process_file(file, args)

    if not files:
        logger.warning("No files to process (directory is likely empty).")
        return

# -----------------------------
# Program starts here, i.e., main call
# -----------------------------

if __name__ == "__main__":
    main()

