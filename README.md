# AI File Renamer (Ollama-based)

Rename files (e.g., .jpg images, .txt, .pdf, .docx, .md) automatically using AI-generated keywords derived from the file's content (and optionally metadata), even from images. It uses **Ollama multimodal models** to analyze files (including images) and generate concise, human-readable filenames. This can **use only local models**, since you control which model to use via ollama.

This project is an extension of [ai-rename-images](https://github.com/PedroLopes/ai-rename-images), but adds functionality such as parsing .txt, .docx, and .pdf files. 

(If you need a interactive batch-renamer that previews a file, then asks you the content (rather than using AI), you can use [peek-rename](https://github.com/PedroLopes/peek-rename))

---

## Features

- Rename files (`.jpg` / `.jpeg` images, `.txt`, `.pdf`, etc)  based on their actual contents
- Supports **any Ollama vision-capable model** (allows selecting different models for text or image files)
- Customizable prompts (append or fully override)
- Reads .pdf, .docx and other text based files
- Optional conversation persistence with the model
- Optional EXIF metadata injection into the prompt
- Optional GPS reverse-geocoding (via `exiftool`)
- Allows custom prefix/postfix, allows parsing timestamps, and more. 

---

## Install and requirements

### What you need
- Python **3.9+**
- [Ollama](https://ollama.com/) running locally (no need for html connections, unless you want to parse GPS metadata from images)
- A vision-capable Ollama model (e.g. `llava-phi3`) and a text-capable model (e.g., ``qwen2.5-coder:7b``)
- To convert word documents, do ``pip install python-docx`` (not ``docx``)

### Installation

Clone the repository and make the installer executable and install it:

```bash
git clone https://github.com/PedroLopes/ai-rename-files
cd ai-rename-files
chmod +x install.sh
./install.sh
```

Alternatively, if you want to do it manually:

```bash
git clone https://github.com/PedroLopes/ai-rename-files
cd ai-rename-files
pip install -r requirements.txt
chmod +x ai_rename_files.py 
```
Now you can invoke it using:

```bash
./ai-rename-files <directory-with-files>
```

Or you can keep invoking it explicitly with ``python3`` (without ``chmod +x`` to create an executable).

```bash
python3 ai_rename_files.py <directory-with-files>
```

## Command-Line Options

### Core Options

| Option | Description |
|------|-------------|
| `directory` | Directory containing files |
| `-vlm, --model-image` | Ollama model to use on images (default: `llava-phi3`) |
| `-llm, --model-text` | Ollama model to use for text files (default: `qwen2.5-coder:7b`) |
| `-n, --number` | Number of keywords (default: `3`) |
| `-r, --readsize | Number of characters to read and pass to prompt on text files (default: `100``) | 
| `-d, --delimiter` | `_`, `-`, or space (default: `-`) |
| `-v, --verbose` | Enable verbose logging |

---

### Prompt Control

| Option | Description |
|------|-------------|
| `-p, --prompt` | Append text to the default prompt |
| `-o, --override` | Replace the entire prompt |

⚠️ **`--prompt` and `--override` are mutually exclusive**

---

### Model Session Control

| Option | Description |
|------|-------------|
| `-k, --keep` | Do not reset the model conversation |

By default, the conversation is reset before processing files.

---

### Metadata / EXIF Options

| Option | Description |
|------|-------------|
| `-mt, --metadata` | Parse metadata using external `exiftool` (requires to install it, e.g., ``brew install exiftool``) |
| `-mp, --metadata-python` | Parse metadata using Python libraries |

When enabled, selected metadata fields (camera, flash, GPS, etc.) are appended to the prompt to improve keyword accuracy. You can configure which EXIF tags are included by editing:

```python
metadata_filter = [
  "Date/Time Original",
  "Flash",
  "Make",
  "Camera Model Name",
  "Orientation",
  "GPS Position"
]
```

Note: tests reveal that using ``exiftool`` enables a more accurate parsing of the tags, especially the GPS location. 

### File date, directory, etc

| Option | Description |
|------|-------------|
| `-dir, --directory-name` | Passes the directory name to the prompt for clues | 
| `--t, --timestamp` | Passes the date of the file (as per file system / OS) to the prompt | 

### Renaming files with prefixes, timestamps, postfixes, etc.

| Option | Description |                                    
|------|-------------|
| `--pre, --prefix` | Passes a string as prefix for all files | 
| `--pretime, --prefix-timestamp` | Passes the file's timestamp string as prefix for each file (if you want a current time, you can consider using ``--prefix $(date +%d-%m-%Y)`` | 
| ``--post, --postfix | Same as above but appends at the end | 
| ``--posttime, --postfix-timestamp | | Same as above but appends at the end | 

## Prompt Design

The tool uses a structured prompt to ensure the AI returns machine-readable output.

### Default Prompt

```python
Describe the file in {number} simple keywords, never use more than {number} words.
Output in JSON format.
Use the following schema: { keywords: List[str] }.
```

- `{number}` is dynamically replaced using the `-n / --number` argument.
- The model **must** return valid JSON matching the specified schema.
- The prompt is intentionally strict to allow automatic parsing.

---

### Modifying the Prompt

You can customize how files are described in two ways:

#### Append to the default prompt

Use `-p / --prompt` to add additional instructions while keeping the original prompt structure:

```bash
python3 ai_rename_files.py ./images -p "Focus on architectural features of the buildings you see."
```

## Supported File Types

  * ✅ .jpg
  * ✅ .jpeg
  * ❌ Everything else is ignored

### Core dependencies include:

  * ``ollama`` 
  * ``pydantic``
  * ``tqdm``

### Optional dependencies (loaded dynamically as needed)

  * ``pillow`` (only if using --metadata-python mode)
  * ``geopy``, ``lat-lon-parser``, ``pandas`` (for GPS parsing while using --metadata mode) 
  * ``Document`` (for ``.docx`` parsing—please do not install ``pip install docx``, instead use ``pip install python-docx`` to prevent a error with python 3.9+)
  * ``PyPDF2`` (for ``.pdf`` parsing)

## Credits

This project is an extension of [ai-rename-images](https://github.com/PedroLopes/ai-rename-images), but adds functionality such as parsing .txt, .docx, and .pdf files. The original is also partially based on a fork of [ollama-rename-img](https://github.com/Tedfulk/ollama-rename-img) but with some major differences: (1) model selection, (2) prompt customization, (3) pip-based dependencies, (4) GPS or metadata passed to prompt, and also, some minor differences: (1) disregards ``.DS_store`` or any non-jpeg files, (2) removes extra files from loading bar. Additionally, the ``ollama-rename-img`` only works for images, and this version works for a variety of text files. 

Additional snippets of code from:
  * A one liner by [Vaibhav K, from stackoverflow](https://stackoverflow.com/questions/21697645/how-to-extract-metadata-from-an-image-using-python) was modified to invoke exiftool and grab metadata tags.
  * Image samples from [exif-samples](https://github.com/ianare/exif-samples/blob/master/jpg/gps/DSCN0010.jpg) were used to test GPS tags when extracted frim exif metadata.

# License

Same license as the original project unless otherwise specified.

