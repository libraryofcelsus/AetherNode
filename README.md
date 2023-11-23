# AetherNode
A Simple Local API for Open-Source LLMs.  Only Llama-2-Chat for now.

This Repository is mostly just an experiment for now.  I may end up using this as a solution for my Ai Assistant/Agent project.  That project can be found here: https://github.com/libraryofcelsus/Aetherius_AI_Assistant

If you need help or have any questions join my discord: https://discord.gg/pb5zcNa7zE


**Changelog:**
- 11/22 Added Public Url Option in settings using ngrok
- 11/22 Added Proper Prompt Truncation
- 11/22 Added Proper Token Counting
- 11/22 Added Model Download from Hugging Face
- 11/22 Added Exllamav2
- Added a Settings Json
- Added Username Parameter
- Added Bot Name Parameter

## How to Install
1. Install Python 3.10.6, Make sure you add it to PATH: https://www.python.org/downloads/release/python-3106/
2. Download the AetherNode github folder and extract it to wherever you want it installed.
3. Run windows_aethernode_installer.bat to install the requirements.
4. Run a run_aethernode.bat to download the model defined in the settings json and start the api.
5. An example script of how to call it can be found in Example_Usage.py
6. To use it with a public host, download ngrok at: https://ngrok.com/download and place the .exe in the project folder.
7. Then, set "Use_Public_Host" in the settings json to true, then use the given ngrok url as the host instead of the localhost and port.
