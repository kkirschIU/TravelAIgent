# TravelAIgent
Study Project for Course “Project: Computer Science DLMCSPCSP01”

TravelAIgent is a two-component system based on the open-weight base model **Llama 3.1 8B** by Meta  
(originally published on Hugging Face: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  
and a domain-specific **fine-tuned adapter** for Iceland travel planning.

The prototype runs locally via **Ollama** and does *not* require any cloud APIs or external booking services.

---

## 1. Hardware requirements (reference setup)

TravelAIgent was developed and tested on the following hardware:

- At least **20 GB** of free SSD storage  
- **16 GB RAM**
- **Intel® Core™ i7-10700F CPU @ 2.90 GHz**
- **NVIDIA GeForce RTX 3070 with 8 GB VRAM**
- Operating system: **Windows 11 Pro (64-bit)**

Similar consumer hardware (modern CPU, ~16 GB RAM, GPU with ~8 GB VRAM) should be sufficient.  
On weaker systems, the model might still run, but slower or with reduced context length.

---

## 2. Installation and setup (no training required)

Follow these steps to run the ready-to-use TravelAIgent model locally.

### Step 1: Install Ollama

1. Download Ollama for your operating system from:  
   https://ollama.com/download
2. Install Ollama using the provided installer.
3. Open a terminal (PowerShell, cmd, macOS Terminal, or Linux shell)

### Step 2: Download the LLaMA 3.1:8B Base Model
4. In your terminal enter
   ollama pull llama3.1:8b

### Step 3: Prepare the TravelAIgent model files
5. Create a folder, for example TravelAIgent at a place of your choice
   mkdir C:\Users\<your_name>\TravelAIgent

6. Download the following files from the GitHub Repository:
   - Modelfile
   - travelaigent_adapter.gguf
7. Copy the files into the TravelAIgent directory you just created. Your directory should look like the following:
   C:\Users\<your_name>\TravelAIgent\
   ├─ Modelfile
   └─ travelaigent_adapter.gguf

### Step 4: Create Travel AIgent
8. To create Travel AIgent first start the Ollama Service
   ollama serve
      Note: if you have installed the windows ollama app and the process is running in your task manager, starting the service is not necessary. Go on with Point 10.
10. In the terminal, change directory to the TravelAIgent directory you just created and where the GitHub Repository data lies in.
10. Enter the command
   ollama create travelaigent -f Modelfile
11. Once the model has been created succesfully you can start Travel AIgent with
    ollama run travelaigent

Enjoy exploring Travel AIgent and plan your first journey!
