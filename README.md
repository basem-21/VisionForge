# VisionForge

**VisionForge** is a powerful local AI application for generating, editing, and enhancing images through an intuitive Gradio web interface.  
It transforms your ideas into visuals using cutting-edge generative AI models with support for text-to-image, image-to-image, inpainting, ControlNet, and more.

---

## ✨ Features
- **Multi-Workflow Generation**  
  - Text-to-Image  
  - Image-to-Image  
  - Inpainting (mask-based editing)  
  - ControlNet (guided generation)  

- **Model & LoRA Support**  
  - Load custom models and LoRAs with adjustable strength.  
  - Built-in downloader for models and LoRAs (Google Drive supported).  

- **Post-Processing**  
  - Super-resolution with RealESRGAN.  
  - Face enhancement with GFPGAN.  

- **History & Metadata**  
  - All images saved with embedded generation parameters.  
  - Built-in history tab with previews, metadata, and reloading parameters.  
  - Export selected or all images as ZIP.  

- **Smart Prompts**  
  - Weighted prompt syntax `(word:1.3)` for emphasis.  
  - Keyboard shortcuts for quick adjustments.  

- **Performance**  
  - Threaded job queue system for smooth, non-blocking UI.  
  - Real-time console log display.  

---

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/VisionForge.git
cd VisionForge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

*For GPU acceleration, ensure you have PyTorch installed with CUDA support.*  
*Optional: install `onnxruntime-gpu` for faster performance if supported.*  

---

## 🖥️ Usage

Run the app:
```bash
python app.py
```

Then open the local URL (typically `http://127.0.0.1:7860`) in your browser.

---

## 📂 Output

All generated images are saved in:
```
~/Documents/AI_Generated_Images/Images
```
Each PNG file includes metadata with the full generation settings.

---

## 📜 License

This project is dual-licensed:  

- **AGPL-3.0** – for personal, academic, and non-profit use.  
- **Commercial License** – required for any for-profit usage. See [`COMMERCIAL.md`](COMMERCIAL.md).  

---

## 🙏 Credits
- Built with [Gradio](https://gradio.app/) for the interface.  
- Powered by Hugging Face [Diffusers](https://huggingface.co/docs/diffusers), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), and [GFPGAN](https://github.com/TencentARC/GFPGAN).  
