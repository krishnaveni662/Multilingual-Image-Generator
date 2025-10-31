# Multilingual-Image-Generator
This project is a Flask-based AI web application that integrates multiple advanced models to enable multilingual image generation, captioning, speech-to-text translation, and content safety filtering — all in one unified platform.

The system allows users to generate AI images in different styles, translate prompts or captions across languages, upload images for automatic captioning, and convert audio speech into translated text.

🚀 Features

✅ Text-to-Image Generation:

Uses Stable Diffusion to generate high-quality, style-based AI images.

Supports multiple visual styles — Realism, Anime, Oil Painting, Natural.

✅ Image Captioning:

Uses BLIP (Salesforce/blip-image-captioning-base) to generate meaningful English captions.

Supports multilingual caption translation using Google Translator API.

✅ Speech-to-Text Conversion:

Converts audio (any format) into text using SpeechRecognition + pydub.

Automatically translates recognized speech into the target language.

✅ Content Safety Filtering:

Detects and blocks unsafe or NSFW text/images before generation or upload.

✅ User Authentication & Dashboard:

Login/register system using Flask-Login and SQLite.

Users can track their history of generations, captions, and speech transcriptions.

🧩 Tech Stack

Backend: Flask, SQLAlchemy, Flask-Login
AI Models: Stable Diffusion (HuggingFace Diffusers), BLIP, Google Translator, SpeechRecognition
Frontend: HTML, CSS, Jinja Templates
Database: SQLite
Libraries: Torch, Transformers, Diffusers, Pydub, PIL
