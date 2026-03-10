import os
import time
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import nltk
from gtts import gTTS
import tempfile
import pygame
from googletrans import Translator  # For translation

nltk.download('punkt')

# -------------------------------
# CONFIGURATION
# -------------------------------
dataset_dir = r'D:\champa\projects\Architectural image\source_code\datasetss - Copy'
csv_file = os.path.join("captions_architecture.csv")
inception_weights = r'D:\champa\projects\Architectural image\source_code\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
features_file = os.path.join(dataset_dir, 'features.pkl')
tokenizer_file = os.path.join(dataset_dir, 'tokenizer.pkl')
checkpoint_file = os.path.join(dataset_dir, 'caption_model_best.h5')

# -------------------------------
# LOAD FEATURE EXTRACTOR
# -------------------------------
base_model = InceptionV3(weights=None, include_top=False, pooling='avg')
base_model.load_weights(inception_weights)
model_cnn = Model(inputs=base_model.input, outputs=base_model.output)
print("✅ InceptionV3 feature extractor ready.")
print(" features.pkl model loaded")
print(" tokenizer.pkl model loaded")
print(" caption_model_best.h5 model loaded")

# -------------------------------
# LOAD CSV
# -------------------------------
try:
    df = pd.read_csv(csv_file, encoding='cp1252')
    df['image_name'] = df['image_name'].astype(str)
except Exception as e:
    print("❌ Failed to load CSV:", e)
    df = pd.DataFrame(columns=['image_name', 'caption'])

# -------------------------------
# INITIALIZE PYGAME
# -------------------------------
pygame.mixer.init()

# -------------------------------
# TRANSLATOR
# -------------------------------
translator = Translator()

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model_cnn.predict(x, verbose=0)
    return feature[0]

# -------------------------------
# GET CSV CAPTION
# -------------------------------
def get_caption(img_path):
    img_name = os.path.basename(img_path)
    row = df[df['image_name'].str.contains(img_name, case=False, na=False)]
    if not row.empty:
        return row.iloc[0]['caption']
    else:
        return "No caption found for this image."

# -------------------------------
# PLAY TTS
# -------------------------------
def play_tts(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_audio_path = fp.name
            tts.save(temp_audio_path)
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            try:
                if not root.winfo_exists():
                    pygame.mixer.music.stop()
                    break
                root.update()
            except tk.TclError:
                pygame.mixer.music.stop()
                break

        pygame.mixer.music.unload()
        time.sleep(0.2)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")

# -------------------------------
# BUTTON FUNCTIONS
# -------------------------------
def tts_english():
    if hasattr(root, 'current_caption'):
        text = root.current_caption
        caption_textbox.config(state='normal')
        caption_textbox.delete(1.0, tk.END)
        caption_textbox.insert(tk.END, text)
        caption_textbox.config(state='disabled')
        play_tts(text, lang='en')

def tts_kannada():
    if hasattr(root, 'current_caption'):
        translated = translator.translate(root.current_caption, dest='kn').text
        caption_textbox.config(state='normal')
        caption_textbox.delete(1.0, tk.END)
        caption_textbox.insert(tk.END, translated)
        caption_textbox.config(state='disabled')
        play_tts(translated, lang='kn')

def tts_hindi():
    if hasattr(root, 'current_caption'):
        translated = translator.translate(root.current_caption, dest='hi').text
        caption_textbox.config(state='normal')
        caption_textbox.delete(1.0, tk.END)
        caption_textbox.insert(tk.END, translated)
        caption_textbox.config(state='disabled')
        play_tts(translated, lang='hi')

# -------------------------------
# PAUSE/RESUME AUDIO
# -------------------------------
def toggle_pause_resume(event=None):
    if pygame.mixer.music.get_busy():
        if toggle_pause_resume.paused:
            pygame.mixer.music.unpause()
            toggle_pause_resume.paused = False
        else:
            pygame.mixer.music.pause()
            toggle_pause_resume.paused = True
toggle_pause_resume.paused = False

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    try:
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        extract_features(file_path)  # optional

        root.current_image_path = file_path
        root.current_caption = get_caption(file_path)

        # Clear textbox
        caption_textbox.config(state='normal')
        caption_textbox.delete(1.0, tk.END)
        caption_textbox.config(state='disabled')

    except Exception as e:
        print(f"⚠️ Error: {e}")

# -------------------------------
# MAIN WINDOW
# -------------------------------
# -------------------------------  
# MODERN Tkinter DESIGN
# -------------------------------  
root = tk.Tk()
root.title("🏛️ Architectural Image Captioning")
root.geometry("750x900")
root.configure(bg="#f0f2f5")  # Soft neutral background

# ---------- TITLE ----------
title_label = tk.Label(
    root,
    text="🏛️ Architectural Image Captioning",
    font=("Helvetica", 20, "bold"),
    bg="#f0f2f5",
    fg="#333"
)
title_label.pack(pady=20)

# ---------- IMAGE DISPLAY ----------
img_frame = tk.Frame(root, bg="#ffffff", bd=2, relief='groove')
img_frame.pack(pady=10)
img_label = tk.Label(img_frame, bg="#ffffff")
img_label.pack(padx=10, pady=10)

# ---------- BUTTONS ----------
btn_frame = tk.Frame(root, bg="#f0f2f5")
btn_frame.pack(pady=15)

def style_button(btn):
    btn.configure(
        font=("Helvetica", 12, "bold"),
        bd=0,
        relief="ridge",
        padx=15,
        pady=8,
        cursor="hand2"
    )
    # Hover effect
    def on_enter(e):
        btn['bg'] = '#555'
        btn['fg'] = 'white'
    def on_leave(e):
        btn['bg'] = original_bg
        btn['fg'] = original_fg
    original_bg = btn['bg']
    original_fg = btn['fg']
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

btn_upload = tk.Button(btn_frame, text="📁 Upload Image", command=upload_image, bg="#0078D7", fg="white")
btn_upload.grid(row=0, column=0, padx=5, pady=5)
style_button(btn_upload)

btn_en = tk.Button(btn_frame, text="🎵 English", bg="#2196F3", fg="white", command=tts_english)
btn_en.grid(row=0, column=1, padx=5, pady=5)
style_button(btn_en)

btn_kn = tk.Button(btn_frame, text="🎵 Kannada", bg="#4CAF50", fg="white", command=tts_kannada)
btn_kn.grid(row=0, column=2, padx=5, pady=5)
style_button(btn_kn)

btn_hi = tk.Button(btn_frame, text="🎵 Hindi", bg="#F44336", fg="white", command=tts_hindi)
btn_hi.grid(row=0, column=3, padx=5, pady=5)
style_button(btn_hi)

btn_pause = tk.Button(btn_frame, text="⏸ Pause/Resume", bg="#FFC107", fg="black")
btn_pause.grid(row=0, column=4, padx=5, pady=5)
style_button(btn_pause)
btn_pause.bind("<Button-1>", toggle_pause_resume)
btn_pause.bind("<Double-Button-1>", toggle_pause_resume)

# ---------- CAPTION BOX ----------
caption_frame = tk.Frame(root, bg="#f0f2f5")
caption_frame.pack(pady=15, fill='both', expand=True)

caption_scroll = tk.Scrollbar(caption_frame)
caption_scroll.pack(side='right', fill='y')

caption_textbox = tk.Text(
    caption_frame,
    height=10,
    font=("Helvetica", 14),
    wrap='word',
    yscrollcommand=caption_scroll.set,
    bg="#ffffff",
    fg="#333",
    bd=2,
    relief="groove",
    padx=10,
    pady=10
)
caption_textbox.pack(side='left', fill='both', expand=True)
caption_textbox.config(state='disabled')
caption_scroll.config(command=caption_textbox.yview)

# ---------- RUN APP ----------
root.mainloop()
