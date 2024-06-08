import json
import os
import re
import sys
import threading
import time
import torch
import tqdm
import whisper
import pysubs2
import subprocess
from typing import Optional
import google.generativeai as genai
from openai import OpenAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config

def progress_indicator():
    while True:
        for cursor in '|/-\\':
            sys.stdout.write(cursor)
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')

def extract_audio_segment(input_path, start_time, end_time, output_path):
    # 檢查目錄是否存在，如果不存在就創建
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output file if it exists
        "-i", input_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-ar", "16000",  # set the sample rate to 16000 Hz
        "-ac", "1",  # set the number of audio channels to 1
        "-vn",  # disable the video stream
        "-acodec", "pcm_s16le",  # set the audio codec to pcm_s16le
        output_path,
        "-loglevel", "quiet"  # hide ffmpeg output
    ]
    subprocess.run(cmd, check=True)

def transcribe_segment(model, temp_audio_path, language):
    result = model.transcribe(temp_audio_path, language=language)
    os.remove(temp_audio_path)
    return result["text"]

def gemini_translate(text, gemini_api_key, gemini_model, prompt, retry_times=0):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(gemini_model)

    try:
        result = model.generate_content(prompt + text, safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE})

        return result.text

    except Exception as e:
        if re.search(r"^429 POST|Remote end closed connection|Connection reset by peer", str(e), flags=re.I) and retry_times < 6:
            print("請求過於頻繁或被斷開，將再次重試，重試次數: %d" % (retry_times + 1))
            time.sleep(min(retry_times + 1, 3))
            return gemini_translate(result.text)
        else:
            return "UnknownError"
        
def chatgpt_translate(text, chatgpt_api_key, chatgpt_model, prompt, retry_times=0):
    client = OpenAI(
        api_key=chatgpt_api_key,
        base_url='https://api.chatanywhere.cn/v1'
    )
    messages = [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': text}
    ]

    try:
        compiletion = client.chat.completions.create(
            model=chatgpt_model,
            messages=messages
        )
        return compiletion.choices[0].message.content
    
    except Exception as e:
        if re.search(r"^429 POST|Remote end closed connection|Connection reset by peer", str(e), flags=re.I) and retry_times < 6:
            print("請求過於頻繁或被斷開，將再次重試，重試次數: %d" % (retry_times + 1))
            time.sleep(min(retry_times + 1, 3))
            return gemini_translate(compiletion.choices[0].message.content)
        else:
            return "UnknownError"

def transcribe_audio(audio_path: str, model_size: str, language: str, output_format: str, 
                     use_ai: str, ass_file: Optional[str] = None, config_path='config.json'):

    # 檢查是否有可用的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 獲取 audio_path 的檔名
    audio_filename = os.path.basename(audio_path)
    # 移除檔名的副檔名
    base_name = os.path.splitext(audio_filename)[0]
    
    # 載入Whisper模型
    print("Whisper正在讀取模型...")
    model = whisper.load_model(model_size).to(device)
    print("模型讀取完畢")

    if use_ai == "g":
        # 載入Gemini api設定檔
        print("載入Gemini api設定檔...")
        config = load_config(config_path)
        gemini_api_key = config['gemini_api_key']
        gemini_model = config['gemini_model']
        prompt = config['prompt']

        print(f'Gemini Api Key: {gemini_api_key}')
        print(f'Gemini Model: {gemini_model}')
        print(f'Prompt: {prompt}')

    if use_ai == "c":
        # 載入ChatGPT設定檔
        print("載入ChatGPT api設定檔...")
        config = load_config(config_path)
        chatgpt_api_key = config['chatgpt_api_key']
        chatgpt_model = config['chatgpt_model']
        prompt = config['prompt']

        print(f'ChatGPT Api Key: {chatgpt_api_key}')
        print(f'ChatGPT Model: {chatgpt_model}')
        print(f'Prompt: {prompt}')

    if ass_file:
        subs = pysubs2.load(ass_file)
        for i, line in tqdm.tqdm(enumerate(subs), total=len(subs), desc="處理字幕"):
            start_time = line.start / 1000  # 起始時間（秒）
            end_time = line.end / 1000  # 結束時間（秒）
            temp_audio_path = f"temp/{i}.wav"
            extract_audio_segment(audio_path, start_time, end_time, temp_audio_path)

            # 轉錄音訊
            result = model.transcribe(temp_audio_path, language=language)

            if use_ai == "g":
                translated_text = gemini_translate(result['text'], gemini_api_key, gemini_model, prompt)
                line.text = result['text'] + "| " + translated_text
            elif use_ai == "c":
                translated_text = chatgpt_translate(result['text'], chatgpt_api_key, chatgpt_model, prompt)
                line.text = result['text'] + "| " + translated_text
            else:
                line.text = result['text']

            print(f"字幕 {i+1}/{len(subs)} ：{line.text}")
            os.remove(temp_audio_path)
            
        # 存檔
        if output_format == "srt":
            subs.save(f"{base_name}.srt")
        elif output_format == "ass":
            subs.save(f"{base_name}.ass")

    else:
        try:
            print("開始進行音訊轉寫，請稍候...")
            progress_thread = threading.Thread(target=progress_indicator)
            progress_thread.start()

            result = model.transcribe(audio_path, language=language)

        finally:
            progress_thread.do_run = False
            progress_thread.join()
            result = model.transcribe(audio_path, language=language)
            lines = result['segments']

            subs = pysubs2.SSAFile()
            for i, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc="處理字幕"):
                start = int(line['start'] * 1000)
                end = int(line['end'] * 1000)
                text = line['text']

                if use_ai == "g":
                    translated_text = gemini_translate(line['text'], gemini_api_key, gemini_model, prompt)
                    print(f"字幕 {i+1}/{len(subs)} ：{line['text']} | {translated_text}")
                    subs.append(pysubs2.SSAEvent(start=start, end=end, text=text + "| " + translated_text))
                elif use_ai == "c":
                    translated_text = chatgpt_translate(result['text'], chatgpt_api_key, chatgpt_model, prompt)
                    print(f"字幕 {i+1}/{len(subs)} ：{line['text']} | {translated_text}")
                    subs.append(pysubs2.SSAEvent(start=start, end=end, text=text + "| " + translated_text))                    
                else:
                    print(f"字幕 {i+1}/{len(subs)} ：{line['text']}")
                    subs.append(pysubs2.SSAEvent(start=start, end=end, text=text))

            # 存檔
            if output_format == "srt":
                subs.save(f"{base_name}.srt")
            elif output_format == "ass":
                subs.save(f"{base_name}.ass")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("model_size", type=str, help="Model size to use (e.g., tiny, base, small, medium, large)")
    parser.add_argument("language", type=str, help="Language of the audio file (e.g., en, ja)")
    parser.add_argument("output_format", type=str, help="Output subtitle format (srt or ass)")
    parser.add_argument("use_ai", type=str, help="Which AI is used for translation (Gemini[g], ChatGPT[c]")
    parser.add_argument("--ass_file", type=str, help="Path to the .ass subtitle file (optional)", default=None)

    args = parser.parse_args()

    transcribe_audio(args.audio_path, args.model_size, args.language, args.output_format, args.use_ai, args.ass_file)
