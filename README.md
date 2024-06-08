# Simple Whisper

* [概述](#overview)
* [更新歷程](#history)
* [套件安裝](#install)
* [使用說明](#use)
 * [config.json](#config)
 * [進行翻譯](#translate)

<h2 id="overview">概述</h2>

基於[Openai Whisper](https://github.com/openai/whisper)所開發的輕量級字幕軸填軸程式。
除了基本的Whisper進行打軸與填入轉錄文本以外，也提供了把打完的空軸交由Whisper來填入轉錄文本，省去後續修軸困擾的功能

並且可調用[Gemini](https://gemini.google.com/app?hl=zh-TW)和[ChatGpt](https://openai.com/index/chatgpt/)進行初步翻譯。

可輸出ass或srt這兩種格式的字幕軸檔案，可透過[Aegisub](https://aegisub.org/)進行字幕軸改動及翻譯。
<br>

<h2 id="history">更新歷程</h2>

2024/06/09:
* 創建 Simple-Whisper。
<br>

<h2 id="install">套件安裝</h2>

```bash
pip install -r requirements.txt
```

<br>

<h2 id="use">使用說明</h2>

<h4 id="#config">config.json</h4>

如果需要使用到AI翻譯，請在 config.json 填入你需要使用的AI的API Key，也可依你的需求去調整其model和prompt設定。

```bash
{
    "gemini_api_key": "YOUR Gemini API Key",
    "gemini_model": "gemini-pro",
    "chatgpt_api_key": "YOUR ChatGPT API Key",
    "chatgpt_model": "gpt-3.5-turbo",
    "prompt": "你是一位擁有多年日語翻譯中文的專業翻譯者，我會給你一段日文內容，你只需要將翻譯好的繁體中文內容告訴我即可"
}

```

<h4 id="#translate">進行翻譯</h4>

點擊 run_transcribe.bat 並依照指示依序輸入。(建議可將需用到的影音檔與字幕軸檔一併放在Simple-Whisper目錄底下)

```bash
請輸入音訊檔案的路徑: 此處填入所需進行轉錄的mp4或mp3等等檔案名稱
請輸入要使用的模型大小 (例如: tiny, base, small, medium, large): 建議使用 medium 或 large 模型，如果是第一次使用則 Whisper 會自動幫你下載該模型
請輸入音訊檔語言代碼 (例如: en, ja): 依照人物說話的語言填入該語言代號
請輸入要儲存的檔案格式 (srt 或 ass): 依照後續要使用的字幕軸檔案格式
請輸入.ass字幕檔案的路徑 (如無則留空): 若是要自行提供空軸則在此處填入空軸的檔案名稱，無則跳過
請輸入要使用的AI翻譯[Gemini(g)/ChatGPT(c)] (如無則留空，必須在config.json設定api key才可使用): 選擇所需使用的AI翻譯，無則跳過
```
![範例截圖](https://imgur.com/zpzFwjk.jpg)

