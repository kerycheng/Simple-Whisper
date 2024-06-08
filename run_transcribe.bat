@echo off

set /p audio_path="請輸入音訊檔案的路徑: "
set /p model_size="請輸入要使用的模型大小 (例如: tiny, base, small, medium, large): "
set /p language="請輸入音訊檔語言代碼 (例如: en, ja): "
set /p output_format="請輸入要儲存的檔案格式 (srt 或 ass): "
set /p ass_file="請輸入.ass字幕檔案的路徑 (如無則留空): "
set /p use_ai="請輸入要使用的AI翻譯[Gemini(g)/ChatGPT(c)] (如無則留空，必須在config.json設定api key才可使用): "

if "%ass_file%"=="" (
    python transcribe.py "%audio_path%" "%model_size%" "%language%" "%output_format%" "%use_ai%"
) else (
    python transcribe.py "%audio_path%" "%model_size%" "%language%" "%output_format%" "%use_ai%" --ass_file "%ass_file%"
)

pause
