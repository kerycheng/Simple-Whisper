@echo off

set /p audio_path="�п�J���T�ɮת����|: "
set /p model_size="�п�J�n�ϥΪ��ҫ��j�p (�Ҧp: tiny, base, small, medium, large): "
set /p language="�п�J���T�ɻy���N�X (�Ҧp: en, ja): "
set /p output_format="�п�J�n�x�s���ɮ׮榡 (srt �� ass): "
set /p ass_file="�п�J.ass�r���ɮת����| (�p�L�h�d��): "
set /p use_ai="�п�J�n�ϥΪ�AI½Ķ[Gemini(g)/ChatGPT(c)] (�p�L�h�d�šA�����bconfig.json�]�wapi key�~�i�ϥ�): "

if "%ass_file%"=="" (
    python transcribe.py "%audio_path%" "%model_size%" "%language%" "%output_format%" "%use_ai%"
) else (
    python transcribe.py "%audio_path%" "%model_size%" "%language%" "%output_format%" "%use_ai%" --ass_file "%ass_file%"
)

pause
