import gradio as gr
import whisper
from moviepy import VideoFileClip
import os


def format_timestamp(seconds):
    """將秒數轉換為 SRT 標準時間格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"


def process_media(file_path, model_size):
    """Gradio 處理流程 (使用 yield 來實現即時進度更新)"""
    if not file_path:
        yield "❌ 請先上傳檔案！", None
        return

    yield f"📂 檔案接收成功，準備處理...", None

    audio_path = "temp_audio_for_whisper.wav"
    # Gradio 上傳的檔案會有個暫存路徑，我們取它的副檔名來判斷
    file_ext = os.path.splitext(file_path)[1].lower()
    srt_path = os.path.splitext(file_path)[0] + ".srt"

    try:
        # 1. 判斷是否需要抽音訊
        if file_ext in ['.mp4', '.mov', '.mkv', '.avi']:
            yield "🎬 偵測到影片檔，正在提取音訊...", None
            video = VideoFileClip(file_path)
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            target_audio = audio_path
        else:
            yield "🎵 偵測到音訊檔，直接進入辨識流程...", None
            target_audio = file_path

        # 2. 載入模型
        yield f"🧠 正在載入 Whisper '{model_size}' 模型...", None
        model = whisper.load_model(model_size)

        # 3. 開始辨識
        yield "🎧 開始語音辨識，這可能需要幾分鐘的時間，請稍候...", None
        result = model.transcribe(target_audio, fp16=False, initial_prompt="這是一段繁體中文字幕。")

        # 4. 生成 SRT
        yield f"📝 辨識完成！正在生成 SRT 檔案...", None
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        yield f"✅ 成功！字幕已生成，請點擊下方按鈕下載。", srt_path

    except Exception as e:
        yield f"❌ 發生錯誤: {str(e)}", None

    finally:
        # 5. 清理暫存的音訊檔
        if os.path.exists(audio_path) and target_audio == audio_path:
            try:
                os.remove(audio_path)
            except:
                pass


# ==========================================
# 🎨 建立 Gradio 網頁介面 (使用 Blocks 讓排版更專業)
# ==========================================
with gr.Blocks(title="AI 自動字幕生成器") as demo:
    gr.Markdown("# 🎙️ Whisper AI 自動字幕生成器")
    gr.Markdown("上傳你的 Vlog 影片或錄音檔，一鍵自動生成 `.srt` 字幕檔，可直接匯入剪輯軟體中！")

    with gr.Row():
        with gr.Column():
            # 輸入區塊
            file_input = gr.File(label="上傳影片或音訊檔 (支援 mp4, mov, mp3, wav)", file_types=["video", "audio"])
            model_dropdown = gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"], value="small",
                                         label="選擇 Whisper 模型大小 (越大越準但越慢)")
            submit_btn = gr.Button("🚀 開始轉錄", variant="primary")

        with gr.Column():
            # 輸出區塊
            status_output = gr.Textbox(label="處理進度", lines=5, interactive=False)
            file_output = gr.File(label="📥 下載 SRT 字幕檔")

    # 綁定按鈕動作
    submit_btn.click(
        fn=process_media,
        inputs=[file_input, model_dropdown],
        outputs=[status_output, file_output]
    )

if __name__ == "__main__":
    # 啟動網頁伺服器
    demo.launch()