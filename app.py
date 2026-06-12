import os
import sys

# 👉 終極必殺技：強制把打包後的暫存資料夾，塞進系統路徑 (PATH) 中
if getattr(sys, 'frozen', False):
    # 當程式被打包成 .app 後，sys._MEIPASS 就是它肚子裡的暫存資料夾
    os.environ["PATH"] += os.pathsep + sys._MEIPASS

# 下面是你原本的匯入
import gradio as gr
import whisper
from moviepy import VideoFileClip
import tempfile
import webview


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
    # 👉 取得系統絕對可以寫入的暫存資料夾路徑，並把檔名接上去
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "temp_audio_for_whisper.wav")
    # audio_path = "temp_audio_for_whisper.wav"
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
    # 1. 啟動 Gradio 伺服器
    # prevent_thread_lock=True: 讓伺服器在背景跑，不要卡住主程式
    # inbrowser=False: 不要呼叫系統預設的 Chrome/Edge 瀏覽器
    # server_port=7860: 綁定固定的通訊埠
    demo.launch(prevent_thread_lock=True, inbrowser=False, server_port=7860)

    # 2. 建立原生桌面視窗，並將 Gradio 的本機網址塞進去
    window = webview.create_window(
        title="🎙️ Whisper AI 字幕轉錄工具",
        url="http://127.0.0.1:7860",
        width=950,   # 設定視窗預設寬度
        height=750,  # 設定視窗預設高度
        resizable=True # 允許使用者縮放視窗
    )

    # 3. 啟動視窗 (程式會停在這裡，直到你點擊右上角的 X 關閉視窗)
    webview.start()

    # 4. 當視窗被關閉後，強制結束背後的 Gradio 伺服器與 Python 行程
    os._exit(0)