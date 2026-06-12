import whisper
from moviepy import VideoFileClip # 修改這行
import os


def extract_audio(video_path, audio_path):
    print(f"🎬 正在從 {video_path} 提取音訊...")
    video = VideoFileClip(video_path)
    # 提取音訊並隱藏 MoviePy 的大量 log
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()
    print("✅ 音訊提取完成！")


def format_timestamp(seconds):
    """將秒數轉換為 SRT 標準時間格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"


def transcribe_to_srt(audio_path, srt_path, model_size="small"):
    print(f"🧠 正在載入 Whisper '{model_size}' 模型...")
    # Whisper 模型大小: tiny, base, small, medium, large (越大越準，但也越慢)
    model = whisper.load_model(model_size)

    print("🎧 開始語音辨識...請耐心等候！")
    # fp16=False 是為了避免在純 CPU 環境下跳出警告
    # initial_prompt 是一個小技巧，強烈暗示 AI 輸出繁體中文
    result = model.transcribe(
        audio_path,
        fp16=False,
        initial_prompt="這是一段繁體中文字幕。"
    )

    print(f"📝 辨識完成！正在生成 {srt_path}...")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()

            # 依照 SRT 格式寫入
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


if __name__ == "__main__":
    # ===== 請在這裡替換你的檔案名稱 =====
    VIDEO_FILE = "test.mp4"  # 你的原始影片檔
    AUDIO_FILE = "temp_audio.wav"  # 暫存音訊檔
    SRT_FILE = "output.srt"  # 輸出的字幕檔名
    MODEL_SIZE = "small"  # 建議從 small 開始測試
    # ==================================

    if os.path.exists(VIDEO_FILE):
        extract_audio(VIDEO_FILE, AUDIO_FILE)
        transcribe_to_srt(AUDIO_FILE, SRT_FILE, model_size=MODEL_SIZE)

        # 處理完畢後，把暫存的音訊檔刪除，保持環境乾淨
        if os.path.exists(AUDIO_FILE):
            os.remove(AUDIO_FILE)

        print("🎉 全部處理完畢！你可以把 SRT 檔丟進 Premiere 或剪映裡了。")
    else:
        print(f"❌ 找不到影片檔案：{VIDEO_FILE}，請確認檔名和路徑是否正確。")