import os
import subprocess

# 画像ファイルが保存されているディレクトリ
image_dir = 'visualize/output_images/1'
output_video = 'visualize/output_movies/output_video1.mp4'

# ffmpegコマンドを作成
ffmpeg_command = [
    'ffmpeg',
    '-framerate', '30',  # フレームレートを30に設定
    '-i', os.path.join(image_dir, 'frame_%04d.png'),  # 入力画像ファイルのパス
    '-c:v', 'libx264',  # ビデオコーデックをlibx264に設定
    '-pix_fmt', 'yuv420p',  # ピクセルフォーマットをyuv420pに設定
    output_video  # 出力ビデオファイルのパス
]

# ffmpegコマンドを実行
subprocess.run(ffmpeg_command)

print(f"ビデオファイルが作成されました: {output_video}")
