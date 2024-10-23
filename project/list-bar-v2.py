from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
import concurrent
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm


def resize_image_to_video(image, video_frame):
    """将图片调整到与视频帧大小相同"""
    return cv2.resize(image, (video_frame.shape[1], video_frame.shape[0]))


def get_gray_image(image):
    """将图片转换为灰度图"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_last_match_frame(video_frames, header_gray, tolerance=0.5):
    """找到与 header 图像匹配的最后一帧"""
    last_matched_frame_index = -1
    for i, frame in enumerate(video_frames):
        gray_frame = get_gray_image(frame)
        res = cv2.matchTemplate(gray_frame, header_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= tolerance:
            last_matched_frame_index = i
    return last_matched_frame_index


def find_first_match_frame(video_frames, tail_gray, tolerance=0.5):
    """找到与 tail 图像匹配的第一帧"""
    for i, frame in enumerate(video_frames):
        gray_frame = get_gray_image(frame)
        res = cv2.matchTemplate(gray_frame, tail_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= tolerance:
            return i
    return -1


def process_video(input_video_path, header_img_path, tail_img_path, output_video_path):
    # 加载视频
    clip = VideoFileClip(input_video_path)
    fps = clip.fps

    # 提取前后10秒视频帧
    front_duration = 10
    back_duration = 10
    front_clip = clip.subclip(0, front_duration)
    back_clip = clip.subclip(clip.duration - back_duration, clip.duration)

    # 加载图片并调整大小
    header_img = cv2.imread(header_img_path)
    tail_img = cv2.imread(tail_img_path)

    first_frame = front_clip.get_frame(0)
    header_resized = resize_image_to_video(header_img, first_frame)
    tail_resized = resize_image_to_video(tail_img, first_frame)

    # 将图片转换为灰度图
    header_gray = get_gray_image(header_resized)
    tail_gray = get_gray_image(tail_resized)

    # 提取视频帧
    front_frames = [front_clip.get_frame(
        t) for t in np.arange(0, front_duration, 1 / fps)]
    back_frames = [back_clip.get_frame(
        t) for t in np.arange(0, back_duration, 1 / fps)]

    # 查找最后一帧匹配 header 的位置
    last_header_frame_idx = find_last_match_frame(front_frames, header_gray)
    header_cut_time = last_header_frame_idx / fps
    print(f"最后一帧匹配 header 的时间: {header_cut_time} 秒")

    # 从最后匹配帧开始，删除该帧前所有帧和该帧之后 0.1 秒的所有帧
    header_final_cut_time = header_cut_time + 0.1

    # 查找第一帧匹配 tail 的位置
    first_tail_frame_idx = find_first_match_frame(back_frames, tail_gray)
    tail_cut_time = clip.duration - back_duration + first_tail_frame_idx / fps
    print(f"第一帧匹配 tail 的时间: {tail_cut_time} 秒")

    # 从第一匹配帧开始，删除该帧后所有帧和该帧之前 0.1 秒的所有帧
    tail_final_cut_time = tail_cut_time - 0.1

    # 删除不需要的帧，导出处理后的视频
    final_clip = clip.subclip(header_final_cut_time, tail_final_cut_time)
    final_clip.write_videofile(
        output_video_path, codec="libx264", audio_codec='aac', audio=True)


def process_video_concurrently(video_file, folder_path, cut_folder_path, output_folder_path):
    header_image_path = os.path.join(cut_folder_path, 'header-after.png')
    tail_image_path = os.path.join(cut_folder_path, 'tail-before.png')
    input_video_path = os.path.join(folder_path, video_file)

    # 修改输出路径，添加 -processing 后缀
    output_video_path = os.path.join(
        output_folder_path, video_file.replace('.mp4', '-processing.mp4'))

    # 检查输出目录是否已存在同名视频文件，且排除 -processing 的文件
    existing_files = os.listdir(output_folder_path)
    if video_file in existing_files or video_file.replace('.mp4', '-processing.mp4') in existing_files:
        print(f"输出目录已存在视频文件，跳过处理: {output_video_path}")
        return

    if os.path.exists(header_image_path) and os.path.exists(tail_image_path):
        process_video(input_video_path, header_image_path,
                      tail_image_path, output_video_path)

        # 处理完成后，重命名输出文件
        final_output_video_path = os.path.join(output_folder_path, video_file)
        os.rename(output_video_path, final_output_video_path)

    else:
        print(f"图片文件缺失，跳过视频文件: {video_file}")


def process_all_videos_concurrently(input_video_dir, cut_image_dir, output_video_dir):
    video_files = []
    for folder_name in os.listdir(input_video_dir):
        folder_path = os.path.join(input_video_dir, folder_name)
        if os.path.isdir(folder_path):
            for video_file in os.listdir(folder_path):
                if video_file.lower().endswith('.mp4'):
                    video_files.append((video_file, folder_path, os.path.join(
                        cut_image_dir, folder_name), os.path.join(output_video_dir, folder_name)))

    with ThreadPoolExecutor(max_workers=10) as executor:
        # 使用tqdm创建进度条
        futures = [executor.submit(
            process_video_concurrently, *video_info) for video_info in video_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass  # 等待所有任务完成


def main_concurrent(input_video_dir, cut_image_dir, output_video_dir):
    process_all_videos_concurrently(
        input_video_dir, cut_image_dir, output_video_dir)


if __name__ == "__main__":
    input_video_dir = "/Users/snowfish/Desktop/demo-cut/source/mp4-in"
    cut_image_dir = "/Users/snowfish/Desktop/demo-cut/source/cut"
    output_video_dir = "/Users/snowfish/Desktop/demo-cut/source/mp4-out"
    main_concurrent(input_video_dir, cut_image_dir, output_video_dir)
