import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import os
from tqdm import tqdm
from queue import Queue, Empty
import threading
import sys
import select
import shutil


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


def process_video(video_file, input_video_path, output_video_base):
    # 加载视频
    clip = VideoFileClip(input_video_path)
    fps = clip.fps

    # 提取前后10秒视频帧
    front_duration = 10
    back_duration = 10
    front_clip = clip.subclip(0, front_duration)
    back_clip = clip.subclip(clip.duration - back_duration, clip.duration)

    for cut_class in subdirectories:
        # 加载图片并调整大小
        header_img = cv2.imread(os.path.join(
            cut_image_dir), cut_class, 'header-after.png')
        tail_img = cv2.imread(os.path.join(cut_image_dir),
                              cut_class, 'tail-before.png')
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
        last_header_frame_idx = find_last_match_frame(
            front_frames, header_gray)
        # 查找第一帧匹配 tail 的位置
        first_tail_frame_idx = find_first_match_frame(back_frames, tail_gray)

        if first_tail_frame_idx > 0 and last_header_frame_idx > 0:
            output_path_final = os.path.join(
                output_video_base, cut_class, (video_file[:-4] + '-processing.mp4'))
            shutil.move(input_video_path, output_path_final)
            return output_path_final

    print(f"匹配失败，视频{video_file}未找到相关水印")
    return ""


def process_video_concurrently(video_file, video_file_path, processing_set):

    # 检查输出目录是否已存在同名视频文件，且排除 -processing 的文件 未分类不知其具体输出目录则全部包含
    existing_files = os.walk(sortover_video_dir)
    if video_file in existing_files or video_file.replace('.mp4', '-processing.mp4') in existing_files:
        print(f"输出目录已存在视频文件，跳过处理: {video_file_path}")
        return

    # 不知其分类则不知其haed与tail对应分类 存在性需要手动确认 程序跳过确认 直接添加
    with processing_lock:
        if video_file in processing_set:
            print(f"视频文件 {video_file} 正在处理中，跳过")
            return
        processing_set.add(video_file)
    try:
        result = process_video(video_file, video_file_path, sortover_video_dir)
        # 处理完成后，重命名输出文件
        if result != "":
            os.rename(result, result.replace(('-processing.mp4', '.mp4')))
            print(f"处理完成: {video_file}")
    finally:
        # 确保处理结束后，从 processing_set 中移除
        with processing_lock:
            processing_set.remove(video_file)


def is_file_completed(filepath, check_interval=0.1):
    """
    检查文件是否传输完成，通过循环检测文件大小是否稳定
    - check_interval: 每次检测文件大小的间隔时间（秒）
    """
    current_size = os.path.getsize(filepath)

    time.sleep(check_interval)

    current_size_now = os.path.getsize(filepath)

    if current_size == current_size_now and current_size_now != 0:  # 如果文件大小不变且不为0，认为传输完成
        return True

    return False


def monitor_directory(input_video_dir, cut_image_dir, output_video_dir, processing_set, queue):
    """监控输入目录以添加新视频文件到处理队列"""
    processed_files = set()
    total_files = 0  # 总文件数
    while True:
        new_files_count = 0  # 新文件计数
        if os.path.isdir(input_video_dir):
            for video_file in os.listdir(input_video_dir):
                if video_file.lower().endswith('.mp4') and video_file not in processed_files:
                    video_file_path = os.path.join(input_video_dir, video_file)

                    # 检查文件是否传输完成（循环检测）
                    if is_file_completed(video_file_path):
                        processed_files.add(video_file)
                        queue.put((video_file, video_file_path))
                        new_files_count += 1
                        total_files += 1  # 更新总文件数
        if new_files_count > 0:
            print(f"检测到 {new_files_count} 个新文件，当前总文件数: {total_files}")
        time.sleep(1)  # 每隔1秒检查一次目录


def worker(queue, processing_set, stop_event):
    """处理队列中的视频文件"""
    while not stop_event.is_set():
        try:
            video_info = queue.get(timeout=3)  # 设置超时防止卡住
            process_video_concurrently(*video_info, processing_set)
            queue.task_done()
        except Empty:
            pass


def main_concurrent(input_video_dir, cut_image_dir, output_video_dir):
    queue = Queue()
    processing_set = set()
    stop_event = threading.Event()  # 新增一个事件用于停止工作线程

    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_directory, args=(
        input_video_dir, cut_image_dir, output_video_dir, processing_set, queue), daemon=True)
    monitor_thread.start()

    with ThreadPoolExecutor(max_workers=10) as executor:
        # 持续监控队列并提交任务
        while True:
            for _ in range(min(10, queue.qsize())):
                executor.submit(worker, queue, processing_set, stop_event)

            # 检查是否有键盘输入
            i, o, e = select.select([sys.stdin], [], [], 0.1)
            if i:
                key = sys.stdin.readline().strip()
                if key == 'q':
                    print("检测到退出信号，准备退出...")
                    # 等待队列处理完所有任务
                    queue.join()

                    # 设置信号停止所有工作线程
                    stop_event.set()

                    # 等待所有任务完成并停止
                    print("等待线程池关闭...")
                    executor.shutdown(wait=True)

                    print("所有任务完成，程序退出")
                    break

            time.sleep(1)  # 短暂休眠，避免CPU空转


if __name__ == "__main__":
    sort_video_dir = "/Users/snowfish/Desktop/demo-cut/source/mp4-sort"
    cut_image_dir = "/Users/snowfish/Desktop/demo-cut/source/cut"
    sortover_video_dir = "/Users/snowfish/Desktop/demo-cut/source/mp4-in"
    processing_lock = threading.Lock()
    # 获取对比截图目录下的一级子目录列表
    subdirectories = [d for d in os.listdir(
        cut_image_dir) if os.path.isdir(os.path.join(cut_image_dir, d))]
    main_concurrent(sort_video_dir, cut_image_dir, sort_video_dir)
