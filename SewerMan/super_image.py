import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime
import json


class VideoSuperImageProcessor:
    def __init__(self, target_frames=9, frame_size=448):
        """
        初始化视频处理器

        Args:
            target_frames: 需要采样的帧数（默认9，生成3x3网格）
            frame_size: 每帧的尺寸（默认224x224）
        """
        self.target_frames = target_frames
        self.frame_size = frame_size
        self.grid_size = int(np.sqrt(target_frames))  # 自动计算网格大小

        # 支持的视频格式
        self.supported_formats = {
            '.mp4', '.avi', '.mov', '.mkv', '.flv',
            '.wmv', '.m4v', '.webm', '.mpg', '.mpeg'
        }

        print(f"初始化完成: {self.grid_size}x{self.grid_size} 网格, 每帧 {frame_size}x{frame_size}")

    def validate_paths(self, input_folder, output_folder):
        """验证输入输出路径"""
        input_folder = os.path.normpath(input_folder)
        output_folder = os.path.normpath(output_folder)

        if not os.path.exists(input_folder):
            raise ValueError(f"输入文件夹不存在: {input_folder}")

        if not os.path.isdir(input_folder):
            raise ValueError(f"输入路径不是文件夹: {input_folder}")

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        return input_folder, output_folder

    def get_video_files(self, folder_path):
        """获取文件夹中所有支持的视频文件"""
        video_files = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in self.supported_formats:
                    video_files.append(file_path)

        return sorted(video_files)

    def extract_frames_uniform(self, video_path):
        """从视频中均匀采样帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        frame_indices = []

        if total_frames == 0:
            raise ValueError("视频文件没有帧数据")

        # 计算采样间隔
        if total_frames <= self.target_frames:
            # 视频帧数不足，尽可能采样
            indices = list(range(total_frames))
            # 如果需要更多帧，重复最后一帧
            indices += [total_frames - 1] * (self.target_frames - total_frames)
        else:
            # 均匀采样
            interval = total_frames // self.target_frames
            indices = [min(i * interval, total_frames - 1) for i in range(self.target_frames)]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换颜色空间并调整大小
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frames.append(frame)
                frame_indices.append(idx)
            else:
                # 读取失败时使用黑色帧
                black_frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
                frames.append(black_frame)
                frame_indices.append(-1)

        cap.release()

        return {
            'frames': frames,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'width': width,
            'height': height,
            'frame_indices': frame_indices
        }

    def create_super_image(self, frames):
        """将帧拼接成超图像"""
        if len(frames) != self.target_frames:
            raise ValueError(f"帧数量不正确: 期望 {self.target_frames}, 实际 {len(frames)}")

        # 创建空白的超图像画布
        super_height = self.frame_size * self.grid_size
        super_width = self.frame_size * self.grid_size
        super_image = np.zeros((super_height, super_width, 3), dtype=np.uint8)

        # 将帧排列到网格中
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx < len(frames):
                    frame = frames[idx]
                    y_start = i * self.frame_size
                    y_end = y_start + self.frame_size
                    x_start = j * self.frame_size
                    x_end = x_start + self.frame_size
                    super_image[y_start:y_end, x_start:x_end] = frame

        return super_image

    def process_single_video(self, video_path, output_path):
        """处理单个视频文件"""
        try:
            # 提取帧
            video_info = self.extract_frames_uniform(video_path)

            # 创建超图像
            super_image = self.create_super_image(video_info['frames'])

            # 保存图像
            cv2.imwrite(output_path, cv2.cvtColor(super_image, cv2.COLOR_RGB2BGR))

            return {
                'status': 'success',
                'video_path': video_path,
                'output_path': output_path,
                'super_image_shape': super_image.shape,
                'video_info': {
                    'total_frames': video_info['total_frames'],
                    'fps': video_info['fps'],
                    'duration': video_info['duration'],
                    'original_resolution': f"{video_info['width']}x{video_info['height']}",
                    'sampled_frames': video_info['frame_indices']
                }
            }

        except Exception as e:
            return {
                'status': 'failed',
                'video_path': video_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def process_folder(self, input_folder, output_folder, max_videos=None):
        """
        处理整个文件夹中的视频

        Args:
            input_folder: 输入视频文件夹路径
            output_folder: 输出文件夹路径
            max_videos: 最大处理视频数量（None表示处理所有）
        """
        start_time = time.time()

        # 验证和规范化路径
        input_folder, output_folder = self.validate_paths(input_folder, output_folder)

        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        print(f"目标帧数: {self.target_frames} ({self.grid_size}x{self.grid_size} 网格)")
        print(f"单帧尺寸: {self.frame_size}x{self.frame_size}")
        print(f"超图像尺寸: {self.frame_size * self.grid_size}x{self.frame_size * self.grid_size}")
        print("-" * 60)

        # 获取视频文件
        video_files = self.get_video_files(input_folder)

        if not video_files:
            print("没有找到支持的视频文件")
            return []

        if max_videos:
            video_files = video_files[:max_videos]
            print(f"限制处理前 {max_videos} 个视频")

        print(f"找到 {len(video_files)} 个视频文件，开始处理...")
        print("-" * 60)

        results = []
        success_count = 0
        failed_count = 0

        # 使用进度条处理每个视频
        for video_path in tqdm(video_files, desc="处理视频", unit="video"):
            try:
                # 生成输出文件名
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_filename = f"{video_name}.jpg"
                output_path = os.path.join(output_folder, output_filename)

                # 处理视频
                result = self.process_single_video(video_path, output_path)
                results.append(result)

                if result['status'] == 'success':
                    success_count += 1
                else:
                    failed_count += 1
                    tqdm.write(f"失败: {os.path.basename(video_path)} - {result['error']}")

            except Exception as e:
                error_result = {
                    'status': 'failed',
                    'video_path': video_path,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
                failed_count += 1
                tqdm.write(f"异常: {os.path.basename(video_path)} - {e}")

        # 计算处理时间
        processing_time = time.time() - start_time

        # 生成报告
        self.generate_report(results, output_folder, processing_time)

        # 打印统计信息
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"总视频数: {len(video_files)}")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}")
        print(f"成功率: {success_count / len(video_files) * 100:.1f}%")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"平均每个视频: {processing_time / len(video_files):.2f} 秒")
        print(f"输出目录: {output_folder}")

        # 显示一些成功案例
        success_results = [r for r in results if r['status'] == 'success']
        if success_results:
            print("\n前5个成功处理的文件:")
            for i, result in enumerate(success_results[:5]):
                print(f"  {i + 1}. {os.path.basename(result['video_path'])}")

        return results

    def generate_report(self, results, output_folder, processing_time):
        """生成处理报告"""
        # JSON报告
        report_data = {
            'processing_date': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'total_videos': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'parameters': {
                'target_frames': self.target_frames,
                'frame_size': self.frame_size,
                'grid_size': self.grid_size
            },
            'results': results
        }

        # 保存JSON报告
        json_report_path = os.path.join(output_folder, 'processing_report.json')
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # 文本报告
        txt_report_path = os.path.join(output_folder, 'processing_summary.txt')
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("视频超图像处理报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {processing_time:.2f} 秒\n")
            f.write(f"总视频数: {len(results)}\n")
            f.write(f"成功: {report_data['successful']}\n")
            f.write(f"失败: {report_data['failed']}\n")
            f.write(f"成功率: {report_data['successful'] / len(results) * 100:.1f}%\n\n")

            f.write("处理参数:\n")
            for key, value in report_data['parameters'].items():
                f.write(f"  {key}: {value}\n")

            f.write("\n失败文件列表:\n")
            for result in results:
                if result['status'] == 'failed':
                    f.write(f"  {os.path.basename(result['video_path'])}: {result['error']}\n")

    def preview_processing(self, input_folder, num_preview=3):
        """预览处理效果"""
        input_folder = os.path.normpath(input_folder)
        video_files = self.get_video_files(input_folder)

        if not video_files:
            print("没有找到视频文件")
            return

        print(f"预览处理效果（前{num_preview}个视频）:")
        print("-" * 50)

        for i, video_path in enumerate(video_files[:num_preview]):
            try:
                video_info = self.extract_frames_uniform(video_path)
                super_image = self.create_super_image(video_info['frames'])

                print(f"{i + 1}. {os.path.basename(video_path)}")
                print(f"   原视频: {video_info['total_frames']}帧, "
                      f"{video_info['duration']:.1f}秒, "
                      f"{video_info['width']}x{video_info['height']}")
                print(f"   超图像: {super_image.shape[1]}x{super_image.shape[0]}")
                print(f"   采样帧: {video_info['frame_indices']}")
                print()

            except Exception as e:
                print(f"{i + 1}. {os.path.basename(video_path)} - 错误: {e}")
                print()


def main():
    """主函数"""
    # 初始化处理器
    processor = VideoSuperImageProcessor(
        target_frames=9,  # 9帧 -> 3x3网格
        frame_size=448  # 每帧224x224
    )

    # 设置您的文件夹路径
    input_folder = r"F:\QV-Data\complete_video\track1_raw_video"  # 修改为您的输入路径
    output_folder = r"D:\QV-Data2"  # 修改为您的输出路径

    try:
        # 先预览
        print("正在预览处理效果...")
        processor.preview_processing(input_folder, num_preview=2)

        # 确认是否继续
        response = input("是否继续处理所有视频？(y/n): ")
        if response.lower() != 'y':
            print("处理已取消")
            return

        # 开始处理
        print("开始处理所有视频...")
        results = processor.process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            max_videos=None  # 处理所有视频，可以设置数字来限制数量
        )

        print("\n处理完成！")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        print("请检查输入路径是否正确")


if __name__ == "__main__":
    # 添加欢迎信息
    print("=" * 60)
    print("视频超图像处理器")
    print("将视频转换为3x3网格的超图像")
    print("=" * 60)

    main()

    # 等待用户按键退出
    input("\n按回车键退出...")