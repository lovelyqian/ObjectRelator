import requests
from tqdm import tqdm
import time


def download_file(url, local_filename):
    # 发送HTTP GET请求
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # 如果请求失败，抛出异常

        # 获取文件总大小
        total_size = int(r.headers.get('content-length', 0))

        # 初始化进度条
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=local_filename)

        # 记录开始时间
        start_time = time.time()

        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉保持连接的空块
                    f.write(chunk)
                    progress_bar.update(len(chunk))

                    # 计算下载速度和剩余时间
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        download_speed = progress_bar.n / elapsed_time
                        remaining_time = (total_size - progress_bar.n) / download_speed
                        progress_bar.set_postfix({'剩余时间': f'{remaining_time:.2f}s'})

        progress_bar.close()
    return local_filename


# 模型文件的实际下载链接
url = "https://huggingface.co/EnmingZhang/PSALM/blob/main/model-00001-of-00002.safetensors"

# 本地保存的文件名
local_filename = "/data/work-gcp-europe-west4-a/yuqian_fu/Ego/huggingface/hub/PSALM/model-00001-of-00002.safetensors"

# 下载文件
download_file(url, local_filename)
print(f"文件已下载并保存为 {local_filename}")