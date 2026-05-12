import csv
import os

def parse_video_csv(file_path):
    """
    读取指定的CSV文件，并返回一个将文件名映射到文本描述的字典。

    Args:
        file_path (str): CSV文件的路径。

    Returns:
        dict: 一个字典，其中键(key)是从文件路径中提取的核心文件名，
              值(value)是对应的文本内容。
              如果文件未找到或发生错误，则返回 None。
    """
    video_data_dict = {}

    try:
        # 使用 'utf-8' 编码打开文件，这是最常见的编码方式
        # newline='' 是处理CSV文件的标准做法，可以防止出现空行
        with open(file_path, mode='r', encoding='utf-8', newline='') as infile:
            # csv.reader 会自动处理被引号包围的字段
            reader = csv.reader(infile)
            
            for row in reader:
                # 确保行不是空的，并且至少有两列
                if not row or len(row) < 2:
                    continue

                # --- 1. 处理第一列：提取文件名作为 Key ---
                full_path = row[0]
                base_name = os.path.basename(full_path)
                file_name, _ = os.path.splitext(base_name)
                
                # --- 2. 处理第二列：文本内容作为 Value ---
                text_content = row[1]
                
                # --- 3. 将键值对直接添加到字典中 ---
                # 假设文件名是唯一的，如果不是，后出现的会覆盖先出现的
                video_data_dict[file_name] = text_content

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return None
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return None

    return video_data_dict

# --- 如何使用 ---

if __name__ == "__main__":
    # 假设您的CSV文件名为 'data.csv'
    csv_file_path = 'data.csv' 
    
    # 调用函数直接获取字典
    video_data = parse_video_csv(csv_file_path)

    # 检查是否成功返回了字典
    if video_data:
        print(f"成功处理并创建了字典，包含 {len(video_data)} 个条目。\n")

        # --- 演示如何使用字典 ---

        # 1. 随机访问一个已知的键
        print("--- 示例1: 查询一个特定的文件名 ---")
        test_filename = "Peasant_Man_8661ee77f3f599eacf8b1df08e416c6c_animate_0_offset16_0080-0143"
        if test_filename in video_data:
            print(f"文件名: {test_filename}")
            # 为了简洁，只打印文本的前100个字符
            print(f"对应文本 (前100字符): {video_data[test_filename][:100]}...")
        else:
            print(f"未在字典中找到文件: {test_filename}")
        
        print("\n" + "="*40 + "\n")

        # 2. 遍历字典的前几项来预览
        print("--- 示例2: 预览字典的前3项 ---")
        # 将字典的 items 转换为列表以进行切片
        items_preview = list(video_data.items())[:3]
        for i, (filename, text) in enumerate(items_preview):
            print(f"[{i+1}] 文件名: {filename}")
            print(f"[{i+1}] 文本 (前50字符): {text[:50]}...\n")