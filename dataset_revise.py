import os
import re


def extract_problem_files_from_log(log_text):
    """
    从日志文本中提取有问题的文件路径
    """
    # 匹配 WARNING 行中的图片路径
    pattern = r'WARNING ⚠️ (C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\.*?\.jpg):'
    problem_image_paths = re.findall(pattern, log_text)

    # 去重
    problem_image_paths = list(set(problem_image_paths))

    return problem_image_paths


def delete_problem_files(problem_image_paths):
    """
    删除有问题的图片文件和对应的标签文件
    """
    deleted_files = []

    for img_path in problem_image_paths:
        # 构建对应的标签文件路径 (.jpg -> .txt)
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

        # 删除图片文件
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                deleted_files.append(f"删除图片: {img_path}")
            except Exception as e:
                deleted_files.append(f"删除图片失败 {img_path}: {str(e)}")

        # 删除标签文件
        if os.path.exists(label_path):
            try:
                os.remove(label_path)
                deleted_files.append(f"删除标签: {label_path}")
            except Exception as e:
                deleted_files.append(f"删除标签失败 {label_path}: {str(e)}")

    return deleted_files


def main():
    # 你的日志内容
    log_content = """
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\labels.cache... 12880 images, 0 backgrounds, 5 corrupt: 100%|██████████| 12880/12880 00:00
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\0_Parade_Parade_0_452.jpg: ignoring corrupt image/label: negative label values [-0.00097656  -0.0013021]
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\2_Demonstration_Political_Rally_2_444.jpg: ignoring corrupt image/label: negative label values [-0.00097656  -0.0015873]
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\2_Demonstration_Protesters_2_231.jpg: 1 duplicate labels removed
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\37_Soccer_Soccer_37_851.jpg: 1 duplicate labels removed
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\39_Ice_Skating_iceskiing_39_380.jpg: ignoring corrupt image/label: negative label values [-0.00097656  -0.0012516]
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\46_Jockey_Jockey_46_576.jpg: ignoring corrupt image/label: negative label values [-0.00097656  -0.0015625]
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\58_Hockey_icehockey_puck_58_947.jpg: ignoring corrupt image/label: negative label values [-0.00067114]
train: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\train\\images\\7_Cheering_Cheering_7_17.jpg: 1 duplicate labels removed
val: Scanning C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\val\\labels.cache... 3226 images, 0 backgrounds, 5 corrupt: 100%|██████████| 3226/3226 00:00
val: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\val\\images\\0_Parade_Parade_0_275.jpg: ignoring corrupt image/label: negative label values [     -0.001     -0.0012]
val: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\val\\images\\21_Festival_Festival_21_604.jpg: 1 duplicate labels removed
val: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\val\\images\\37_Soccer_soccer_ball_37_281.jpg: ignoring corrupt image/label: negative label values [     -0.001     -0.0015]
val: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\val\\images\\50_Celebration_Or_Party_houseparty_50_715.jpg: ignoring corrupt image/label: negative label values [     -0.001     -0.0015]
val: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\val\\images\\7_Cheering_Cheering_7_171.jpg: ignoring corrupt image/label: negative label values [    -0.0015]
val: WARNING ⚠️ C:\\Users\\mc\\Desktop\\yolov5-7.0_revise\\WIDERFace_yolo\\val\\images\\7_Cheering_Cheering_7_426.jpg: ignoring corrupt image/label: negative label values [     -0.001     -0.0007]
    """

    # 提取有问题的文件
    print("正在提取有问题的文件路径...")
    problem_files = extract_problem_files_from_log(log_content)
    print(f"共找到 {len(problem_files)} 个有问题的图片文件")

    if not problem_files:
        print("未找到任何有问题的文件")
        return

    # 显示将要删除的文件
    print("\n即将删除以下文件：")
    for i, file_path in enumerate(problem_files, 1):
        print(f"{i}. {file_path}")

    # 确认删除
    confirm = input("\n是否确认删除这些文件及其对应的标签文件？(y/n): ")
    if confirm.lower() == 'y':
        # 执行删除
        deleted = delete_problem_files(problem_files)

        # 显示删除结果
        print("\n删除结果：")
        for msg in deleted:
            print(msg)
        print(f"\n操作完成！共处理 {len(deleted) // 2} 组文件（图片+标签）")
    else:
        print("取消删除操作")


if __name__ == "__main__":
    main()