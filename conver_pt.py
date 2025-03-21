import time

import torch
import argparse
from pathlib import Path
from tqdm import tqdm


def strip_optimizer(input_model='best.pt', out_model='out.pt'):

    # 处理模型
    x = torch.load(input_model, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']
    for k in ['optimizer', 'best_fitness', 'ema', 'updates']:
        x[k] = None
    x['epoch'] = -1
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False

    # 确定保存路径
    save_path = out_model or input_model
    torch.save(x, save_path)

    # 获取原始文件大小
    orig_size = Path(input_model).stat().st_size / 1e6  # MB
    # 获取新文件大小
    new_size = Path(save_path).stat().st_size / 1e6

    # 打印对比结果
    print(f"压缩完成: {orig_size:.1f}MB → {new_size:.1f}MB")
    print(f"保存优化后的模型 {out_model or input_model}, 大小为: {new_size:.1f} MB")


def strip_optimizer_visualization(input_model='best.pt', out_model='out.pt', progress=False):

    # 原始文件信息
    orig_path = Path(input_model)
    orig_size = orig_path.stat().st_size / 1e6  # MB

    # ====================== 处理阶段 ======================
    def update_pbar(pbar, step, desc):
        if progress and pbar:
            pbar.set_postfix_str(desc)
            pbar.update(step)
        elif not progress:
            print(f"▶ {desc}")

    time.sleep(0.0001)  # 模拟操作延迟

    # 主处理流程
    if progress:
        main_pbar = tqdm(total=3, desc="🛠️ 模型处理", unit="step")
    else:
        main_pbar = None

    # 1. 加载模型
    x = torch.load(input_model, map_location=torch.device('cpu'))
    update_pbar(main_pbar, 1, "模型加载完成")

    # 2. 处理EMA权重
    if x.get('ema'):
        x['model'] = x['ema']
        update_pbar(main_pbar, 1, "EMA权重已应用")
    else:
        update_pbar(main_pbar, 1, "未检测到EMA权重")

    # 3. 清理元数据
    for k in ['optimizer', 'best_fitness', 'ema', 'updates']:
        x[k] = None
    x['epoch'] = -1
    update_pbar(main_pbar, 1, "元数据清理完成")

    if main_pbar:
        main_pbar.close()

    # ====================== 转换阶段 ======================
    x['model'].half()  # 转换为FP16

    # 获取参数列表以确定总数
    model_params = list(x['model'].parameters())
    num_params = len(model_params)

    if progress:
        params = tqdm(model_params, desc="🔁 精度转换", unit="param", total=num_params)
    else:
        params = model_params
        print("▶ 开始精度转换")

    for p in params:
        p.requires_grad = False

    # ====================== 保存阶段 ======================
    save_path = Path(out_model) if out_model else orig_path

    if progress:
        with tqdm(total=1, desc="💾 保存模型") as save_pbar:
            torch.save(x, save_path)
            save_pbar.update(1)
    else:
        print("▶ 正在保存模型...")
        torch.save(x, save_path)

    # ====================== 结果输出 ======================
    new_size = save_path.stat().st_size / 1e6
    print(f"\n✅ 压缩完成: {orig_size:.1f}MB → {new_size:.1f}MB (-{100 * (orig_size - new_size) / orig_size:.1f}%)")
    print(f"📁 输出路径: {save_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='移除PyTorch模型中的优化器信息并压缩模型大小')
    parser.add_argument('--input', type=str, default='best.pt', help='输入模型路径')
    parser.add_argument('--output', type=str, default='out.pt', help='输出路径（留空则覆盖输入文件）')
    parser.add_argument('--progress', type=int, default= False, help='显示进度条')

    args = parser.parse_args()

    print(f"🚀 开始处理: {args.input}")
    #strip_optimizer(input_model=args.input, out_model=args.output)
    strip_optimizer_visualization(input_model=args.input, out_model=args.output, progress=args.progress)