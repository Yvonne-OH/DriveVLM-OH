import argparse  # 用于解析命令行参数
import os  # 文件操作和路径管理
import time  # 用于计时和性能分析
import numpy as np  # 科学计算库，主要用于数组和数值运算
from utils import *  # 自定义工具函数，用于数据加载和保存
from gpt3_score import gpt3score  # 自定义 GPT3 评分模块
from transformers import GPT2Tokenizer  # 从 HuggingFace 的 transformers 库导入 GPT2 分词器
import json  # 用于 JSON 数据处理

"""
代码实现了一个多模型评估框架，可以使用 GPT3 系列、OPT 系列、GPT2 系列和 FLAN-T5 系列模型
来对数据进行打分，支持不同指标（如质量、自然性、信息量）的评估。
"""

class Scorer:
    """ 评分器类：支持多种语言模型（GPT3、OPT、GPT2、FLAN-T5）的评分 """
    def __init__(self, args=None):
        """
        初始化评分器类。
        参数：
            args: 从命令行解析的参数
        """
        self.args = args  # 保存命令行参数
        self.device = self.args.device  # 运行设备（如 "cuda:0"）
        self.eval_asp = self.args.aspect  # 要评估的指标（如 "quality"）
        self.data = read_pickle(self.args.file_path)  # 从文件路径加载数据
        self.demos, self.asp_dfs = read_demos(self.args.demo_path)  # 加载示例数据和指标描述

        # 优先对少量数据进行评估（减少开销）
        print('由于 GPT3 模型调用成本较高，我们可以先测试少量样本。')
        print('默认测试样本数量为 2。')
        import random
        random.seed(2)  # 设置随机种子
        N = 2  # 选取样本的数量
        idxs = random.sample(range(0, len(self.data) - 1), N)  # 随机选取样本索引
        new_data = {idx: self.data[idx] for idx in idxs}  # 创建子集数据
        self.data = new_data  # 更新数据为子集
        print('评估样本数量: ', len(self.data))

    def save_data(self, path):
        """ 保存评估结果到文件 """
        save_pickle(self.data, path)

    def demo_convert(self, demos, template):
        """
        转换示例数据为特定模板格式，用于生成评分的输入。
        参数：
            demos: 示范数据（包含源文本、参考摘要和生成摘要等）
            template: 模板字符串，用于格式化示范数据
        返回：
            refhyp_demos: 参考到生成的示例
            hypref_demos: 生成到参考的示例
        """
        refhyp_demos = []  # 保存 "参考 -> 生成" 格式的示例
        hypref_demos = []  # 保存 "生成 -> 参考" 格式的示例
        for demo in demos:
            src_line = demo["src"].strip()  # 源文本
            ref_line = demo["ref_summ"].strip()  # 参考摘要
            hyp_line = demo["sys_summ"].strip()  # 系统生成摘要
            polar = demo["polarity"].strip()  # 极性描述
            # 使用模板替换占位符生成示例
            refhyp_demo = template.replace("XXXXX", ref_line).replace("YYYYY", hyp_line)
            refhyp_demos.append(refhyp_demo)
            hypref_demo = template.replace("XXXXX", hyp_line).replace("YYYYY", ref_line)
            hypref_demos.append(hypref_demo)
        return refhyp_demos, hypref_demos

    def score(self, metrics):
        """
        根据指定的评分指标列表，对数据进行打分。
        参数：
            metrics: 指标列表，例如 ["gpt3_score", "flan_small_score"]
        """
        for metric_name in metrics:

            if metric_name == 'gpt3_score':
                """ 使用 GPT3 模型进行评分 """
                print(f'执行 GPT3 模型评分...')
                start = time.time()  # 记录开始时间
                print('样本数量: ', len(self.data))
                demo = self.demos[self.eval_asp]  # 加载示范数据
                asp_df = self.asp_dfs[self.eval_asp]  # 获取指标描述
                print('示范数据: ', demo)
                print('指标描述: ', asp_df)
                refhyp_templates = ["XXXXX In other words , \nYYYYY", ]  # 模板
                template = refhyp_templates[0]  # 选择第一个模板
                refhyp_demos, hypref_demos = self.demo_convert(demo, template)  # 转换示例
                for samp_id, doc_id in enumerate(self.data):
                    print('样本 ID: ', samp_id)
                    ref_summs = self.data[doc_id]['ref_summs']  # 加载参考摘要
                    ref_summs = [detokenize(line) for line in ref_summs]  # 去掉标记化
                    sys_summ = detokenize(self.data[doc_id]['sys_summ'])  # 去掉标记化
                    ref_hypo_scores = []  # 保存参考到生成的分数
                    hypo_ref_scores = []  # 保存生成到参考的分数
                    keep_seen_refsumm_score = {}  # 缓存已经计算过的参考摘要分数

                    for k, ref_summ in enumerate(ref_summs):
                        print(f'当前评估指标: {self.eval_asp}; 样本 ID: {samp_id}; 参考摘要索引: {k}/{len(ref_summs)}')
                        ref_summ = add_dot(ref_summ)  # 确保末尾有句号
                        sys_summ = add_dot(sys_summ)  # 确保末尾有句号
                        if ref_summ in keep_seen_refsumm_score:
                            # 如果参考摘要已经计算过，跳过重复计算
                            ref_hypo_score = keep_seen_refsumm_score[ref_summ][0]
                            hypo_ref_score = keep_seen_refsumm_score[ref_summ][1]
                            ref_hypo_scores.append(ref_hypo_score)
                            hypo_ref_scores.append(hypo_ref_score)

                        else:
                            # 参考到生成评分
                            if self.args.use_ist and self.args.use_demo:
                                refhyp_demos_str = "\n\n".join(refhyp_demos)
                                prefix = asp_df + '\n\n' + refhyp_demos_str + '\n\n'
                            elif self.args.use_ist and not self.args.use_demo:
                                prefix = asp_df + '\n'
                            elif not self.args.use_ist and not self.args.use_demo:
                                prefix = ''

                            input1 = template.replace("XXXXX", ref_summ).replace("YYYYY", "")
                            input1 = prefix + input1
                            output1 = lower_check(sys_summ)
                            ref_hypo_score = gpt3score(input1, output1, self.args.gpt3model, self.args.api_key)
                            ref_hypo_scores.append(ref_hypo_score)
                            # 生成到参考评分
                            input2 = template.replace("XXXXX", sys_summ).replace("YYYYY", "")
                            input2 = prefix + input2
                            output2 = lower_check(ref_summ)
                            hypo_ref_score = gpt3score(input2, output2, self.args.gpt3model, self.args.api_key)
                            hypo_ref_scores.append(hypo_ref_score)
                            keep_seen_refsumm_score[ref_summ] = [ref_hypo_score, hypo_ref_score]

                    # 转换为 NumPy 数组并计算分数
                    ref_hypo_scores = np.array(ref_hypo_scores)
                    hypo_ref_scores = np.array(hypo_ref_scores)
                    ref_hypo = ref_hypo_scores.max()
                    hypo_ref = hypo_ref_scores.max()
                    avg_f = (0.5 * (ref_hypo_scores + hypo_ref_scores)).max()
                    harm_f = (ref_hypo_scores * hypo_ref_scores / (ref_hypo_scores + hypo_ref_scores)).max()
                    print('参考到生成最大分数: ', ref_hypo)
                    print('生成到参考最大分数: ', hypo_ref)
                    print('平均值: ', avg_f)
                    print('调和平均值: ', harm_f)

                    # 保存分数到数据
                    if self.args.use_ist:
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_ref_hypo'] = ref_hypo
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_hypo_ref'] = hypo_ref
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_avg_f'] = avg_f
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_harm_f'] = harm_f

                    else:
                        self.data[doc_id]['scores'][f'{metric_name}_ref_hypo'] = ref_hypo
                        self.data[doc_id]['scores'][f'{metric_name}_hypo_ref'] = hypo_ref
                        self.data[doc_id]['scores'][f'{metric_name}_avg_f'] = avg_f
                        self.data[doc_id]['scores'][f'{metric_name}_harm_f'] = harm_f

                print(f'GPT3 模型评分完成，耗时 {time.time() - start}s。')


            else:
                raise NotImplementedError

def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file_path', type=str, default='XXX',
                        help='The data to load from.')
    parser.add_argument('--demo_path', type=str, default='XXX',
                        help='The demonstrated samples to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--dataname', type=str, default='BAGEL',
                        required=False, help='The name of the evaluated dataset.')
    parser.add_argument('--aspect', type=str, default='Engaging',
                        required=False, help='The evaluated aspect considered.')
    parser.add_argument('--gpt3model', type=str, default='ada',
                        required=False, help='Set which GPT3-based model to use.')
    parser.add_argument('--api_key', type=str, default='YOUR_OPENAI_API_KEY',
                        required=False, help='The OPENAI API key.')
    parser.add_argument('--use_ist', type=str2bool, default=False,
                        required=True, help='If set to True, use instruction.')
    parser.add_argument('--use_demo', type=str2bool, default=False,
                        required=False, help='If set to True, use demonstrated samples.')
    parser.add_argument('--output', type=str, default="XXXX",
                        help='The output path to save the calculated scores.')
    parser.add_argument('--out_dir_name', type=str, default="XXXX",
                        required=False, help='The output folder name to save the calculated scores.')

    parser.add_argument('--gpt3_score', type=str2bool,  default=False,
                        help='Whether to calculate gpt3_score.')
    parser.add_argument('--opt125m_score', type=str2bool, default=False,
                        help='Whether to calculate facebook/opt-125m.')
    parser.add_argument('--opt350m_score', type=str2bool, default=False,
                        help='Whether to calculate facebook/opt-350m.')
    parser.add_argument('--opt1_3B_score', type=str2bool, default=False,
                        help='Whether to calculate facebook/opt-1.3b.')
    parser.add_argument('--opt2_7B_score', type=str2bool, default=False,
                        help='Whether to calculate opt2_7B_score.')
    parser.add_argument('--opt6_7B_score', type=str2bool, default=False,
                        help='Whether to calculate opt6_7B_score.')
    parser.add_argument('--opt13B_score', type=str2bool, default=False,
                        help='Whether to calculate opt13B_score.')
    parser.add_argument('--opt66B_score', type=str2bool, default=False,
                        help='Whether to calculate opt66B_score.')
    parser.add_argument('--gpt2_medium_score', type=str2bool, default=False,
                        help='Whether to calculate gpt2_medium_score.')
    parser.add_argument('--gpt2_large_score', type=str2bool, default=False,
                        help='Whether to calculate gpt2_large_score.')
    parser.add_argument('--gpt2_xl_score', type=str2bool, default=False,
                        help='Whether to calculate gpt2_xl_score.')
    parser.add_argument('--gptJ6B_score', type=str2bool, default=False,
                        help='Whether to calculate gptJ6B_score.')
    parser.add_argument('--flan_small_score', type=str2bool, default=False,
                        help='Whether to calculate flan_small_score.')
    parser.add_argument('--flan_base_score', type=str2bool, default=False,
                        help='Whether to calculate flan_base_score.')
    parser.add_argument('--flan_large_score', type=str2bool, default=False,
                        help='Whether to calculate flan_large_score.')
    parser.add_argument('--flan_xl_score', type=str2bool, default=False,
                        help='Whether to calculate flan_xl_score.')
    parser.add_argument('--flan_xxl_score', type=str2bool, default=False,
                        help='Whether to calculate flan_xxl_score.')
    args = parser.parse_args()


    METRICS = []
    if args.gpt3_score:
        METRICS.append('gpt3_score')
    if args.opt350m_score:
        METRICS.append('opt350m_score')
    if args.opt1_3B_score:
        METRICS.append('opt1_3B_score')
    if args.opt6_7B_score:
        METRICS.append('opt6_7B_score')
    if args.opt13B_score:
        METRICS.append('opt13B_score')
    if args.opt66B_score:
        METRICS.append('opt66B_score')

    if args.flan_small_score:
        METRICS.append('flan_small_score')
    if args.flan_base_score:
        METRICS.append('flan_base_score')
    if args.flan_large_score:
        METRICS.append('flan_large_score')
    if args.flan_xl_score:
        METRICS.append('flan_xl_score')
    if args.flan_xxl_score:
        METRICS.append('flan_xxl_score')

    if args.gpt2_medium_score:
        METRICS.append('gpt2_medium_score')
    if args.gpt2_large_score:
        METRICS.append('gpt2_large_score')
    if args.gpt2_xl_score:
        METRICS.append('gpt2_xl_score')
    if args.gptJ6B_score:
        METRICS.append('gptJ6B_score')

    print('METRICS: ',METRICS)

    out_dir1 = args.out_dir_name

    data_dir = "datas/meta_datas/d2t/"
    demon_dir = "datas/demos/d2t/"
    out_dir = './analysis/d2t/'+out_dir1
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_name = args.dataname
    print('##### eval data_name: ', data_name)
    print()
    file_path = data_dir + data_name+'/data.pkl'
    args.file_path = file_path
    args.demo_path =demon_dir + data_name + '_demos.json'

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # args.output = out_dir + '/' + data_name + '_15plainBaseline_score.pkl'
    args.output = out_dir + '/' + data_name + args.out_dir_name + '_usedemo[' + str(args.use_demo) + ']' + '_useist[' + str(args.use_ist) + ']_OptFlanGpt2_score.pkl'
    # args.output = out_dir + '/' + data_name + '_' + args.aspect + '_' + args.out_dir_name + '_usedemo[' + str(args.use_demo) + ']' + '_useist[' + str(args.use_ist) + ']_' + args.gpt3model + '.pkl'
    print('args.output: ', args.output)

    scorer = Scorer(args)
    scorer.score(METRICS)
    scorer.save_data(args.output)

if __name__ == '__main__':
    main()

