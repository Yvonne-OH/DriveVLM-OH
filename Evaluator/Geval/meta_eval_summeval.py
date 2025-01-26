from scipy.stats import spearmanr, pearsonr, kendalltau
import json
import re
import argparse
from prettytable import PrettyTable


# 计算预测分数和人工分数的相关性
def calculate_correlation(pred_score, human_score, result):
    # 确保预测分数和人工分数长度一致
    assert len(pred_score) == len(human_score)

    # 初始化相关性结果字典
    if len(result) == 0:
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}

    # 计算 Pearson、Spearman 和 Kendalltau 相关系数
    result['pearson'] += pearsonr(pred_score, human_score)[0]
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result


# 打印最终的相关性结果
def print_correlations(result, n):
    # 创建表格显示相关性结果
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    if n == 0:
        n = 1
    # 计算平均相关性值并添加到表格
    table.add_row(
        [round(result['pearson'] / n, 4), round(result['spearman'] / n, 4), round(result['kendalltau'] / n, 4)])
    print(table)


# 从字符串输出中解析得分
def parse_output(output):
    # 使用正则表达式提取浮点数
    matched = re.search("^ ?([\d\.]+)", output)
    if matched:
        try:
            score = float(matched.group(1))
        except:
            score = 0  # 如果解析失败，返回 0
    else:
        score = 0  # 如果未匹配到，返回 0
    return score


if __name__ == '__main__':
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fp', type=str, default='results/gpt4_rel_detailed.json',
                        help='包含评估结果的输入 JSON 文件路径')
    parser.add_argument('--dimension', type=str, default='relevance',
                        help='评估的维度，例如 relevance 或 fluency')
    args = parser.parse_args()

    # 加载 JSON 文件
    jobj = json.load(open(args.input_fp))
    pred_scores, human_scores = {}, {}

    print("Calculating correlation for G-Eval")
    # 遍历 JSON 文件中的每个项目
    for item in jobj:
        doc_id = item["doc_id"]  # 获取文档 ID
        if doc_id not in pred_scores:
            pred_scores[doc_id] = []  # 初始化预测分数列表
            human_scores[doc_id] = []  # 初始化人工分数列表

        # 获取所有 GPT 生成的响应
        all_responses = item["all_responses"]
        # 解析每个响应的分数
        all_scores = [parse_output(x) for x in all_responses]
        # 计算当前文档的平均预测分数
        score = sum(all_scores) / len(all_scores)

        # 将分数添加到对应的字典中
        pred_scores[doc_id].append(score)
        human_scores[doc_id].append(item['scores'][args.dimension])  # 使用指定维度的人工分数

    # 打印预测分数和人工分数的总长度
    print('len(pred_scores): {}'.format(len(pred_scores)))
    print('len(human_scores): {}'.format(len(human_scores)))

    # 初始化相关性结果
    results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    d_ctr = 0  # 用于计数有效文档的计数器
    for doc_id in pred_scores:
        pred_scores_doc = pred_scores[doc_id]  # 当前文档的预测分数
        human_scores_doc = human_scores[doc_id]  # 当前文档的人工分数
        # 跳过无效分数（如所有分数相同）
        if (len(set(human_scores_doc)) <= 1) or (len(set(pred_scores_doc)) <= 1):
            continue

        # 计算当前文档的相关性
        results = calculate_correlation(pred_scores_doc, human_scores_doc, results)
        d_ctr += 1  # 有效文档计数器 +1

    # 打印最终的相关性结果
    print_correlations(results, n=d_ctr)
