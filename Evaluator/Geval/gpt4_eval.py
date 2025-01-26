import openai
import json
import argparse
import tqdm
import time


if __name__ == '__main__':

    test_sample = 10

    # 初始化命令行参数解析器
    argparser = argparse.ArgumentParser()
    # 设置命令行参数
    argparser.add_argument('--prompt_fp', type=str, default='prompts\summeval\con_detailed.txt',
                           help='包含评估提示模板的文件路径')
    argparser.add_argument('--save_fp', type=str, default='results\gpt4_con_detailed_openai.json',
                           help='保存结果的 JSON 文件路径')
    argparser.add_argument('--summeval_fp', type=str, default='data\summeval.json',
                           help='SummEval 数据集文件路径')
    argparser.add_argument('--key', type=str, required=True,
                           help='OpenAI API 密钥，必须提供')
    argparser.add_argument('--model', type=str, default='gpt-4-0613',
                           help='用于评估的 OpenAI 模型名称')
    args = argparser.parse_args()

    # 设置 OpenAI API 密钥
    openai.api_key = args.key

    # 加载 SummEval 数据集
    summeval = json.load(open(args.summeval_fp))
    # 读取提示模板
    prompt = open(args.prompt_fp).read()

    # 计数器初始化
    ct, ignore = 0, 0

    # 存储处理后的数据
    new_json = []

    # 遍历 SummEval 数据集中的每个实例
    for instance in tqdm.tqdm(summeval[0:test_sample]):
        source = instance['source']  # 获取原始文档
        system_output = instance['system_output']  # 获取系统生成的摘要
        # 根据模板替换占位符，生成当前实例的完整提示
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt  # 将生成的提示保存到当前实例中

        while True:
            try:
                # 调用 OpenAI ChatCompletion 接口生成评估结果
                _response = openai.ChatCompletion.create(
                    model=args.model,  # 使用的模型名称
                    messages=[{"role": "system", "content": cur_prompt}],  # 提供的对话上下文
                    temperature=2,  # 生成文本的随机性，值越高越随机
                    max_tokens=5,  # 最大生成的 token 数
                    top_p=1,  # nucleus sampling 的概率阈值
                    frequency_penalty=0,  # 减少重复生成的权重
                    presence_penalty=0,  # 增加生成新内容的权重
                    stop=None,  # 不设置停止符
                    n=20  # 生成的候选数量
                )
                time.sleep(0.5)  # 避免 API 调用频率过高

                # 提取所有生成的候选响应
                all_responses = [_response['choices'][i]['message']['content'] for i in
                                 range(len(_response['choices']))]
                instance['all_responses'] = all_responses  # 保存所有响应到实例中
                new_json.append(instance)  # 将实例加入结果列表
                ct += 1  # 成功处理计数器 +1
                break
            except Exception as e:
                # 捕获异常，打印错误信息
                print(e)
                if ("limit" in str(e)):  # 如果是调用频率限制错误
                    time.sleep(2)  # 等待 2 秒后重试
                else:  # 其他错误，跳过当前实例
                    ignore += 1
                    print('ignored', ignore)
                    break

    print('ignored total', ignore)  # 打印忽略的实例总数
    # 将所有结果保存到指定的 JSON 文件
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
