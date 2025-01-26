import time
import sys
from transformers import GPT2Tokenizer
import openai

class GPT3Model(object):

    def __init__(self, model_name, api_key, logger=None):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger

    def do_inference(self, input, output, max_length=2048):
        losses = []
        data = input.strip() + "\n\n" + output.strip()  # 加入明确的分隔符

        response = self.gpt3(data)
        out = response["choices"][0]

        print(f"输入长度: {len(input)}")
        print(f"返回的 text_offset: {out['logprobs']['text_offset']}")
        print(f"完整返回的文本: {out['text']}")

        try:
            # 查找输入结束位置的偏移量
            i = out['logprobs']['text_offset'].index(len(input) - 1)
        except ValueError:
            # 如果找不到精确位置，回退到最近的偏移量
            closest_offset = max([offset for offset in out['logprobs']['text_offset'] if offset <= len(input) - 1],
                                 default=0)
            print(f"找不到确切匹配，使用最近的偏移量: {closest_offset}")
            try:
                i = out['logprobs']['text_offset'].index(closest_offset)
            except ValueError:
                print("错误：即使使用最近的偏移量，仍然无法找到索引。")
                return None

        # 打印用于调试的偏移信息
        print('评估文本:', out['logprobs']['tokens'][i: -1])

        # 计算平均损失
        try:
            loss = -sum(out['logprobs']["token_logprobs"][i:-1])  # 忽略最后标点符号
            avg_loss = loss / (len(out['logprobs']['text_offset']) - i - 1)
            print('平均损失: ', avg_loss)
            losses.append(avg_loss)
        except Exception as e:
            print(f"计算损失时出错: {e}")
            return None

        return avg_loss

    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(engine=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                print('prompt: ',prompt)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response

