def llava_conversation(conservation, role, content, is_image=False):
    """
    根据输入角色和内容更新Llava对话。

    :param dialogue: 当前对话列表
    :param role: 对话的角色 ('user' 或 'assistant'或'system')
    :param content: 内容，文本或图片
    :param is_image: 是否是图片内容，默认为 False
    """
    # 创建新的消息条目
    message = {
        "role": role,
        "content": []
    }

    if is_image:
        message["content"].append({"type": "image"})

    # 如果是文本内容，添加文本到消息
    if isinstance(content, str):
        message["content"].append({"type": "text", "text": content})
    elif isinstance(content, list):
        for c in content:
            if isinstance(c, str):
                message["content"].append({"type": "text", "text": c})
            elif isinstance(c, dict) and c.get("type") == "image":
                message["content"].append(c)

    # 将消息添加到对话中
    conservation.append(message)

if __name__ == "__main__":
    # 示例使用
    llava_dialogue = [ ]

    # 添加新的用户对话和回答
    llava_conversation(llava_dialogue, "user", "What about this image? How many birds do you see?", is_image=True)
    llava_conversation(llava_dialogue, "assistant", "There are three birds in the image, perched on a fence.")

    # 打印更新后的对话
    for message in llava_dialogue:
        print(f"Role: {message['role']}")
        for content in message['content']:
            if content['type'] == 'text':
                print(f"Text: {content['text']}")
            elif content['type'] == 'image':
                print("Image: <image>")
