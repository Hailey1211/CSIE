// 获取所需元素
var inputContentField = document.getElementById('input-content');
var inputEndBtn = document.getElementById('input-end-btn');
var receivedChatsContent = document.getElementById('received-chats-content');
var inputText = document.getElementById('input-text');

// 当用户点击输入结束按钮时，将输入的内容添加到 received-chats 中
inputEndBtn.addEventListener('click', function() {
    // 创建一个新的段落元素
    var newParagraph = document.createElement('p');
    // 设置新段落的文本内容为输入框的值
    newParagraph.textContent = inputText.value;
    // 将新段落添加到 received-chats 容器中
    receivedChatsContent.appendChild(newParagraph);
    // 将输入的内容存储在隐藏字段中，以备表单提交时使用
    inputContentField.value = inputText.value;
    // 清空输入框
    inputText.value = '';
    // 提交表单
    document.querySelector('form').submit();
});