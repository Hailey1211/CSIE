// 获取输入框元素
var userInput = document.querySelector(".form-control");

// 给输入框绑定 keydown 事件监听器
userInput.addEventListener("keydown", function(event) {
    // 检查用户是否按下了 Enter 键，并且输入框不为空
    if (event.key === "Enter" && userInput.value.trim() !== "") {
        // 调用发送消息函数，并将输入框的内容作为参数传递
        sendMessage(userInput.value);

        // 清空输入框
        userInput.value = "";
    }
});

// 获取发送图标元素
var sendIcon = document.querySelector(".send-icon");

// 给发送图标绑定点击事件监听器
sendIcon.addEventListener("click", function() {
    // 检查输入框不为空
    if (userInput.value.trim() !== "") {
        // 调用发送消息函数，并将输入框的内容作为参数传递
        sendMessage(userInput.value);

        // 清空输入框
        userInput.value = "";
    }
});

// 定义发送消息的函数
function sendMessage(message) {
    // 获取当前时间
    var currentTime = getCurrentTime();
    // 将时间格式化为所需的格式（如：18:06 PM | July 24）
    var formattedTime = currentTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }) + " | " + currentTime.toLocaleDateString('en-US', { month: 'long', day: 'numeric' });

    // 创建一个新的出站消息元素
    var newOutgoingChat = document.createElement("div");
    newOutgoingChat.classList.add("outgoing-chats");

    var outgoingChatsImg = document.createElement("div");
    outgoingChatsImg.classList.add("outgoing-chats-img");
    outgoingChatsImg.innerHTML = '<img src="https://img2.imgtp.com/2024/03/15/WVaaAPNU.jpg" />';

    var outgoingMsg = document.createElement("div");
    outgoingMsg.classList.add("outgoing-msg");

    var outgoingChatsMsg = document.createElement("div");
    outgoingChatsMsg.classList.add("outgoing-chats-msg");

    var messageParagraph = document.createElement("p");
    messageParagraph.classList.add("multi-msg");
    messageParagraph.textContent = message; // 这里需要获取输入框的内容，而不是将参数作为消息内容

    var timeSpan = document.createElement("span");
    timeSpan.classList.add("time");
    timeSpan.textContent = formattedTime;

    // 将消息内容和时间添加到消息元素中
    outgoingChatsMsg.appendChild(messageParagraph);
    outgoingChatsMsg.appendChild(timeSpan);

    // 将头像和消息元素添加到出站消息元素中
    outgoingMsg.appendChild(outgoingChatsMsg);
    newOutgoingChat.appendChild(outgoingChatsImg);
    newOutgoingChat.appendChild(outgoingMsg);

    // 将新的出站消息添加到聊天页面中
    var chatPage = document.querySelector(".msg-page");
    chatPage.appendChild(newOutgoingChat);

    // 调用情感识别函数处理输入
    handleInput(message);
}

function handleInput(message) {
    var inputText = message.trim();
    // 模拟情感识别并获取结果
    var emotionResult = simulateEmotionRecognition(inputText);
}

function simulateEmotionRecognition(input) {
    fetch('/chatRecognition', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: input }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json(); // 解析JSON响应
    })
    .then(result => {
        // 提取情感属性
        var emotion = result.emotion;
        // 调用 output 函数显示情感结果
        output("Emotion Recognition Result: " + emotion, getCurrentTime());
    })
    .catch(error => {
        console.error('There was a problem with your fetch operation:', error);
    });
}


function getCurrentTime() {
    var currentTime = new Date();
    return currentTime;
}

function output(message, time) {
    // 创建新的消息元素
    var newMessage = document.createElement("div");
    newMessage.classList.add("received-chats");

    var messageImg = document.createElement("div");
    messageImg.classList.add("received-chats-img");
    messageImg.innerHTML = '<img src="https://img2.imgtp.com/2024/03/15/WVaaAPNU.jpg" />';

    var messageContent = document.createElement("div");
    messageContent.classList.add("received-msg");

    var messageInbox = document.createElement("div");
    messageInbox.classList.add("received-msg-inbox");

    var messageParagraph = document.createElement("p");
    messageParagraph.textContent = message;

    var timeSpan = document.createElement("span");
    timeSpan.classList.add("time");
    timeSpan.textContent = time.toLocaleTimeString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }) + " | " + time.toLocaleDateString('en-US', { month: 'long', day: 'numeric' });

    // 将消息内容和时间添加到消息元素中
    messageInbox.appendChild(messageParagraph);
    messageInbox.appendChild(timeSpan);

    // 将头像和消息元素添加到消息容器中
    messageContent.appendChild(messageInbox);
    newMessage.appendChild(messageImg);
    newMessage.appendChild(messageContent);

    // 将新的消息添加到聊天页面中
    var chatPage = document.querySelector(".msg-page");
    chatPage.appendChild(newMessage);
}
