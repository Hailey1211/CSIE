// 获取显示消息时间的 span 元素
var incomingTimeSpan = document.getElementById("incoming-time-1");
var outgoingTimeSpan = document.getElementById("outgoing-time-1");

// 定义一个函数来更新时间
function updateTime() {
    // 获取当前时间
    var currentTime = new Date();

    // 将时间格式化为所需的格式（如：18:06 PM | July 24）
    var formattedTime = currentTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }) + " | " + currentTime.toLocaleDateString('en-US', { month: 'long', day: 'numeric' });

    // 将格式化后的时间设置到 span 元素中
    incomingTimeSpan.textContent = formattedTime;
    outgoingTimeSpan.textContent = formattedTime;
}

// 每秒更新一次时间
setInterval(updateTime, 1000);

// 页面加载时调用一次以初始化时间
updateTime();
