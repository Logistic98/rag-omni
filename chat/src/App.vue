<template>
  <div style="height: 100vh; display: flex; justify-content: center; align-items: center; background-color: rgb(223, 223, 227)">
    <div class="chat">
      <div class="chat_left">
        <div class="chat_left_head">聊天记录</div>
        <div class="chat_left_body">
          <div v-for="(item, index) in messages" :key="'record_card' + index" :class="item.id === currentId ? 'record_card card_choose' : 'record_card'" @click="recordCardClick(item)">
            <div class="record_card_title">
              <span v-if="!item.edit">{{ item.name }}</span>
              <Input v-else v-model="item.name" placeholder="请输入名称" style="width: 150px" @change="saveMessages" />
              <Button shape="circle" type="primary" icon="md-create" size="small" style="position: absolute; top: 7px; right: 38px; height: 26px; width: 26px" v-if="!item.edit" @click.stop="changeMessageEditStatus(item, true)" />
              <Button shape="circle" type="success" icon="md-checkmark" size="small" style="position: absolute; top: 7px; right: 38px; height: 26px; width: 26px" v-else @click.stop="changeMessageEditStatus(item, false)" />
              <Button shape="circle" type="error" icon="ios-trash" size="small" style="position: absolute; top: 7px; right: 7px; height: 26px; width: 26px" @click.stop="deleteRecord(index, item.id)" />
            </div>
            <div class="record_card_info">
              <div>{{ item.value.length }}&nbsp;条对话</div>
              <div>{{ item.time }}</div>
            </div>
          </div>
        </div>
        <div class="chat_left_foot">
          <Button icon="md-add" class="chat_left_foot_button" @click="addNewDialog('新的聊天')">新的聊天</Button>
        </div>
      </div>
      <div class="chat_right">
        <div class="chat_right_head">
          <div style="line-height: 65px; font-size: 20px; font-weight: bolder;">{{ currentRecord.name }}</div>
          <div style="font-size: 12px; position: relative; top: -16px;" v-if="currentRecord.sending">正在生成回复...</div>
        </div>
        <div class="chat_right_body" ref="messages">
          <div v-for="(message, index) in currentRecord.value" :key="'message' + index" :class="message.isSelf ? 'message self' : 'message'">
            <img v-bind:src="message.avatar" class="avatar" />
            <div :class="message.isSelf ? 'content self' : 'content'">
              <div class="name">{{ message.name }}</div>
              <div style="margin: auto 0 10px auto;" v-if="!message.isSelf">
                <span style="margin: 10px">渲染</span>
                <i-switch v-model="message.isMarkdown" size="default"></i-switch>
              </div>
              <VueMarkdown v-if="message.isMarkdown && !message.loading" class="markdown markdown-body" :source="message.text" />
              <div class="text markdown-body" v-else-if="!message.isMarkdown && !message.loading" v-html="message.text"></div>
              <div class="text_log" v-else>
                <div class="log-loading">
                  <div class="dot-flashing"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <Button shape="circle" style="position: absolute; bottom: 120px; right: 20px" @click="stopGenerate" v-show="currentRecord.sending" type="info">停止生成</Button>
        <div class="chat_right_send">
          <div class="chat_input_action">
            <div :class="currentRecord.sending ? 'voice-input-button-wrapper disable' : 'voice-input-button-wrapper'">
              <voice-input-button
                color="#fff"
                appId="d9ecd998"
                apiKey="f57978586f51f6c9da23b227c29eb3d0"
                apiSecret="ZmNkMzBkMTRlZDRmZWQ1Y2NiMmJiNjVj"
                v-model="text"
                @record="showResult"
                @record-start="recordStart"
                @record-stop="recordStop"
                @record-blank="recordNoResult"
                @record-failed="recordFailed"
                @record-ready="recordReady"
                @record-complete="recordComplete"
              >
                <template slot="no-speak">没听清您说的什么</template>
              </voice-input-button>
            </div>
          </div>
          <div>
            <textarea class="chat_input_area" v-model="text" @keyup.enter="sendMessage"></textarea>
            <Button class="chat_input_button" type="success" shape="circle" icon="ios-send" @click="sendMessage" :disabled="currentRecord.sending">发送</Button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import voiceInputButton from 'voice-input-button2';
import 'github-markdown-css';
import VueMarkdown from 'vue-markdown';

export default {
  components: {
    VueMarkdown,
    voiceInputButton
  },
  mounted() {
    // 使用浏览器localstorage存储之前的对话信息
    this.storageKey = 'chatInfo';
    if (localStorage.getItem(this.storageKey) !== null) {
      // 如果有存储历史信息的话取出
      this.messages = JSON.parse(localStorage.getItem(this.storageKey));
      this.messages.forEach(item => {
        item.edit = false;
        item.sending = false;
      });
      // 选中第一个对话
      this.recordCardClick(this.messages[0]);
    } else {
      // 没有历史信息则自动添加一个空的对话信息
      this.addNewDialog('新的聊天');
    }
  },
  data() {
    return {
      messages: [], // 存放所有的对话信息
      currentRecord: [], // 当前选中的对话的信息
      currentId: 0, // 当前选中的对话的id
      options: {
        apiUrl: 'http://127.0.0.1:5002/api/rag/summary', // 更新为新的API URL
        serverName: '机器人',
        userName: '我',
        prologue: '你好，请问有什么可以帮助您?',
        paramName: '输入'
      },
      text: '',
      storageKey: '' // 存放在localstorage中的key
    };
  },
  methods: {
    saveMessages() {
      window.localStorage.setItem(this.storageKey, JSON.stringify(this.messages));
    },
    deleteRecord(index, id) {
      this.messages.splice(index, 1);
      if (this.currentRecord.id === id) {
        if (this.messages.length >= index + 1) {
          this.currentRecord = this.messages[index];
        } else if (index > 0) {
          this.currentRecord = this.messages[index - 1];
        } else {
          this.addNewDialog('新的聊天');
        }
      }
      this.saveMessages();
    },
    changeMessageEditStatus(item, status) {
      item.edit = status;
    },
    recordCardClick(item) {
      this.currentRecord = item;
      this.currentId = item.id;
    },
    stopGenerate() {
      let lastIndex = this.currentRecord.value.length - 1;
      this.currentRecord.value[lastIndex].text = '已停止生成';
      this.currentRecord.value[lastIndex].loading = false;
      this.currentRecord.sending = false;
      this.saveMessages();
    },
    addNewDialog(title) {
      let unique = 0;
      const time = Date.now();
      const random = Math.floor(Math.random() * 1000000000);
      // eslint-disable-next-line no-undef
      unique++;
      let newId = random + unique + String(time);
      let nowData = new Date();
      let nowTime = nowData.getFullYear() + '/' + (nowData.getMonth() + 1) + '/' + nowData.getDate() + ' ' + nowData.getHours() + ':' + nowData.getMinutes() + ':' + nowData.getSeconds();
      let newRecord = {
        id: newId,
        name: title,
        time: nowTime,
        edit: false,
        sending: false,
        value: [
          {
            id: this.messages.length + 1,
            name: this.options.serverName,
            avatar: require('./assets/robot.png'),
            text: this.options.prologue,
            isSelf: false,
            isMarkdown: false,
            loading: false
          }
        ]
      };
      this.messages.push(newRecord);
      this.currentRecord = newRecord;
      this.currentId = newId;
      console.log('this.messages: ', this.messages);
      console.log('this.currentRecord: ', newRecord);
      this.saveMessages();
    },
    scrollToBottom() {
      this.$nextTick(() => {
        let scrollElem = this.$refs.messages;
        scrollElem.scrollTo({
          top: scrollElem.scrollHeight,
          behavior: 'smooth'
        });
      });
    },
    recordReady() {
      console.info('按钮就绪!');
    },
    recordStart() {
      console.info('录音开始');
    },
    showResult(text) {
      console.info('收到识别结果：', text);
    },
    recordStop() {
      console.info('录音结束');
    },
    recordNoResult() {
      console.info('没有录到什么，请重试');
    },
    recordComplete() {
    },
    recordFailed(error) {
      console.info('识别失败，错误栈：', error);
    },
    sendMessage() {
      if (!this.options.apiUrl) {
        alert('请先输入地址！');
        return;
      }
      if (this.currentRecord.sending || this.text.replaceAll(' ', '') === '' || this.text.replaceAll(' ', '').replaceAll('\n', '') === '') {
        return;
      }
      let newMessage = {
        id: this.messages.length + 1,
        name: this.options.userName,
        avatar: require('./assets/people.png'),
        text: this.text,
        isSelf: true,
        isMarkdown: false
      };
      this.currentRecord.value.push(newMessage);
      let chooseRecord = [];
      this.messages.forEach(message => {
        if (message.id === this.currentId) {
          chooseRecord = message;
        }
      });
      chooseRecord.value.push({
        id: this.messages.length + 1,
        name: this.options.serverName,
        avatar: require('./assets/robot.png'),
        text: '',
        isSelf: false,
        loading: true,
        isMarkdown: true
      });
      let sendMsg = this.text;
      this.text = '';
      chooseRecord.sending = true;
      this.scrollToBottom();
      this.getSummary(this.options.apiUrl, sendMsg, this.generateSessionId());
    },
    async getSummary(url, user_prompt, session_id) {
      let headers = {
        'Content-Type': 'application/json'
      };
      let data = {
        user_prompt: user_prompt,
        session_id: session_id
      };
      try {
        let response = await fetch(url, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify(data)
        });
        let result = await response.json();
        let chooseRecord = [];
        this.messages.forEach(message => {
          if (message.id === this.currentId) {
            chooseRecord = message;
          }
        });
        let lastIndex = chooseRecord.value.length - 1;
        chooseRecord.value[lastIndex].text = result.data.response || '未返回任何内容';
        chooseRecord.value[lastIndex].loading = false;
        chooseRecord.sending = false;
        this.saveMessages();
      } catch (error) {
        console.error('Error:', error);
        let chooseRecord = [];
        this.messages.forEach(message => {
          if (message.id === this.currentId) {
            chooseRecord = message;
          }
        });
        let lastIndex = chooseRecord.value.length - 1;
        chooseRecord.value[lastIndex].text = '请求失败，请重试';
        chooseRecord.value[lastIndex].loading = false;
        chooseRecord.sending = false;
        this.saveMessages();
      }
    },
    generateSessionId() {
      return 'session_' + Math.random().toString(36).substr(2, 9);
    }
  }
};
</script>

<style scoped>

/* 整个滚动条 */
::-webkit-scrollbar {
  /* 对应纵向滚动条的宽度 */
  width: 5px;
  /* 对应横向滚动条的宽度 */
  height: 5px;
}

/* 滚动条上的滚动滑块 */
::-webkit-scrollbar-thumb {
  background-color: #b2b2b2;
  border-radius: 32px;
}

/* 滚动条轨道 */
::-webkit-scrollbar-track {
  background-color: #dbeffd;
  border-radius: 32px;
}

.card_choose {
  border: 2px solid #1d93ab !important;
}
.chat {
  background-color: #F4F4F4;
  border-radius: 20px;
  box-shadow: 0 0 20px rgba(128, 128, 128, 1);
  overflow: hidden;
  text-align: left;
  width: 1200px;
  height: 95%;
}
.chat_left {
  float: left;
  height: 100%;
  top: 0;
  width: 280px;
  box-sizing: border-box;
  padding: 20px;
  background-color: #e7f8ff;
  display: flex;
  flex-direction: column;
  box-shadow: inset -2px 0 2px 0 rgba(0,0,0,.05);
  position: relative;
  transition: width .05s ease;
}
.chat_left_head {
  position: relative;
  padding-top: 20px;
  padding-bottom: 20px;
  font-size: 20px;
  font-weight: 700;
  animation: home_slide-in__gYZA0 .3s ease;
}
.chat_left_body {
  flex: 1 1;
  overflow: auto;
  overflow-x: hidden;
}
.chat_left_foot {
  padding-top: 20px;
}
.chat_left_foot_button {
  background-color: #fff;
  border-radius: 20px;
  outline: none;
  border: none;
  color: #303030;
  float: right;
  height: 40px;
}
.chat_left_foot_button:hover {
  background-color: #f3f3f3;
}
.chat_right {
  float: left;
  width: calc(100% - 300px);
  background-color: #fff;
  display: flex;
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}
.chat_right_body {
  flex: 1 1;
  overflow: auto;
  padding: 20px 20px 40px;
  position: relative;
  overscroll-behavior: none;
}
.chat_right_send {
  position: relative;
  width: 100%;
  padding: 10px 20px 20px;
  box-sizing: border-box;
  flex-direction: column;
  border-top-left-radius: 10px;
  border-top-right-radius: 10px;
  border-top: 1px solid #dedede;
  box-shadow: 0px 2px 4px 0px rgba(0,0,0,.05);
}
.chat_right_head {
  border-bottom: 1px solid rgba(0,0,0,.1);
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: center;
  text-align: center;
  height: 70px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: block;
}
.chat_input_action {
  position: absolute;
  top: 30px;
  left: 10px;
}
.chat_input_area {
  height: 80px;
  width: calc(100% - 35px);
  margin-left: 40px;
  border-radius: 10px;
  border: 1px solid #dedede;
  box-shadow: 0 -2px 5px rgba(0,0,0,.03);
  background-color: white;
  color: black;
  font-family: inherit;
  padding: 10px 90px 10px 14px;
  resize: none;
  outline: none;
}
.chat_input_area:focus {
  border: 1px solid #1d93ab;
}
.chat_input_button {
  position: absolute;
  right: 30px;
  bottom: 32px
}
.record_card {
  padding: 10px 14px;
  background-color: #fff;
  border-radius: 10px;
  margin-bottom: 10px;
  box-shadow: 0px 2px 4px 0px rgba(0,0,0,.05);
  transition: background-color .3s ease;
  cursor: pointer;
  -webkit-user-select: none;
  -moz-user-select: none;
  user-select: none;
  border: 2px solid transparent;
  position: relative;
}
.record_card:hover {
  background-color: #f3f3f3;
}
.record_card_selected {
  border-color: #1d93ab;
}
.record_card_title {
  font-size: 14px;
  font-weight: bolder;
  display: block;
  width: 200px;
  height: 32px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  animation: home_slide-in__gYZA0 .3s ease;
}
.record_card_info {
  display: flex;
  justify-content: space-between;
  color: #a6a6a6;
  font-size: 12px;
  animation: home_slide-in__gYZA0 .3s ease;
}
.voice-input-button-wrapper {
  width: 40px;
  height: 40px;
  font-size: 12px;
  background-color: #07C160;
  border-radius: 50%;
}

.voice-input-button-wrapper.disable {
  background-color: #999;
}

.messages {
  height: 84%;
  overflow-y: auto;
}

.message {
  display: flex;
  margin: 10px 20px;
  padding: 10px;
}

/deep/ .el-collapse {
  font-size: 18px;
  background-color: transparent;
}

.message.self {
  flex-direction: row-reverse;
}

.avatar {
  width: 50px;
  height: 50px;
  border-radius: 50%;
}

.content {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin: 0 10px;
}

.content.self {
  align-items: flex-end;
}

.name {
  font-size: 14px;
  font-weight: bold;
}

.text {
  margin-top: 10px;
  font-size: 14px;
  text-align: left;
  border-radius: 10px;
  padding: 10px;
  max-width: 600px;
  word-wrap:break-word;
  word-break:break-all;
  white-space: pre-wrap;
  //white-space: pre-line;
  border: 1px solid #dedede;
}
.markdown {
  margin-top: 10px;
  font-size: 14px;
  text-align: left;
  border-radius: 10px;
  padding: 10px;
  max-width: 600px;
  word-wrap:break-word;
  word-break:break-all;
  border: 1px solid #dedede;
}

.text_log {
  margin-top: 10px;
  font-size: 14px;
  text-align: left;
  border-radius: 10px;
  padding: 3px 10px;
  background-color: #FFF;
  max-width: 600px;
  word-wrap:break-word;
  word-break:break-all;
  white-space: pre-wrap;
  background-color: rgba(0,0,0,.05);
  border: 1px solid #dedede;
}

.head {
  height: 6%;
  padding-top: 1.5%;
  font-size: 18px;
  font-weight: bold;
  border-bottom: 1px solid #CCC;
  position: relative;
}

.input {
  display: flex;
  border-top: 1px solid #CCC;
  padding-top: 3%;
  height: 10%;
  font-size: 18px;
}

.text-input {
  height: 40px;
  flex: 1;
  padding: 0 10px;
  outline: 0;
  border: none;
  border-radius: 4px;
  background-color: #F4F4F4;
}

.send-button {
  height: 40px;
  width: 80px;
  margin-left: 10px;
  border: none;
  border-radius: 4px;
  background-color: #07C160;
  color: #FFF;
  font-size: 18px;
  font-weight: bold;
  cursor: pointer;
}
.send-button.disable {
  background-color: #999;
}
.record-input-wrapper .el-textarea__inner{
  background: transparent!important;
  color: white !important;
  border: none;
  height: 100%;
}

.record-input-wrapper svg{
  transform: scale(0.8) translateY(-10%);
  fill: #ffffff;
}
.dot-elastic {
  position: relative;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  animation: dotElastic 1s infinite linear
}

.dot-elastic::before, .dot-elastic::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-elastic::before {
  left: -15px;
  animation: dotElasticBefore 1s infinite linear
}

.dot-elastic::after {
  left: 15px;
  animation: dotElasticAfter 1s infinite linear
}

@keyframes dotElasticBefore {
  0% {
    transform: scale(1, 1)
  }
  25% {
    transform: scale(1, 1.5)
  }
  50% {
    transform: scale(1, 0.67)
  }
  75% {
    transform: scale(1, 1)
  }
  100% {
    transform: scale(1, 1)
  }
}

@keyframes dotElastic {
  0% {
    transform: scale(1, 1)
  }
  25% {
    transform: scale(1, 1)
  }
  50% {
    transform: scale(1, 1.5)
  }
  75% {
    transform: scale(1, 1)
  }
  100% {
    transform: scale(1, 1)
  }
}

@keyframes dotElasticAfter {
  0% {
    transform: scale(1, 1)
  }
  25% {
    transform: scale(1, 1)
  }
  50% {
    transform: scale(1, 0.67)
  }
  75% {
    transform: scale(1, 1.5)
  }
  100% {
    transform: scale(1, 1)
  }
}

.dot-pulse {
  position: relative;
  left: -9999px;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160;
  animation: dotPulse 1.5s infinite linear
}

@keyframes dotPulse {
  0% {
    box-shadow: 9984px 0 0 -5px #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 2px #07C160
  }
  25% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 2px #07C160, 10014px 0 0 0 #07C160
  }
  50% {
    box-shadow: 9984px 0 0 2px #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 -5px #07C160
  }
  75% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 -5px #07C160, 10014px 0 0 0 #07C160
  }
  100% {
    box-shadow: 9984px 0 0 -5px #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 2px #07C160
  }
}

.dot-flashing {
  position: relative;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  animation: dotFlashing 1s infinite linear alternate;
  animation-delay: .5s
}

.dot-flashing::before, .dot-flashing::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-flashing::before {
  left: -15px;
  animation: dotFlashing 1s infinite alternate;
  animation-delay: 0s
}

.dot-flashing::after {
  left: 15px;
  animation: dotFlashing 1s infinite alternate;
  animation-delay: 1s
}

@keyframes dotFlashing {
  0% {
    background-color: #ffffff
  }
  100% {
    background-color: #07C160;
  }
}

.dot-collision {
  position: relative;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-collision::before, .dot-collision::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-collision::before {
  left: -10px;
  animation: dotCollisionBefore 2s infinite ease-in
}

.dot-collision::after {
  left: 10px;
  animation: dotCollisionAfter 2s infinite ease-in;
  animation-delay: 1s
}

@keyframes dotCollisionBefore {
  0%, 50%, 75%, 100% {
    transform: translateX(0)
  }
  25% {
    transform: translateX(-15px)
  }
}

@keyframes dotCollisionAfter {
  0%, 50%, 75%, 100% {
    transform: translateX(0)
  }
  25% {
    transform: translateX(15px)
  }
}

.dot-revolution {
  position: relative;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-revolution::before, .dot-revolution::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-revolution::before {
  left: 0;
  top: -15px;
  transform-origin: 5px 20px;
  animation: dotRevolution 1.4s linear infinite
}

.dot-revolution::after {
  left: 0;
  top: -30px;
  transform-origin: 5px 35px;
  animation: dotRevolution 1s linear infinite
}

@keyframes dotRevolution {
  0% {
    transform: rotateZ(0deg) translate3d(0, 0, 0)
  }
  100% {
    transform: rotateZ(360deg) translate3d(0, 0, 0)
  }
}

.dot-carousel {
  position: relative;
  left: -9999px;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160;
  animation: dotCarousel 1.5s infinite linear
}

@keyframes dotCarousel {
  0% {
    box-shadow: 9984px 0 0 -1px #07C160, 9999px 0 0 1px #07C160, 10014px 0 0 -1px #07C160
  }
  50% {
    box-shadow: 10014px 0 0 -1px #07C160, 9984px 0 0 -1px #07C160, 9999px 0 0 1px #07C160
  }
  100% {
    box-shadow: 9999px 0 0 1px #07C160, 10014px 0 0 -1px #07C160, 9984px 0 0 -1px #07C160
  }
}

.dot-typing {
  position: relative;
  left: -9999px;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160;
  animation: dotTyping 1.5s infinite linear
}

@keyframes dotTyping {
  0% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160
  }
  16.667% {
    box-shadow: 9984px -10px 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160
  }
  33.333% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160
  }
  50% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px -10px 0 0 #07C160, 10014px 0 0 0 #07C160
  }
  66.667% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160
  }
  83.333% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px -10px 0 0 #07C160
  }
  100% {
    box-shadow: 9984px 0 0 0 #07C160, 9999px 0 0 0 #07C160, 10014px 0 0 0 #07C160
  }
}

.dot-windmill {
  position: relative;
  top: -15px;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  transform-origin: 5px 20px;
  animation: dotWindmill 2s infinite linear
}

.dot-windmill::before, .dot-windmill::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-windmill::before {
  left: -12.99px;
  top: 22.5px
}

.dot-windmill::after {
  left: 12.99px;
  top: 22.5px
}

@keyframes dotWindmill {
  0% {
    transform: rotateZ(0deg) translate3d(0, 0, 0)
  }
  100% {
    transform: rotateZ(720deg) translate3d(0, 0, 0)
  }
}

.dot-bricks {
  position: relative;
  top: 8px;
  left: -9999px;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  box-shadow: 9991px -16px 0 0 #07C160, 9991px 0 0 0 #07C160, 10007px 0 0 0 #07C160;
  animation: dotBricks 2s infinite ease
}

@keyframes dotBricks {
  0% {
    box-shadow: 9991px -16px 0 0 #07C160, 9991px 0 0 0 #07C160, 10007px 0 0 0 #07C160
  }
  8.333% {
    box-shadow: 10007px -16px 0 0 #07C160, 9991px 0 0 0 #07C160, 10007px 0 0 0 #07C160
  }
  16.667% {
    box-shadow: 10007px -16px 0 0 #07C160, 9991px -16px 0 0 #07C160, 10007px 0 0 0 #07C160
  }
  25% {
    box-shadow: 10007px -16px 0 0 #07C160, 9991px -16px 0 0 #07C160, 9991px 0 0 0 #07C160
  }
  33.333% {
    box-shadow: 10007px 0 0 0 #07C160, 9991px -16px 0 0 #07C160, 9991px 0 0 0 #07C160
  }
  41.667% {
    box-shadow: 10007px 0 0 0 #07C160, 10007px -16px 0 0 #07C160, 9991px 0 0 0 #07C160
  }
  50% {
    box-shadow: 10007px 0 0 0 #07C160, 10007px -16px 0 0 #07C160, 9991px -16px 0 0 #07C160
  }
  58.333% {
    box-shadow: 9991px 0 0 0 #07C160, 10007px -16px 0 0 #07C160, 9991px -16px 0 0 #07C160
  }
  66.666% {
    box-shadow: 9991px 0 0 0 #07C160, 10007px 0 0 0 #07C160, 9991px -16px 0 0 #07C160
  }
  75% {
    box-shadow: 9991px 0 0 0 #07C160, 10007px 0 0 0 #07C160, 10007px -16px 0 0 #07C160
  }
  83.333% {
    box-shadow: 9991px -16px 0 0 #07C160, 10007px 0 0 0 #07C160, 10007px -16px 0 0 #07C160
  }
  91.667% {
    box-shadow: 9991px -16px 0 0 #07C160, 9991px 0 0 0 #07C160, 10007px -16px 0 0 #07C160
  }
  100% {
    box-shadow: 9991px -16px 0 0 #07C160, 9991px 0 0 0 #07C160, 10007px 0 0 0 #07C160
  }
}

.dot-floating {
  position: relative;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  animation: dotFloating 3s infinite cubic-bezier(0.15, 0.6, 0.9, 0.1)
}

.dot-floating::before, .dot-floating::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-floating::before {
  left: -12px;
  animation: dotFloatingBefore 3s infinite ease-in-out
}

.dot-floating::after {
  left: -24px;
  animation: dotFloatingAfter 3s infinite cubic-bezier(0.4, 0, 1, 1)
}

@keyframes dotFloating {
  0% {
    left: calc(-50% - 5px)
  }
  75% {
    left: calc(50% + 105px)
  }
  100% {
    left: calc(50% + 105px)
  }
}

@keyframes dotFloatingBefore {
  0% {
    left: -50px
  }
  50% {
    left: -12px
  }
  75% {
    left: -50px
  }
  100% {
    left: -50px
  }
}

@keyframes dotFloatingAfter {
  0% {
    left: -100px
  }
  50% {
    left: -24px
  }
  75% {
    left: -100px
  }
  100% {
    left: -100px
  }
}

.dot-fire {
  position: relative;
  left: -9999px;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  box-shadow: 9999px -15px 0 0;
  animation: dotFire 1s infinite linear
}

.dot-fire::before, .dot-fire::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-fire::before {
  box-shadow: 9999px -5px 0 0;
  animation: dotFireBefore 1s infinite linear
}

.dot-fire::after {
  box-shadow: 9999px 15px 0 0;
  animation: dotFireAfter 1s infinite linear
}

@keyframes dotFire {
  0% {
    box-shadow: 9999px -15px 0 -3px
  }
  1%, 50% {
    box-shadow: 9999px 15px 0 -5px
  }
  100% {
    box-shadow: 9999px -5px 0 2px
  }
}

@keyframes dotFireBefore {
  0% {
    box-shadow: 9999px -5px 0 2px
  }
  50% {
    box-shadow: 9999px -15px 0 -3px
  }
  51%, 100% {
    box-shadow: 9999px 15px 0 -5px
  }
}

@keyframes dotFireAfter {
  1% {
    box-shadow: 9999px 15px 0 -5px
  }
  50% {
    box-shadow: 9999px -5px 0 2px
  }
  100%, 0% {
    box-shadow: 9999px -15px 0 -3px
  }
}

.dot-spin {
  position: relative;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: transparent;
  color: transparent;
  box-shadow: 0 -18px 0 0 #07C160, 12.72984px -12.72984px 0 0 #07C160, 18px 0 0 0 #07C160, 12.72984px 12.72984px 0 0 rgba(152, 128, 255, 0), 0 18px 0 0 rgba(152, 128, 255, 0), -12.72984px 12.72984px 0 0 rgba(152, 128, 255, 0), -18px 0 0 0 rgba(152, 128, 255, 0), -12.72984px -12.72984px 0 0 rgba(152, 128, 255, 0);
  animation: dotSpin 1.5s infinite linear
}

@keyframes dotSpin {
  0%, 100% {
    box-shadow: 0 -18px 0 0 #07C160, 12.72984px -12.72984px 0 0 #07C160, 18px 0 0 0 #07C160, 12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), 0 18px 0 -5px rgba(152, 128, 255, 0), -12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), -18px 0 0 -5px rgba(152, 128, 255, 0), -12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0)
  }
  12.5% {
    box-shadow: 0 -18px 0 -5px rgba(152, 128, 255, 0), 12.72984px -12.72984px 0 0 #07C160, 18px 0 0 0 #07C160, 12.72984px 12.72984px 0 0 #07C160, 0 18px 0 -5px rgba(152, 128, 255, 0), -12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), -18px 0 0 -5px rgba(152, 128, 255, 0), -12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0)
  }
  25% {
    box-shadow: 0 -18px 0 -5px rgba(152, 128, 255, 0), 12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0), 18px 0 0 0 #07C160, 12.72984px 12.72984px 0 0 #07C160, 0 18px 0 0 #07C160, -12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), -18px 0 0 -5px rgba(152, 128, 255, 0), -12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0)
  }
  37.5% {
    box-shadow: 0 -18px 0 -5px rgba(152, 128, 255, 0), 12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0), 18px 0 0 -5px rgba(152, 128, 255, 0), 12.72984px 12.72984px 0 0 #07C160, 0 18px 0 0 #07C160, -12.72984px 12.72984px 0 0 #07C160, -18px 0 0 -5px rgba(152, 128, 255, 0), -12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0)
  }
  50% {
    box-shadow: 0 -18px 0 -5px rgba(152, 128, 255, 0), 12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0), 18px 0 0 -5px rgba(152, 128, 255, 0), 12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), 0 18px 0 0 #07C160, -12.72984px 12.72984px 0 0 #07C160, -18px 0 0 0 #07C160, -12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0)
  }
  62.5% {
    box-shadow: 0 -18px 0 -5px rgba(152, 128, 255, 0), 12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0), 18px 0 0 -5px rgba(152, 128, 255, 0), 12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), 0 18px 0 -5px rgba(152, 128, 255, 0), -12.72984px 12.72984px 0 0 #07C160, -18px 0 0 0 #07C160, -12.72984px -12.72984px 0 0 #07C160
  }
  75% {
    box-shadow: 0 -18px 0 0 #07C160, 12.72984px -12.72984px 0 -5px rgba(152, 128, 255, 0), 18px 0 0 -5px rgba(152, 128, 255, 0), 12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), 0 18px 0 -5px rgba(152, 128, 255, 0), -12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), -18px 0 0 0 #07C160, -12.72984px -12.72984px 0 0 #07C160
  }
  87.5% {
    box-shadow: 0 -18px 0 0 #07C160, 12.72984px -12.72984px 0 0 #07C160, 18px 0 0 -5px rgba(152, 128, 255, 0), 12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), 0 18px 0 -5px rgba(152, 128, 255, 0), -12.72984px 12.72984px 0 -5px rgba(152, 128, 255, 0), -18px 0 0 -5px rgba(152, 128, 255, 0), -12.72984px -12.72984px 0 0 #07C160
  }
}

.dot-falling {
  position: relative;
  left: -9999px;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  box-shadow: 9999px 0 0 0 #07C160;
  animation: dotFalling 1s infinite linear;
  animation-delay: .1s
}

.dot-falling::before, .dot-falling::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-falling::before {
  animation: dotFallingBefore 1s infinite linear;
  animation-delay: 0s
}

.dot-falling::after {
  animation: dotFallingAfter 1s infinite linear;
  animation-delay: .2s
}

@keyframes dotFalling {
  0% {
    box-shadow: 9999px -15px 0 0 rgba(152, 128, 255, 0)
  }
  25%, 50%, 75% {
    box-shadow: 9999px 0 0 0 #07C160
  }
  100% {
    box-shadow: 9999px 15px 0 0 rgba(152, 128, 255, 0)
  }
}

@keyframes dotFallingBefore {
  0% {
    box-shadow: 9984px -15px 0 0 rgba(152, 128, 255, 0)
  }
  25%, 50%, 75% {
    box-shadow: 9984px 0 0 0 #07C160
  }
  100% {
    box-shadow: 9984px 15px 0 0 rgba(152, 128, 255, 0)
  }
}

@keyframes dotFallingAfter {
  0% {
    box-shadow: 10014px -15px 0 0 rgba(152, 128, 255, 0)
  }
  25%, 50%, 75% {
    box-shadow: 10014px 0 0 0 #07C160
  }
  100% {
    box-shadow: 10014px 15px 0 0 rgba(152, 128, 255, 0)
  }
}

.dot-stretching {
  position: relative;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160;
  animation: dotStretching 2s infinite ease-in;
  transform: scale(1.25, 1.25)
}

.dot-stretching::before, .dot-stretching::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #07C160;
  color: #07C160
}

.dot-stretching::before {
  animation: dotStretchingBefore 2s infinite ease-in
}

.dot-stretching::after {
  animation: dotStretchingAfter 2s infinite ease-in
}

@keyframes dotStretching {
  0% {
    transform: scale(1.25, 1.25)
  }
  50%, 60% {
    transform: scale(.8, .8)
  }
  100% {
    transform: scale(1.25, 1.25)
  }
}

@keyframes dotStretchingBefore {
  0% {
    transform: translate(0) scale(.7, .7)
  }
  50%, 60% {
    transform: translate(-20px) scale(1, 1)
  }
  100% {
    transform: translate(0) scale(.7, .7)
  }
}

@keyframes dotStretchingAfter {
  0% {
    transform: translate(0) scale(.7, .7)
  }
  50%, 60% {
    transform: translate(20px) scale(1, 1)
  }
  100% {
    transform: translate(0) scale(.7, .7)
  }
}

.dot-shuttle {
  position: relative;
  width: 12px;
  height: 12px;
  border-radius: 6px;
  background-color: #000;
  color: transparent;
  margin: -1px 0;
  filter: blur(2px)
}

.dot-shuttle::before, .dot-shuttle::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  left: -50px;
  width: 12px;
  height: 12px;
  border-radius: 6px;
  background-color: #000;
  color: transparent;
  opacity: 0;
  filter: blur(2px);
  animation: dotShuttle 2s infinite ease-in
}

.dot-shuttle::after {
  animation-delay: 1s
}

@keyframes dotShuttle {
  0% {
    opacity: 0;
    transform: translateX(0)
  }
  50% {
    opacity: 1;
    transform: translateX(50px)
  }
  100% {
    opacity: 0;
    transform: translateX(100px)
  }
}

.log-loading {
  height: 20px;
  width: 45px;
  padding-left: 20px;
  margin-top: 10px
}

.dot-hourglass {
  position: relative;
  top: -15px;
  width: 12px;
  height: 12px;
  border-radius: 6px;
  background-color: #000;
  color: transparent;
  margin: -1px 0;
  filter: blur(2px);
  transform-origin: 5px 20px;
  animation: dotHourglass 2.4s infinite ease-in-out;
  animation-delay: .6s
}

.dot-hourglass::before, .dot-hourglass::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  left: 0;
  width: 12px;
  height: 12px;
  border-radius: 6px;
  background-color: #000;
  color: transparent;
  filter: blur(2px)
}

.dot-hourglass::before {
  top: 30px
}

.dot-hourglass::after {
  animation: dotHourglassAfter 2.4s infinite cubic-bezier(0.65, 0.05, 0.36, 1)
}

@keyframes dotHourglass {
  0% {
    transform: rotateZ(0deg)
  }
  25% {
    transform: rotateZ(180deg)
  }
  50% {
    transform: rotateZ(180deg)
  }
  75% {
    transform: rotateZ(360deg)
  }
  100% {
    transform: rotateZ(360deg)
  }
}

@keyframes dotHourglassAfter {
  0% {
    transform: translateY(0)
  }
  25% {
    transform: translateY(30px)
  }
  50% {
    transform: translateY(30px)
  }
  75% {
    transform: translateY(0)
  }
  100% {
    transform: translateY(0)
  }
}

.dot-overtaking {
  position: relative;
  width: 12px;
  height: 12px;
  border-radius: 6px;
  background-color: transparent;
  color: #000;
  margin: -1px 0;
  box-shadow: 0 -20px 0 0;
  filter: blur(2px);
  animation: dotOvertaking 2s infinite cubic-bezier(0.2, 0.6, 0.8, 0.2)
}

.dot-overtaking::before, .dot-overtaking::after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  left: 0;
  width: 12px;
  height: 12px;
  border-radius: 6px;
  background-color: transparent;
  color: #000;
  box-shadow: 0 -20px 0 0;
  filter: blur(2px)
}

.dot-overtaking::before {
  animation: dotOvertaking 2s infinite cubic-bezier(0.2, 0.6, 0.8, 0.2);
  animation-delay: .3s
}

.dot-overtaking::after {
  animation: dotOvertaking 1.5s infinite cubic-bezier(0.2, 0.6, 0.8, 0.2);
  animation-delay: .6s
}

@keyframes dotOvertaking {
  0% {
    transform: rotateZ(0deg)
  }
  100% {
    transform: rotateZ(360deg)
  }
}

.dot-emoji {
  position: relative;
  height: 10px;
  font-size: 10px
}

.dot-emoji::before {
  content: '⚽🏀🏐';
  display: inline-block;
  position: relative;
  animation: dotEmoji 1s infinite
}

@keyframes dotEmoji {
  0% {
    top: -20px;
    animation-timing-function: ease-in
  }
  34% {
    transform: scale(1, 1)
  }
  35% {
    top: 20px;
    animation-timing-function: ease-out;
    transform: scale(1.5, 0.5)
  }
  45% {
    transform: scale(1, 1)
  }
  90% {
    top: -20px
  }
  100% {
    top: -20px
  }
}
/*# sourceMappingURL=three-dots.min.css.map */
.markdown-body {
  box-sizing: border-box;
  min-width: 200px;
  max-width: 600px;
  margin: 0 auto;
  padding: 15px;
}


.output {
  overflow: auto;
  width: 50%;
  height: 100%;
  box-sizing: border-box;
  padding: 0 20px;
}

.markdown-body, [data-theme="light"] {
  /*light*/
  color-scheme: light;
  --color-prettylights-syntax-comment: #6e7781;
  --color-prettylights-syntax-constant: #0550ae;
  --color-prettylights-syntax-entity: #6639ba;
  --color-prettylights-syntax-storage-modifier-import: #24292f;
  --color-prettylights-syntax-entity-tag: #116329;
  --color-prettylights-syntax-keyword: #cf222e;
  --color-prettylights-syntax-string: #0a3069;
  --color-prettylights-syntax-variable: #953800;
  --color-prettylights-syntax-brackethighlighter-unmatched: #82071e;
  --color-prettylights-syntax-invalid-illegal-text: #f6f8fa;
  --color-prettylights-syntax-invalid-illegal-bg: #82071e;
  --color-prettylights-syntax-carriage-return-text: #f6f8fa;
  --color-prettylights-syntax-carriage-return-bg: #cf222e;
  --color-prettylights-syntax-string-regexp: #116329;
  --color-prettylights-syntax-markup-list: #3b2300;
  --color-prettylights-syntax-markup-heading: #0550ae;
  --color-prettylights-syntax-markup-italic: #24292f;
  --color-prettylights-syntax-markup-bold: #24292f;
  --color-prettylights-syntax-markup-deleted-text: #82071e;
  --color-prettylights-syntax-markup-deleted-bg: #ffebe9;
  --color-prettylights-syntax-markup-inserted-text: #116329;
  --color-prettylights-syntax-markup-inserted-bg: #dafbe1;
  --color-prettylights-syntax-markup-changed-text: #953800;
  --color-prettylights-syntax-markup-changed-bg: #ffd8b5;
  --color-prettylights-syntax-markup-ignored-text: #eaeef2;
  --color-prettylights-syntax-markup-ignored-bg: #0550ae;
  --color-prettylights-syntax-meta-diff-range: #8250df;
  --color-prettylights-syntax-brackethighlighter-angle: #57606a;
  --color-prettylights-syntax-sublimelinter-gutter-mark: #8c959f;
  --color-prettylights-syntax-constant-other-reference-link: #0a3069;
  --color-fg-default: #1F2328;
  --color-fg-muted: #656d76;
  --color-fg-subtle: #6e7781;
  --color-canvas-default: #ffffff;
  --color-canvas-subtle: #f6f8fa;
  --color-border-default: #d0d7de;
  --color-border-muted: hsla(210,18%,87%,1);
  --color-neutral-muted: rgba(175,184,193,0.2);
  --color-accent-fg: #0969da;
  --color-accent-emphasis: #0969da;
  --color-attention-fg: #9a6700;
  --color-attention-subtle: #fff8c5;
  --color-danger-fg: #d1242f;
  --color-done-fg: #8250df;
  --color-prettylights-syntax-comment: #6e7781;
  --color-prettylights-syntax-constant: #0550ae;
  --color-prettylights-syntax-entity: #6639ba;
  --color-prettylights-syntax-storage-modifier-import: #24292f;
  --color-prettylights-syntax-entity-tag: #116329;
  --color-prettylights-syntax-keyword: #cf222e;
  --color-prettylights-syntax-string: #0a3069;
  --color-prettylights-syntax-variable: #953800;
  --color-prettylights-syntax-brackethighlighter-unmatched: #82071e;
  --color-prettylights-syntax-invalid-illegal-text: #f6f8fa;
  --color-prettylights-syntax-invalid-illegal-bg: #82071e;
  --color-prettylights-syntax-carriage-return-text: #f6f8fa;
  --color-prettylights-syntax-carriage-return-bg: #cf222e;
  --color-prettylights-syntax-string-regexp: #116329;
  --color-prettylights-syntax-markup-list: #3b2300;
  --color-prettylights-syntax-markup-heading: #0550ae;
  --color-prettylights-syntax-markup-italic: #24292f;
  --color-prettylights-syntax-markup-bold: #24292f;
  --color-prettylights-syntax-markup-deleted-text: #82071e;
  --color-prettylights-syntax-markup-deleted-bg: #ffebe9;
  --color-prettylights-syntax-markup-inserted-text: #116329;
  --color-prettylights-syntax-markup-inserted-bg: #dafbe1;
  --color-prettylights-syntax-markup-changed-text: #953800;
  --color-prettylights-syntax-markup-changed-bg: #ffd8b5;
  --color-prettylights-syntax-markup-ignored-text: #eaeef2;
  --color-prettylights-syntax-markup-ignored-bg: #0550ae;
  --color-prettylights-syntax-meta-diff-range: #8250df;
  --color-prettylights-syntax-brackethighlighter-angle: #57606a;
  --color-prettylights-syntax-sublimelinter-gutter-mark: #8c959f;
  --color-prettylights-syntax-constant-other-reference-link: #0a3069;
  --color-fg-default: #1F2328;
  --color-fg-muted: #656d76;
  --color-fg-subtle: #6e7781;
  --color-canvas-default: #ffffff;
  --color-canvas-subtle: #f6f8fa;
  --color-border-default: #d0d7de;
  --color-border-muted: hsla(210,18%,87%,1);
  --color-neutral-muted: rgba(175,184,193,0.2);
  --color-accent-fg: #0969da;
  --color-accent-emphasis: #0969da;
  --color-attention-fg: #9a6700;
  --color-attention-subtle: #fff8c5;
  --color-danger-fg: #d1242f;
  --color-done-fg: #8250df;
}
</style>
