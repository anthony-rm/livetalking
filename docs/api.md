# LiveTalking API 接口文档

基础路径：`http://<host>:<listenport>`

所有接口统一返回格式：

```json
{ "code": 0, "msg": "ok", "data": {} }
```

`code` 为 0 表示成功，非 0 表示错误。

---

## 1. WebRTC Offer (JSON)

交换 SDP 以建立 WebRTC 连接，扩展参数在 JSON body 中。

```
POST /offer
```

**Content-Type**: `application/json`

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `sdp` | 是 | string | — | WebRTC Offer SDP |
| `type` | 是 | string | — | 必须为 `offer` |
| `avatar` | 否 | string | 启动参数值 | 指定数字人 ID |
| `refaudio` | 否 | string | — | 参考音频 |
| `reftext` | 否 | string | — | 参考文本 |

**响应** (200):

```json
{
  "sdp": "v=0\r\n...",
  "type": "answer",
  "sessionid": "session-uuid"
}
```

---

## 2. WebRTC Offer (WHEP)

符合 [WHEP 协议](https://datatracker.ietf.org/doc/draft-ietf-wish-whep/)（WebRTC HTTP Egress Protocol）。
SDP offer 以 `application/sdp` 裸文本发送，扩展参数通过 query string 传递。

```
POST /whep
```

**Content-Type**: `application/sdp`

**Query 参数**:

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `avatar` | 否 | string | 启动参数值 | 指定数字人 ID |
| `refaudio` | 否 | string | — | 参考音频 |
| `reftext` | 否 | string | — | 参考文本 |
| `tts` | 否 | string | — | TTS 引擎 |
| `tts_server` | 否 | string | — | TTS 服务地址 |
| `tts_speed` | 否 | number | — | TTS 语速 |

**Body**: SDP offer 裸文本，例如：

```
v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n...
```

**响应** (201):

- `Content-Type`: `application/sdp`
- `X-Session-ID`: 生成的会话 ID（UUID）

Body 为 SDP answer 裸文本：

```
v=0\r\no=- ...\r\n...
```

**客户端示例**:

```javascript
const params = new URLSearchParams({
  avatar: 'wav2lip256_avatar1',
  refaudio: 'zh-CN-YunxiaNeural',
});

const res = await fetch('/whep?' + params.toString(), {
  method: 'POST',
  headers: { 'Content-Type': 'application/sdp' },
  body: pc.localDescription.sdp,
});

const answerSdp = await res.text();
const sessionid = res.headers.get('X-Session-ID');
await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
```

---

## 3. 文本驱动 (Human)

发送文本驱动数字人说话，支持直接复读或 LLM 对话。

```
POST /human
```

**Content-Type**: `application/json`

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `sessionid` | 是 | string | — | 会话 ID |
| `text` | 是 | string | — | 输入文本 |
| `type` | 是 | string | — | `echo`: 直接复读; `chat`: 触发 LLM 回答 |
| `interrupt` | 否 | bool | false | 是否打断当前播报 |
| `tts` | 否 | object | — | 透传给 TTS 的配置（如 `voice`, `emotion`） |
| `emotion` | 否 | string | — | 面部表情（ditto 后端）: `angry` `disgust` `fear` `happy` `neutral` `sad` `surprise` `contempt`，或 0-7 索引，或索引数组。仅作用于本句，句末自动清空 |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 4. 音频驱动 (Human Audio)

上传音频文件驱动数字人。

```
POST /humanaudio
```

**Content-Type**: `multipart/form-data`

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |
| `file` | 是 | file | 音频文件 |
| `emotion` | 否 | string | 同 `/human` 的 `emotion`（ditto 后端） |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 4b. 设置表情 (Set Emotion)

设置会话级表情（ditto 后端）。与 `/human` 的 `emotion` 参数（仅本句生效）不同，本接口持续生效直到再次调用。

```
POST /set_emotion
```

**Content-Type**: `application/json`

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |
| `emotion` | 是 | string/int/list | 表情名或 0-7 索引；索引数组为软混合；`null` 恢复默认 |
| `gain` | 否 | float | 表情强度。`1`（默认）为上游原始向量；`2` 为距离 neutral 两倍；`0` 归零。省略则保持当前强度 |

**响应**:

```json
{ "code": 0, "msg": "ok", "data": { "emotion": "happy@2" } }
```

`data.emotion` 在强度不为 1 时带 `@强度` 后缀，界面据此点亮对应强度按钮。

**关于 `gain` 的实测结论**（重要）: 该参数沿 neutral→label 轴缩放 8 维条件向量，实测对画面的影响很小（盒子上同一音频、同一种子下平均仅 1.07 倍，部分表情甚至略降）。原因是一维 one-hot 经 `softmax(one_hot*8)` 已达 99.77%，继续放大只会把其它维推向负概率——分布之外，模型没见过。真正的强度旋钮在 `guidance_weight`（`MotionDecoder.guided_forward` 的 `weight`，上游 2），它是整段渲染的全局参数而非逐表情参数，因此未暴露在 HTTP 接口上；离线实验用 `scripts/ditto/emotion_lab.py --guidance`。

非 ditto 后端返回 `emotion is only supported by the ditto backend`。

## 4c. 表情词表 (Emotion vocabulary)

返回 ditto 的表情标签表，以及指定会话当前生效的表情。页面用这一份列表渲染下拉框和视频下方的按钮，标签顺序只在 `avatars/ditto/emotion.py` 维护一处。

```
GET /api/emotions?sessionid=<sid>
```

**响应**:

```json
{
  "code": 0, "msg": "ok",
  "data": {
    "emotions": ["angry","disgust","fear","happy","neutral","sad","surprise","contempt"],
    "default": "neutral",
    "supported": true,
    "current": "happy"
  }
}
```

`sessionid` 可省略；不带会话或后端不支持表情时 `supported` 为 `false`、`current` 为 `null`，页面据此禁用按钮而不是等第一次点击才报错。

## 5. 打断播报

立即清空当前会话的音频队列。

```
POST /interrupt_talk
```

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |

**响应**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 6. 查询说话状态

```
POST /is_speaking
```

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": true
}
```

---

## 7. SSE 事件流

```
GET /sse?sessionid=<sessionid>
```

**协议**: Server-Sent Events (SSE)

用于接收**服务器→客户端**的异步状态推送（播报事件、状态变化等）

**请求参数 (Query)**:

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `sessionid` | 是 | string | 会话 ID |

**响应格式**: `Content-Type: text/event-stream`

每条事件一行 JSON，格式为：

```
data: {"status": "start"}

```

**客户端示例**:

```javascript
const es = new EventSource(`/sse?sessionid=${sessionid}`);
es.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data 为服务器推送的播报状态/事件
};
es.onerror = () => {
    // 连接出错或服务端主动断开，EventSource 会自动重连
};

// 断开时
es.close();
```