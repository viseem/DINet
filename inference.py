import time
from utils.deep_speech import DeepSpeech
from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DINetInferenceOptions
from models.DINet import DINet
from scipy.io import wavfile
import threading
import queue
import numpy as np
import glob
import os
import pika
import cv2
import torch
import subprocess
import random
import nls
from collections import OrderedDict
import asyncio
import json
import logging
import os
import ssl
import threading
import time
import traceback
import websockets
import pika
import uuid
from queue import Queue, Empty
from typing import Union

import aiortc
import av
import requests
from aiohttp import web
from aiortc import MediaStreamTrack, RTCSessionDescription, RTCPeerConnection, RTCRtpSender, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamError
from av.frame import Frame
from av.packet import Packet
from pyee.asyncio import AsyncIOEventEmitter

# 文件的 user_id 与 count 的分割
FILE_NAME_SPLIT = "#"
  
# 定义rabitmq的链接
channel = None

# 音频处理设置信息
data_queue = queue.Queue()
URL= "wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1"
TOKEN= "89253ea12c624e78a0ccd3453d4799e9"
APPKEY="FCW8uluerIsGU24l"
sample_rate = 16000
bytes_per_sample = 2  # 16-bit PCM
data_buffer = np.array([])
file_count = 0

# webrtc 信息
players = {}
pcs = set()
playlist = [
    "asserts/examples/base.mp4",
]

############################################## webrtc start ##############################################

def extract_frames_from_video(video_path,save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)
    return (int(frame_width),int(frame_height))


class MediaFrameHolder:
    def __init__(self, url: str) -> None:
        if (url == "asserts/examples/base.mp4"):
            self.audio_holder = FrameHolder('audio', url)
            self.video_holder = FrameHolder('video', url)
        else: 
            self.audio_holder = FrameHolder('audio', url + '.wav')
            self.video_holder = FrameHolder('video', url + '.mp4')


class FrameHolder:
    def __init__(self, kind, url: str) -> None:
        self._kind = kind
        # 视频url
        self._url = url
        self._is_net_file = url.startswith("http")
        # 如果是http的视频，先缓存到本地。否则av打开网上的视频，很久不读之后会有问题，造成画面卡死
        if self._is_net_file:
            self.local_filename = "video_temp/" + os.path.basename(url)
            # 如果目录不存在，就创建目录
            if not os.path.exists(os.path.dirname(self.local_filename)):
                os.makedirs(os.path.dirname(self.local_filename))
            # 发送 GET 请求获取视频二进制数据
            response = self._retry_get(url, 3)
            # 将二进制数据写入本地文件
            with open(self.local_filename, "wb") as f:
                f.write(response.content)
            self._container = av.open(self.local_filename)
        else:
            self._container = av.open(url)
        self.buf = []
        self.last_frame = None
        self._sampler = av.AudioResampler(
            format="s16",
            layout="stereo",
            rate=48000,
            frame_size=int(48000 * 0.02),
        )

    def _retry_get(self, url: str, count: int):
        for _ in range(count):
            try:
                return requests.get(url, timeout=(1, 3))
            except Exception:
                traceback.print_stack()
                logging.error("get video error: %s, %d", url, count)

    def clear(self):
        def close_resources():
            self._container.close()
            if self._is_net_file:
                os.remove(self.local_filename)

        try:
            close_resources()
        except Exception:
            traceback.print_stack()
            logging.warning("remove %s failed, try remove again 5s later", self.local_filename)
            timer = threading.Timer(5, close_resources)
            timer.start()  # 启动定时器

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self._kind == 'audio':
                audio_stream = self._container.streams.audio[0]
                frame = next(self._container.decode(audio_stream))
                for re_frame in self._sampler.resample(frame):
                    self.buf.append(re_frame)
            else:
                video_stream = self._container.streams.video[0]
                frame = next(self._container.decode(video_stream))
                self.buf.append(frame)
        except Exception:
            pass
        try:
            # Remove and return item at index (default last). fuck！！！
            f = self.buf.pop(0)
            self.last_frame = f
            return False, self.last_frame
        except Exception:
            self._container.close()
            return True, self.last_frame


class Sentinel:
    def __init__(self, player) -> None:
        self._player = player
        # 记录播放列表的游标
        self._play_list_cursor = 0
        self._thread = None
        self._sig_stop = False

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                name='sentinel',
                target=self.maintain_push_queue,
                args=()
            )
            self._thread.start()

    def stop(self):
        self._sig_stop = True

    def next_video(self):
        # # 有插播就先选择插播
        # jump_in_list = self._player.jump_in_list
        # jump_in = None
        # try:
        #     jump_in = jump_in_list.get_nowait()
        # except Empty:
        #     pass
        # if jump_in:
        #     return jump_in

        # 没有插播的话要拿到当前播放的下一个
        play_list = self._player.playlist
        idx = self._play_list_cursor % len(play_list)
        url = play_list[idx]
        self._play_list_cursor += 1
        return url

    def maintain_push_queue(self):
        push_queue = self._player.push_queue
        while True:
            if self._sig_stop:
                break
            url = self.next_video()
            push_queue.put(MediaFrameHolder(url))
            print("successfully put to push_queue", url)


class PlayListTrack(MediaStreamTrack):
    def __init__(self, player, kind):
        super().__init__()
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self._start = None
        self._time = 0.
        self.fps = 25

    async def recv(self) -> Union[Frame, Packet]:
        if self.readyState != "live":
            raise MediaStreamError
        while self._player.push_queue.empty():
            await asyncio.sleep(0.01)
        from_jump_in = False
        if not self._player.push_queue.empty() or not self._player.jump_in_list.empty():
            if not self._player.jump_in_list.empty():
                media_frame_holder = self._player.jump_in_list.queue[0]
                from_jump_in = True
            else:
                media_frame_holder = self._player.push_queue.queue[0]

            if self.kind == 'audio':
                finished, data = next(media_frame_holder.audio_holder)
                data.pts = int(self._time * 48000)
                self._time += 0.02
                result = data
                if finished:
                    if not from_jump_in:
                        self._player.push_queue.get_nowait()
                    else:
                        self._player.jump_in_list.get_nowait()
                    media_frame_holder.video_holder.clear()
            else:
                finished, data = next(media_frame_holder.video_holder)
                data.pts = int(self._time / data.time_base)
                self._time += 0.04
                result = data
                if finished:
                    if not from_jump_in:
                        self._player.push_queue.get_nowait()
                    else:
                        self._player.jump_in_list.get_nowait()
                    media_frame_holder.audio_holder.clear()
        else:
            raise Exception('result is None')
        if result is None:
            raise Exception('result is None')
        data_time = float(result.pts * result.time_base)

        if self._start is None:
            self._start = time.time() - data_time
        else:
            wait = self._start + data_time - time.time()
            await asyncio.sleep(wait)

        return result

    def pause(self):
        pass

    def resume(self):
        pass

    def stop(self):
        super().stop()
        self._queue.empty()


class SegmentPlayer:
    def __init__(self, playlist: []) -> None:
        super().__init__()
        self.__audio = PlayListTrack(self, 'audio')
        self.__video = PlayListTrack(self, 'video')
        self.__loop = asyncio.get_event_loop()
        self.__thread = None
        self.__stop = False
        self.__sentinel = None
        # 循环播放队列
        self.__playlist = playlist
        # 插播队列
        self.jump_in_list = Queue()
        # 等待推送队里了
        self.push_queue = Queue(maxsize=1)
        self.last_jumpin_tiem = time.time()

    @property
    def is_stopped(self):
        return self.__stop

    @property
    def playlist(self) -> []:
        return self.__playlist


    @property
    def audio(self) -> PlayListTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> PlayListTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    # 开始
    def start(self):
        if self.__sentinel is None:
            self.__sentinel = Sentinel(self)
            self.__sentinel.start()

    # 停止
    def stop(self):
        self.__stop = True
        self.audio.stop()
        self.video.stop()
        self.__sentinel.stop()

    def jump_in(self, url):
        self.jump_in_list.put(MediaFrameHolder(url))
        # 计算上次插播时间
        current_time = time.time()
        print("jump duration:", current_time - self.last_jumpin_tiem)
        self.last_jumpin_tiem = current_time


async def on_message(ws, message):
    json_message = json.loads(message)
    sdp = json_message["sdp"]
    type = json_message["type"]
    client_id = json_message["client_id"]

    conn_id = client_id
    offer = RTCSessionDescription(sdp=sdp, type=type)
    pc = RTCPeerConnection(configuration=RTCConfiguration(
        iceServers=[
            RTCIceServer('turn:stun.viseem.com:3478', username='test', credential='123456'),
            RTCIceServer('stun:stun.viseem.com:3478'),

        ])
    )
    pcs.add(pc)

    player = SegmentPlayer(playlist=playlist)
    print("[###] user {} connected.".format(conn_id))
    players[conn_id] = player

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            player.stop()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechante():
        logging.info("ice connection stat change: %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            player.stop()

    sender = pc.addTrack(player.video)
    pc.addTrack(player.audio)

    aiortc.codecs.h264.MIN_BITRATE = 3000000
    aiortc.codecs.h264.DEFAULT_BITRATE = 3000000
    aiortc.codecs.h264.MAX_BITRATE = 4000000
    codecs = RTCRtpSender.getCapabilities("video").codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == "video/H264"]
    )
    
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await ws.send(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "id": conn_id}))

    player.start()

async def init_websocket():
    print("init websocket ok !!!")
    uri = "wss://cofi-ws.viseem.com?pid=server&client_type=server"
    async with websockets.connect(uri) as websocket:
        while True:
            msg = await websocket.recv()
            await on_message(websocket, msg)
            await asyncio.sleep(0.01)

############################################## webrtc end ##############################################


if __name__ == '__main__':

    # load config
    opt = DINetInferenceOptions().parse_args()
    if not os.path.exists(opt.source_video_path):
        raise ('wrong video path : {}'.format(opt.source_video_path))
    
    ############################################## extract frames from source video ##############################################
    print('extracting frames from video: {}'.format(opt.source_video_path))
    video_frame_dir = opt.source_video_path.replace('.mp4', '')
    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)
    video_size = extract_frames_from_video(opt.source_video_path,video_frame_dir)
    
    ############################################## load facial landmark ##############################################
    print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
    if not os.path.exists(opt.source_openface_landmark_path):
        raise ('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))
    video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(np.int)

    ############################################## load pretrained model weight ##############################################
    print('loading pretrained model from: {}'.format(opt.pretrained_clip_DINet_path))
    model = DINet(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    if not os.path.exists(opt.pretrained_clip_DINet_path):
        raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DINet_path))
    state_dict = torch.load(opt.pretrained_clip_DINet_path)['state_dict']['net_g']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    if not os.path.exists(opt.deepspeech_model_path):
        raise ('pls download pretrained model of deepspeech')
    DSModel = DeepSpeech(opt.deepspeech_model_path)
    DSModel.compute_audio_feature(opt.driving_audio_path)
    ref_img_tensor = torch.load("five_selected_ref_img.pt")

    ############################################## 核心推理函数 ##############################################
    def infer_process(wav_data, wav_file, mp4_file):
        __start_time = time.time()
        # 获取当前的时间戳，按照毫秒
        time_stamp = time.time()
        ds_feature = DSModel.compute_raw_wav_feature(wav_data)
        res_frame_length = ds_feature.shape[0]
        ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
        
        ############################################## align frame with driving audio ##############################################
        print('aligning frames with driving audio')
        time_stamp = time.time()

        video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
        if len(video_frame_path_list) != video_landmark_data.shape[0]:
            raise ('video frames are misaligned with detected landmarks')
        video_frame_path_list.sort()
        video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
        video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
        video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
        if video_frame_path_list_cycle_length >= res_frame_length:
            res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
            res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
        else:
            divisor = res_frame_length // video_frame_path_list_cycle_length
            remainder = res_frame_length % video_frame_path_list_cycle_length
            res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
            res_video_landmark_data = np.concatenate([video_landmark_data_cycle]* divisor + [video_landmark_data_cycle[:remainder, :, :]],0)
        res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                        + res_video_frame_path_list \
                                        + [video_frame_path_list_cycle[-1]] * 2 
        res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
        assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
        pad_length = ds_feature_padding.shape[0]
        print('安排视频帧(秒):', time.time() - time_stamp)

        ############################################## randomly select 5 reference images ##############################################
        print('selecting five reference images')
        time_stamp = time.time()
        
        # ref_img_list = []
        resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
        resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
        # ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
        # for ref_index in ref_index_list:
        #     crop_flag,crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
        #     if not crop_flag:
        #         raise ('our method can not handle videos with large change of facial size!!')
        #     crop_radius_1_4 = crop_radius // 4
        #     ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index- 3])[:, :, ::-1]
        #     ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
        #     ref_img_crop = ref_img[
        #             ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
        #             ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[33, 0] + crop_radius +crop_radius_1_4,
        #             :]
        #     ref_img_crop = cv2.resize(ref_img_crop,(resize_w,resize_h))
        #     ref_img_crop = ref_img_crop / 255.0
        #     ref_img_list.append(ref_img_crop)
        # ref_video_frame = np.concatenate(ref_img_list, 2)
        # ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()
        # torch.save(ref_img_tensor, "five_selected_ref_img.pt")

        print('随机选择5个图片的时间开销(秒):', time.time() - time_stamp)

        ############################################## inference frame by frame ##############################################
        time_stamp = time.time()

        # if not os.path.exists(opt.res_video_dir):
        #     os.mkdir(opt.res_video_dir)
        
        if os.path.exists(mp4_file):
            os.remove(mp4_file)
        videowriter = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc(*'XVID'), 25, video_size)

        # res_face_path = res_video_path.replace('_facial_dubbing.mp4', '_synthetic_face.mp4')
        # if os.path.exists(res_face_path):
        #     os.remove(res_face_path)
        # videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (resize_w, resize_h))
        time_stamp = time.time()
        for clip_end_index in range(5, pad_length, 1):
            # print('synthesizing {}/{} frame'.format(clip_end_index - 5, pad_length - 5))
            crop_flag, crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],random_scale = 1.05)
            if not crop_flag:
                raise ('our method can not handle videos with large change of facial size!!')
            crop_radius_1_4 = crop_radius // 4
            frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
            frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
            crop_frame_data = frame_data[
                                frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                                frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[33, 0] + crop_radius +crop_radius_1_4,
                                :]
            crop_frame_h,crop_frame_w = crop_frame_data.shape[0],crop_frame_data.shape[1]
            crop_frame_data = cv2.resize(crop_frame_data, (resize_w,resize_h))  # [32:224, 32:224, :]
            crop_frame_data = crop_frame_data / 255.0
            crop_frame_data[opt.mouth_region_size//2:opt.mouth_region_size//2 + opt.mouth_region_size,
                            opt.mouth_region_size//8:opt.mouth_region_size//8 + opt.mouth_region_size, :] = 0

            crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
            deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
                pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
            # videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
            pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w,crop_frame_h))
            frame_data[
            frame_landmark[29, 1] - crop_radius:
            frame_landmark[29, 1] + crop_radius * 2,
            frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
            frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
            :] = pre_frame_resize[:crop_radius * 3,:,:]
            videowriter.write(frame_data[:, :, ::-1])
        
        # print('一共推理图片: ', pad_length - 5)
        # print('时间开销(秒):', time.time() - time_stamp)
        # print('总的帧率: ', (pad_length - 5) / (time.time() - time_stamp))

        videowriter.release()
        # videowriter_face.release()

        # video_add_audio_path = res_video_path.replace('_video.mp4', '.mp4')
        # if os.path.exists(video_add_audio_path):
        #     os.remove(video_add_audio_path)
        # cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
        #     res_video_path,
        #     wav_file,
        #     video_add_audio_path)
        
        print('一共推理图片: ', pad_length - 5)
        print('总的帧率: ', (pad_length - 5) / (time.time() - time_stamp))

        # 计算下面的耗时
        # ffmepg_start_time = time.time()
        # subprocess.call(cmd, shell=True)
        # print('ffmepg_start_time  合成耗时: ', time.time() - ffmepg_start_time)
        
        # 删除 res_video_path
        # os.remove(res_video_path)
        # os.remove(wav_file)
        
        end_time = time.time()
        # 计算程序的总耗时
        print("总耗时: ", end_time - __start_time)

    ############################################## TTS 核心逻辑 ##############################################
    def save_wav(filename, data):
        wavfile.write(filename, sample_rate, data.astype(np.int16))

    def tts_on_error(message, *args):
        print("on_error args=>{}".format(message))

    def tts_on_close(*args):
        print("on_close: args=>{}".format(args))

    def tts_on_completed(message, *args):
        user_id = args[0]
        global data_buffer
        if len(data_buffer) > 0:
            data_queue.put({'rawdata': data_buffer[:sample_rate], 'user_id': user_id})  # put any remaining data in the queue
            data_buffer = data_buffer[sample_rate:]

    def tts_on_data(data, *args):
        user_id = args[0]
        global data_buffer
        current_data = np.frombuffer(data, dtype=np.int16)
        data_buffer = np.concatenate((data_buffer, current_data))
        
        while len(data_buffer) >= sample_rate:  # 1 second of audio data
            data_queue.put({'rawdata': data_buffer[:sample_rate], 'user_id': user_id})  # put the data in the queue
            data_buffer = data_buffer[sample_rate:]

    def process_data():
        global file_count, FILE_NAME_SPLIT
        while True:
            data = data_queue.get()
            wav_file = "/home/ubuntu/code/DINet/asserts/inference_result/{}{}{}.wav".format(data['user_id'], FILE_NAME_SPLIT, file_count)
            mp4_file = "/home/ubuntu/code/DINet/asserts/inference_result/{}{}{}.mp4".format(data['user_id'], FILE_NAME_SPLIT, file_count)
            jump_file = "/home/ubuntu/code/DINet/asserts/inference_result/{}{}{}".format(data['user_id'], FILE_NAME_SPLIT, file_count)
            save_wav(wav_file, data['rawdata'])
            infer_process(data['rawdata'], wav_file, mp4_file) # 推理

            # 选择对的 player jump in 数据
            try:
                players[data['user_id']].jump_in(jump_file)
            except print(0):
                pass
            file_count += 1

    
    ############################################## 开始推理  开始推理  开始推理  开始推理 ##############################################

    ############################################## 监听队列消息 ##############################################
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='49.234.229.39', port='5672'))
    channel = connection.channel()
    channel.queue_declare(queue='q_tts')

    # 接收 gpt 文本，进行 tts 语音合成
    def tts_cb(ch, method, properties, body, test_str = None):
        gpt_text = body.decode('utf-8') if body is not None else None
        tts_data = json.loads(body)
        
        tts = nls.NlsSpeechSynthesizer(
            url=URL,
            token=TOKEN,
            appkey=APPKEY,
            on_data=tts_on_data,
            on_completed=tts_on_completed,
            on_error=tts_on_error,
            on_close=tts_on_close,
            callback_args=[tts_data['user_id']])

        
        print(" [x] 开始处理文字： %r" % tts_data['text'])
        if gpt_text is not None:
            tts.start(tts_data['text'], voice="ailun", aformat="wav", sample_rate=16000)

    # 启动socket服务 
    def run_asyncio_ws():
        asyncio.run(init_websocket())

    threading.Thread(target=run_asyncio_ws).start()

    # callback(1, 2, 3, 4, opt.driving_audio_path)
    # callback(1, 2, 3, 4, "utils/tmptz3ewf1z.wav")
    # print(" [*] Waiting for process_data")
    threading.Thread(target=process_data).start()
    print(" [*] Waiting for rabbitmq messages.")

    channel.basic_consume(queue='q_tts', on_message_callback=tts_cb, auto_ack=True)
    channel.start_consuming()

 
   







