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

  
# 定义rabitmq的链接
channel = None

# 音频处理设置信息
data_queue = queue.Queue()
URL= "wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1"
TOKEN= "fb3f966388324e2189f553919a782e07"
APPKEY="FCW8uluerIsGU24l"
sample_rate = 16000
bytes_per_sample = 2  # 16-bit PCM
data_buffer = np.array([])
file_count = 0


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

    ############################################## 核心推理函数 ##############################################
    def infer_process(wav_data, wav_file, file_count):
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

        ref_img_list = []
        resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
        resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
        ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
        for ref_index in ref_index_list:
            crop_flag,crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
            if not crop_flag:
                raise ('our method can not handle videos with large change of facial size!!')
            crop_radius_1_4 = crop_radius // 4
            ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index- 3])[:, :, ::-1]
            ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
            ref_img_crop = ref_img[
                    ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                    ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[33, 0] + crop_radius +crop_radius_1_4,
                    :]
            ref_img_crop = cv2.resize(ref_img_crop,(resize_w,resize_h))
            ref_img_crop = ref_img_crop / 255.0
            ref_img_list.append(ref_img_crop)
        ref_video_frame = np.concatenate(ref_img_list, 2)
        ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

        print('随机选择5个图片的时间开销(秒):', time.time() - time_stamp)

        ############################################## inference frame by frame ##############################################
        time_stamp = time.time()

        if not os.path.exists(opt.res_video_dir):
            os.mkdir(opt.res_video_dir)
        
        res_video_path = os.path.join(opt.res_video_dir, os.path.basename(opt.source_video_path)[:-4] + f'_{file_count}_video.mp4')
        if os.path.exists(res_video_path):
            os.remove(res_video_path)
        videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, video_size)

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

        video_add_audio_path = res_video_path.replace('_video.mp4', '.mp4')
        if os.path.exists(video_add_audio_path):
            os.remove(video_add_audio_path)
        cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
            res_video_path,
            wav_file,
            video_add_audio_path)
        
        print('一共推理图片: ', pad_length - 5)
        print('时间开销(秒):', time.time() - time_stamp)
        print('总的帧率: ', (pad_length - 5) / (time.time() - time_stamp))
        subprocess.call(cmd, shell=True)
        
        # 删除 res_video_path
        os.remove(res_video_path)
        os.remove(wav_file)
        
        end_time = time.time()
        # 计算程序的总耗时
        print("总耗时: ", end_time - __start_time)

    ############################################## TTS 核心逻辑 ##############################################
    def save_wav(filename, data):
        wavfile.write(filename, sample_rate, data.astype(np.int16))

    def tts_on_error(message, *args):
        print("on_error args=>{}".format(args))

    def tts_on_close(*args):
        print("on_close: args=>{}".format(args))

    def tts_on_completed(message, *args):
        global data_buffer
        if data_buffer:
            data_queue.put(data_buffer)  # put any remaining data in the queue
            data_buffer.clear()
        print("on_completed:args=>{} message=>{}".format(args, message))

    def tts_on_data(data):
        global data_buffer
        current_data = np.frombuffer(data, dtype=np.int16)
        data_buffer = np.concatenate((data_buffer, current_data))
        
        while len(data_buffer) >= sample_rate:  # 1 second of audio data
            data_queue.put(data_buffer[:sample_rate])  # put the data in the queue
            data_buffer = data_buffer[sample_rate:]

    def process_data():
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='49.234.229.39', port='5672'))
        q_video_channel = connection.channel()
        q_video_channel.queue_declare(queue='q_video')
        
        global file_count
        while True:
            data = data_queue.get(block=True, timeout=None)
            if data is None:  # sentinel value to exit the loop
                continue
            wav_file = "ali_tts_part_{}.wav".format(file_count)
            save_wav(wav_file, data)
            infer_process(data, wav_file, file_count) # 推理

            # 发送消息
            filename = f"/home/ubuntu/code/DINet/asserts/inference_result/cofi_{file_count}.mp4"
            q_video_channel.basic_publish(exchange='', routing_key='q_video', body=filename)
            file_count += 1

    tts = nls.NlsSpeechSynthesizer(
        url=URL,
        token=TOKEN,
        appkey=APPKEY,
        on_data=tts_on_data,
        on_completed=tts_on_completed,
        on_error=tts_on_error,
        on_close=tts_on_close)
    ############################################## 开始推理  开始推理  开始推理  开始推理 ##############################################

    ############################################## 监听队列消息 ##############################################
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='49.234.229.39', port='5672'))
    channel = connection.channel()
    channel.queue_declare(queue='q_tts')

    # 接收 gpt 文本，进行 tts 语音合成
    def tts_cb(ch, method, properties, body, test_str = None):
        gpt_text = body.decode('utf-8') if body is not None else None
        print(" [x] 开始处理文字： %r" % gpt_text)
        if gpt_text is not None:
            tts.start(gpt_text, voice="ailun", aformat="wav", sample_rate=16000)


    # callback(1, 2, 3, 4, opt.driving_audio_path)
    # callback(1, 2, 3, 4, "utils/tmptz3ewf1z.wav")

    data_thread = threading.Thread(target=process_data)
    data_thread.start()

    channel.basic_consume(queue='q_tts', on_message_callback=tts_cb, auto_ack=True)
    channel.start_consuming()

   







