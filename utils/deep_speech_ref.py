import os
import time

import numpy as np
import warnings
import resampy
from scipy.io import wavfile
from python_speech_features import mfcc
import tensorflow as tf
# import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.reset_default_graph()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)



# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)


class DeepSpeech():
    def __init__(self,model_path):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        self.graph, self.logits_ph, self.input_node_ph, self.input_lengths_ph \
            = self._prepare_deepspeech_net(model_path)
        self.target_sample_rate = 16000
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto(device_count={'GPU': 1}))

    def _prepare_deepspeech_net(self,deepspeech_pb_path):
        with tf.io.gfile.GFile(deepspeech_pb_path, "rb") as f:
            # graph_def = tf.compat.v1.GraphDef()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        # graph = tf.compat.v1.get_default_graph()
        graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name="deepspeech")
        logits_ph = graph.get_tensor_by_name("deepspeech/logits:0")
        input_node_ph = graph.get_tensor_by_name("deepspeech/input_node:0")
        input_lengths_ph = graph.get_tensor_by_name("deepspeech/input_lengths:0")

        return graph, logits_ph, input_node_ph, input_lengths_ph

    def conv_audio_to_deepspeech_input_vector(self,
                                              audio,
                                              sample_rate,
                                              num_cepstrum,
                                              num_context):
        # Get mfcc coefficients:
        features = mfcc(
            signal=audio,
            samplerate=sample_rate,
            numcep=num_cepstrum)

        # We only keep every second feature (BiRNN stride = 2):
        features = features[::2]

        # One stride per time step in the input:
        num_strides = len(features)

        # Add empty initial and final contexts:
        empty_context = np.zeros((num_context, num_cepstrum), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future):
        window_size = 2 * num_context + 1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            shape=(num_strides, window_size, num_cepstrum),
            strides=(features.strides[0],
                     features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions:
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / \
                       np.std(train_inputs)

        return train_inputs


    def compute_audio_feature_from_1s_16k(self, audio):
        # saver = tf.train.Saver()
        resampled_audio = audio # np.frombuffer(audio)
        # resampled_audio = resampy.resample(
        #         x=audio.astype(np.float),
        #         sr_orig=16000,
        #         sr_new=self.target_sample_rate)
        import time
        # with tf.Session(graph=self.graph, config=tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})) as sess:
        with tf.compat.v1.Session(graph=self.graph) as sess:
            s = time.thread_time_ns()

            input_vector = self.conv_audio_to_deepspeech_input_vector(
                audio=resampled_audio.astype(np.int16),
                sample_rate=self.target_sample_rate,
                num_cepstrum=26,
                num_context=9)

            e = time.thread_time_ns()
            print('conv_audio_to_deepspeech_input_vector time: {}ms'.format((e - s) / 1000000))

            s = time.thread_time_ns()

            network_output = sess.run(
                self.logits_ph,
                feed_dict={
                    self.input_node_ph: input_vector[np.newaxis, ...],
                    self.input_lengths_ph: [input_vector.shape[0]]
                }
            )

            e = time.thread_time_ns()
            print('network_output time: {}ms'.format((e - s) / 1000000))

            # saver_path = saver.save(sess, "save/model.ckpt")
            ds_features = network_output[::2,0,:]
        return ds_features


    def compute_audio_feature_from_frames(self, audio_frames):
        audio = audio_frames.astype(np.float)
        input_vector = self.conv_audio_to_deepspeech_input_vector(
            audio=audio.astype(np.int16),
            sample_rate=self.target_sample_rate,
            num_cepstrum=26,
            num_context=9)
        network_output = self.sess.run(
            self.logits_ph,
            feed_dict={
                self.input_node_ph: input_vector[np.newaxis, ...],
                self.input_lengths_ph: [input_vector.shape[0]]
            }
        )
        ds_features = network_output[::2, 0, :]
        return ds_features

    def compute_audio_feature(self, audio_path):
        audio_sample_rate, audio = wavfile.read(audio_path)
        if audio.ndim != 1:
            warnings.warn(
                "Audio has multiple channels, the first channel is used")
            audio = audio[:, 0]
        if audio_sample_rate != self.target_sample_rate:
            resampled_audio = resampy.resample(
                x=audio.astype(np.float),
                sr_orig=audio_sample_rate,
                sr_new=self.target_sample_rate)
        else:
            resampled_audio = audio.astype(np.float)

        # saver = tf.train.Saver()
        # with tf.compat.v1.Session(graph=self.graph) as sess:
        # with tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto(device_count={'GPU': 1})) as sess:
        input_vector = self.conv_audio_to_deepspeech_input_vector(
            audio=resampled_audio.astype(np.int16),
            sample_rate=self.target_sample_rate,
            num_cepstrum=26,
            num_context=9)
        time_stamp = time.time()
        network_output = self.sess.run(
            self.logits_ph,
            feed_dict={
                self.input_node_ph: input_vector[np.newaxis, ...],
                self.input_lengths_ph: [input_vector.shape[0]]
            }
        )
        print('初始化 seesion 的时间(秒):', time.time() - time_stamp)
        # saver_path = saver.save(sess, "save/model.ckpt")
        ds_features = network_output[::2,0,:]
        return ds_features

def test():
    audio_path = r'../asserts/examples/1s.wav'
    model_path = r'../asserts/output_graph.pb'

    DSModel = DeepSpeech(model_path)
    
    ds_feature = DSModel.compute_audio_feature(audio_path)
    # ds_feature = DSModel.compute_audio_feature_from_1s_16k(np.random.randint(0, 65530, size=(64000,)))
    # print(ds_feature)
    



if __name__ == '__main__':
    # audio_path = r'asserts/examples/tts2.wav'
    # model_path = r'asserts/output_graph.pb'
    # DSModel = DeepSpeech(model_path)
    # # ds_feature = DSModel.compute_audio_feature(audio_path)
    # ds_feature = DSModel.compute_audio_feature_from_1s_16k(np.random.randint(0, 65530, size=(64000,)))
    # print(ds_feature)

    import threading
    threading.Thread(target=test).start()