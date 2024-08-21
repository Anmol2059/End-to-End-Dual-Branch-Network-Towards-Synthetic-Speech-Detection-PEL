import os
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy import signal
import torch.nn as nn
# from resnet2 import ResNet, TypeClassifier, SelfEnhancementModule, MutualEnhancementModule
from resnet2 import ResNet, SelfEnhancementModule, MutualEnhancementModule , TypeClassifier

import torch.nn.functional as F

# Use appropriate default dtype and device
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(device)

def load_wav_snf(path):
    wav, sr = sf.read(path, dtype=np.float32)
    return wav

def preemphasis(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

def logpowcqt(wav_path, sr=16000, hop_length=512, n_bins=84, bins_per_octave=12, window="hann", fmin=1.0, pre_emphasis=0.97, ref=1.0, amin=1e-30, top_db=None):
    wav = load_wav_snf(wav_path)
    if pre_emphasis is not None:
        wav = preemphasis(wav, k=pre_emphasis)
    cqtfeats = librosa.cqt(y=wav, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, window=window, fmin=fmin)
    magcqt = np.abs(cqtfeats)
    powcqt = np.square(magcqt)
    logpowcqt = librosa.power_to_db(powcqt, ref=ref, amin=amin, top_db=top_db)
    return logpowcqt

def extract_lfcc(audio_path, sr=16000, n_mfcc=20):
    wav, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def getFakeFeature(feature, label):
    f = []
    l = []
    for i in range(0, label.shape[0]):
        if label[i] != 20:
            l.append(label[i])
            f.append(feature[i])
    f = torch.stack(f)
    l = torch.stack(l)
    return f, l


# Instead of making adjustments to the original resnet I did this
def calculate_target_size(num_nodes, min_time_steps=3):
    freq_dim = num_nodes * 24  # Multiply by 24 to account for the initial stride
    freq_dim = max(freq_dim, 84)  # 84 is the original CQT dimension,
    time_dim = max(min_time_steps * 8, 313)  # Multiply by 8 to account for the three stride-2 layers
    return freq_dim, time_dim

def pad_tensor(tensor, target_freq, target_time):
    current_freq, current_time = tensor.size(2), tensor.size(3)
    freq_padding = max(0, target_freq - current_freq)
    time_padding = max(0, target_time - current_time)
    return F.pad(tensor, (0, time_padding, 0, freq_padding))
def train_pipeline():
    audio_path = 'sample_audio.flac'  

    print("Initializing audio processing...")

    # CQT extraction
    print("Extracting CQT features...")
    cqt_features = logpowcqt(audio_path)
    cqt_tensor = torch.tensor(cqt_features).unsqueeze(0).unsqueeze(0).float().to(device)
    print("CQT features shape:", cqt_tensor.shape)

    # LFCC extraction
    print("Extracting LFCC features...")
    lfcc_features = extract_lfcc(audio_path)
    lfcc_tensor = torch.tensor(lfcc_features).unsqueeze(0).unsqueeze(0).float().to(device)
    print("LFCC features shape:", lfcc_tensor.shape)

    # Initialize models
    print("Initializing models...")
    num_nodes = 256  # This should match the value used when initializing ResNet
    resnet_lfcc = ResNet(num_nodes, 256, resnet_type='18', nclasses=2).to(device)
    resnet_cqt = ResNet(num_nodes, 256, resnet_type='18', nclasses=2).to(device)
    classifier_lfcc = TypeClassifier(256, 6, 0.05, ADV=True).to(device)
    classifier_cqt = TypeClassifier(256, 6, 0.05, ADV=True).to(device)
    print("Models initialized.")

    # Calculate target sizes and pad tensors
    target_size_freq, target_size_time = calculate_target_size(num_nodes)

    lfcc_tensor_padded = F.pad(lfcc_tensor, (0, max(0, target_size_time - lfcc_tensor.size(3)),
                                             0, max(0, target_size_freq - lfcc_tensor.size(2))))
    cqt_tensor_padded = F.pad(cqt_tensor, (0, max(0, target_size_time - cqt_tensor.size(3)),
                                           0, max(0, target_size_freq - cqt_tensor.size(2))))

    print("Padded LFCC tensor shape:", lfcc_tensor_padded.shape)
    print("Padded CQT tensor shape:", cqt_tensor_padded.shape)

    # Forward pass through models
    print("Passing LFCC features through model...")
    feature_lfcc, out_lfcc, _ = resnet_lfcc(lfcc_tensor_padded)
    print("LFCC features shape after ResNet:", feature_lfcc.shape)
    
    print("Passing CQT features through model...")
    feature_cqt, out_cqt, _ = resnet_cqt(cqt_tensor_padded)
    print("CQT features shape after ResNet:", feature_cqt.shape)

    # Forward pass through classifiers
    print("Extracting fake features and passing through classifiers...")
    feature_fake_lfcc, fakelabel_lfcc = getFakeFeature(feature_lfcc, torch.tensor([1]))  # Assuming fake label 1 for testing
    typepred_lfcc = classifier_lfcc(feature_fake_lfcc)
    print("Type prediction for LFCC:", typepred_lfcc)

    feature_fake_cqt, fakelabel_cqt = getFakeFeature(feature_cqt, torch.tensor([1]))  # Assuming fake label 1 for testing
    typepred_cqt = classifier_cqt(feature_fake_cqt)
    print("Type prediction for CQT:", typepred_cqt)

    print("LFCC features shape after processing:", feature_lfcc.shape if 'feature_lfcc' in locals() else "Not available")
    print("CQT features shape after processing:", feature_cqt.shape if 'feature_cqt' in locals() else "Not available")

if __name__ == '__main__':
    train_pipeline()
