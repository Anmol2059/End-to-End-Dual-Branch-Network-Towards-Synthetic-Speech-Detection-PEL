import os
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy import signal
import torch.nn as nn
from resnet1 import ResNet, TypeClassifier  # Ensure resnet.py is in the same directory or in the PYTHONPATH
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

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

def calculate_target_size(num_nodes, min_time_steps=3):
    freq_dim = num_nodes * 24  # Multiply by 24 for the initial stride
    freq_dim = max(freq_dim, 84)  # 84 is the original CQT dimension, adjust if needed
    time_dim = max(min_time_steps * 8, 313)  # Multiply by 8  for the three stride-2 layers
    return freq_dim, time_dim

def pad_tensor(tensor, target_freq, target_time):
    current_freq, current_time = tensor.size(2), tensor.size(3)
    freq_padding = max(0, target_freq - current_freq)
    time_padding = max(0, target_time - current_time)
    return F.pad(tensor, (0, time_padding, 0, freq_padding))

def test_pipeline():
    audio_path = 'sample_audio.flac'  # Replace with your audio file path
    
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
    
    target_size_freq, target_size_time = calculate_target_size(num_nodes)
    
    lfcc_tensor_padded = F.pad(lfcc_tensor, (0, max(0, target_size_time - lfcc_tensor.size(3)), 
                                             0, max(0, target_size_freq - lfcc_tensor.size(2))))
    cqt_tensor_padded = F.pad(cqt_tensor, (0, max(0, target_size_time - cqt_tensor.size(3)), 
                                           0, max(0, target_size_freq - cqt_tensor.size(2))))
    
    print("Padded LFCC tensor shape:", lfcc_tensor_padded.shape)
    print("Padded CQT tensor shape:", cqt_tensor_padded.shape)
    
    # Forward pass through models
    print("Passing LFCC features through model...")
    try:
        feature_lfcc, out_lfcc = resnet_lfcc(lfcc_tensor_padded)
        print("LFCC features enhanced.")
        print("LFCC features shape after ResNet:", feature_lfcc.shape)
        print("LFCC output shape after ResNet:", out_lfcc.shape)
        print("LFCC forward pass successful:", feature_lfcc is not None and out_lfcc is not None)
    except Exception as e:
        print("Error during LFCC forward pass:", e)
        import traceback
        traceback.print_exc()
    
    print("Passing CQT features through model...")
    try:
        feature_cqt, out_cqt = resnet_cqt(cqt_tensor_padded)
        print("CQT features enhanced.")
        print("CQT features shape after ResNet:", feature_cqt.shape)
        print("CQT output shape after ResNet:", out_cqt.shape)
        print("CQT forward pass successful:", feature_cqt is not None and out_cqt is not None)
    except Exception as e:
        print("Error during CQT forward pass:", e)
        import traceback
        traceback.print_exc()
    
    print("Extracting fake features and passing through classifiers...")
    try:
        feature_fake_lfcc, fakelabel_lfcc = getFakeFeature(feature_lfcc, torch.tensor([1]))  # Assuming fake label 1 for testing
        typepred_lfcc = classifier_lfcc(feature_fake_lfcc)
        print("Type prediction for LFCC:", typepred_lfcc)
    except Exception as e:
        print("Error during LFCC classification:", e)
        import traceback
        traceback.print_exc()
    
    try:
        feature_fake_cqt, fakelabel_cqt = getFakeFeature(feature_cqt, torch.tensor([1]))  # Assuming fake label 1 for testing
        typepred_cqt = classifier_cqt(feature_fake_cqt)
        print("Type prediction for CQT:", typepred_cqt)
    except Exception as e:
        print("Error during CQT classification:", e)
        import traceback
        traceback.print_exc()
    
    print("LFCC features shape after processing:", feature_lfcc.shape if 'feature_lfcc' in locals() else "Not available")
    print("CQT features shape after processing:", feature_cqt.shape if 'feature_cqt' in locals() else "Not available")

if __name__ == '__main__':
    test_pipeline()