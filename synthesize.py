import torch
import torch.nn as nn
import torch.multiprocessing as mp
from scipy.io import wavfile
import numpy as np
import hparams as hp
import os

import argparse
import re
from string import punctuation

from fastspeech2 import FastSpeech2
from vocoder import vocgan_generator

from text import text_to_sequence, sequence_to_text
import utils
import audio as Audio

import codecs
from g2pk import G2p
from jamo import h2j

device = torch.device('cuda')

def cut_idx(signal):
    for i in range(int(len(signal)/2), len(signal)):
        if (np.absolute(signal[i:i+5000]) < 100).all():
            return i
    return len(signal)

def pause(text):
	out = []
	for i in range(len(text)):
		if text[i:i+2] in  ['가 ', '이 ', '에 ', '는 ', '은 ']: #'를 ', '을 ' '의 '
			out += text[i]
			out += ','
		else:
			out += text[i]
	return "".join(out)

def split_text(text):
    text = text.split(" ")
    texts = []
    for i in range(1, len(text)):
        text[i] = ";;"+text[i] if i % 4 == 0 else text[i]
    for i in range(0, len(text), 4):
        joint = " ".join(text[i:i+5])
        texts.append(joint if not joint[:2] == ";;" else joint[2:] )
    if len(texts[-1].split(" ")) == 1:
        texts[-2] = texts[-2].split(";;")[0] + texts[-1]
        del texts[-1]
    return texts

def kor_preprocess(text):
    text = text.rstrip(punctuation)
    
    g2p=G2p()
    phone = g2p(text)
    print('after g2p: ',phone)
    phone = h2j(phone)
    print('after h2j: ',phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ',phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    print('after re.sub: ',phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone,hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, vocoder, text, sentence, prefix=''):
    sentence = sentence[:10] # long filename will result in OS Error

    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
        
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len)
    
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    f0_output = f0_output[0]
    energy_output = energy_output[0]

    mel_torch = utils.de_norm(mel_torch.transpose(1, 2), mean_mel, std_mel)
    mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(1, 2), mean_mel, std_mel).transpose(1, 2)
    f0_output = utils.de_norm(f0_output, mean_f0, std_f0).squeeze().detach().cpu().numpy()
    energy_output = utils.de_norm(energy_output, mean_energy, std_energy).squeeze().detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    if vocoder is not None:
        if hp.vocoder.lower() == "vocgan":
            return utils.vocgan_infer(mel_postnet_torch, vocoder, path=os.path.join(""))   
    
if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()

    
    model = get_FastSpeech2(args.step).to(device)
    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
    else:
        vocoder = None   
 
    g2p=G2p()

    print('Input sentence: ')
    sentence = input()

    sentence = split_text(sentence)
    cuts = []
    for e, s in enumerate(sentence):   
        text = kor_preprocess(s)
        audio = synthesize(model, vocoder, text, sentence, prefix='{}'.format(e))
        cuts.append(audio[:cut_idx(audio)])
    joint = np.hstack(cuts)
    wavfile.write("result.wav", hp.sampling_rate, joint)
