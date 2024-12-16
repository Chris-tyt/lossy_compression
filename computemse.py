#!/usr/bin/env python

# Needed for struct.pack

import struct
import wave
import numpy as np
import scipy as sp

# Open the input and output files. 'wb' is needed on some platforms
# to indicate that 'compressed' is a binary file.

fin = wave.open('step.wav','r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

samplesint = [struct.unpack('<h',inbytes[2*i:2*i+2]) for i in range(nframes)]

samplesfloat = [float(x[0])/(2**15) for x in samplesint]
samples1 = np.array(samplesfloat)

fin = wave.open('out.wav','r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

samplesint = [struct.unpack('<h',inbytes[2*i:2*i+2]) for i in range(nframes)]
samplesfloat = [float(x[0])/(2**15) for x in samplesint]
samples2 = np.array(samplesfloat)

# # 截断第二个样本到第一个样本的长度
# samples2 = samples2[:len(samples1)]
# samples1 = samples1[:len(samples2)]

dimens = samples1.shape
print('MSE: {}'.format(sum((samples1 - samples2)**2)/dimens[0]))
# print('MSE: {}'.format(sum((samples1[0:-368] - samples2[0:-368])**2)/(dimens[0]-368)))
