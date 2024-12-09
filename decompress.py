#!/usr/bin/env python

import struct
import wave
import numpy as np
import scipy as sp
from scipy import fftpack

NUMCOEFF = 326000

fin = open('compressed','rb')

inbytes = fin.read();
fin.close()

coeffintlist = [struct.unpack('<h',inbytes[2*i:2*i+2]) for i in range(70000)]

coeffint = np.array(coeffintlist[0:35000]) + np.array(coeffintlist[35000:])*1j

allcoeffint = np.append(coeffint, np.zeros(NUMCOEFF - 2*(len(coeffint)-1) - 1))
allcoeffint = np.append(allcoeffint, coeffint[len(coeffint)-1:0:-1].conjugate())

samples = sp.fftpack.ifft(allcoeffint)

samplesint = [ int(round(32768*x)) for x in samples.real]
outbytes = [struct.pack('<h',i) for i in samplesint]
strout = b''.join(e for e in outbytes)

fp = wave.open('out.wav','wb')

fp.setparams((1, 2, 44100, 326000, 'NONE', 'not compressed'))
fp.writeframes(strout)
fp.close()
