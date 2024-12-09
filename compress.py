#!/usr/bin/env python

# Needed for struct.pack

import struct
import wave
import numpy as np
import scipy as sp
from scipy import fftpack

# Open the input and output files. 'wb' is needed on some platforms
# to indicate that 'compressed' is a binary file.

fin = wave.open('step.wav','r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = fin.getparams()
inbytes = fin.readframes(nframes)
fin.close()

samplesint = [struct.unpack('<h',inbytes[2*i:2*i+2]) for i in range(nframes)]

samplesfloat = [float(x[0])/(2**15) for x in samplesint]
samples = np.array(samplesfloat)

F = sp.fftpack.fft(samples)

truncreal = [int(round(x.real)) for x in F[0:35000]]
truncimag = [int(round(x.imag)) for x in F[0:35000]]

writedata = truncreal+truncimag

#outbytes = bytearray()
#or i in writedata:
#   outbytes.append(i)
#
outbytes = [struct.pack('<h',i) for i in writedata] 
strout = b''.join(e for e in outbytes)

fout = open('compressed','wb')
fout.write(strout)
fout.close()
