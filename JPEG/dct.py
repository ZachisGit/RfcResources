import numpy as np
from PIL import Image


im = Image.open("test.png")
im = im.convert("RGB")
im = np.array(im)

im = im[:128,:128,0]

# Save ori
im_out = Image.fromarray(im)
im_out.save("ori.png")


# VARIABLES
N = 8

# Divide in 8x8 blocks
blocks = []

# Works only on image sizes divisible by N
for y in range(0,im.shape[0],N):
	blocks.append([])
	for x in range(0,im.shape[1],N):
		blocks[y//N].append(im[y:y+N,x:x+N])

# Compute matrix T
T = np.zeros([N,N],dtype=np.float32)

for j in range(N):
	for i in range(N):

		ai = 1.0#/np.sqrt(N) if i == 0 else np.sqrt(2.0/N)
		aj = 1.0/np.sqrt(N) if j == 0 else np.sqrt(2.0/N)

		cos_0 = np.cos((np.pi * ((2.0*j)+1.0) * j)/(2.0*N))
		cos_1 = np.cos((np.pi * ((2.0*i)+1.0) * i)/(2.0*N))

		T[j,i] = ai*aj*cos_0#*cos_1

		'''

		if j == 0:
			T[j,i] = 1.0 / np.sqrt(N)
		else:
			T[j,i] = np.sqrt(2.0/N) * (np.cos(((2.0*j)+1)*i*np.pi)/(2.0*N))
		'''

print (np.around(T,4))
