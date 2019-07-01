import numpy as np
from PIL import Image
from scipy.fftpack import dct


im = Image.open("test.png")
im = im.convert("L")
im = np.array(im)

#im = im[:128,:128,0]

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
blocks = np.array(blocks).astype(np.float32) -128.0


# Compute matrix T
T = np.zeros([N,N],dtype=np.float32)

for j in range(N):
	for i in range(N):
		if j == 0:
			T[j,i] = 1.0 / np.sqrt(N)
		else:
			T[j,i] = 0.5 * np.cos((np.pi * (j * (2.0*i+1.0))) / (2.0*N))

print (np.around(T,4))


# Modify image

def DCT(block):
	return np.matmul(np.matmul(T,block),T.T)
def IDCT(block):
	return np.matmul(np.matmul(T.T,block),T)


print (IDCT(DCT(im[:8,:8])).astype(np.uint8))
print (im[:8,:8])

# Perform Dct on blocks
dct_blocks = np.zeros_like(blocks)
for y in range(blocks.shape[0]):
	for x in range(blocks.shape[1]):
		dct_blocks[y,x] = DCT(blocks[y,x])

# DCT Image
dct_changed_im_out = np.zeros_like(im)
for y in range(dct_blocks.shape[0]):
	for x in range(dct_blocks.shape[1]):
		dct_changed_im_out[y*N:y*N+N,x*N:x*N+N] = dct_blocks[y,x]

dct_changed_im_out = Image.fromarray(dct_changed_im_out)
dct_changed_im_out.save("dct_changed_im_out.png")


Q_mat = np.array([	[16, 11, 10, 16, 24, 40, 51, 61],
					[12,12,14,19,26,58,60,55],
					[14,13,16,24,40,57,69,56],
					[14,17,22,29,51,87,80,62],
					[18,22,37,56,68,109,103,77],
					[24,35,55,64,81,104,113,92],
					[49,64,78,87,103,121,120,101],
					[72,92,95,98,112,100,103,99]],dtype=np.float32)
Q_mat_flip = np.flipud(np.fliplr(Q_mat))

Z_mat_dist = 1
Z_mat = np.ones([N,N],dtype=np.float32)
for y in range(N):
	for x in range(N):
		if y >= Z_mat_dist or x >= Z_mat_dist:
			Z_mat[y,x] = 1024
print (Z_mat)
'''
Z_mat = np.array([	[1, 1, 1, 1, 255, 255, 255, 255],
					[1, 1, 1, 1, 255, 255, 255, 255],
					[1, 1, 1, 1, 255, 255, 255, 255],
					[1, 1, 1, 1, 255, 255, 255, 255],
					[255, 255, 255, 255, 255, 255, 255, 255],
					[255, 255, 255, 255, 255, 255, 255, 255],
					[255, 255, 255, 255, 255, 255, 255, 255],
					[255, 255, 255, 255, 255, 255, 255, 255]],dtype=np.float32)
'''
Z_mat_flip = np.flipud(np.fliplr(Z_mat))


def Q_im(mat,name):
	dct_blocks_MAT = np.zeros_like(dct_blocks)
	for y in range(dct_blocks.shape[0]):
		for x in range(dct_blocks.shape[1]):
			dct_blocks_MAT[y,x] = mat*np.round(dct_blocks[y,x]/mat)

	# DCT Out Image
	reconst_out_im = np.zeros_like(im)
	for y in range(dct_blocks_MAT.shape[0]):
		for x in range(dct_blocks_MAT.shape[1]):
			reconst_out_im[y*N:y*N+N,x*N:x*N+N] = dct_blocks_MAT[y,x]

	reconst_out_im = Image.fromarray(reconst_out_im)
	reconst_out_im.save(name+"_dct_out.png")

	# Reconstructed Image
	reconst_mat_im = np.zeros_like(im)
	for y in range(dct_blocks_MAT.shape[0]):
		for x in range(dct_blocks_MAT.shape[1]):
			reconst_mat_im[y*N:y*N+N,x*N:x*N+N] = (IDCT(dct_blocks_MAT[y,x])+128.0).astype(np.uint8)

	reconst_mat_im = Image.fromarray(reconst_mat_im)
	reconst_mat_im.save(name+".png")


Q_im(Q_mat,"q")
Q_im(Q_mat_flip, "q_flip")

Q_im(Z_mat,"z")
Q_im(Z_mat_flip, "z_flip")
