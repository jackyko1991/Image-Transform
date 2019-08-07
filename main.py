import SimpleITK as sitk
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def load_img(path,dimension,view=False):
	# load image data
	image = Image.open(path)

	# resize to desired size
	old_size = image.size

	ratio = float(dimension)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	image = image.resize(new_size, Image.ANTIALIAS)

	# create image and paste the resize image
	image_new = Image.new("RGB",(dimension,dimension))
	image_new.paste(image,((dimension-new_size[0])//2,(dimension-new_size[1])//2))

	if view:
		image_new.show()

	return image_new

def load2D():
	DIMS = 400
	IMAGE1 = './data/lena.png'
	IMAGE2 = './data/lisa.jpg'

	img1 = load_img(IMAGE1,DIMS,view=False)
	img2 = load_img(IMAGE2,DIMS,view=False)

	# stack into single tensor
	img_conc = np.stack([img1,img2], axis=0)

	return img_conc

def load3D():
	return None

def transform2D(image, affine_matrix):
	# grab the shape of the image
	B, H, W, C = image.shape
	M = affine_matrix

	# mesh grid generation
	x = np.linspace(-1, 1, W)
	y = np.linspace(-1, 1, H)
	x_t, y_t  = np.meshgrid(x,y)

	# augment the dimensions to create homogeneous coordinates
	# reshape to (xt, yt, 1)
	ones = np.ones(np.prod(x_t.shape))
	sampling_grid = np.vstack([x_t.flatten(),y_t.flatten(), ones])
	# repeat to number of batches
	sampling_grid = np.resize(sampling_grid, (B, 3, H*W))

	# transform the sampling grid, i.e. batch multiply
	batch_grids = np.matmul(M, sampling_grid) # the batch grid has the shape (B, 2, H*W)

	# reshape to (B, H, W, 2)
	batch_grids = batch_grids.reshape(B, 2, H, W)
	batch_grids = np.moveaxis(batch_grids,1,-1)

	# bilinear resampler
	x_s = batch_grids[:,:,:,0:1].squeeze()
	y_s = batch_grids[:,:,:,1:2].squeeze()\

	# rescale x and y to [0, W/H]
	x = ((x_s+1.)*W)*0.5
	y = ((y_s+1.)*H)*0.5

	# for each coordinate we need to grab the corner coordinates
	x0 = np.floor(x).astype(np.int64)
	x1 = x0+1
	y0 = np.floor(y).astype(np.int64)
	y1 = y0+1

	# clip to fit actual image size
	x0 = np.clip(x0, 0, W-1)
	x1 = np.clip(x1, 0, W-1)
	y0 = np.clip(y0, 0, H-1)
	y1 = np.clip(y1, 0, H-1)

	# grab the pixel value for each corner coordinate
	Ia = image[np.arange(B)[:,None,None], y0, x0]
	Ib = image[np.arange(B)[:,None,None], y1, x0]
	Ic = image[np.arange(B)[:,None,None], y0, x1]
	Id = image[np.arange(B)[:,None,None], y1, x1]

	# calculated the weighted coefficients and actual pixel value
	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	# add dimension for addition
	wa = np.expand_dims(wa, axis=3)
	wb = np.expand_dims(wb, axis=3)
	wc = np.expand_dims(wc, axis=3)
	wd = np.expand_dims(wd, axis=3)

	# compute output
	image_out = wa*Ia + wb*Ib + wc*Ic + wd*Id
	image_out = image_out.astype(np.int64)

	return image_out

def transform3D(image):
	# grab the shape of the image
	B, H, W, D, C = input_img.shape
	np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])	
	print("3d image under development")
	exit()

def main():
	MODE = '2D'

	if MODE == '2D':
		# 2D
		input_img = load2D()

		# define the affine matrix
		# initialize M to identity transform
		M = np.array([[1., 0., 0.], [0., 1., 0.]])
		# repeat num_batch times
		M = np.resize(M, (input_img.shape[0], 2, 3))

		# change affine matrix values
		# translation
		M[0,:,:] = [[1,0,0.5],[0,1,0.25]]
		img_translate = transform2D(input_img, M)

		# rotation
		angle = 45 #degree
		M[0,:,:] = [[math.cos(angle/180*math.pi),-math.sin(angle/180*math.pi),0],[math.sin(angle/180*math.pi),math.cos(angle/180*math.pi),0]]
		img_rotate = transform2D(input_img, M)

		# shear
		M[0,:,:] = [[1,0.5,0],[0.5,1,0]]
		img_shear = transform2D(input_img, M)

		plt.figure(1)
		ax1 = plt.subplot(221)
		plt.imshow(input_img[0,:,:,:], cmap="gray")
		ax1.title.set_text('Original')
		plt.axis("off")

		ax2 =plt.subplot(222)
		plt.imshow(img_translate[0,:,:,:], cmap="gray")
		ax2.title.set_text('Translation')
		plt.axis("off")

		ax3 = plt.subplot(223)
		plt.imshow(img_rotate[0,:,:,:], cmap="gray")
		ax3.title.set_text('Rotation')
		plt.axis("off")

		ax4 = plt.subplot(224)
		plt.imshow(img_shear[0,:,:,:], cmap="gray")
		ax4.title.set_text('Shear')
		plt.axis("off")

		plt.show()
	else:
		# 3D
		input_img = load3D() 
		transform3D(input_img)


if __name__=="__main__":
	main()