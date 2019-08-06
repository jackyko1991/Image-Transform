import SimpleITK as sitk
import numpy as np
from PIL import Image

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

def transform2D(image):
	# grab the shape of the image
	B, H, W, C = image.shape
	# initialize M to identity transform
	M = np.array([[1., 0., 0.], [0., 1., 0.]])
	# repeat num_batch times
	M = np.resize(M, (B, 2, 3))

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

	print(sampling_grid.shape)

	# transform the sampling grid, i.e. batch multiply
	batch_grids = np.matmul(M, sampling_grid)
	exit()


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
		transform2D(input_img)
	else:
		# 3D
		input_img = load3D() 
		transform3D(input_img)



if __name__=="__main__":
	main()