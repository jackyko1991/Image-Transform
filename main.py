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

def load_nifti(path, dimension, view=False):
	# load the data
	reader = sitk.ImageFileReader()
	reader.SetFileName(path)
	image = reader.Execute()

	# resize to desire size
	old_spacing = image.GetSpacing()
	old_size = image.GetSize()
   
	new_spacing = []
	for i in range(3):
		new_spacing.append(int(math.ceil(old_spacing[i]*old_size[i]/dimension[i])))
	new_size = tuple(dimension)

	resampler = sitk.ResampleImageFilter()
	resampler.SetInterpolator(2)
	resampler.SetOutputSpacing(new_spacing)
	resampler.SetSize(new_size)

	# resample on image
	resampler.SetOutputOrigin(image.GetOrigin())
	resampler.SetOutputDirection(image.GetDirection())
	# print("Resampling image...")
	image_new = resampler.Execute(image)
	image_new = sitk.GetArrayFromImage(image_new)

	# to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
	image_new = np.transpose(image_new,(2,1,0))

	if view:
		fig = plt.figure(1)

		for layer in range(image_new.shape[2]):
			ax = fig.add_subplot(math.ceil(image_new.shape[2]/4), 4, layer+1)
			ax.imshow(image_new[:,:,layer], cmap="gray")
			ax.axis("off")

		plt.show()

	return image_new

def load2D(DIMS=400):
	IMAGE1 = './data/lena.png'
	IMAGE2 = './data/lisa.jpg'

	img1 = load_img(IMAGE1,DIMS,view=False)
	img2 = load_img(IMAGE2,DIMS,view=False)

	# stack into single tensor
	img_conc = np.stack([img1,img2], axis=0)

	return img_conc

def load3D(layer_num):
	DIMS = [256,256,layer_num]
	IMAGE1 = './data/ch2.nii.gz'
	IMAGE2 = './data/brodmann.nii.gz'

	img1 = load_nifti(IMAGE1,DIMS,view=False)
	img2 = load_nifti(IMAGE2,DIMS,view=False)

	# stack into single tensor
	img_conc = np.stack([img1,img2], axis=0)
	img_conc = np.expand_dims(img_conc, axis=-1)

	return img_conc

def transform2D(image, affine_matrix):
	# grab the shape of the image
	B, H, W, C = image.shape
	M = affine_matrix

	# mesh grid generation
	# use x = np.linspace(-1, 1, W)  if you want to rotate about center
	x = np.linspace(0, 1, W) 
	y = np.linspace(0, 1, H)
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
	y_s = batch_grids[:,:,:,1:2].squeeze()

	# rescale x and y to [0, W/H]
	# use this function if you want to rotate about center
	# x = ((x_s+1.)*W)*0.5
	# y = ((y_s+1.)*H)*0.5
	x = ((x_s)*W)
	y = ((y_s)*H)

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

def transform3D(image, affine_matrix):
	# grab the shape of the image
	B, H, W, D, C = image.shape
	M = affine_matrix
	
	# mesh grid generation
	x = np.linspace(0, 1, W)
	y = np.linspace(0, 1, H)
	z = np.linspace(0, 1, D)
	x_t, y_t, z_t  = np.meshgrid(x,y,z)

	# augment the dimensions to create homogeneous coordinates
	# reshape to (xt, yt, zt, 1)
	ones = np.ones(np.prod(x_t.shape))
	sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), z_t.flatten(), ones])
	# repeat to number of batches
	sampling_grid = np.resize(sampling_grid, (B, 4, H*W*D))

	# transform the sampling grid, i.e. batch multiply
	batch_grids = np.matmul(M, sampling_grid) # the batch grid has the shape (B, 3, H*W*D)

	# reshape to (B, H, W, D, 3)
	batch_grids = batch_grids.reshape(B, 3, H, W, D)
	batch_grids = np.moveaxis(batch_grids,1,-1)

	# bilinear resampler
	x_s = batch_grids[:,:,:,:,0:1].squeeze()
	y_s = batch_grids[:,:,:,:,1:2].squeeze()
	z_s = batch_grids[:,:,:,:,2:3].squeeze()

	# rescale x, y and z to [0, W/H/D]
	x = ((x_s)*W)
	y = ((y_s)*H)
	z = ((z_s)*D)

	# for each coordinate we need to grab the corner coordinates
	x0 = np.floor(x).astype(np.int64)
	x1 = x0+1
	y0 = np.floor(y).astype(np.int64)
	y1 = y0+1
	z0 = np.floor(z).astype(np.int64)
	z1 = z0+1

	# clip to fit actual image size
	x0 = np.clip(x0, 0, W-1)
	x1 = np.clip(x1, 0, W-1)
	y0 = np.clip(y0, 0, H-1)
	y1 = np.clip(y1, 0, H-1)
	z0 = np.clip(z0, 0, D-1)
	z1 = np.clip(z1, 0, D-1)

	# grab the pixel value for each corner coordinate
	Ia = image[np.arange(B)[:,None,None,None], y0, x0, z0]
	Ib = image[np.arange(B)[:,None,None,None], y1, x0, z0]
	Ic = image[np.arange(B)[:,None,None,None], y0, x1, z0]
	Id = image[np.arange(B)[:,None,None,None], y1, x1, z0]
	Ie = image[np.arange(B)[:,None,None,None], y0, x0, z1]
	If = image[np.arange(B)[:,None,None,None], y1, x0, z1]
	Ig = image[np.arange(B)[:,None,None,None], y0, x1, z1]
	Ih = image[np.arange(B)[:,None,None,None], y1, x1, z1]

	# calculated the weighted coefficients and actual pixel value
	wa = (x1-x) * (y1-y) * (z1-z)
	wb = (x1-x) * (y-y0) * (z1-z)
	wc = (x-x0) * (y1-y) * (z1-z)
	wd = (x-x0) * (y-y0) * (z1-z)
	we = (x1-x) * (y1-y) * (z-z0)
	wf = (x1-x) * (y-y0) * (z-z0)
	wg = (x-x0) * (y1-y) * (z-z0)
	wh = (x-x0) * (y-y0) * (z-z0)

	# add dimension for addition
	wa = np.expand_dims(wa, axis=4)
	wb = np.expand_dims(wb, axis=4)
	wc = np.expand_dims(wc, axis=4)
	wd = np.expand_dims(wd, axis=4)
	we = np.expand_dims(we, axis=4)
	wf = np.expand_dims(wf, axis=4)
	wg = np.expand_dims(wg, axis=4)
	wh = np.expand_dims(wh, axis=4)

	# compute output
	image_out = wa*Ia + wb*Ib + wc*Ic + wd*Id + we*Ie + wf*If + wg*Ig + wh*Ih
	image_out = image_out.astype(np.int64)

	return image_out

def mutual_information(hgram):
	""" Mutual information for joint histogram
	"""
	# Convert bins counts to probability values
	pxy = hgram / float(np.sum(hgram))
	px = np.sum(pxy, axis=1) # marginal for x over y
	py = np.sum(pxy, axis=0) # marginal for y over x
	px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
	# Now we can do the calculation using the pxy, px_py 2D arrays
	nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
	return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def image_matching_metric(image1, image2, title="",plot=False):
	corr = np.corrcoef(image1.ravel(), image2.ravel())[0,1]

	# 3d histogram
	hist_2d, x_edges, y_edges = np.histogram2d(
		image1.ravel(), image2.ravel(), bins=20)
	mi = mutual_information(hist_2d)

	if plot:
		# histogram and mutual information
		plt.figure(1)
		ax1 = plt.subplot(231)
		plt.hist(image1.ravel(), bins=20)
		ax1.set_title('Input Image')

		ax1 = plt.subplot(232)
		plt.hist(image2.ravel(), bins=20)
		ax1.set_title(title)

		ax2 = plt.subplot(233)
		plt.plot(image1.ravel(),image2.ravel(),'.')
		plt.xlabel("Image1")
		plt.ylabel("Image2")
		
		ax2.set_title("Correlation:" + str(corr))

		# plot as image
		ax3 = plt.subplot(234)
		ax3.imshow(hist_2d.T, origin='lower')
		ax3.set_title("Mutual Information: "+ str(mi))
		plt.xlabel("Image1 bin")
		plt.ylabel("Image2 bin")
		# log histogram
		hist_2d_log = np.zeros(hist_2d.shape)
		non_zeros = hist_2d != 0
		hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
		ax4 = plt.subplot(235)
		ax4 = plt.imshow(hist_2d_log.T, origin='lower')
		plt.xlabel("Image1 bin (log)")
		plt.ylabel("Image2 bin (log")

		plt.show()

	return corr, mi

def affine_transform(mode):
	if mode == '2D':
		# 2D
		input_img = load2D()

		# define the affine matrix
		# initialize M to identity transform
		M = np.array([[1., 0., 0.], [0., 1., 0.]])
		# repeat num_batch times
		M = np.resize(M, (input_img.shape[0], 2, 3))

		# change affine matrix values
		# translation
		M[0,:,:] = [[1.,0.,0.],[0.,1.,0.]]
		img_translate = transform2D(input_img, M)

		# rotation
		angle = 45 #degree
		M[0,:,:] = [[math.cos(angle/180*math.pi),-math.sin(angle/180*math.pi),0],[math.sin(angle/180*math.pi),math.cos(angle/180*math.pi),0]]
		img_rotate = transform2D(input_img, M)

		# shear
		M[0,:,:] = [[1,0.5,0],[0.5,1,0]]
		img_shear = transform2D(input_img, M)

		image_matching_metric(input_img[0,:,:,:], img_translate[0,:,:,:], title="Translate",plot=True)
		image_matching_metric(input_img[0,:,:,:], img_rotate[0,:,:,:], title="Rotate",plot=True)
		image_matching_metric(input_img[0,:,:,:], img_shear[0,:,:,:], title="Shear",plot=True)

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
		layer_num = 8
		input_img = load3D(layer_num)

		# define the affine matrix
		# initialize M to identity transform
		M = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.] , [0., 0., 1., 0.]])
		# repeat num_batch times
		M = np.resize(M, (input_img.shape[0], 3, 4))

		# change affine matrix values
		# translation
		M[0,:,:] = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
		img_translate = transform3D(input_img, M)

		# rotation
		alpha = 20 #degree
		beta = 0
		gamma = 0

		# convert from degree to radian
		alpha = alpha*math.pi/180
		beta = beta*math.pi/180
		gamma = gamma*math.pi/180
		# Tait-Bryan angles in homogeneous form, reference: https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf
		Rx = [[1,0,0,0],[0,math.cos(alpha),-math.sin(alpha),0],[0,math.sin(alpha),math.cos(alpha),0],[0.,0.,0,1.]]
		Ry = [[math.cos(beta),0,math.sin(beta),0],[0,1,0,0],[-math.sin(beta),0,math.cos(beta),0],[0.,0.,0.,1.]]
		Rz = [[math.cos(gamma),-math.sin(gamma),0,0],[math.sin(gamma),math.cos(gamma),0,0],[0,0,1,0],[0.,0.,0.,1.]]

		print("Rx",Rx)
		print("Ry",Ry)
		print("Rz",Rz)

		M[0,:,:] = np.matmul(Rz,np.matmul(Ry,Rx))[0:3,:]
		print(M)

		img_rotate = transform3D(input_img, M)

		# shear
		M[0,:,:] = [[1,0.5,0,0],[0.5,1,0,0],[0,0,1,0]]
		img_shear = transform3D(input_img, M)

		image_matching_metric(input_img[0,:,:,:,:], img_translate[0,:,:,:,:], title="Translate",plot=False)
		image_matching_metric(input_img[0,:,:,:,:], img_rotate[0,:,:,:,:], title="Rotate",plot=False)
		image_matching_metric(input_img[0,:,:,:,:], img_shear[0,:,:,:,:], title="Shear",plot=False)

		fig = plt.figure(1)

		for layer in range(input_img.shape[3]):
			ax0 = fig.add_subplot(4, input_img.shape[3], layer+1)
			ax0.imshow(input_img[0,:,:,layer,0], cmap="gray")
			ax0.axis("off")

			ax1 = fig.add_subplot(4, input_img.shape[3], input_img.shape[3]*1 + layer+1)
			ax1.imshow(img_translate[0,:,:,layer,0], cmap="gray")
			ax1.axis("off")

			ax2 = fig.add_subplot(4, input_img.shape[3], input_img.shape[3]*2 + layer+1)
			ax2.imshow(img_rotate[0,:,:,layer,0], cmap="gray")
			ax2.axis("off")

			ax3 = fig.add_subplot(4, input_img.shape[3], input_img.shape[3]*3 + layer+1)
			ax3.imshow(img_shear[0,:,:,layer,0], cmap="gray")
			ax3.axis("off")

		plt.show()

def random_deform_2d(image):
	# grab the shape of the image
	if len(image.shape) == 4:
		B, H, W, C = image.shape
		mode = "2D"
	# mesh grid generation
	# use x = np.linspace(-1, 1, W)  if you want to rotate about center
	x = np.linspace(0, 1, W) 
	y = np.linspace(0, 1, H)
	x_t, y_t  = np.meshgrid(x,y)

	# augment the dimensions to create homogeneous coordinates
	# reshape to (xt, yt, 1)
	ones = np.ones(np.prod(x_t.shape))
	sampling_grid = np.vstack([x_t.flatten(),y_t.flatten()])

	# repeat to number of batches
	sampling_grid = np.resize(sampling_grid, (B, 2, H*W))

	# transform the sampling grid by random addition
	if W > H:
		grid_deformation = (np.random.rand(sampling_grid.shape[0],sampling_grid.shape[1],sampling_grid.shape[2])-0.5)/W*1
	else:
		grid_deformation = (np.random.rand(sampling_grid.shape[0],sampling_grid.shape[1],sampling_grid.shape[2])-0.5)/H*1

	# constant direction
	grid_deformation = np.zeros(sampling_grid.shape)
	grid_deformation[0,0,:] = 0
	# grid_deformation[0,0,:] = 1/W*10
	grid_deformation[0,1,:] = 1/H*10
	# grid_deformation[0,1,:] = 0

	batch_grids = sampling_grid + grid_deformation

	# reshape to (B, H, W, 2)
	batch_grids = batch_grids.reshape(B, 2, H, W)
	batch_grids = np.moveaxis(batch_grids,1,-1)

	sampling_grid = sampling_grid.reshape(B, 2, H, W)
	sampling_grid = np.moveaxis(sampling_grid,1,-1)

	deformation_field = -(batch_grids - sampling_grid) # note that the deformation field is the inverse of grid deformation

	# bilinear resampler
	x_s = batch_grids[:,:,:,0:1].squeeze()
	y_s = batch_grids[:,:,:,1:2].squeeze()

	# rescale x and y to [0, W/H]
	# use this function if you want to rotate about center
	# x = ((x_s+1.)*W)*0.5
	# y = ((y_s+1.)*H)*0.5
	x = ((x_s)*W)
	y = ((y_s)*H)

	# sampling grid and batch grid to image size
	batch_grids[:,:,:,0:1] = batch_grids[:,:,:,0:1]*W
	batch_grids[:,:,:,1:2] = batch_grids[:,:,:,1:2]*H
	sampling_grid[:,:,:,0:1] = sampling_grid[:,:,:,0:1]*W
	sampling_grid[:,:,:,1:2] = sampling_grid[:,:,:,1:2]*H

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

	return image_out, sampling_grid, batch_grids, deformation_field

def vector_transform(mode):
	if mode == "2D":
		# 2D
		input_img = load2D(DIMS=25)
		image_out, sampling_grids, batch_grids, deformation_field = random_deform_2d(input_img)

		plt.figure(1)
		ax1 = plt.subplot(221)
		plt.imshow(input_img[0,:,:,:], cmap="gray")
		ax1.title.set_text('Original')
		plt.axis("off")

		ax2 = plt.subplot(222)
		plt.imshow(image_out[0,:,:,:], cmap="gray")
		ax2.title.set_text('Vector Deformation')
		plt.axis("off")

		ax3 = plt.subplot(223)
		plt.plot(sampling_grids[0,:,:,0].ravel(),sampling_grids[0,:,:,1].ravel(),'.')
		plt.xlim([0,input_img.shape[1]])
		plt.ylim([0,input_img.shape[2]])

		ax4 = plt.subplot(224)
		M = np.hypot(deformation_field[0,:,:,0], deformation_field[0,:,:,1])
		plt.quiver(
			sampling_grids[0,:,:,0].ravel(),
			sampling_grids[0,:,:,1].ravel(),
			deformation_field[0,:,:,0].ravel(), 
			-deformation_field[0,:,:,1].ravel(), ## note that Cartesian coordinate and pixel coordinate are in opposite direction for y axis
			M.ravel(),
			width=0.5,
			scale=1/2,
			units='xy')

		# plt.plot(batch_grids[0,:,:,0].ravel(),batch_grids[0,:,:,1].ravel(),'.')
		ax4.scatter(sampling_grids[0,:,:,0].ravel(),sampling_grids[0,:,:,1].ravel(), color='0.5', s=1)
		# ax4.scatter(batch_grids[0,:,:,0].ravel(),batch_grids[0,:,:,1].ravel(), color='r', s=1)
		plt.xlim([0,input_img.shape[1]])
		plt.ylim([0,input_img.shape[2]])

		plt.show()
	else:
		print("3D vector transform under development")

	return

def main():
	MODE = '2D'
	METHOD = 'VECTOR'
	# METHOD = 'AFFINE'

	if METHOD == 'AFFINE':
		affine_transform(MODE)
	else:
		vector_transform(MODE)
	


if __name__=="__main__":
	main()