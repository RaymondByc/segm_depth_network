from scipy import misc 
import glob
import os 
import numpy as np

gt_imgs = glob.glob('/home/zhanwj/Desktop/scenflow/flownet2/DispNet/citycapes/gtFine/val/*/*_gtFine_color.png')
baseline_root = '/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/result/cs/0.85/sem_1.0/'
compare_root = '/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/result/cs/0.85/sem_0.1/'
output_root = 'compare_output/'

for gt_img in gt_imgs:

	basename = os.path.basename(gt_img)
	print(gt_img)
	img_gt = misc.imread(gt_img, mode='RGB')
	basename = basename.replace('_gtFine_color', '_leftImg8bit')
	img_b = misc.imread(baseline_root + basename)
	img_c = misc.imread(compare_root + basename)
	print(np.shape(img_gt))	
	img_gt = misc.imresize(img_gt, (512, 1024, 3))
	# img_b = np.transpose(Image.open(baseline_root + basename), (1,0,2))
	# img_c = np.transpose(Image.open(compare_root + basename),(1,0,2))
	img_error =  np.where(img_c == img_b, img_gt, 255)
	img_all = np.concatenate([img_gt, img_b, img_c, img_error], axis=0)
	misc.imsave(output_root + basename, img_all)