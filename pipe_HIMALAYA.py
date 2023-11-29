import numpy as np
import matplotlib.pyplot as plt
from func import *
import os
from astropy.io import fits
import sys

if len(sys.argv) != 3:
    print("Usage: python your_script.py icase nx")
    sys.exit(1)

icase = sys.argv[1]
nx = int(sys.argv[2])  #  'nx' is an integer value for cropped image size 

nz = 151
ny = nx


#icase='1' # 1-6
n_realization=20
n_comp=30





case = {
    '1': [0,   151, "106.0MHz-121.0MHz", 0.106, 0.121, 201.1650072190865,  1071.0513542809060],
    '2': [150, 301, "121.0MHz-136.0MHz", 0.121, 0.136, 188.99951594411843, 1039.9247330274152],
    '3': [300, 451, "136.0MHz-151.0MHz", 0.136, 0.151, 178.78423268555557, 1010.6101653459351],
    '4': [450, 601, "151.0MHz-166.0MHz", 0.151, 0.166, 170.04232435195007, 982.82461128417330],
    '5': [600, 751, "166.0MHz-181.0MHz", 0.166, 0.181, 162.44501789577873, 956.35324591089910],
    '6': [750, 901, "181.0MHz-196.0MHz", 0.181, 0.196, 155.75723976772431, 931.02844428601920]
}


BMAJ= 0.0553607952110976*3600
BMIN= 0.0411477702279064*3600

bins_kper_center=np.array([5.000000e-02,1.000000e-01,1.500000e-01,2.000000e-01,
                           2.500000e-01,3.000000e-01,3.500000e-01,4.000000e-01,4.500000e-01,5.000000e-01])

bins_klos_center=  bins_kper_center
bin_edges_kper=bin_centers_to_edges(bins_kper_center)
bin_edges_klos=bin_centers_to_edges(bins_klos_center)

start_value = case[icase][3] # GHz
end_value = case[icase][4] # GHz
total_bins = 151
freq_array = np.linspace(start_value, end_value, num=total_bins)

# Empirical formula derived from the PSF fitting
scale= (0.106/freq_array)**0.5
scale_factor= convert_units_jy2k([BMAJ*scale,BMIN*scale],freq_array)

print(freq_array[0],freq_array[-1],scale_factor[0],scale_factor[-1])

res_los=  case[icase][5]/151
res_per = case[icase][6]/2048
dlos= res_los*nz
dper= res_per*nx
print(dlos,dper,scale_factor[0],scale_factor[-1])


# In[101]:


file= f"sdc3-images/npy/ZW3.msn_psf_{case[icase][2]}.npy"
print("reading",file)
data = np.load(file).astype('float64')
#scale_area=  np.pi/4/np.log(2)*BMAJ/16*BMIN/16*scale
scale_area=crop_image(data[0],nx).sum()#
psf=crop_data_cube(data,nx)/scale_area
print("psf shape and psf_area:", psf.shape, psf[0].sum() )

psf_kernal= psf[0]


# In[102]:


print("reading",file)
file= f"sdc3-images/npy/ZW3.msn_image_{case[icase][2]}.npy"
data=np.load(file).astype('float64')
y=scale_factor[:,np.newaxis,np.newaxis]*crop_data_cube(data,nx)
print('y shape and std:', y.shape, y.std())


# In[103]:


#y_conv= convolution_with_psf_astropy(y, psf_kernal, boundary='fill')
y_conv= convolution_with_psf_scipy(y, psf_kernal)
rec_y_conv, svd_data= svd_fg_removal(y_conv,n_comp=n_comp)
ps2d_y_rec,_,_=calculate_2d_power_spectrum(rec_y_conv,[res_los,res_per], bin_edges_klos=bin_edges_klos,
                                           bin_edges_kper=bin_edges_kper)

rec_y, svd_ydata= svd_fg_removal(y,n_comp=n_comp)
ps2d_yraw_rec,_,_=calculate_2d_power_spectrum(rec_y,[res_los,res_per], bin_edges_klos=bin_edges_klos,
                                           bin_edges_kper=bin_edges_kper)


print("conv y std=", y_conv.std(), 'y std=', y.std() )
print('std(cleaned conv y)=', rec_y_conv.std())




path="sdc3-images/test-data/"
file="TestDataset.msn_image.fits" 
with fits.open(os.path.join(path, file)) as hdl:
    data = np.array(hdl[0].data)
y_dirty21=scale_factor[:,np.newaxis,np.newaxis]*crop_data_cube(data,nx) # for transfer func, factor cancelled
ps2d_test_true = np.loadtxt('sdc3-images/test-data/TestDatasetTRUTH_166MHz-181MHz.data')
print(file,y_dirty21.shape)
ps2d_dirty21,los_edge,per_edge=calculate_2d_power_spectrum(y_dirty21,[res_los,res_per], nbin=50)




#rec_ysvd, svd_data0= svd_fg_removal(y,n_comp=n_comp)
#rec_ydirty21cm, svd_data1= svd_fg_removal(convolution_with_psf_scipy(y_dirty21, psf_kernal),n_comp=0)
#rec_y21cm, svd_data2= svd_fg_removal(y_dirty21,n_comp=0)
#nn=np.arange(1,nz+1)

#plt.figure(figsize=(20, 4))
#plt.subplot(151)
#plt.plot(nn,np.sqrt(svd_data0[0]),label='svd y')
#plt.plot(nn,np.sqrt(svd_data[0]),label='svd conv y')
#plt.plot(nn,np.sqrt(svd_data1[0]),label='svd conv dirty 21')
#plt.plot(nn,np.sqrt(svd_data2[0]),label='svd dirty 21')
#plt.yscale('log')
#plt.xscale('log')
#plt.legend()

#plt.subplot(152)
#plt.imshow(rec_ysvd[0])
#plt.colorbar()
#plt.title('rec_ysvd')

#rec_ysvd_conv= convolution_with_psf_scipy(rec_ysvd, psf_kernal)
#plt.subplot(153)
#plt.imshow(rec_ysvd_conv[0])
#plt.colorbar()
#plt.title('rec_ysvd+conv')

#plt.subplot(154)
#plt.imshow(rec_y_conv[0])
#plt.title('rec_y_conv')
#plt.colorbar()

#plt.subplot(155)
#plt.imshow(y_conv[0])
#plt.title('y')
#plt.colorbar()

#np.std(rec_y_conv),np.std(rec_y)


# In[107]:


#ps_ysvd,kper= calculate_image_1dpower_spectrum(rec_ysvd_conv[0], pixel_length=res_per, nbin=100)
#ps_rec_y_conv,kper= calculate_image_1dpower_spectrum(rec_y_conv[0], pixel_length=res_per, nbin=100)
#ps_rec_y21_conv,kper= calculate_image_1dpower_spectrum(rec_ydirty21cm[0], pixel_length=res_per, nbin=100)


# In[108]:


#plt.plot(kper[:-1],ps_ysvd,label='ysvd')
#plt.plot(kper[:-1],ps_rec_y_conv,label='rec_y_conv')
#plt.plot(kper[:-1],ps_rec_y21_conv,label='true y21_conv')
#plt.yscale('log')
#plt.xscale('log')
#plt.legend()


# In[109]:


ps2d = []
y_res,_= svd_fg_removal(y_conv, n_comp=n_comp)

 
for i in range(n_realization):
    image_sim = generate_cube_from_ps2d(ps2d_dirty21, [nz, nx, ny], [res_los, res_per], 
                                       bin_edges_klos=los_edge, bin_edges_kper=per_edge)
    image_conv = convolution_with_psf_scipy(image_sim, psf_kernal)
    y_convsim=y_conv-y_res+image_conv
    image_rec, svd_data = svd_fg_removal(y_convsim, n_comp=n_comp)
    ps2d_rec, _, _ = calculate_2d_power_spectrum(image_rec, [res_los, res_per], 
                                               bin_edges_klos=bin_edges_klos, bin_edges_kper=bin_edges_kper)
    ps2d.append(ps2d_rec)
    print(f"{n_comp}-comp: {i}-th done, ps2d(rec)={ps2d_rec.mean():.5f}, std(21image)={image_sim.std():.4f}, std(conv21image)={image_conv.std():.4f}")
    

''' 
# SVD y
for i in range(n_realization):
    image_sim = generate_cube_from_ps2d(ps2d_dirty21, [nz, nx, ny], [res_los, res_per], 
                                       bin_edges_klos=los_edge, bin_edges_kper=per_edge)
    image_rec, svd_data = svd_fg_removal(image_sim, svd_data=svd_ydata)
    ps2d_rec, _, _ = calculate_2d_power_spectrum(image_rec, [res_los, res_per], 
                                               bin_edges_klos=bin_edges_klos, bin_edges_kper=bin_edges_kper)
    ps2d.append(ps2d_rec)
    print(f"{n_comp}-comp: {i}-th done, ps2d(rec)={ps2d_rec.mean():.5f}, std(21image)={image_sim.std():.4f}, std(conv21image)={image_conv.std():.4f}")
'''      


tf_mean, tf_std = transfer_mean_and_std(ps2d, ps2d_test_true)



#ps2d_mean=tf_mean*ps2d_rec
#ps2d_std = tf_std*ps2d_rec
#plt.errorbar(bins_kper_center, y=ps2d_mean[0], yerr=ps2d_std[0])
#plt.plot(bins_kper_center, ps2d_test_true[0])

#plt.errorbar(bins_kper_center, y=ps2d_mean[3], yerr=ps2d_std[3])
#plt.plot(bins_kper_center, ps2d_test_true[3])

#plt.errorbar(bins_kper_center, y=ps2d_mean[-1], yerr=ps2d_std[-1])
#plt.plot(bins_kper_center, ps2d_test_true[-1])
#plt.yscale('log')


# In[116]:


ps2d_final_mean= tf_mean*ps2d_y_rec # for SVD y ps2d_yraw_rec
ps2d_final_std = tf_std*ps2d_y_rec  # for SVD y ps2d_yraw_rec
plt.errorbar(bins_kper_center, y=ps2d_final_mean[0], yerr=ps2d_final_std[0],label=f'svd-{n_comp} bin0')
plt.plot(bins_kper_center, ps2d_test_true[0],label="true bin0")

#plt.errorbar(bins_kper_center, y=ps2d_final_mean[5], yerr=ps2d_final_std[5],label=f'svd-{n_comp} bin5')
#plt.plot(bins_kper_center, ps2d_test_true[5],label="true bin5")

plt.errorbar(bins_kper_center, y=ps2d_final_mean[9], yerr=ps2d_final_std[9],label=f'svd-{n_comp} bin9')
plt.plot(bins_kper_center, ps2d_test_true[9],label="true bin9")

plt.yscale('log')
plt.ylim(1e-4,10)
plt.legend()
ps2d_final_mean.sum(),ps2d_final_std.sum()
file=f"plot/msn_image_{nx}_{case[icase][2]}_svdfg_{n_comp}.png"
plt.savefig(file)

def save_matrix_to_file(matrix, filename):
    np.savetxt(filename, matrix, delimiter=' ', fmt='%1.6f', comments='')

save_matrix_to_file(ps2d_final_mean, f"plot/teamHIMALAYA_{case[icase][2]}.data_svdfg_{n_comp}_{nx}")
save_matrix_to_file(ps2d_final_std, f"plot/teamHIMALAYA_{case[icase][2]}_errors.data_svdfg_{n_comp}_{nx}")



#ps2d_final_mean.sum(),ps2d_final_std.sum()


# In[113]:


#hist_data = np.array(ps2d)
#plt.hist(hist_data[:,5,5], bins=20, color='blue')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Histogram of ps2d[:, 3, 3]')
#plt.grid(True)
# Show the plot
#plt.show()



