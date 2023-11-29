import numpy as np
from astropy.io import fits
import os


case = {
    '1': [0, 151, "106.0MHz-121.0MHz", 0.106, 0.121, 201.1650072190865, 1071.0513542809060],
    '2': [150, 301, "121.0MHz-136.0MHz", 0.121, 0.136, 188.99951594411843, 1039.9247330274152],
    '3': [300, 451, "136.0MHz-151.0MHz", 0.136, 0.151, 178.78423268555557, 1010.6101653459351],
    '4': [450, 601, "151.0MHz-166.0MHz", 0.151, 0.166, 170.04232435195007, 982.82461128417330],
    '5': [600, 751, "166.0MHz-181.0MHz", 0.166, 0.181, 162.44501789577873, 956.35324591089910],
    '6': [750, 901, "181.0MHz-196.0MHz", 0.181, 0.196, 155.75723976772431, 931.02844428601920]
}


path = "fits"



file = "ZW3.msn_image"
for icase in range(1, 7):  # Change the range to include case 6
    i1 = case[str(icase)][0]
    i2 = case[str(icase)][1]

    with fits.open(os.path.join(path, file+".fits")) as hdl:
        data = np.array(hdl[0].data[i1:i2, :, :])

    filename = f"{file}_{case[str(icase)][2]}"
    np.save(filename, data)
    print(filename,i1,i2)


file = "ZW3.msn_psf"
for icase in range(1, 7):  # Change the range to include case 6
    i1 = case[str(icase)][0]
    i2 = case[str(icase)][1]

    with fits.open(os.path.join(path, file+".fits")) as hdl:
        data = np.array(hdl[0].data[i1:i2, :, :])

    filename = f"{file}_{case[str(icase)][2]}"
    np.save(filename, data)
    print(filename,i1,i2)

exit()

file = "ZW3.msw_image"
for icase in range(1, 7):  # Change the range to include case 6
    i1 = case[str(icase)][0]
    i2 = case[str(icase)][1]

    with fits.open(os.path.join(path, file+".fits")) as hdl:
        data = np.array(hdl[0].data[i1:i2, :, :])

    filename = f"{file}_{case[str(icase)][2]}"
    np.save(filename, data)
    print(filename,i1,i2)


file = "ZW3.msw_psf"
for icase in range(1, 7):  # Change the range to include case 6
    i1 = case[str(icase)][0]
    i2 = case[str(icase)][1]

    with fits.open(os.path.join(path, file+".fits")) as hdl:
        data = np.array(hdl[0].data[i1:i2, :, :])

    filename = f"{file}_{case[str(icase)][2]}"
    np.save(filename, data)
    print(filename,i1,i2)

file = "station_beam"
for icase in range(1, 7):  # Change the range to include case 6
    i1 = case[str(icase)][0]
    i2 = case[str(icase)][1]

    with fits.open(os.path.join(path, file+".fits")) as hdl:
        data = np.array(hdl[0].data[i1:i2, :, :])

    filename = f"{file}_{case[str(icase)][2]}"
    np.save(filename, data)
    print(filename,i1,i2)
