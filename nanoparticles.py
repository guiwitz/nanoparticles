"""
This module is a collection of functions to detect gold particles
in EM images. It performs template matching with disks with a range
of possible radii to find the best match.
 """
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: MIT License

import glob, os
import numpy as np
import skimage
import skimage.feature
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

plt.style.use('ggplot')
font = FontProperties()
font.set_style('normal')
font.set_size(15)


def create_disk_template(radius, image_size):
    """Create a disk image
    
    Parameters
    ----------
    radius : int
        radius of disk
    image_size: int
        size of output image
    
    Returns
    -------
    template : 2D numpy array
        binary image of a disk
    """
    
    template = np.zeros((image_size,image_size))
    center = [(template.shape[0]-1)/2,(template.shape[1]-1)/2]
    Y, X = np.mgrid[0:template.shape[0],0:template.shape[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    template[dist_from_center<=radius] = 1
    
    return template


def create_templates(minrad, maxrad, image_size):
    """Create a series of disk templates of varying radii and
    varying shift around the middle
    
    Parameters
    ----------
    minrad : int
        minimal radius of disk
    maxrad : int
        maximal radius of disk
    image_size: int
        size of output image
    
    Returns
    -------
    template : list of numpy arrays
        binary disk templates
    """
    all_templates = []
    for ind, x in enumerate(range(minrad,maxrad)):
        templ = create_disk_template(x, image_size)

        templ_norm = templ-np.mean(templ)
        templ_norm = templ_norm/np.sqrt(np.sum(templ_norm*templ_norm))
        
        #all_templates.append(templ_norm)
        for i in range(-2,3):
            for j in range(-2,3):
                
                temp_roll = np.roll(templ_norm,i,axis=0)
                temp_roll = np.roll(temp_roll,j,axis=1)
                all_templates.append({'template':temp_roll, 'shift_x':i,'shift_y':j,'radius':x})
    
    return all_templates


def init_filtering(image, radii):
    """Create a disk image
    
    Parameters
    ----------
    radius : int
        radius of disk
    image_size: int
        size of output image
    
    Returns
    -------
    template : 2D numpy array
        binary image of a disk
    """
    #all_filt = np.zeros((image.shape[0],image.shape[1],3))
    
    all_filt = [skimage.feature.match_template(image, skimage.morphology.disk(x),pad_input=True) for x in radii]
    #all_filt = [pool.apply(skimage.feature.match_template, args=(row, 4, 8)) for row in data]
    all_filt = np.stack(all_filt, axis = 2)

    im_filt = np.max(all_filt,axis = 2)
    
    return im_filt


def muli_radius_fitting(image, im_filt, minrad, maxrad, match_quality = 0.2):
    """Create a disk image
    
    Parameters
    ----------
    image : numpy array 2D
        image to analyze
    im_filt : numpy array 2D
        pre-filtered image with maxima at particle locations
    minrad: int
        minimum radius to consider
    maxrad: int
        maximum radius to consider
    match_quality: float
        threshold on matching quality. 0 means all local maxima are selected
    
    Returns
    -------
    all_radii : list
        mesuread radii
    circles: list
        list of triplet with x,y and radius of detected particles
    intensities: list
        list of average intensity of each particle
    
    """
    #minrad = 30
    #maxrad = 80
    image_size = 2*maxrad+41#201
    image_halft_size = int((image_size-1)/2)
    all_templates = create_templates(minrad, maxrad, image_size)

    local_max_indices = skimage.feature.peak_local_max(im_filt, min_distance=int(0.6*maxrad),indices=True, 
                                                       threshold_abs = match_quality)

    mask = np.zeros(im_filt.shape)

    all_radii = []
    circles = []
    intensities = []
    for particle_loc in local_max_indices:
        if (particle_loc[0]-image_halft_size>0)&(particle_loc[0]+image_halft_size<im_filt.shape[0])&\
        (particle_loc[1]-image_halft_size>0)&(particle_loc[1]+image_halft_size<im_filt.shape[1]):

            sub_im = image[particle_loc[0]-image_halft_size:particle_loc[0]+image_halft_size+1,
                   particle_loc[1]-image_halft_size:particle_loc[1]+image_halft_size+1]

            im_norm = sub_im-np.mean(sub_im)
            im_norm = im_norm/np.sqrt(np.sum(im_norm*im_norm))

            #products = [[all_templates[ind]['radius'],all_templates[ind]['shift_x'],all_templates[ind]['shift_y'],
            #             np.sqrt(np.abs(np.sum(all_templates[ind]['template']*im_norm)))] for ind in range(len(all_templates))]
            
            products = [[all_templates[ind]['radius'],all_templates[ind]['shift_x'],all_templates[ind]['shift_y'],
                         np.sum(all_templates[ind]['template']*im_norm)] for ind in range(len(all_templates))]
            
            products = np.array(products)

            pos_max = np.argmax(products[:,3])
            fit_rad = products[pos_max,0]
            fit_shift_x = int(products[pos_max,1])
            fit_shift_y = int(products[pos_max,2])
            all_radii.append(fit_rad)
            templ = create_disk_template(int(fit_rad), image_size)
            templ = np.roll(np.roll(templ,fit_shift_x,axis=0),fit_shift_y,axis = 1)
            
            #calculate mean intensity of each particle
            masked_sum = np.sum(image[particle_loc[0]-image_halft_size:particle_loc[0]+image_halft_size+1,
                   particle_loc[1]-image_halft_size:particle_loc[1]+image_halft_size+1]*templ)
            intensities.append(masked_sum/np.sum(templ))
            #create a maks of all particls
            mask[particle_loc[0]-image_halft_size:particle_loc[0]+image_halft_size+1,
                   particle_loc[1]-image_halft_size:particle_loc[1]+image_halft_size+1]+=templ
            #recover location of all particles
            circles.append([particle_loc[0]+fit_shift_x,particle_loc[1]+fit_shift_y,fit_rad])
    return all_radii, circles, intensities

def analyze_particles(path_to_data, min_rad, max_rad, scale, match_quality = 0.2, single_image = False):
    """Main function analyzing particles size in EM images
    
    Parameters
    ----------
    path_to_data : str
        path to folder containing tif files
    minrad: int
        minimum radius to consider
    maxrad: int
        maximum radius to consider
    scale: float
        image scale (in nanometer per pixel)
    match_quality: float
        threshold on matching quality. 0 means all local maxima are selected
    
    Returns
    -------
    all_radii: list
        list of all measured radii in all images
    
    """
    tif_files = glob.glob(path_to_data+'/*tif')
    if single_image:
        tif_files = tif_files[0:1]
        
    all_radii = []
    for tif in tif_files:
        image = skimage.io.imread(tif)
        if len(image.shape)==3:
            image = image[:,:,1]+0.001
        im_filt = init_filtering(image, np.arange(min_rad, max_rad,10))
        radii, circles, intensities = muli_radius_fitting(image, im_filt, min_rad, max_rad, match_quality)
        plot_detection(image, circles, radii, scale)
        pd_temp = pd.DataFrame({'radii': np.array(radii)*scale,'intensities': intensities, 'filename': os.path.basename(tif)})
        all_radii.append(pd_temp)
    all_radii = pd.concat(all_radii)
    return all_radii


def plot_detection(image, circles, radii, scale):
    """Create a disk image
    
    Parameters
    ----------
    image : numpy array 2D
        analyzed image
    circles: list
        list with positions and radii of particles, output of muli_radius_fitting
    radii : list
        mesuread radii
    scale: float
        image scale (in nanometer per pixel)
    
    Returns
    -------
    
    
    """
    
    fig, ax = plt.subplots(1,2,figsize=(15,7))
    ax[0].imshow(image, cmap = 'gray')

    if len(circles) > 0:
        for x in circles:
            plot_circ = plt.Circle((x[1], x[0]), x[2], color='r', linewidth = 1, faceColor = [1,1,1,0])
            ax[0].add_artist(plot_circ)
        ax[0].set_axis_off()
    
    if len(radii) > 0:
        ax[1].hist(np.array(radii)*scale, bins = np.arange(20,80,2)*scale)
        ax[1].set_xlabel('Radius [nm]',fontproperties = font)
        ax[1].set_ylabel('Count',fontproperties = font)
        ax[1].tick_params(labelsize=12)
    plt.show()

