#!/usr/bin/env python
# coding: utf-8
# Center for Microscopy and Image Analysis
#Version 0
#TBD: make sure, target directory is not the same as the raw data, make sure only correct directories are accepted. Capture if selection is cancelled.
#Works on MD single round, multiple acquisition rounds. all rounds must have the same dimensions. 3D, 2D and Time allowed. 


#not all libraries used - needs to be updated......
from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox
#import math
import os
import re
import glob
#import json
import time
import dask.array as da
import dask.delayed as dl
#import dask_image.imread
import multiprocessing
from multiprocessing import Pool
import imageio.v2 as imageio
import numpy as np
#import napari
#from IPython.display import clear_output
from tifffile import imwrite
from datetime import datetime
import itertools
from pathlib import Path
#from saveParallel import save_hyperstack_image


class Gui():
    def __init__(self):
        self.root = Tk()
        self.root.geometry('800x400')
        self.root.attributes('-topmost', True)
        self.root.title ('Data conversion')
        #Variables
        self.textOrigDir = StringVar()
        self.textOrigDir.set("")
        
        self.textDestDir = StringVar()
        self.textDestDir.set("")
        
        self.textTextBox = StringVar()
        self.textTextBox.set('Data conversion of Molecular devices files from Tiff -> Fiji Tiff Hyperstacks\n\nSelect top directory containing one or multiple acquistions from the MD system. All data must have the same dimensions.')
        
        self.textOKGo = StringVar()
        self.textOKGo.set('Do you want to proceed? This may take some minutes depending on the size of your data.')
        #Elements,
        self.info = ttk.Label(self.root, textvariable=self.textTextBox)
        self.buttonOrigDir = ttk.Button(self.root,
                                text="Select raw data",
                                command=self.selectOrigDir)
        self.labelOrigDir = ttk.Label(self.root, textvariable=self.textOrigDir)
        self.buttonDestDir = ttk.Button(self.root,
                                text="Select destination directory",
                                command=self.selectDestDir)
        self.labelDestDir = ttk.Label(self.root, textvariable=self.textDestDir)
        self.buttonProceed = ttk.Button(self.root,
                                        text='Proceed',
                                        command=self.okGo)
        #Placement
        self.info.grid(column=0, row=0, columnspan=3, rowspan=2,padx=5, pady=20,
                        sticky=(N, S, E, W))
        self.buttonOrigDir.grid(column=0, row=2, columnspan=3,
                               sticky=(N, W), padx=5)
        self.labelOrigDir.grid(column=0, row=3, columnspan=2,
                               sticky=(N, W), padx=5)
        self.buttonDestDir.grid(column=0, row=4, columnspan=2,
                               sticky=(N, W), padx=5)
        self.labelDestDir.grid(column=0, row=5, columnspan=2,
                               sticky=(N, W), padx=5)
        self.buttonProceed.grid(column=0, row=6, columnspan=2,
                                sticky=(N,W,E), padx=5, pady=10)

                
        self.root.mainloop()
            

    def selectOrigDir(self):
        oriPath = filedialog.askdirectory(title='select directory with data')
        self.textOrigDir.set(oriPath)  
        
    def selectDestDir(self):
        destPath = filedialog.askdirectory(title='select destination directory')
        self.textDestDir.set(destPath)  
    
    def okGo(self):
        if len(self.textOrigDir.get()) == 0:
            messagebox.showwarning(title=('Attention'), message=('Please select a directory with data'))
        elif len(self.textDestDir.get())== 0: 
            messagebox.showwarning(title=('Attention'), message=('Please select a destination directory'))
        elif self.textOrigDir.get() in self.textDestDir.get():
             messagebox.showwarning(title=('Attention'), message=('Please select a destination directory outside your raw data'))
        else:
            okgo = messagebox.askokcancel(title='Proceed?', message='Do you want to proceed and convert the data?')
            self.textOKGo.set(okgo)
            self.root.destroy()

def save_hyperstack_image(destPath, filename, img, deltaT, calibration_x, calibration_y, nfield, field, nwell, well):
    
    #change names to strings if they are coming as int from filenames        
    if type(field) == int:
        field = '_f_' + str(f'{field:03}')
    elif field is None:
        #TBD: introduce for 1 field
        field = ''
    
    if type(well) == int:
        well = str(f'{well:03}')
    
    
    filename = filename +'_w_'+ well +  field + '.tif'

    #check if onlyn1 field
    if nfield is None:
        img_s= img[:, :, :, 0, nwell, :, :]
    else:
        img_s= img[:, :, :, nfield, nwell, :, :]
    imwrite(
        os.path.join(destPath,filename),
        img_s,
        imagej=True,
        photometric='minisblack',
        resolution= (1/calibration_x, 1/calibration_y),
        metadata={'axes': 'TZCYX',
                    'spacing': 5*calibration_x,
                    'unit': 'um',
                    'finterval': deltaT}
        )
    return print(f'Saved: {filename}')

if __name__ == '__main__':
    print('Do not close the window until saving is complete!\n')
    
    directoriesGui = Gui()
    parentDirectory = directoriesGui.textOrigDir.get()
    destPath = directoriesGui.textDestDir.get()
    okGo = directoriesGui.textOKGo.get()
    if okGo != '1':
        print('Data conversion aborted.')
    else:     
        #get all filenames
        filenames = glob.glob(os.path.join(parentDirectory,'**','*.TIF'), recursive=True)
    
        print('Number of files in dataset: ' + str(len(filenames)))
        #read one image to determine the shape / size. Assuming all images having the same size
        sample = imageio.imread(filenames[0])
        #get metadata of image
        pattern = ' type(.\*) value'
        meta_data_raw = str.replace(sample.meta.get('description'), '\"', '').split("/>\n<prop id=")
        pattern = ' type(.*?) value='
        meta_data_key = [re.sub(pattern, ',', x).split(',')[0] for x in meta_data_raw]
        meta_data_val = [re.sub(pattern, ',', x).split(',')[1] for x in meta_data_raw]
        meta_data = dict(zip(meta_data_key,meta_data_val))
        calibration_x = float(meta_data.get('spatial-calibration-x'))
        calibration_y = float(meta_data.get('spatial-calibration-y'))
        calibration_unit = meta_data.get('spatial-calibration-units')
        pixel_size_x = int(meta_data.get('pixel-size-x'))
        pixel_size_y = int(meta_data.get('pixel-size-y'))
        acquisition_time_local_start = meta_data.get('acquisition-time-local')
        acquisition_time_local_start =datetime.strptime(acquisition_time_local_start, '%Y%m%d %H:%M:%S.%f')
        print(f'{pixel_size_x} x {pixel_size_y} pixels, pixel size: {calibration_x} {calibration_unit}')
        
        #check if file with HTD extension is present. If yes, a single folder with data was selected.
        pattern = '*.HTD'
        htd_files = glob.glob(os.path.join(parentDirectory, pattern))
        if len(htd_files) == 0:
            multiround_flag = 1
            print('Multiple directories will be processed')
        else:
            multiround_flag = 0
            print('Only one directory is processed')
            
        # Get various dimensions from the filename
        fn_comp_sets = dict()
        fn_comp_multidimensions = dict()
        for fn in filenames:
           #get the tail  of the filepath (the filename), extract all information from the filename like wells, fields, channels
            filename = os.path.split(fn)[1]
            #print(fn)
            #split the string to get parts containing all information regarding the data set
            for i, comp in enumerate(re.split(r'_(?=[A-H])|_s|_w|\\|\.',filename)):
                #print(f'i: {i}, {comp}')
                if comp.isdecimal():
                    #convert to int if needed
                    comp = int(comp)
                fn_comp_sets.setdefault(i, set())
                fn_comp_sets[i].add(comp)
            path_parts = re.split(r'\\', os.path.normpath(os.path.split(fn)[0]))
            if 'ZStep' in path_parts[-1]:
                s = -3 #3D Data
                dim_flag = 1
            elif 'TimePoint' in path_parts[-1]:
                s = -2 #2D Data
                dim_flag = 0 #to check later 2D vs 3D case
            for i in range(s, 0):
                if i > s:
                    comp = re.split(r'_', path_parts[i])[-1]
                else: 
                    comp = path_parts[i]
                if comp.isdecimal():
                    comp = int(comp)
                fn_comp_multidimensions.setdefault(abs(i)+1, set())
                fn_comp_multidimensions[abs(i)+1].add(comp)
                    
        fn_comp_sets = list(map(sorted, fn_comp_sets.values()))
        #generate a list of dictonaries with all dataset information
        #filename -> remap_comps[0]
        #well -> remap_comps[1]
        #fields -> remap_comps[2]
        #channel -> remap_comps[3]
        #extension -> remap_comps[4]
        remap_comps = [dict(map(reversed, enumerate(x))) for x in fn_comp_sets]
        
        #check if filenames contain _s -> more than 1 field, _w ->more than 1 channel, only first filename needed to be analyzed
        if not re.search('_w\d.', filename):
            print('only one channel was acquired')
            remap_comps.insert(3, {None: None})
        if not re.search('_s\d_', filename):
            print('only one field was acquired')
            remap_comps.insert(2, {None: None})
    
        fn_comp_multidimensions = list(map(sorted, fn_comp_multidimensions.values()))
        
            
        #generate a list of dictonaries with all dataset information
        #imaging rounds -> remap_comps_multidim [0]
        #timelapse folders in each round -> remap_comps_multidim [1]
        #zSteps in each round -> remap_comps_multidim [2], is not present if 2D files!
        remap_comp_multidim = [dict(map(reversed, enumerate(x))) for x in fn_comp_multidimensions]
        if dim_flag == 0:
            remap_comp_multidim.append({1:0})
        
        if multiround_flag==0:
            parentDirectory=os.path.split(parentDirectory)[0]
            
        #print(f'Multidimension: {remap_comp_multidim}')
        #print(f'Dataset in each round: {remap_comps}')
        
        
        #construct filenames for loading in sequences    
        
        sorted_tags = list(itertools.product(
            list(remap_comps[1].keys()), #wells, 
            list(remap_comps[2].keys()), #fields, 
            list(remap_comp_multidim[0].keys()), #rounds of acquistions, 
            list(remap_comp_multidim[1].keys()), #timelapse folders, 
            list(remap_comps[3].keys()), #channel, 
            list(remap_comp_multidim[2].keys()) #zStep
        ))
        
        sortedFilenames = []
        for ii, i in enumerate(sorted_tags):
            #print(f'{ii}:{parentDirectory}/{i[2]}/TimePoint_{i[3]}/Zstep_{i[5]}/{filename[0]}_{i[0]}_s{i[1]}_w{i[4]}')
            field_tag = '' if i[1] is None else '_s'+str(i[1])
            channel_tag = '' if i[4] is None else '_w'+str(i[4])
            file = str(list(remap_comps[0].keys())[0] #filename
                       +'_'+i[0] #Well
                       +field_tag #Field
                       +channel_tag #Channel
                       +'.'+list(remap_comps[4].keys())[0] #Extension
                      )
            if dim_flag ==0:
                path =  Path(os.path.join(parentDirectory, i[2],  #parent directory
                                          'TimePoint_'+ str(i[3]), #timepoint directories
                                          file))
            else:
                path =  Path(os.path.join(parentDirectory, i[2],  #parent directory
                                          'TimePoint_'+ str(i[3]), #timepoint directories
                                          'ZStep_' + str(i[5]),  #3D directories
                                          file))        
            #print(path)
            if path.exists: 
                sortedFilenames.append(path) 
        '''sortedFilenames: sort order is 1. Wells, 2. Fields, 3. Timepoints, 4. Channels, 5. Planes'''
        
        # #read images delayed in single chunked arrays
        imgData = [dl(imageio.imread)(fn) for fn in sortedFilenames]
        imgData = [da.from_delayed(x, shape=sample.shape, dtype=sample.dtype)
                        for x in imgData]
        
        #determine the number of fields, wells, etc (just for easier reading)
        #TBD for calculation of tiling layouts for viewing all fields, this can be handy
        nfields = len(remap_comps[2]) #number of fields
        print(f'Number of fields: {nfields}')
        ntimes = len(list(remap_comp_multidim[0].keys())) + len(list(remap_comp_multidim[1].keys())) -1 #number of times
        print(f'Number of timepoints: {ntimes}')
        nwells = len(remap_comps[1]) #number of wells
        print(f'Number of wells: {nwells}')
        nchannels = len(remap_comps[3]) #number of channels
        print(f'Number of channels: {nchannels}')
        nsteps = len(list(remap_comp_multidim[2].keys())) #zStep)
        print(f'Number of planes: {nsteps}')
        
        # Create an empty object array to organize each chunk that loads a TIFF
        dataset = np.empty((ntimes, nsteps, nchannels, nfields, nwells, 1, 1), dtype=object)
        print(f'Dataset TZCFW: {dataset.shape}')
        
        #get the first image of the last timepoint - to approximate the deltaT
        sample = imageio.imread(sortedFilenames[(ntimes-1)*nchannels*nwells*nfields])
        #get metadata of image
        pattern = ' type(.\*) value'
        meta_data_raw = str.replace(sample.meta.get('description'), '\"', '').split("/>\n<prop id=")
        pattern = ' type(.*?) value='
        meta_data_key = [re.sub(pattern, ',', x).split(',')[0] for x in meta_data_raw]
        meta_data_val = [re.sub(pattern, ',', x).split(',')[1] for x in meta_data_raw]
        meta_data = dict(zip(meta_data_key,meta_data_val))
        acquisition_time_local_end = meta_data.get('acquisition-time-local')
        acquisition_time_local_end =datetime.strptime(acquisition_time_local_end, '%Y%m%d %H:%M:%S.%f')
        deltaT = (acquisition_time_local_end-acquisition_time_local_start).total_seconds()/ntimes
        print(f'Average deltaT of the timeseries: {deltaT:.2f} s')
        
        '''sortedFilenames: sortorder is 1. Wells, 2. Fields, 3. Timepoints, 4. Channels, 5. Planes'''
        i=0
        totalFiles = nsteps * nchannels * nwells * nfields * ntimes
        print(f'Total files to be processed: {totalFiles}')
        for fn, im in zip(sortedFilenames, imgData):
            plane = int((i%nsteps))
            #print(f'plane: {plane}')
            channel = int(i/nsteps)%nchannels
            #print(f'channel: {channel}')
            timepoint = int(i/(nsteps*nchannels))%ntimes
            #print(f'timepoint: {timepoint}')
            field = int(i/(nsteps*nchannels*ntimes))%nfields
            #print(f'field: {field}')
            well = int(i/(nsteps*nchannels*ntimes*nfields))%nwells
            #print(f'well: {well}')
            dataset[timepoint, plane, channel, field, well, 0, 0] = im
            i+=1
            
        # Stitch together the many blocks into a single array
        lzy_img=da.block(dataset.tolist())
        lzy_img
        
        
        
        
        filename = list(remap_comps[0].keys())[0]
        img = lzy_img 
        
        
        fields = list(remap_comps[2].values())
        field_list = list(remap_comps[2].keys())
        wells = list(remap_comps[1].values())
        well_list = list(remap_comps[1].keys())
        star_fields = list(itertools.zip_longest(fields, field_list))
        star_wells = list(itertools.zip_longest(wells, well_list),)
        star_fields_wells = list(itertools.product(star_fields, star_wells))
        star_fields_wells=[list(x) for x in star_fields_wells ]
        star = []
        #compile the variable list for the workers and function
        # numbers and name descriptions are all added to have the full information available in the function
        #destPath, filename, img, calibration_x, calibration_y, nmintime, mintime, nmaxtime, maxtime, nminchannel, minchannel, nmaxchannel, maxchannel, nfield, field, nwell, well 
        for x, x_li in enumerate(star_fields_wells):
            temp = list(x_li[0] + x_li[1])    
            temp.insert(0,calibration_y)        
            temp.insert(0,calibration_x)
            temp.insert(0,deltaT)
            temp.insert(0, img)
            temp.insert(0,filename)
            temp.insert(0,destPath)
            star.append(tuple(temp))
        
        starli = []
        for x, x_li in enumerate(star_fields_wells):
            temp = list(x_li[0] + x_li[1])
            temp.insert(0,filename)
            starli.append(tuple(temp))
            
        ncpu= multiprocessing.cpu_count()
        #generate for number of fields a worker
        #ncpu = len(field_list)
        if ncpu > 40:
            ncpu = 40
        start_time= time.time()
        pl = Pool(ncpu)
        
    
        
    
        with pl as p:
             p.starmap(save_hyperstack_image, star)
        pl.close()
        pl.join()
        end_time = time.time()
        print(f'Saving done. It took {end_time- start_time: 0.2f} second(s) to complete.\n') 
        print(f'Saved files to {destPath}.')


