# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:03:33 2019

Code to analyze and plot the data coming from Triton but can also be used
with other machines. Just data coming in a text file from lockin measuring
transport.

For image saving three options are given jpg 300 dpi, jpg 600 dpi or pdf
PDF always gives the highest quality but to quickly look at the data using 
jpg might be more convenient. Also to do ppts jpg is easier to handle 

@author: Jaime Diez Merida (s1812831)
jaime.diezmerida@gmail.com
Master student University of Twente
ICE/QTM group 
"""

'''Import the necessary libraries'''
import numpy as np
import matplotlib.pyplot as plt 
import time
import glob
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

'''
The first code loads the data from linecans and plots it with the right units
The units depend on the system used. 
dc convert give the right conversion for the bias applied
ac convert gives the right conversion for the lock in amplitude
the data columns are the columns where the data is located, in here two lockins
are always used and lockin1 is in column 4 and lockin 2 in column 9
new box command appeared because there was a change in the summing module so 
the conversion factors changed
The code is thought with the scan type parameter so that different types of 
scans, IV, gate sweep, field sweep...will yield a graph with right units and 
right axis
The way originally ran in jupyter notebook is shown below. 
The idea is you can open many
files and save all the data from them in dictionaries so they are easily accesed
later
'''
#Example code in jupyter notebook
'''
files=['A_IV_15021033.dn','Tri2_IV_15021043.up','A_IV_15021033.dn']
# All IV curves
# scan type = 0
dIV = {} #stores all the data
xIV ={} #store the xaxis data for plotting
pIV={} #store the data to plot in the right units
ac_excit=[0.1,0.1]
#lockin = 1
for i in range(0,len(files)):
    for j in range(1,3):
        dIV['data'+str(i)+'sr'+str(j)],xIV['data'+str(i)+'sr'+str(j)],
        pIV['data'+str(i)+'sr'+str(j)]=triton_data.plot_linescans_file
        (files[i],lockin=j,ac_excit=ac_excit)


'''
directory = 'C:/Users/s1812831/data/Triton/'
sample = 'BSTS5'
directory= directory+sample+'/'
dc_convert = [10,10]
hd=0

def plot_linescans_file(dataset,conductance=False,save=False,scan_type=0,
                        lockin=1, ac_excit= [0.1,0.02],new_box=False,
                        symmetrize=False,curr_volt=True,center=0,
                        directory=directory,skip=3,delimiter=None,
                        dc_convert=dc_convert,hd=hd):
    '''
    #scan type can be IV sr1, dvdifieldsweep or dvdi gate sweep
    #skip the first three lines of the data because they are just text
    #set lockin to start from 0 to translate to python array numbering
    #sr and scan name chosen with scan type and lock in 
    '''
    
    ac_convert= np.array([100*10/ac_excit[0],100*10/ac_excit[1]],dtype='float')
    if new_box:        
        ac_convert= np.array([100*10/ac_excit[0],100/(3*ac_excit[1])],dtype='float')

    data_columns=[4,9]
    #title = ['Device Tri2 IV', 'Device B IV']
    #skip = 3
    lockin=lockin-1
    
    #open the file
    
    file = directory+dataset
    #dataset.split('_')
    save_name = sample+dataset
    raw_data = np.loadtxt(file,skiprows=skip,delimiter=delimiter)
    data = np.zeros((len(raw_data[0]),len(raw_data[:,0])))
    col_num=len(raw_data[0])
    for i in range(col_num):
        data[i] = raw_data[:,i]
        
    #Plot
    #depending on the scan type x_axis to plot changes
    x_axis=[data[2+lockin]*dc_convert[lockin],data[0]*688,data[0],data[0]]
    #print(x_axis[0])
    '''
    When the current is converted to a voltage in the Triton there is an offset
    from 0. This code aims to fix this offset. The code works. Center determines 
    which value to substract, the exactly one that is 0 in the current or the one
    after. 
    '''
    if curr_volt:
        if scan_type==0:
            for i in range(len(data[0])):
                if data[0][i]==0:
                    if center==0:
                        x_zero=i
                    elif center==1:
                        x_zero=i+1
                    #print(data[0])
                    #print(len(data[0]))
                    break
            #print(x_zero)
            x_axis[scan_type]=(data[2+lockin]-data[2+lockin][x_zero])*dc_convert[lockin]
    #print(x_axis[0])
    
    plot_data=data[data_columns[lockin]]*ac_convert[lockin]
    if symmetrize:
        sym_data=(plot_data+np.flip(plot_data))/2
        plot_data=sym_data
    
    #All these parameters are so that different types of scans are plotted
    #correctly
    sr=['sr1','sr2']
    col = ['k','b']
    scan_name=['IV','dvdi field','dvdi field','dvdi gate']
    x_label= [r'$Bias\ [mV]$',r'$Field\ [G]$' , r'$Field\ [T]$', r'$Gate\ [V]$']
 
    plt.figure(figsize=(6,4))
    plt.plot(x_axis[scan_type],plot_data, col[lockin])
    plt.title(save_name+' '+scan_name[scan_type]+' '+sr[lockin])
    plt.xlabel(x_label[scan_type])
    plt.ylabel(r'$R\ [k\Omega]$')
    plt.grid()
    if save:
        if hd==0:
            plt.savefig(save_name+scan_name[scan_type]+sr[lockin]+'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=300)
        elif hd==1:
            plt.savefig(save_name+scan_name[scan_type]+sr[lockin]+'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=600)
        elif hd==2:
            plt.savefig(save_name+scan_name[scan_type]+sr[lockin]+'_'+time.strftime("%H%M%d%m%y")+'.pdf')   
    plt.show()
    
    '''
    plt.figure(figsize=(6,4))
    plt.plot(x_axis[scan_type[scan_type+lockin]]*dc_convert[1],data[9]*ac_convert[1])
    plt.title((save_name+' IV sr2'))
    plt.grid()
    plt.xlabel(r'$Bias\ [mV]$')
    plt.ylabel(r'$R\ [k\Omega]$')
    if save:
        plt.savefig(save_name +' IV sr2'+'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=300)
    plt.show()
    '''
    
    return data, x_axis, plot_data

#check which lockin is in each column of your data
#column 7 is first lockin and column 12 is second locking
#usually the first three rows are just text, so we skip it
#save saves the image
'''
The second code is to plot megasweeps of data, so it can be plotted as maps
Similar parameters as before, now the column of the lockins is different from 
before
Also it gives general directory where the data usually is. This can be changed,
also it can be changed if a jupyter notebook is used
The idea is the same, just that different type of scans can be specified and the
right map and units will be plotted
The idea was similar to above, in jupyter put many files together in 
dictionaries so they are easy to access. Example code below:
    
'''
#Example code in jupyter notebook
'''
# B field scans

files=['mTri1Tri2_dvdiB_G56V.up',
       'mTri1Tri2_dvdiB_G43V.up',]

dIVB2 = {} #data to plot the maps
rIVB2 ={} #raw data for doing figures and also to have it
sIVB2= {} #the save_name data
#nIVB2 = {} #data that comes out from the normalization function

save=True
conductance=False
lockin = 1
scan_type = 0
ac_excit=[0.2,0.2]
sweep=100
for i in range(0,len(files)):
    for j in range(1,3):
        dIVB2['data'+str(i)+'sr'+str(j)],rIVB2['data'+str(i)+'sr'+str(j)],
        sIVB2['data'+str(i)+'sr'+str(j)]=triton_data.dvdi_maps(files[i],
        sweep = sweep,lockin=j,scan_type=scan_type,cmap='RdBu',ac_excit=ac_excit,save=save)
        #nIVB['data'+str(i)+'sr'+str(j)]=
        triton_data.normalize(dGB['data'+str(i)+'sr'+str(j)],rGB['data'+str(i)
        +'sr'+str(j)],sGB['data'+str(i)+'sr'+str(j)], scan_type=scan_type,sweep=sweep)

'''

#Parameters used for the code
save= False
conductance = False
gate= False 
ac_excit = [0.1,0.02]
dc_convert = [10,10]
#ac conversion with new summing module on 27/02/19
ac_convert= np.array([100*10/ac_excit[0],100/(3*ac_excit[1])],dtype='float')
#previous ac conversion is 
#ac_convert = np.array([100*10/ac_excit[0],100*10/(ac_excit[1])],dtype='float')
column = [7,12]
title = ['Device Tri2 IV', 'Device B IV']
directory = 'C:/Users/s1812831/data/Triton/'
sample = 'BSTS5'
directory= directory+sample+'/'

#ac_excit given in v



def dvdi_maps(dataset,sweep = 50, lockin=1 ,scan_type=0, column=column,directory=directory, cmap='RdBu',ac_excit=[0.1,0.02],
              plotting = True, save = False,conductance= False, eh=False, new_box=False,
              symmetrize=False,aspect='auto',font=15,title_name='',title=False,
              curr_volt=True, center=0,hd=hd):
    '''
    #say which lockin you are using
    #types of scans are 
        dvdi vs field Keithley, 
        dvdi vs gate, 
        gate vs field keithley
        dvdi vs field oxford
        gate vs field oxford
    #scan_type=[0,1,2,3,4]
    '''
    lockin=lockin-1 #convert to python order starting from 0
    skip = 3
    ac_convert= np.array([100*10/ac_excit[0],100*10/ac_excit[1]],dtype='float')
    if new_box:        
        ac_convert= np.array([100*10/ac_excit[0],100/(3*ac_excit[1])],dtype='float')

    #which file to open
    file = directory+dataset
    
    #loads the file
    raw_data = np.loadtxt(file,skiprows=skip)
    #create array with zeros to fill with the data in the right way
    data = np.zeros((len(raw_data[0]),len(raw_data[:,0])))
    col_num=len(raw_data[0])
    for i in range(col_num):
        data[i] = raw_data[:,i]
    #print(new[0])
    #find the values for the maps
    #print(range(0,len(new[0])))
    #make the IV into maps
    map_data = []
    map_line= []
    #for first lockin choose column 7
    #for second lockin column 12
    value = 0
    for i in range(0,len(data[0])):
        #print(i)
        value= value+1
        try: 
            a = data[0,value]-data[0,value-1]
        except(IndexError):
            a = 1
        if a == 0.0:
            #map_line.append(data[column[lockin],i]*ac_convert[lockin])
            if symmetrize:
                sym_data=(data[column[lockin],i]*ac_convert[lockin]+np.flip(data[column[lockin],i]*ac_convert[lockin]))/2
                map_line.append(sym_data)
            else:
                map_line.append(data[column[lockin],i]*ac_convert[lockin])
        else:
            map_data.append(map_line)
            map_line=[]

    #convert the array into a numpy array. easier to handle
    map_data=np.asarray(map_data)
    
    #plot the data obtained
    sr = ['sr1', 'sr2']
    save_name = sample+'_'+dataset+'_' +sr[lockin]
    scan_name=['_dvdi field map','_dvdi gate map','_dvdi field vs gate','_dvdi field map','_dvdi field vs gate']
    x_label= [r'$Bias\ [mV]$', r'$Bias\ [mV]$', r'$Field\ [G]$',r'$Bias\ [mV]$',r'$Field\ [T]$']
    y_label= [r'$Field\ [G]$', r'$Gate\ [V]$', r'$Gate\ [V]$',r'$Field\ [T]$',r'$Gate\ [V]$']
    #label to put in x axis
    #depending on the scan type x_axis to plot changes
    #column 3 is keith r
    #substract the 0 value 
    
    bias = data[3+lockin]*dc_convert[lockin]
    print(bias)
    #print(bias)
    #print(len(bias))
    #print(int(len(bias)/sweep))
    '''
    When the current is converted to a voltage in the Triton there is an offset
    from 0. This code aims to fix this offset. The code works. Center determines 
    which value to substract, the exactly one that is 0 in the current or the one
    after. 
    '''
    #print(data[2][0:sweep+1])
    curr_scan=[0,1,3]
    if curr_volt:
        if scan_type in curr_scan:
            for i in range(sweep):
                #print(i)
                #print(data[2][i])
                if data[2][i]==0:
                    if center==0:
                        x_zero=i
                    elif center==1:
                        x_zero=i+1
                    break
            print(x_zero)
            bias=(data[3+lockin]-data[3+lockin][x_zero])*dc_convert[lockin]
            print(bias)
            print(bias[x_zero])
    #for i in range(data[:,0]):
        
    #bias=bias-bias[int(sweep/2)]
    x_axis=[bias,bias,data[2]*688,bias,data[2]]
    extent_y0=[data[0,-1]*688,data[0,-1],data[0,-1],data[0,-1],data[0,-1]]
    extent_y1=[data[0,0]*688,data[0,0],data[0,0],data[0,0],data[0,0]]
    
    extent=[x_axis[scan_type][0],x_axis[scan_type][sweep],
    extent_y0[scan_type],extent_y1[scan_type]]

    
    if plotting == True:
        plt.figure(figsize=(10,7))
        plt.imshow(map_data,cmap=cmap ,extent=extent,aspect=aspect)
        plt.xlabel(x_label[scan_type],fontsize=font)
        plt.ylabel(y_label[scan_type],fontsize=font)
        if title:
            plt.title(title_name,fontsize=font)
        else:
            plt.title(save_name+scan_name[scan_type],fontsize=font)
        plt.tick_params(axis='both',which='major',labelsize=font)
        cb=plt.colorbar()
        cb.set_label(r'$R [kOhm]$',fontsize=font)
        cb.ax.tick_params(labelsize=font)
        
        if save == True:
            if hd==0:
                plt.savefig(save_name +'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=300)
            elif hd==1:
                plt.savefig(save_name +'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=600)
            elif hd==2:
                plt.savefig(save_name +'_'+time.strftime("%H%M%d%m%y")+'.pdf')
            
        plt.show()
    
    #Plot conductance
  
    if conductance or eh:
        cond = np.zeros_like(map_data)
        x_length = len(cond[0,:])
        y_length = len(cond[:,0])
        save_name = save_name
        cond_name=r'$G\ [mS]$'
        for i in range(y_length):
            for j in range(x_length):
                if map_data[i,j] != 0:
                    cond[i,j] = 1/(map_data[i,j])
                    if eh:
                        cond[i,j]=cond[i,j]*1E-3/(2*3.877E-5)
                        cond_name=r'$G\ [2e^2/h]$'
                else:
                    cond[i,j] = map_data[i,j]

        plt.figure(figsize=(10,7))
        plt.imshow(cond,cmap=cmap,extent=extent,aspect=aspect)
        plt.tick_params(axis='both',which='major',labelsize=font)
        plt.xlabel(x_label[scan_type],fontsize=font)
        plt.ylabel(y_label[scan_type],fontsize=font)
        if title:
            plt.title(title_name,fontsize=font)
        else:
            plt.title('dI/dV '+save_name+scan_name[scan_type],fontsize=font)
        cb=plt.colorbar()
        cb.set_label(cond_name,fontsize=font)
        cb.ax.tick_params(labelsize=font)
        if save == True:
            if hd==0:
                plt.savefig('G_'+save_name +'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=300)
            elif hd==1:
                plt.savefig('G_'+save_name +'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=600)
            elif hd==2:
                plt.savefig('G_'+save_name +'_'+time.strftime("%H%M%d%m%y")+'.pdf')
        plt.show()
        try:
            return map_data, cond, data, save_name
        except ValueError:
            return map_data, data, save_name
    else: 
        return map_data, data, save_name

#now the in data is a martix not the data file
'''
Here the data from the maps is normalized and then plotted in a normalized 
fashion. This code was used to compare easier some maps in which there is 
some decaying background but features repeate in one of the axis
'''


def normalize(plot_data, extent_data,save_name,name='',scan_type=2,sweep=300,
              cmap='jet',max_value_x=0.0, max_value_y=0.0,lockin=0,font=15,
              plotting=True,save=False,hd=hd):
    #requires an np.array as input
    #Now I want to normalize all the values of a given set to have
    #same axis values to look for pattern
    nor_x = np.zeros_like(plot_data)
    avg_x = np.zeros_like(plot_data)
    nor_y = np.zeros_like(plot_data)
    avg_y = np.zeros_like(plot_data)
    b_length = len(plot_data[0,:])
    gate_length = len(plot_data[:,0])
    
    scan_name=['dvdi field map','dvdi gate map','dvdi field vs gate','dvdi field map','dvdi field vs gate']
    x_label= [r'$Bias\ [mV]$', r'$Bias\ [mV]$', r'$Field\ [G]$',r'$Bias\ [mV]$',r'$Field\ [T]$']
    y_label= [r'$Field\ [G]$', r'$Gate\ [V]$', r'$Gate\ [V]$',r'$Field\ [T]$',r'$Gate\ [V]$']
    data=extent_data
    bias = data[3+lockin]*dc_convert[lockin]
    x_axis=[bias,bias,data[2]*688,bias,data[2]]
#    x_axis=[data[0],data[0],data[2]*688,data[0],data[2]]
    extent_y0=[data[0,-1]*688,data[0,-1],data[0,-1],data[0,-1],data[0,-1]]
    extent_y1=[data[0,0]*688,data[0,0],data[0,0],data[0,0],data[0,0]]
    extent=[x_axis[scan_type][0],x_axis[scan_type][sweep],
    extent_y0[scan_type],extent_y1[scan_type]]
    
    #extent = [extent_data[2,0],extent_data[2,300],extent_data[0,-1],extent_data[0,0]]
    for i in range(gate_length):
        for j in range(b_length):
            avg_x[i,j] = sum(plot_data[i,:])/len(plot_data[i,:])
            avg_y[i,j] = sum(plot_data[:,j])/len(plot_data[:,j])
            #if i == 26:
             #   avg_map[i][j] = sum(data_map[i][0:150])/len(data_map[i][0:150])
            nor_x[i,j] = plot_data[i,j]-avg_x[i,j]
            nor_y[i,j] = plot_data[i,j]-avg_y[i,j]
    if plotting == True:
        plt.figure(figsize=(10,7))
        if max_value_x == 0.0:
            plt.imshow(nor_x,cmap=cmap,extent=extent,aspect='auto')
        else:
            plt.imshow(nor_x,cmap=cmap,extent=extent,
                           aspect='auto',vmax=max_value_x,vmin=-max_value_x)
        plt.title(name+' Field normalized '+ scan_name[scan_type],fontsize=font)
        plt.tick_params(axis='both',which='major',labelsize=font)
        plt.xlabel(x_label[scan_type],fontsize=font)
        plt.ylabel(y_label[scan_type],fontsize=font)
        cb=plt.colorbar()
        cb.set_label('Normalized',fontsize=font)
        cb.ax.tick_params(labelsize=font)
        if save == True:
            if hd==0:
                plt.savefig(save_name+'Bnorm'+'.png',dpi=300)
            elif hd==1:
                plt.savefig(save_name+'Bnorm'+'.png',dpi=600)
            elif hd==2:
                plt.savefig(save_name+'Bnorm'+'.pdf')
        plt.show()
        '''
        plt.figure(figsize=(12,8))
        if max_value_y == 0.0:
            plt.imshow(nor_y,cmap='jet',extent=extent,aspect='auto')
        else:
            plt.imshow(nor_y,cmap='jet',extent=extent,
                           aspect='auto',vmax=max_value_y,vmin=-max_value_y)
        plt.title(name+' Gate normalized dV/dI map field vs gate')
        plt.xlabel('Field [mA]')
        plt.ylabel('Gate [V]')
        plt.colorbar()
        if save == True:
            print(1)
            plt.savefig(save_name+ 'Gnorm'+'.png')
        plt.show()
        '''

    #return nor_x, nor_y
    return nor_x




#column 7 is first lockin and column 12 is second locking

'''
Extract files from a folder depending on some keyword. This can be useful
when many different scans are taken with similar name so that you can open
all of them together

'''
def files_from_folder(directory=directory, keyword="*_*"):
    file_list = glob.glob(os.path.join(os.getcwd(),directory, keyword))
    
    file_matrix = []
    
    for file_path in file_list:
        if 'batch' in file_path:
            continue
        print(file_path)
        raw_data=np.loadtxt(file_path,skiprows=3)
        data = np.zeros((len(raw_data[0]),len(raw_data[:,0])))
        col_num=len(raw_data[0])
        for i in range(col_num):
            data[i] = raw_data[:,i]
        file_matrix.append(data)

    return file_matrix
'''
Gives back the data ordered from smaller to larger value of Temperature.
When doing the many files it is necessary to reorder the files from lower to 
higher value (if order is important). In this case temperature was the ordering
parameter but if the column is changed from 18 (which is the temperature 
column), to some other column then the files could be ordered following another
criterium
'''

def reorder_temp(data):
    length=len(data)
    new_data = np.zeros((length,1))

    for i in range (length):
        new_data[i]= data[i][18,0]
    ordered_data = np.sort(new_data ,0)
    positions =np.zeros_like(ordered_data , dtype='int')
    full_order_data =[]
    for i in range(length):
        for j in range(length):
            if data[j][18,0]==ordered_data [i]:
                positions [i]=j
        full_order_data .append(data[positions[i,0]])
    return full_order_data
    
#ac_excit = [0.1,0.02]
#dc_convert = [10,10]
#ac_convert= np.array([100*10/ac_excit[0],100/(3*ac_excit[1])],dtype='float')
#data_columns=[4,9]
#title = ['Device Tri2 IV', 'Device B IV']


'''
#the load many files is mostly used for the temperature scans. Basically open
many files and plots them after each other. By using the code below then you can 
store all the data in the same dictionary which is then easy to access

'''
#Copy these lines on top of the notebook code
'''
directory = 'C:/Users/s1812831/data/Triton/'
sample = 'BSTS5'
folder='/IV_T/'
directory= directory+sample+folder
'''
#keyword is what differences the files from other files in same folder
def load_manyfiles(keyword='**',directory='',sort_par=18):
    file_list = glob.glob(os.path.join(os.getcwd(),directory, keyword))
    #store the data
    data=[]
    for file_path in file_list:
        raw_data=np.loadtxt(file_path,skiprows=3)
        line_data=np.zeros((len(raw_data[0]),len(raw_data[:,0])))
        col_num=len(raw_data[0])
        for i in range(col_num):
            line_data[i]=raw_data[:,i]
        data.append(line_data)
    #now sort the data from min to max value of the data you want
    #if temperature, then col 18 is the sorting parameter
    minim_data=np.zeros((len(data),1))
    for i in range(len(data)):
        minim_data[i]=data[i][18,0]
    ordered_data=np.sort(minim_data,0)
    positions=np.zeros_like(ordered_data,dtype='int')
    'Puts all the files in the right order'
    full_order=[]
    files_order=[]
    for i in range((len(data))):
        for j in range((len(data))):
            if data[j][18,0]==ordered_data[i]:
                positions [i]=j
                full_order.append(data[positions[i,0]])
                files_order.append(file_list[positions[i,0]])
    
    return files_order, full_order


'''
This finally plots the many files loaded. The way to excute the code is shown 
below (if used with a jupyter notebook)
files, data=triton_data.load_manyfiles('*_0V*',directory=directory)
#print(files)
ac_excit=[0.2,0.02]
dIVT0V = {}
xIVT0V={}
#lockin = 1
scan_type = 0
for i in range(0,len(files)):
    for j in range(1,3):
        dIVT0V['data'+str(i)+'sr'+str(j)],xIVT0V['data'+str(i)+'sr'+str(j)]=
        triton_data.plot_linescans_directory(files[i],lockin=j,
        scan_type=scan_type,ac_excit=ac_excit,new_box=True)

'''

def plot_linescans_directory(dataset,conductance=False,save=False,scan_type=0,
                             lockin=1, ac_excit= [0.1,0.02],new_box=False,
                             curr_volt=True,center=0,df=False,hd=0):
    #dataset includes directory in this case
    #scan type can be IV sr1, dvdifieldsweep or dvdi gate sweep
    #skip the first three lines of the data because they are just text
    #set lockin to start from 0 to translate to python array numbering
    #sr and scan name chosen with scan type and lock in 
    dc_convert = [10,10]
    ac_convert= np.array([100*10/ac_excit[0],100*10/ac_excit[1]],dtype='float')
    if new_box:        
        ac_convert= np.array([100*10/ac_excit[0],100/(3*ac_excit[1])],dtype='float')
        #dc_convert = [10,3.3]
    data_columns=[4,9]
    #title = ['Device Tri2 IV', 'Device B IV']
    skip = 3
    lockin=lockin-1
    
    #open the file
    
    file = dataset
    #dataset.split('_')
    splitted=dataset.split('/')
       
    save_name = sample+splitted[-1]
    
    raw_data = np.loadtxt(file,skiprows=skip)
    data = np.zeros((len(raw_data[0]),len(raw_data[:,0])))
    col_num=len(raw_data[0])
    for i in range(col_num):
        data[i] = raw_data[:,i]
        
    #Plot
    #depending on the scan type x_axis to plot changes
    x_axis=[data[2+lockin]*dc_convert[lockin],data[0]*688,data[0],data[0]]
    #print(x_axis[0])
    '''
    When the current is converted to a voltage in the Triton there is an offset
    from 0. This code aims to fix this offset. The code works. Center determines 
    which value to substract, the exactly one that is 0 in the current or the one
    after. 
    '''
    
    if curr_volt:
        if scan_type==0:
            for i in range(len(data[0])):
                if data[0][i]==0:
                    if center==0:
                        x_zero=i
                    elif center==1:
                        x_zero=i+1
                    #print(data[0])
                    #print(len(data[0]))
                    break
            #print(x_zero)
            x_axis[scan_type]=(data[2+lockin]-data[2+lockin][x_zero])*dc_convert[lockin]
    #print(x_axis[0])
    
        
    sr=['sr1','sr2']
    col = ['k','b']
    scan_name=['IV','dvdi field','dvdi field','dvdi gate']
    x_label= [r'$Bias\ [mV]$',r'$Field\ [G]$' , r'$Field\ [T]$', r'$Gate\ [V]$']
 
    plt.figure(figsize=(6,4))
    plt.plot(x_axis[scan_type],data[data_columns[lockin]]*ac_convert[lockin], col[lockin],label='T: {0:.3f}K' .format(data[18][0]))
    plt.title(save_name+' '+scan_name[scan_type]+' '+sr[lockin])
    plt.xlabel(x_label[scan_type])
    plt.ylabel(r'$R\ [k\Omega]$')
    plt.legend()
    plt.grid()
    if save:
        if hd==0:
            plt.savefig(save_name+scan_name[scan_type]+sr[lockin]+'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=300)
        elif hd==1:
            plt.savefig(save_name+scan_name[scan_type]+sr[lockin]+'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=600)
        elif hd==2:
            plt.savefig(save_name+scan_name[scan_type]+sr[lockin]+'_'+time.strftime("%H%M%d%m%y")+'.pdf')

    
    plt.show()
    
    '''
    plt.figure(figsize=(6,4))
    plt.plot(x_axis[scan_type[scan_type+lockin]]*dc_convert[1],data[9]*ac_convert[1])
    plt.title((save_name+' IV sr2'))
    plt.grid()
    plt.xlabel(r'$Bias\ [mV]$')
    plt.ylabel(r'$R\ [k\Omega]$')
    if save:
        plt.savefig(save_name +' IV sr2'+'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=300)
    plt.show()
    '''
    
    return data, x_axis


'Matplotlib inline kind of programmes, here to code it correctly but needs to be used'
'online in the notebook'

#Data to be used
#data_dict=dGB
#data_dict=pIVB
#x_dict=xIVB
#dev_names = ['' ,'Tri2 ','Tri1 ']
#extra_name=['1T','1.2T','1.4T','1.6T','1.8T','2T','2.2T','2.4T']

'This is to make interactive plots from single linescans'
#example code
'''
%matplotlib inline

data_dict=pIV
x_dict=xIV
dev_names = ['' ,'Tri2 ','Tri1 ']

def interactive_linescans(dataset=0, lockin=1, scan_type=0,save=False, 
                          conduc=False, eh=False):
    
    #Choose the device you want to plot
    dev_data=data_dict['data'+str(dataset)+'sr'+str(lockin)]
    dev_plot=x_dict['data'+str(dataset)+'sr'+str(lockin)][scan_type]
    #dev_norm=norm_dict['data'+str(dataset)+'sr'+str(lockin)]

    triton_data.int_singlelinescans(dev_data, dev_plot,dev_names ,
                                    lockin,scan_type, save, conduc, eh)
    
    return 
interactive(interactive_linescans,dataset=(0,30,1), lockin=(0,2,1), scan_type=(0,6,1))
'''

def int_singlelinescans(dev_data, x_data, dev_names, lockin, 
                        scan_type, save=False, conduc=False, eh=False,
                        extra_name='',hd=0):
    
    scan_name=['IV ','field sweep', 'gate sweep ','field sweep ']
    x_axis= [r'$Bias\ [mV]$', r'$Field\ [G]$',r'$Gate\ [V]$',r'$Field\ [T]$']
    
    
    plt.figure(figsize=(10,7))
    cond=np.zeros_like(dev_data)
    cond_eh=np.zeros_like(dev_data)
    for i in range(len(dev_data)):
        cond[i] = 1/(dev_data[i])  #units in mS
        cond_eh[i] =cond[i]*1E-3/(2*3.877E-5)
    if conduc==False and eh==False:
        plt.plot(x_data,dev_data,'b')
        plt.ylabel(r'$dV/dI\ [k\Omega]$')
    elif conduc==True and eh==False:
        plt.plot(x_data,cond,'b')
        plt.ylabel(r'$dI/dV\ [mS]$')
    elif eh==True:
        plt.plot(x_data,cond_eh,'b')
        plt.ylabel(r'$2e^2/h$')    
    
    plt.title('BSTS5 '+dev_names[lockin]+scan_name[scan_type]+extra_name )
    plt.xlabel(x_axis[scan_type])
    #plt.ylabel(r'$dV/dI\ [k\Omega]$')
    plt.grid()
    if save:
        if hd==0:
            plt.savefig('BSTS5 '+dev_names[lockin+1]+scan_name[scan_type]+extra_name +'_'+time.strftime("%H%M%d%m%y")+'.png',dpi=300)
           
        elif hd==1:
            plt.savefig('BSTS5 '+dev_names[lockin+1]+scan_name[scan_type]+extra_name +'_'+time.strftime("%H%M%d%m%y")+'.png', dpi=600)
        elif hd==2:
            plt.savefig('BSTS5 '+dev_names[lockin+1]+scan_name[scan_type]+extra_name +'_'+time.strftime("%H%M%d%m%y")+'.pdf')
    plt.show()
    return

#Data to be used
#data_dict=dGB
#raw_dict={}
#data_dict={}
#dev_names = ['' ,'Tri2','Tri1']

'''
This makes the interactive linescans from the maps. It allows you to access
different lines of the maps, both in the vertical and horizontal direction
Again with scan_type it should ensure that the correct map and correct units 
are plotted 
The code to run such a structure is shown below:
'''
#Example code
'''
%matplotlib inline
#field_dev = [Tri2_norb, B_norb, Tri2lft_norb,Tri1_norb, Tri2_dc005_norb]
#gate_dev = [Tri2_norG,B_norG,Tri2lft_norG,Tri1_norG] 

#Data to be used
#data_dict=dGB
data_dict=pIVB
x_dict=xIVB
dev_names = ['' ,'Tri2 ','Tri1 ']
extra_name=['1T','1.2T','1.4T','1.6T','1.8T','2T','2.2T','2.4T']
def interactive_linescans(dataset=0, lockin=1, scan_type=0,save=False, conduc=False, eh=False):
    
    #Choose the device you want to plot
    dev_data=data_dict['data'+str(dataset)+'sr'+str(lockin)]
    dev_plot=x_dict['data'+str(dataset)+'sr'+str(lockin)][scan_type]
    #dev_norm=norm_dict['data'+str(dataset)+'sr'+str(lockin)]

    triton_data.int_singlelinescans(dev_data, dev_plot,dev_names ,lockin,scan_type, save, conduc, eh,extra_name=extra_name[dataset])
    
    return 
interactive(interactive_linescans,dataset=(0,30,1), lockin=(0,2,1), scan_type=(0,6,1))
'''

# Create a new colormap

nipyspbig = plt.get_cmap('nipy_spectral', 512)
newcmp = ListedColormap(nipyspbig(np.linspace(0.15, 0.9, 256)))

def make_int_linescans(dev_data, dev_plot, dev_names,lockin,
                       scan_type=0, hor_st=0 ,hor_ext=20, vert_st=0, 
                       vert_ext=10,hor_offset=0,vert_offset=0, hor_int=1, 
                       vert_int=1,plot_hor=True,plot_vert=False, 
                       save=False, conduc=False, eh=False,legend=False,interlocked=False,
                       extra_name=False,expand=10,symmetrize=False,background=False,
                       backg_line=0,font=15,name='',curr_volt=True,center=0,fontleg=15,
                       hd=0):
    
    #Choose the device you want to plot
    #dev_data=raw_dict['data'+str(dataset)+'sr'+str(lockin)]
    #dev_plot=data_dict['data'+str(dataset)+'sr'+str(lockin)]
    
    '''
    Make the datasets for x axis. Depending on the scan type different columns taken
    hor correspond to x axis, thess can be bias,bias,field[G],bias, field[T] 
    vert correspond to y axis, these are field [G], gate, gate, field[T], gate
    '''
    
    hor_column=[3+lockin-1,3+lockin-1,2,3+lockin-1,2] #Columns to be chosen
    hor_mult=[10,10,688,10,1] #Multipliers to get the right units
    vert_column=[0,0,0,0,0] #Columns to be chosen   
    vert_mult=[688,1,1,1,1] #Multipliers to get the right units
    
    # Make the arrays of data for horizontal and vertical from the image
    
    hor = np.linspace(dev_data[hor_column[scan_type]][0],dev_data[hor_column[scan_type]][-1],len(dev_plot[hor_column[scan_type]]))*hor_mult[scan_type]
    #print(hor)
    '''
    When the current is converted to a voltage in the Triton there is an offset
    from 0. This code aims to fix this offset. The code works. Center determines 
    which value to substract, the exactly one that is 0 in the current or the one
    after. 
    '''
    curr_scan=[0,1,3]
    if curr_volt:
        if scan_type in curr_scan:
            for i in range(len(dev_plot[hor_column[scan_type]])):
                if dev_data[2][i]==0:
                    if center==0:
                        x_zero=i
                    elif center==1:
                        x_zero=i+1
                    #print(data[2])
                    #print(len(data[2]))
                    break
            print(x_zero)
            hor = (np.linspace(dev_data[hor_column[scan_type]][0],dev_data[hor_column[scan_type]][len(dev_plot[hor_column[scan_type]])],len(dev_plot[hor_column[scan_type]]))-dev_data[hor_column[scan_type]][x_zero])*hor_mult[scan_type]
            #print(hor)
    #   '''
    #print(len(hor))
    vert= np.linspace(dev_data[vert_column[scan_type]][0],dev_data[vert_column[scan_type]][-1],len(dev_plot[:,1]))*vert_mult[scan_type]
    #print(vert)
    #print(len(dev_plot[:,1]))
    #print(len(hor))
    '''
    #Choose which lines you wish to plot. There should be a max limit of lines to avoid errors that
    you want to plot over the limit of the array
    start is where it starts and ext(extension) adds how many more lines
    If the end is further than the real end then we put a limit on that. This has to do in the 
    plotting stage
    '''
    
    hor_end=hor_st+hor_ext 
    vert_end=vert_st+vert_ext
    if hor_end>=len(hor):
        hor_end=len(hor)
        print('Index'+ str(hor_end)+' out of range')
    if vert_end>=len(vert):
        vert_end=len(vert)
        print('Index'+ str(vert_end)+' out of range')
    #print(len(vert))
    #print(hor_end)
    '''
    Choose the lines to be plotted. There are two ways, interconnected or not. If interconnected
    then the lines on one plot correspond to the amount plotted on the other plot. Otherwise
    you just plot a limit without being the same as the other plot
    '''
    #if interlocked:
    vert_lines=np.arange(hor_st,hor_end,hor_int)
    hor_lines=np.arange(vert_st,vert_end,vert_int)
    
   
    '''
    Create a conductance array to plot conductance . Both in mS and in e^2/h units
    '''
    cond = np.zeros_like(dev_plot)
    cond_eh=np.zeros_like(cond)

    for i in range(len(vert)):
        for j in range(len(hor)):
            cond[i,j] = 1/(dev_plot[i,j])  #units in mS
            cond_eh[i,j] =cond[i,j]*1E-3/(2*3.877E-5) #in units of e^2/h
    
    '''
    Could make a plot with the background substracted
    backg=np.zeros_like(dev_plot)
    
    for i in range(len(hor)):
        backg[:,i]=cond[:,i]-np.mean(cond[:,i])
    ''' 
    
    '''
    For the colorprogression of the linescans. The idea is to have a progressing color code
    with always the same limits, to make it clear in which direciton the plotting advances
    '''
    hor_num=len(hor_lines)
    vert_num=len(vert_lines)
    
    
    cmap = plt.get_cmap(newcmp)
    hor_colors = [cmap(i) for i in np.linspace(0, 1, hor_num)]
    vert_colors = [cmap(i) for i in np.linspace(0, 1, vert_num)]
    
    #Prepare all the labels to make good plots
    if extra_name==False:
        scan_name=['_dvdi field map','_dvdi gate map','_dvdi field vs gate','_dvdi field map','_dvdi field vs gate']
    else:
        scan_name=[name,name,name,name,name]
    hor_x= [r'$Bias\ [mV]$', r'$Bias\ [mV]$', r'$Field\ [G]$',r'$Bias\ [mV]$',r'$Field\ [T]$']
    vert_x= [r'$Field\ [G]$', r'$Gate\ [V]$', r'$Gate\ [V]$',r'$Field\ [T]$',r'$Gate\ [V]$']
    hor_labels=['Field[G]', 'Gate[V]', 'Gate[V]','Field[T]','Gate[V]']
    vert_labels=['Bias mV]', 'Bias[mV]', 'Field[G]','Bias[mV]','Field[T]']
    
    i=0
    #Make the horizontal plot
    if plot_hor:
        if conduc==False and eh==False:
            int_horplot(hor,dev_plot,hor_st,hor_end,hor_lines,vert,hor_offset,
                        hor_labels[scan_type],hor_colors,interlocked,expand=expand,
                        symmetrize=symmetrize,background=background,backg_line=backg_line)
        elif conduc==True:
            int_horplot(hor,cond,hor_st,hor_end,hor_lines,vert,hor_offset,hor_labels[scan_type],
                        hor_colors,interlocked,expand=expand,symmetrize=symmetrize,
                        background=background,backg_line=backg_line)            
        elif eh==True:
            int_horplot(hor,cond_eh,hor_st,hor_end,hor_lines,vert,hor_offset,
                        hor_labels[scan_type],hor_colors,interlocked,expand=expand,
                        symmetrize=symmetrize,background=background,backg_line=backg_line)
        '''  
        plt.figure(figsize=(10,7))
        hor_count =0
        for i in hor_lines:
            if conduc==False and eh==False:
                if len(hor_lines)>10 and hor_count%10==0:
                    plt.plot(hor[hor_st:hor_end],dev_plot[i,hor_st:hor_end]+hor_offset*hor_count, label=hor_labels[scan_type]+'{0:.2f}'.format(vert[i]),color=hor_colors[hor_count])
                elif len(hor_lines)>10:
                    plt.plot(hor[hor_st:hor_end],dev_plot[i,hor_st:hor_end]+hor_offset*hor_count ,color=hor_colors[hor_count])
                else:
                    plt.plot(hor[hor_st:hor_end],dev_plot[i,hor_st:hor_end]+hor_offset*hor_count, label=hor_labels[scan_type]+'{0:.2f}'.format(vert[i]),color=hor_colors[hor_count])
                    
            elif conduc==True:
                if len(hor_lines)>10 and hor_count%20==0:
                    plt.plot(hor[hor_st:hor_end],cond[i,hor_st:hor_end]+hor_offset*hor_count, label=hor_labels[scan_type]+'{0:.2f}'.format(vert[i]),color=hor_colors[hor_count])
                elif len(hor_lines)>10:
                    plt.plot(hor[hor_st:hor_end],cond[i,hor_st:hor_end]+hor_offset*hor_count,color=hor_colors[hor_count])
                else:
                    plt.plot(hor[hor_st:hor_end],cond[i,hor_st:hor_end]+hor_offset*hor_count, label=hor_labels[scan_type]+'{0:.2f}'.format(vert[i]),color=hor_colors[hor_count])
                    
            elif eh==True:
                if len(hor_lines)>10 and hor_count%20==0:
                    plt.plot(hor[hor_st:hor_end],cond_eh[i,hor_st:hor_end]+hor_offset*hor_count, label=hor_labels[scan_type]+'{0:.2f}'.format(vert[i]),color=hor_colors[hor_count])
                else:
                    plt.plot(hor[hor_st:hor_end],cond_eh[i,hor_st:hor_end]+hor_offset*hor_count,color=hor_colors[hor_count])
            hor_count=hor_count+1
        '''         
        #title='BSTS5 '+dev_names[lockin]+scan_name[scan_type]+ 'offseted by' +str(hor_offset)+extra_name+'backg line'+str(backg_line)
        if hor_offset==0.0 and background==False:
            title='BSTS5 '+dev_names[lockin]+scan_name[scan_type] 
        elif background==True and hor_offset==0.0:
            title='BSTS5 '+dev_names[lockin]+scan_name[scan_type]+ ' backg '+hor_labels[scan_type]+' '+str(vert[backg_line])
        elif background==False and hor_offset!=0.0:
            title='BSTS5 '+dev_names[lockin]+scan_name[scan_type]+ ' offseted by ' +str(hor_offset)
        else:
            title='BSTS5 '+dev_names[lockin]+scan_name[scan_type]+ ' offseted by ' +str(hor_offset)+ ' backg '+hor_labels[scan_type]+' ' +str(vert[backg_line])
        plt.title(title,fontsize=font)
        #plt.xlabel('Field [G]')
        plt.xlabel(hor_x[scan_type],fontsize=font)
        #plt.locator_params(axis='x', nbins=15)
        plt.ylabel(r'$dVdI\ [k\Omega]$',fontsize=font)
        plt.tick_params(axis='both',which='major',labelsize=font)
        if conduc:
            plt.ylabel(r'$dIdV\ [mS]$',fontsize=font)
        if eh:
            plt.ylabel((r'$dI/dV\ [2e^2/h]$'),fontsize=font)   
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(1, 1), shadow=True, ncol=1,fontsize=fontleg)
        plt.grid()
        if save:
            if hd==0:
                plt.savefig(title+time.strftime("%H%M%d%m%y")+'.png', dpi=300)
            elif hd==1:
                plt.savefig(title+time.strftime("%H%M%d%m%y")+'.png', dpi=600)
            elif hd==2:
                plt.savefig(title+time.strftime("%H%M%d%m%y")+'.pdf')
        plt.show()
    

    if plot_vert:
        if conduc==False and eh==False:
            int_vertplot(vert,dev_plot,vert_st,vert_end,vert_lines,hor,vert_offset,vert_labels[scan_type],vert_colors,interlocked,expand=expand)
        elif conduc==True:
            int_vertplot(vert,cond,vert_st,vert_end,vert_lines,hor,vert_offset,vert_labels[scan_type],vert_colors,interlocked,expand=expand)          
        elif eh==True:
            int_vertplot(vert,cond_eh,vert_st,vert_end,vert_lines,hor,vert_offset,vert_labels[scan_type],vert_colors,interlocked,expand=expand)
        '''
        plt.figure(figsize=(12,8))
        vert_count =0
        for i in vert_lines:
            if conduc==False and eh==False:
                if len(vert_lines)>10 and vert_count%10==0:
                    plt.plot(vert[vert_st:vert_end],dev_plot[vert_st:vert_end,i]+vert_offset*vert_count, label=vert_labels[scan_type]+'{0:.2f}'.format(hor[i]),color=vert_colors[vert_count])
                else:
                    plt.plot(vert[vert_st:vert_end],dev_plot[vert_st:vert_end,i]+vert_offset*vert_count, color=vert_colors[vert_count])
 
            elif conduc==True:
                if len(vert_lines)>10 and hor_count%20==0:
                    plt.plot(vert[vert_st:vert_end],cond[vert_st:vert_end,i]+vert_offset*vert_count, label=vert_labels[scan_type]+'{0:.2f}'.format(hor[i]),color=vert_colors[vert_count])
                else:
                    plt.plot(vert[vert_st:vert_end],cond[vert_st:vert_end,i]+vert_offset*vert_count,color=vert_colors[vert_count])
            elif eh==True:
                if len(vert_lines)>10 and vert_count%20==0:
                    plt.plot(vert[vert_st:vert_end],cond_eh[vert_st:vert_end,i]+vert_offset*vert_count, label=vert_labels[scan_type]+'{0:.2f}'.format(hor[i]),color=vert_colors[vert_count])
                else:
                    plt.plot(vert[vert_st:vert_end],cond_eh[vert_st:vert_end,i]+vert_offset*vert_count,color=vert_colors[vert_count])
            vert_count=vert_count+1
            '''
        title='BSTS5 '+dev_names[lockin]+scan_name[scan_type]+ 'offseted by' +str(vert_offset)
        if vert_offset==0.0:
            title='BSTS5 '+dev_names[lockin]+scan_name[scan_type]
        plt.title(title,fontsize=font)
        plt.xlabel(vert_x[scan_type],fontsize=font)
        plt.ylabel(r'$dVdI\ [k\Omega]$',fontsize=font)
        plt.tick_params(axis='both',which='major',labelsize=font)
        if conduc:
            plt.ylabel(r'$dIdV\ [mS]$',fontsize=font)
        if eh:
            plt.ylabel((r'$dI/dV\ [2e^2/h]$'),fontsize=font) 
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(1, 1), shadow=True, ncol=1,fontsize=fontleg)
        plt.grid()
        if save:
            if hd==0:
                plt.savefig(title+time.strftime("%H%M%S%d%m%y")+'.png', dpi=300)
            elif hd==1:
                plt.savefig(title+time.strftime("%H%M%S%d%m%y")+'.png', dpi=600)
            elif hd==2:
                plt.savefig(title+time.strftime("%H%M%S%d%m%y")+'.pdf')
    
        plt.show()

    return

''' 
Plotting for interactive linescans. Code usedin the above lines for doing all
the plots
'''
def int_horplot(x,y,init,end,lines_to_plot,lines_label,offset,labels,colors,
                interlocked,expand,symmetrize,background, backg_line):
    plt.figure(figsize=(10,7+expand*offset))
    count =0
    for i in lines_to_plot:
        if interlocked:
            #if len(lines_to_plot)>10 and count%10==0:
            #    plt.plot(x[init:end],y[i,init:end]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
            #elif len(lines_to_plot)>10:
            #    plt.plot(x[init:end],y[i,init:end]+offset*count ,color=colors[count])
            #else:
            if symmetrize:
                plt.plot(x[init:end],(y[i,init:end]+np.flip(y[i,init:end]))/2+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
                count=count+1
            if background:
                plt.plot(x[init:end],y[i,init:end]-y[backg_line,init:end]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
                count=count+1
            else:
                plt.plot(x[init:end],y[i,init:end]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
                count=count+1
                
        else:
            #if len(lines_to_plot)>10 and count%10==0:
            #    plt.plot(x,y[i,:]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
            #elif len(lines_to_plot)>10:
            #    plt.plot(x,y[i,:]+offset*count ,color=colors[count])
            #else:
            if symmetrize:
                plt.plot(x,(y[i,:]+np.flip(y[i,:]))/2+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
                count=count+1
            if background:
                plt.plot(x[:],y[i,:]-y[backg_line,:]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
                count=count+1
            else:
                plt.plot(x,y[i,:]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
                count=count+1
    
    return

def int_vertplot(x,y,init,end,lines_to_plot,lines_label,offset,labels,colors,interlocked,expand):
    plt.figure(figsize=(10,7+expand*offset))
    count =0
    for i in lines_to_plot:
        if interlocked:
            #if len(lines_to_plot)>10 and count%10==0:
            #    plt.plot(x[init:end],y[init:end,i]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
            #elif len(lines_to_plot)>10:
            #    plt.plot(x[init:end],y[init:end,i]+offset*count ,color=colors[count])
            #else:
            plt.plot(x[init:end],y[init:end,i]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
            count=count+1
        else:
            #if len(lines_to_plot)>10 and count%10==0:
            #    plt.plot(x,y[:,i]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
            #elif len(lines_to_plot)>10:
            #    plt.plot(x,y[:,i]+offset*count ,color=colors[count])
            #else:
            plt.plot(x,y[:,i]+offset*count, label=labels+'{0:.2f}'.format(lines_label[i]),color=colors[count])
            count=count+1
    
    return



'''
Interactive with initial being many linescans. The way to use it in the 
jupyter notebook is shown below:
  
    
data_dict=[dIVT0V,dIVT60V,dIVTh,dIVThb]
x_dict=[xIVT0V,xIVT60V,xIVTh,xIVThb]
ac_convert= np.array([100*10/ac_excit[0],100/(3*ac_excit[1])],dtype='float')
dc_convert = [10,3.3]
data_column=[4,9]
lockins=[1,2]
gen_name='BSTS5 '
name=['Tri2','Tri1']
name_cond=['Gate 0V','Gate 60V','high T','high T']

def interactive_linescans(dataset=0,lockin=0,init=0,end=31,interv=1,offset=0,
                          conduc=False,eh=False,legend=True,save=False):
    
    triton_data.int_compare_lins(data_dict=data_dict,x_dict=x_dict,dataset=dataset,
                                 lockin=lockin,init=init,end=end,interv=interv,
                                 offset=offset,conduc=conduc,
                                 eh=eh,name=name,name_cond=name_cond,
                                 legend=legend,save=save)
    return
                     
interactive(interactive_linescans,dataset=(0,5,1),lockin=(0,1),init=(0,60,1),
            end=(0,60,1),interv=(0,60,1),offset=(0,10,0.001))    
'''

def int_compare_lins(data_dict, x_dict,scan_type=0,dataset=0,lockin=0,init=0,ext=30,interv=1,
                     data_column=[4,9], offset=0, font=15,
                     conduc=False,eh=False,gen_name='BSTS5',name=['',''],
                     name_cond=['',''],legend=True,save=False,temperature=False,
                     ac_convert=ac_convert,legfont=15):
    count=0
    plt.figure(figsize=(10,7))
    '''Create the colormap for the linescans'''
    cmap=plt.get_cmap('jet')
    if init+ext>int(len(data_dict[dataset])/2):
        colors=[cmap(i) for i in np.linspace(0, 1, int(len(data_dict[dataset])/2))]
    else:
        colors=[cmap(i) for i in np.linspace(0, 1, int((ext)))]
        
    '''Name of axis'''
    x_labels= [r'$Bias\ [mV]$',r'$Field\ [G]$' , r'$Field\ [T]$', r'$Gate\ [V]$']    
    for j in np.arange(init,init+ext,interv):
        #print(j)
        #print(count)
        try:
            data=data_dict[dataset]['data'+str(j)+'sr'+str(lockin+1)][data_column[lockin]]*ac_convert[lockin]
            if temperature:
                temp=data_dict[dataset]['data'+str(j)+'sr'+str(lockin+1)][18][0]
            else:
                temp=count
            x_ax=x_dict[dataset]['data'+str(j)+'sr'+str(lockin+1)][scan_type]
        except KeyError:
            print('Too many scans, reduce end value')
            break
           
        cond=np.zeros_like(data)
        cond_eh=np.zeros_like(data)
        for i in range(len(data)):
            cond[i] = 1/(data[i])  #units in mS
            cond_eh[i] =cond[i]*1E-3/(2*3.877E-5)
        if conduc==False and eh==False:
            plt.plot(x_ax,data+offset*j,label='T={0:.3f}K' .format(temp),color=colors[count])
        elif conduc==True and eh==False:
            plt.plot(x_ax,cond+offset*j,label='T={0:.3f}K' .format(temp),color=colors[count])
        elif eh==True:
            plt.plot(x_ax,cond_eh+offset*j,label='T={0:.3f}K' .format(temp),color=colors[count])
        count=count+1*interv
    if offset== 0.0:
        plt.title(gen_name+name[lockin]+name_cond[dataset],fontsize=font)
    else:
        plt.title(gen_name+name[lockin]+name_cond[dataset]+' offseted by '+str(offset),fontsize=font)
        
    plt.tick_params(axis='both',which='major',labelsize=font)
    
    plt.xlabel(x_labels[scan_type],fontsize=font)
    plt.ylabel(r'$dV/dI\ [k\Omega]$',fontsize=font)
    if conduc==True and eh==False:
        plt.ylabel(r'$dI/dV\ [mS]$',fontsize=font)
    elif eh==True:
        plt.ylabel(r'$dI/dV\ [2e^2/h]$',fontsize=font)
    plt.grid()
    if legend:
        plt.legend(loc='right',fontsize=legfont)
    if save:
        plt.savefig(gen_name+name[lockin]+name_cond[dataset]+str(offset)+time.strftime("%H%M%S%d%m%y")+'.png', dpi=300)
    plt.show()
    return
    
    