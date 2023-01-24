# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:50:17 2023

@author: lewislab
"""

import json

def input_from_json(json_name):
    
    # Opening JSON file
    f = open(json_name)
      
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
      
    scan_param = {}
    scan_param['slice_width'] = data['SliceThickness']
    scan_param['repetition_time'] = data['RepetitionTime']
    scan_param['flip_angle'] = data['FlipAngle']
    scan_param['t1_time'] = 4 # default for water
    scan_param['num_pulse'] = 100 # default
    scan_param['alpha_list'] = data['SliceTiming']
    scan_param['num_slice'] = 10 # default
      
    # Closing file
    f.close()
    
    return scan_param
    

# def set_pos_type(func_type):

#         match func_type:
#             case 'linear'
#                 print(func_type)
        
    
#     # options:
#     # linear, sine, box, trap, fourier
    
    
# set_pos_type('linear')
    
    
    

    