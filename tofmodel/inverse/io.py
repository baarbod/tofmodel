# -*- coding: utf-8 -*-

import tofmodel.inverse.utils as utils
import os
import numpy as np
import json


# load area and corresponding slice positions for a subject
def load_area(subject, slc1, root_dir='/om/user/bashen/repositories/inflow-analysis'):
    filepath = os.path.join(root_dir, 'data', subject, 'area.txt')
    area = np.loadtxt(filepath) # mm^2
    xarea = 1 * np.arange(np.size(area)) # mm
    xarea_new = 0.1 * np.arange(10 * np.size(area)) # mm
    area = np.interp(xarea_new, xarea, area) # mm
    xarea = xarea_new - slc1 # mm
    xarea *= 0.1 # convert the depth to cm
    return xarea, area


def get_slice_from_subject(inputs, subject, slcpos):
    for sub in inputs:
        if sub['name'] == subject:
            if sub['postype'] == slcpos:
                slc = sub['slc']
    return slc


def load_subject_area_matrix(config_data_path, area_size):
    with open(config_data_path, "r") as jsonfile:
            param_data = json.load(jsonfile)
    inputs = param_data['human_data']
    subjects = []
    slice_mode = []
    for subject_info in inputs:
        subjects.append(subject_info['name'])
        slice_mode.append(subject_info['postype'])
    Ay = np.zeros((area_size, len(subjects) + 1)) 
    Ax = np.zeros((area_size, len(subjects) + 1)) 
    for idx, (subject, slc_mode)  in enumerate(zip(subjects, slice_mode)):
        slc = get_slice_from_subject(inputs, subject, slc_mode)
        xarea, area = load_area(subject, slc)
        xarea = utils.downsample(xarea, area_size)
        area = utils.downsample(area, area_size)
        Ax[:, idx] = xarea.flatten()
        Ay[:, idx] = area.flatten()
    # add straight tube case
    subjects.append('straight_tube')
    Ax[:, -1] = Ax[:, -2]
    Ay[:, -1] = np.ones(Ay.shape[0])
    
    return Ax, Ay, subjects
