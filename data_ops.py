import numpy as np
import cv2
import initialization as init

tc = init.init_params()
    
def find_nearest(self,a):
    
    A = np.asarray(self)
    idx = (np.abs(A-a)).argmin()
    return A[idx],idx 

#SENSOR-PLANE DESIGN based on SENSOR REGION WIDTH, CLASS/LABEL-SIZE and NUMBER of DETECTORS
def sensorplane_geometry():
    
    fl_num_det_row = int(np.floor(tc.NUM_CLASS**(0.5)))
    ce_num_det_row = int(np.ceil(tc.NUM_CLASS**(0.5)))

    choices = np.asarray([fl_num_det_row,ce_num_det_row])
    square_arrgmnt = choices**2
    remain_det = np.abs(tc.NUM_CLASS-square_arrgmnt)
    slct = np.argmin(remain_det)
    slct_ = np.mod(slct+1,2)
    rows_label = choices[slct]
    col_distribution = np.ones((rows_label,1))*rows_label
    col_distribution[rows_label//2-remain_det[slct]//2:rows_label//2-remain_det[slct]//2+remain_det[slct]] = choices[slct_]
    nDet = int(np.amin(col_distribution))
    NDet = int(np.amax(col_distribution))

    LABEL_ROW = int(np.round(tc.CLASS_SIZE_X/tc.DX)) #Vertical label size
    LABEL_COL = int(np.round(tc.CLASS_SIZE_Y/tc.DY)) #Horizontal label size

    marginX = (tc.SENSOR_WX-nDet*tc.CLASS_SIZE_X)/2/nDet
    marginY_1 = (tc.SENSOR_WY-nDet*tc.CLASS_SIZE_Y)/2/nDet
    marginY_2 = (tc.SENSOR_WY-NDet*tc.CLASS_SIZE_Y)/2/NDet

    cont_cell_locs = np.zeros((tc.NUM_CLASS,2))
    cell_locs = np.zeros((tc.NUM_CLASS,2))
    sensorx = (np.arange(tc.SENSOR_ROW)-(tc.SENSOR_ROW-1)/2)*tc.DX
    sensory = (np.arange(tc.SENSOR_COL)-(tc.SENSOR_COL-1)/2)*tc.DY
    ULC_sensorX = -(tc.SENSOR_ROW-1)/2*tc.DX
    ULC_sensorY = -(tc.SENSOR_COL-1)/2*tc.DY
    det_count = 0
    label_fields = np.zeros((tc.NUM_CLASS,tc.SENSOR_ROW,tc.SENSOR_COL))
    for rr in range(rows_label):
        mdet = int(col_distribution[rr])
        for md in range(mdet):
            if(mdet==nDet):
                cont_cell_locs[det_count,:] = np.asarray([ULC_sensorX+rr*tc.CLASS_SIZE_X+(2*rr+1)*marginX,ULC_sensorY+md*tc.CLASS_SIZE_Y+(2*md+1)*marginY_1])
                vx,qx = find_nearest(sensorx,cont_cell_locs[det_count,0])
                vy,qy = find_nearest(sensory,cont_cell_locs[det_count,1])
                cell_locs[det_count,:] = [qx,qy] 
                label_fields[det_count,qx:qx+LABEL_ROW,qy:qy+LABEL_COL] = 1
            else:
                cont_cell_locs[det_count,:] = np.asarray([ULC_sensorX+rr*tc.CLASS_SIZE_X+(2*rr+1)*marginX,ULC_sensorY+md*tc.CLASS_SIZE_Y+(2*md+1)*marginY_2])
                vx,qx = find_nearest(sensorx,cont_cell_locs[det_count,0])
                vy,qy = find_nearest(sensory,cont_cell_locs[det_count,1])
                cell_locs[det_count,:] = [qx,qy]
                label_fields[det_count,qx:qx+LABEL_ROW,qy:qy+LABEL_COL] = 1
            det_count = det_count+1    

    cell_locs = cell_locs.astype('int')        
    thecell=np.ones((LABEL_ROW,LABEL_COL))

    return cell_locs, thecell, label_fields, col_distribution

if tc.APPLICATION == 'classification':
        global cell_locs
        global thecell
        cell_locs, thecell, label_fields, col_distribution = sensorplane_geometry()

def object_rotation(self):
    u = self
    angle = np.random.randint(-90,90)
    R = cv2.getRotationMatrix2D((tc.N/2,tc.M/2),angle,1)
    u = cv2.warpAffine(u,R,(tc.N,tc.M))
    return u

def object_resampling_DATA2OBJ(self):
    
    u = self
    up_u = np.zeros((tc.OBJECT_ROW, tc.OBJECT_COL), dtype=np.uint8)
    up_u = cv2.resize(u, (np.int(tc.OBJECT_COL),np.int(tc.OBJECT_ROW)))
    return up_u

def object_resampling_DATA2SENSOR(self):
    
    u = self
    up_u = np.zeros((tc.SENSOR_ROW, tc.SENSOR_COL), dtype=np.uint8)
    up_u = cv2.resize(u, (np.int(tc.SENSOR_COL),np.int(tc.SENSOR_ROW)))
    return up_u
    
def gt_generator_classification(cls):
    
    gt_sensor = np.zeros((tc.SENSOR_ROW, tc.SENSOR_COL), dtype=np.float32)
    cellindex = cls
    ulcorner = cell_locs[cellindex,0:2]
    gt_sensor[ulcorner[0]:ulcorner[0]+thecell.shape[0],
            ulcorner[1]:ulcorner[1]+thecell.shape[1]] = thecell
                                    
    return gt_sensor


def int_to_one_hot(cls):
    oh = np.zeros((tc.NUM_CLASS))
    oh[cls] = 1
    return oh   

def create_validation(label_train, label_test, ratio):
    N_TEST = np.amax(label_test.shape)
    N_TRAIN = np.amax(label_train.shape)
    N_VAL = int(N_TEST*ratio)
    N_cls_val = int(np.round(N_VAL/tc.NUM_CLASS))
    ind_val = np.zeros((N_VAL))
    indexes = np.arange(N_TRAIN)
    for cls in range(tc.NUM_CLASS):
        ind_cls = indexes[label_train==cls]
        Ncls = int(np.sum([label_train==cls]))
        begin = cls*N_cls_val
        rand_ind = ind_cls[np.random.randint(0,Ncls,size=[2*N_cls_val,1])]
        R = np.random.permutation(np.unique(rand_ind))
        ind_val[begin:begin+N_cls_val] = R[0:N_cls_val]
        
    return ind_val.astype('int32')