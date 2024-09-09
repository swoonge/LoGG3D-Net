from scipy.spatial.distance import cdist
import logging
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.mulran.mulran_dataset import load_poses_from_csv, load_timestamps_csv
from utils.data_loaders.kitti.kitti_dataset import load_poses_from_txt, load_timestamps
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
from statistics import mean

__all__ = ['evaluate_sequence_reg']

def save_pickle(data_variable, file_name):
    dbfile = open(file_name, 'wb')
    pickle.dump(data_variable, dbfile)
    dbfile.close()
    # logging.info(f'Finished saving: {file_name}')

def load_pickle(file_name):
    dbfile = open(file_name, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    # logging.info(f'Finished loading: {file_name}')
    return db

@torch.no_grad()
def evaluate_sequence_reg(model, cfg):
    save_info = cfg.eval_save_info
    save_dir_seq = str(cfg.checkpoint_name).split('/')[-1]
    
    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        cfg.kitti_data_split['test'] = [eval_seq]
        eval_seq = '%02d' % eval_seq        
        sequence_path = cfg.kitti_dir + 'sequences/' + eval_seq + '/'        
        pose_database_full, positions_database = load_poses_from_txt(
            sequence_path + 'poses.txt')
        timestamps = load_timestamps(sequence_path + 'times.txt')        
        
        save_dir = os.path.join(os.path.dirname(__file__), f'result/Kitti/{eval_seq}/{save_dir_seq}_s{str(cfg.select_super)}')
        
    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        cfg.mulran_data_split['test'] = [eval_seq]
        sequence_path = cfg.mulran_dir + eval_seq
        pose_database_full, positions_database = load_poses_from_csv(
            sequence_path + '/scan_poses.csv')
        timestamps = load_timestamps_csv(sequence_path + '/scan_poses.csv')
        
        save_dir = os.path.join(os.path.dirname(__file__), f'result/MulRan/{eval_seq}/{save_dir_seq}_s{str(cfg.select_super)}')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    logging.info(f"save_dir : {save_dir}")

    logging.info(f'Evaluating sequence {eval_seq} at {sequence_path}')
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))

    num_queries = len(positions_database)
    num_thresholds = len(thresholds)

    # Databases of previously visited/'seen' places.
    seen_poses = []
    seen_descriptors = []    
    argMaxIdx_seen_desc = []    
    corruptList = []    

    # Store results of evaluation.
    num_true_positive = torch.zeros(num_thresholds)
    num_false_positive = torch.zeros(num_thresholds)
    num_true_negative = torch.zeros(num_thresholds)
    num_false_negative = torch.zeros(num_thresholds)

    prep_timer = Timer()
    desc_timer = Timer()
    ret_timer = Timer()
    ret_timer2 = Timer()
    ret_timer3 = Timer()

    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0
    start_time = timestamps[0]

    select_super = cfg.select_super
    yaw_error_list = []    
    corruptCnt = 0
                
    if(cfg.compare == False):
        test_loader = make_data_loader(cfg,
                                    cfg.test_phase,
                                    cfg.eval_batch_size,
                                    num_workers=cfg.test_num_workers,
                                    shuffle=False)
        iterator = test_loader.__iter__()
        logging.info(f'len_dataloader {len(test_loader.dataset)}')                
    else:
        contextDB = np.load(save_dir + '/contextDB.npy')
        logging.info(f"ContextDB shape {contextDB.shape}")
        
        argMaxDB = np.load(save_dir + '/argMaxDB.npy')
        logging.info(f"ArgMaxDB shape {argMaxDB.shape}")
        
        corruptList = load_pickle(save_dir + '/corruptList.pickle')
        logging.info(f"corruptList length {len(corruptList)}")
        
        # eval csv 저장
        eval_csv_file = save_dir + '/eval_s' + str(cfg.select_super) + '.csv'
        eval_csv = open(eval_csv_file, 'w', newline='')
        eval_csv_writer = csv.writer(eval_csv)    
        logging.info(f"Writing eval_csv_file {eval_csv_file}")
        
    
    for query_idx in tqdm(range(num_queries)): # 지나온 길 중에 내가 왔던 적이 있는지 비교하는 코드
    # for query_idx in range(num_queries): # 지나온 길 중에 내가 왔던 적이 있는지 비교하는 코드
    
        # target_idx = 2415
        # if(query_idx > target_idx):
        #     break
        
        # if(query_idx==10):
        #     break

        ############### Generate Descc ###############
        if(cfg.compare == False):
            input_data = next(iterator)
            lidar_pc = input_data[0][0]  # .cpu().detach().numpy()
            
            if not len(lidar_pc) > 0:
                logging.info(f'Corrupt cloud id: {query_idx}')
                corruptList.append(query_idx)
                continue

            # input data를 sparse tensor로 변환
            prep_timer.tic()            
            input_st, validRow = make_sparse_tensor(lidar_pc, cfg.voxel_size, num_sector=60)            
            input_st = input_st.cuda()
            prep_timer.toc()            
            
            desc_timer.tic()
            context_desc = model(input_st, validRow)  # .squeeze() # 여기서 model에 scan input만 넣어주면 global descriptor 만듦 
            desc_timer.toc()
            
            context_desc = context_desc.cpu().detach().numpy()        
            if len(context_desc) < 1:
                corruptList.append(query_idx)
                continue
            
            seen_descriptors.append(context_desc) 
            argMaxIdx_seen_desc.append(np.argmax(context_desc, axis=0))        
                        
            # plt.imsave(save_dir + "/context_" + str(query_idx) + '.png', context_desc)            
            continue        
        
        
        
        ############### Evaluation ###############
        if(query_idx in corruptList):
            corruptCnt += 1
        
        context_desc = contextDB[query_idx - corruptCnt]     
        argMax_desc = argMaxDB[query_idx - corruptCnt]
        query_pose = positions_database[query_idx]
        query_time = timestamps[query_idx]

        seen_descriptors.append(context_desc)            
        argMaxIdx_seen_desc.append(argMax_desc)        
        seen_poses.append(query_pose)        
        
        # if(query_idx < target_idx):
        #     continue
        
        # if(query_idx < 1400):
        #     continue
        
        if (query_time - start_time - cfg.skip_time) < 0: # Build retrieval database using entries 30s prior to current query. 
            continue        

        tt = next(x[0] for x in enumerate(timestamps)
            if x[1] > (query_time - cfg.skip_time))
        
        # Find top-1 candidate.
        nearest_idx = 0
        min_dist = math.inf

        # query 
        q_context_desc = seen_descriptors[-1] # 60,256
        q_global_desc_maxIdx = argMaxIdx_seen_desc[-1]  

        # ref DB
        db_context_desc = np.copy(seen_descriptors)[:tt+1]                 
        db_global_desc_maxIdx = np.copy(argMaxIdx_seen_desc)[:tt+1]
        db_seen_poses = np.copy(seen_poses)[:tt+1]
        
        ret_timer.tic()      
        
        # maxPooled Idx로 얼마나 회전했는지 파악  
        ret_timer2.tic()      
        maxIdxSubt = np.subtract(q_global_desc_maxIdx, db_global_desc_maxIdx)
        maxIdxSubt[maxIdxSubt < 0] += 60     
        
        binCount = np.apply_along_axis(np.bincount, axis=1, arr=maxIdxSubt, minlength=60)
        rotation_row = 60 - np.argmax(binCount, axis=1)    
        ret_timer2.toc()                     
        
        # 회전량 계산
        ret_timer3.tic()    
        context_feat_dists = diffSC_mostSimilar_db_np_cdist_rotList(q_context_desc, db_context_desc, rotation_row)                
        ret_timer3.toc()        
        
        min_dist = round(np.min(context_feat_dists),3) # min_dist : 유사도 차이
        nearest_idx = np.argmin(context_feat_dists)
        min_shift = rotation_row[np.argmin(context_feat_dists)]             
        ret_timer.toc()    
        
        # print(ret_timer.avg)
        
        # print(ret_timer.avg, ret_timer2.avg, ret_timer3.avg)          
          
        # query와 후보간 거리 계산
        p_dist = np.linalg.norm(query_pose - db_seen_poses[nearest_idx]) # p_dist : 제일 유사하다고 판단한 global desc의 pose와 실제 pose 사이 거리
        p_dist = round(p_dist, 3)
        
        # 모델이 추정한 yaw값 (ref 입장에서 shift)
        yaw_infer = min_shift * (360 / 60) # min_shift : 60 = yaw_infer : 360    
        if(yaw_infer > 180):
            yaw_infer = yaw_infer - 360            
        yaw_infer = round(yaw_infer, 3)
                

        #gt yaw와 차이 계산
        yaw_error = 0
        gt_yaw = 0
        
        query_rot = pose_database_full[query_idx][:3,:3]                
        ref_rot = pose_database_full[nearest_idx][:3,:3]
        # 각 Yaw 각도의 차이를 계산
        mat_diff = np.matmul(ref_rot.T, query_rot)
        gt_yaw = np.degrees(np.arctan2(mat_diff[1, 0], mat_diff[0, 0]))
                
        # Yaw 각도를 [-π, π) 범위로 조정
        if(gt_yaw > 180):
            gt_yaw = gt_yaw - 360                    
        gt_yaw = round(gt_yaw, 3)               
        
        ############## yaw error 
        yaw_error = abs(gt_yaw - yaw_infer)
        if(yaw_error >= 180):
            yaw_error = yaw_error - 360
        
        yaw_error = round(yaw_error, 3)
                

        # ####### # 매칭된 pair 저장
        # # query, ref, ref(row shift한거)
        # ref_sc = seen_descriptors[nearest_idx]
        # result_stack = np.vstack((global_descriptor_sc, ref_sc))
        # result_stack = np.vstack((result_stack, np.roll(ref_sc, min_shift, axis=0)))

        # plt.imsave(sc_pair_save_dir + "/" + str(query_idx) + "_" + str(nearest_idx) + "_yaw" + str(yaw_infer)
        #             + "_super" + str(select_super) + '.png'
        #             , result_stack)
        
        # plt.imsave(sc_pair_save_dir + "/" + str(query_idx) + '/.png', ref_sc)

        
        ######################## revisit list 안쓰도록 하고 코드 추가함 ##########################
        
        is_revisit = 0
        gt_dist = np.linalg.norm(query_pose.reshape(1,3) - db_seen_poses.reshape(-1,3), axis=1)
        nearList = gt_dist[gt_dist <= 3]
        if(len(nearList) > 0):
            is_revisit = 1
        # print('revisit', is_revisit, np.min(nearList))     
            
        ########################
        
        
        is_correct_loc = 0          
        if is_revisit:        
            num_revisits += 1
            if p_dist <= cfg.revisit_criteria: # 오차 3미터 내이면 맞은걸로 처리
                num_correct_loc += 1
                is_correct_loc = 1
                
                yaw_error_list.append(yaw_error)            
                
                
            # #### 상위 후보들 중에 3m 이내에 들어오는 애들 몇 개 있는지?  
            # print('[GT distance]', gt_dist_filtered)                           
            # # nv_dist = np.linalg.norm(query_pose - db_seen_poses[super_selected], axis=1)
            # # print('[Netvlad Desc distance]', nv_dist)            
            # context_dist = np.linalg.norm(query_pose - db_seen_poses[np.argsort(context_feat_dists)], axis=1)
            # print('[context_dist distance]', context_dist)          
            # print()
            #######################################################                

        eval_csv_writer.writerow([query_idx, nearest_idx, is_revisit, is_correct_loc, min_dist, gt_yaw, yaw_infer, yaw_error, p_dist])
        # logging.info(f'id: {query_idx} n_id: {nearest_idx} is_rev: {is_revisit} is_correct_loc: {is_correct_loc} min_dist: {min_dist} gt_yaw :{gt_yaw} yaw_infer:{yaw_infer} yaw_error:{yaw_error} p_dist: {p_dist}')
        

        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]

            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= cfg.revisit_criteria:
                    num_true_positive[thres_idx] += 1 

                elif p_dist > cfg.not_revisit_criteria:                
                    num_false_positive[thres_idx] += 1

            else:  # Negative Prediction
                if(is_revisit == 0):
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1

      

    if(cfg.compare == False):            
        contextDB_np = np.array(seen_descriptors)
        np.save(save_dir + '/contextDB', contextDB_np)      
        logging.info(f'Saving contextDB {save_dir}, ContextDB shape {contextDB_np.shape}')  
        
        argMaxDB_np = np.array(argMaxIdx_seen_desc)
        np.save(save_dir + '/argMaxDB', argMaxDB_np)      
        logging.info(f'Saving argMaxDB {save_dir}, argMaxDB shape {argMaxDB_np.shape}')       
                
        save_pickle(corruptList, save_dir + '/corruptList.pickle')    
        logging.info(f'Saving corruptList {save_dir}, len(corruptList) : {len(corruptList)}')               
        
        with open(save_dir + "/prepTime_descTime.txt", 'w') as timeFile:
            timeFile.write(f"--- Prep: {prep_timer.avg}s Desc: {desc_timer.avg}s ---")
        logging.info(f"--- Prep: {prep_timer.avg}s Desc: {desc_timer.avg}s ---")                       
        return    
    else:
        eval_csv.close()    

    F1max = 0.0
    Precisions, Recalls = [], []
    
    for ithThres in range(num_thresholds):
        nTrueNegative = num_true_negative[ithThres]
        nFalsePositive = num_false_positive[ithThres]
        nTruePositive = num_true_positive[ithThres]
        nFalseNegative = num_false_negative[ithThres]

        Precision = 0.0
        Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)

            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1
            F1_TN = nTrueNegative
            F1_FP = nFalsePositive
            F1_TP = nTruePositive
            F1_FN = nFalseNegative
            F1_thresh_id = ithThres
        Precisions.append(Precision)
        Recalls.append(Recall)
        
    logging.info(f'num_revisits: {num_revisits}')
    logging.info(f'num_correct_loc: {num_correct_loc}')
    logging.info(f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}')
    logging.info(f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
    logging.info(f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
    logging.info(f'F1_thresh_id: {F1_thresh_id}')
    logging.info(f'F1max: {F1max}')
    logging.info(f"--- Ret: {ret_timer.avg}s ---")
    logging.info(f'Avg yaw error {mean(yaw_error_list)}')    
    
    
    if save_info:        
        
        # save pr curve
        plt.title('Seq: ' + str(eval_seq) + '    F1Max: ' + "%.4f" % (F1max))
        plt.plot(Recalls, Precisions, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis([0, 1, 0, 1.1])
        plt.xticks(np.arange(0, 1.01, step=0.1))
        plt.grid(True)
        
        pr_curve_file = save_dir + '/pr_curve_s' + str(cfg.select_super) + '.png'
        plt.savefig(pr_curve_file)
        logging.info(f'Saving {pr_curve_file}')      
                
        # save tp fp tn fn
        nTP_file = save_dir + f'/num_true_positive_{cfg.select_super}.pickle'
        nFP_file = save_dir + f'/num_false_positive_{cfg.select_super}.pickle'
        nTN_file = save_dir + f'/num_true_negative_{cfg.select_super}.pickle'
        nFN_file = save_dir + f'/num_false_negative_{cfg.select_super}.pickle'        
        save_pickle(num_true_positive, nTP_file)
        save_pickle(num_false_positive, nFP_file)
        save_pickle(num_true_negative, nTN_file)
        save_pickle(num_false_negative, nFN_file)
        
        # save info & ret time
        with open(save_dir + "/evalInfo.txt", 'w') as evalInfo_file:     
            evalInfo_file.write(f'num_revisits: {num_revisits}\n')
            evalInfo_file.write(f'num_correct_loc: {num_correct_loc}\n')
            evalInfo_file.write(f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}\n')
            evalInfo_file.write(f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}\n')
            evalInfo_file.write(f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}\n')
            evalInfo_file.write(f'F1_thresh_id: {F1_thresh_id}\n')
            evalInfo_file.write(f'F1max: {F1max}\n')
            evalInfo_file.write(f"--- Ret: {ret_timer.avg}s ---\n")
            evalInfo_file.write(f"Avg yaw error : {mean(yaw_error_list)} degrees\n")            
            
    
    return F1max
