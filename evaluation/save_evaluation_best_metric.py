# 분석할 폴더의 웨이트 파일 목록 확인하기
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 피클 파일들의 데이터를 데이터프레임으로 변환하는 함수
def load_pickles_and_convert_df(path, pickle_files):
    # 피클 파일들의 데이터를 데이터프레임으로 변환
    dataframes_dict = {}

    for file_name in pickle_files:
        file_path = os.path.join(path, file_name)  # 피클 파일 경로
        
        # 피클 파일 불러오기
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # 데이터가 리스트로 구성되어 있는지 확인 후 데이터프레임으로 변환
        df = pd.DataFrame(data)

        # expanded_df = pd.json_normalize(df)  # 각 행의 딕셔너리를 개별 컬럼으로 변환
        thresholds = np.linspace(0.001, 0.8, 500)
        df.insert(0, 'Thresholds', thresholds[:len(df)])
        
        dataframes_dict[file_name.split('.')[0]] = df  # 파일 이름을 키로 데이터프레임 리스트 저장
    return dataframes_dict

def main(path_list):
    for path in path_list:
        print("********** metric result analyze for {} **********".format(path))
        # 특정 폴더의 피클 파일들을 불러오기
        pickle_files = [f for f in os.listdir(path) if f.endswith('.pkl')]  # .pkl 확장자 파일들
        pickle_files.sort()
        for idx, p in enumerate(pickle_files):
            print('file {}:'.format(idx), p)

        metric_df = load_pickles_and_convert_df(path=path, pickle_files=pickle_files)
        metric_df_keys = list(metric_df.keys())

        ## 각 모델별 단순 F1-Score의 최고값을 분석 
        best_model = ''
        best_f1_score = 0
        # find and plot for f1max
        for key in metric_df_keys:
            if isinstance(metric_df[key], list):
                df = metric_df[key][0]
            else:
                df = metric_df[key]
            max_f1_score = df["F1-Score"].max()
            if max_f1_score > best_f1_score:
                best_f1_score = max_f1_score
                best_model = key
            corresponding_threshold = df[df["F1-Score"] == max_f1_score]["Thresholds"].values[0]
            corresponding_recall = df[df["F1-Score"] == max_f1_score]["Recall (TPR)"].values[0]
            print(key + f"'s Best F1-Score:\t {max_f1_score}, \tRecall: {corresponding_recall}, \tat thresholds: {corresponding_threshold}")

    # 미완

if __name__ == '__main__':
    path_list = ['./OverlapTransformer/2024-10-14_16-40-13']
    main(path_list)