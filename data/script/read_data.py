import os
import glob
import pandas as pd 
import sys

def read_csv_from_folder(path:str) -> pd.DataFrame:
    csv_list = glob.glob(os.path.join(path,'*.csv'))
    data =  pd.DataFrame(columns= ['experiment_number','normal','t','h_1','h_2','h_3'])
    exp_number = 0
    for csv in csv_list:
        read_data = pd.read_csv(csv)
        path, file = os.path.split(csv)
        split_name = file.split('_')
        if split_name[1] == '1':
            read_data['normal'] = [True for i in range(0,read_data.shape[0])]
        else:
            read_data ['normal'] = [False for i in range(0,read_data.shape[0])]

        read_data['experiment_number'] = [exp_number for i in range(0,read_data.shape[0])]
        data = data.append(read_data, ignore_index=True)
        exp_number += 1
    data = data.rename(columns = {'t':'time','h_1':'h1','h_2':'h2','h_3':'h3'})
    return data 

def read_csv_scen(path:str,scen_num:int) -> pd.DataFrame:
    csv_list = glob.glob(os.path.join(path,'*_scen{0}_*.csv'.format(scen_num)))
    data =  pd.DataFrame(columns= ['experiment_number','normal','t','h_1','h_2','h_3'])
    exp_number = 0
    for csv in csv_list:
        read_data = pd.read_csv(csv)
        path, file = os.path.split(csv)
        split_name = file.split('_')
        if scen_num == 0:
            read_data['normal'] = [True for i in range(0,read_data.shape[0])]
        else:
            read_data ['normal'] = [False for i in range(0,read_data.shape[0])]

        read_data['experiment_number'] = [exp_number for i in range(0,read_data.shape[0])]
        data = data.append(read_data, ignore_index=True)
        exp_number += 1
    data = data.rename(columns = {'t':'time','h_1':'h1','h_2':'h2','h_3':'h3'})
    return data 

def save_result_csv(data:pd.DataFrame, dir_path:str, file_name:str) -> None:
    data.to_csv(os.path.join(dir_path,file_name), index = False)
    
def remove_mediana_sample(data:pd.DataFrame) -> (pd.DataFrame, dict):
    time = data[data['experiment_number'] == data.iloc[0]['experiment_number']]['time'].array
    data_normal = data[data['normal'] == True][['time','h1','h2','h3']]
    med_dic = {}
    for i in time:
        med_dic[i] = data_normal[data_normal['time'] == i][['h1','h2','h3']].median()
    dict_ = {
            'experiment_number':[],
            'normal':[],
            'time':[],
            'h1':[],
            'h2':[],
            'h3':[]}
    for i in range(0, data.shape[0]):
        med_h1 = data.iloc[i]['h1'] - med_dic[data.iloc[i]['time']]['h1']
        med_h2 = data.iloc[i]['h2'] - med_dic[data.iloc[i]['time']]['h2']
        med_h3 = data.iloc[i]['h3'] - med_dic[data.iloc[i]['time']]['h3']
        dict_['experiment_number'].append(data.iloc[i]['experiment_number'])
        dict_['normal'].append(data.iloc[i]['normal'])
        dict_['time'].append(data.iloc[i]['time'])
        dict_['h1'].append(med_h1)
        dict_['h2'].append(med_h2)
        dict_['h3'].append(med_h3)
    return pd.DataFrame(dict_), med_dic

def remove_mediana(data:pd.DataFrame, med_dic:dict) -> pd.DataFrame:
    dict_ = {
            'experiment_number':[],
            'normal':[],
            'time':[],
            'h1':[],
            'h2':[],
            'h3':[]}
    for i in range(0, data.shape[0]):
        med_h1 = data.iloc[i]['h1'] - med_dic[data.iloc[i]['time']]['h1']
        med_h2 = data.iloc[i]['h2'] - med_dic[data.iloc[i]['time']]['h2']
        med_h3 = data.iloc[i]['h3'] - med_dic[data.iloc[i]['time']]['h3']
        dict_['experiment_number'].append(data.iloc[i]['experiment_number'])
        dict_['normal'].append(data.iloc[i]['normal'])
        dict_['time'].append(data.iloc[i]['time'])
        dict_['h1'].append(med_h1)
        dict_['h2'].append(med_h2)
        dict_['h3'].append(med_h3)
    return pd.DataFrame(dict_)

if __name__ == "__main__":
    if len(sys.argv)>1:
        data = read_csv_from_path(sys.argv[1])
        read_csv_from_folder(data, sys.argv[2], sys.argv[3])
    else:
        data_600 = read_csv_from_folder('../raw/data_600/')
        save_result_csv(data_600, '../result/', 'result_600.csv')
        
        data_1000 = read_csv_from_folder('../raw/data_1000/')
        save_result_csv(data_1000, '../result/', 'result_1000.csv')
        
        data_scen0 = read_csv_scen('../raw/data_1000/',0)
        save_result_csv(data_scen0, '../result/', 'result_scen0.csv')
        
        data_scen1 = read_csv_scen('../raw/data_1000/',1)
        save_result_csv(data_scen1, '../result/', 'result_scen1.csv')
        
        data_scen2 = read_csv_scen('../raw/data_1000/',2)
        save_result_csv(data_scen2, '../result/', 'result_scen2.csv')
        
        data_med_600, _ = remove_mediana_sample(data_600)
        save_result_csv(data_med_600, '../result/', 'result_med_600.csv')
        
        data_med_1000, _  = remove_mediana_sample(data_1000)
        save_result_csv(data_med_1000, '../result/', 'result_med_1000.csv')
        
        data_med_scen0, med_dic = remove_mediana_sample(data_scen0)
        save_result_csv(data_med_scen0, '../result/', 'result_med_scen0.csv')
        
        data_med_scen1 = remove_mediana(data_scen1, med_dic)
        save_result_csv(data_med_scen1, '../result/', 'result_med_scen1.csv')
        
        data_med_scen2 = remove_mediana(data_scen2, med_dic)
        save_result_csv(data_med_scen2, '../result/', 'result_med_scen2.csv')
        