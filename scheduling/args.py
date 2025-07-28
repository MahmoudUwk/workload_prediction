def get_paths():
    # data_path
    base_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/'
    # base_path = "/home/student/CPU_project/"
    processed_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/Proccessed_Alibaba"
    feat_stats_step1 = base_path+"feat_stats_step1"
    feat_stats_step2 = base_path+"feat_stats_step2_divide"
    feat_stats_step3 = base_path+"feat_stats_ready"
    sav_path = base_path+"pred_results_all"
    sav_path_plots = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/IEEE paper/figs/Alibaba"
    
    return base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots

# from args import get_paths
# base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()