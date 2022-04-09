"""
Main file: defines steps to evaluate categorical encoders / classifier pairs on a given task.
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold

# custom library
from src import columnar as col





def main(task: str, configpath: str):
    """runs a 5-task classification benchmark on all transformer / classifier
    pairs (incl. 5 transformers and 4 classifiers). The benchmark is run
    using a 5-fold cross validation strategy.
    
    Args:
        task (str): name of the task to run the benchmark on.
            currently supports 'petfinder', 'hr_analytics', 'adults',
            'mushrooms' and 'titanic'
        config (str): path to the config file defining the benchmark configuration"""
    
    cfg = col.config.BenchmarkConfig.load(configpath)
    
    # setup matplotlib style for figures
    plt.style.use(cfg.plot.style)
    
    print(f"Comparing (model, encoder) pairs for the {task} classification task")
    
    # load data
    print(f"\tloading dataset")
    loader = col.DataLoader(root=cfg.paths.root, task=task)
    data = loader.load_data()
    
    # define scoring metrics of interest
    scorer = col.Scorer(**col.config.get_metrics_from_config(cfg))

    # define cross validation strategy
    kf = KFold(n_splits=cfg.cross_validation.n_splits)
    
    # define features to select
    feature_selection = col.FeatureSelection(**loader.get_selected_features(data))
    
    # get transformers and classifiers from config file
    transformers = col.config.get_transformers_from_config(cfg)
    classifiers = col.config.get_classifiers_from_config(cfg)
    
    
    print(f"""\t  {cfg.cross_validation.n_splits} cross-validation folds
    \tx {len(transformers)} transformers
    \tx {len(classifiers)} models 
    \tcombinations to train and test""")
    
    
    # initialize benchmark runner
    runner = col.BenchmarkRunner(features=feature_selection,
                                 cat_transformers=transformers,
                                 classifiers=classifiers,
                                 scorer = scorer,
                                )
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        
        print(f"\tRunning benchmark on fold number {i+1} / {cfg.cross_validation.n_splits}", end='\r')
        start = time.time()
        
        # split train and test data using the CV fold
        cv_train, cv_test = data.iloc[train_idx], data.iloc[test_idx]
        
        X_train, y_train = feature_selection.select_features(cv_train)
        X_test, y_test = feature_selection.select_features(cv_test)
        
        runner.run(X_train, y_train, X_test, y_test)
        print(f"\tRunning benchmark on fold number {i+1} / {cfg.cross_validation.n_splits} - time = {col.utils.convert_time(time.time() - start)}")
            
                
    print("Evaluation completed")
    
    reporter = runner.create_reporter()

    
    # saving report as csv
    save_directory = os.path.join(cfg.paths.root, cfg.paths.reports)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
        print(f"{save_directory} path created")
        
    save_path = os.path.join(save_directory, f'{task}.csv')
    reporter.save(save_path)
    print(f"results summary saved in {save_path}")
    
    # saving summary plot model
    figpath = os.path.join(cfg.paths.root, cfg.paths.figures, f'{task}.png')
    fig = col.plot_model_encoder_pairs(reporter, figpath=figpath, title=f"{task} dataset".capitalize())
    print(f"visualization saved in {figpath}")
    
    return reporter
            

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", 
                        help="the dataset and classification task to evaluate categorical encoders and models against",
                        type=str)
    parser.add_argument("-c", "--configpath", 
                        default='./config.yml', 
                        nargs='?', 
                        type=str, 
                        help="the path to the YML file defining the benchmark configuration")

    args = parser.parse_args()
    reporter = main(args.task, args.configpath)