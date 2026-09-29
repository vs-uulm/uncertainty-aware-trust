import os
from torch.utils.data import Dataset
import torch

from NN_datasets import feature_preparation_next_gen
from .VeReMiNextGen import VeReMiNextGen


class VeReMiNextGenPredictionDatasetRelPos(Dataset, VeReMiNextGen):
    def __init__(self, 
                 context_len, 
                 training = False, 
                 scenario = ["highway", "urban"],
                 attack_types = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], 
                 traffic_densitys = [2,7],
                 data_type="Train", 
                 dataset_type="VeReMiNextGen",
                 use_sliding_window=False,
                 sliding_window_step=1,
                 ground_truth_only=False
                 ):
        self.context_len = context_len
        self.training = training
        self.scenario = scenario
        self.attack_types = attack_types
        self.traffic_densitys = traffic_densitys
        self.NN_input = []
        self.data_type = data_type
        self.dataset_type = dataset_type
        self.use_sliding_window = use_sliding_window
        self.sliding_window_step = sliding_window_step
        self.ground_truth_only = ground_truth_only

        print("create VeReMiNextGenPredictionDatasetRelPos")

        # include out_only in filename to avoid loading incompatible cached files
        self.save_path = f"NN_input_data/PRED_rel_pos_cl{context_len}_td{traffic_densitys}_at{attack_types}_dt{data_type}_ds{dataset_type}.pt"

        if os.path.exists(self.save_path):
            print("load: " + self.save_path)
            try:
                self.NN_input = torch.load(self.save_path)
            except:
                print("load failed")
                VeReMiNextGen.__init__(self, scenario = scenario, traffic_density = traffic_densitys, attack_types = attack_types, data_type=data_type, dataset_type=dataset_type, use_sliding_window=use_sliding_window, ground_truth_only=ground_truth_only)
        else:
            print("create & save: " + self.save_path)
            VeReMiNextGen.__init__(self, scenario = scenario, traffic_density = traffic_densitys, attack_types = attack_types, data_type=data_type, dataset_type=dataset_type, use_sliding_window=use_sliding_window, ground_truth_only=ground_truth_only)
                
            # try:
            #    torch.save(self.NN_input, self.save_path)
            # except:
            #    print("save failed")

        
    def __len__(self):
        return len(self.NN_input)

    def __getitem__(self, idx):
        x = self.NN_input[idx]
        return x
    

    def _prepare_sim(self, sim):
        import time
        start = time.time()
        

        sim_NN_input = feature_preparation_next_gen.generate_nn_input_data(
            data=sim["vehicle_data"],
            context_len=self.context_len,
            remove_attacker=self.training,
            attack_type=sim["at"],
            data_type=self.data_type,
            use_sliding_window=self.use_sliding_window,
            sliding_window_step=self.sliding_window_step,
            scenario=sim["scenario"],
            traffic_density=sim["td"]
        )
        self.NN_input.extend(sim_NN_input)
        elapsed = time.time() - start
        print(f"Completed: td {sim['td']}, at {sim['at']} - Feature prep took {elapsed:.1f}s")
        self.data.remove(sim)
