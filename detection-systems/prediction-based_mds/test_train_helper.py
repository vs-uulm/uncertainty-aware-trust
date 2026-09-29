import torch
from torch.utils.data import DataLoader
from NN_classes import MLP_PRED_REL_POS
from NN_datasets.VeReMiNextGenPredictionDatasetRelPos import VeReMiNextGenPredictionDatasetRelPos

FEATURE_LEN = 16

TRAINED_MODEL_DIR = './trained_models/'

def load_model_with_state_dict(
        context_len, 
        path = None,
        hidden_sizes=None,
    ):
    model = get_model(
        context_len,
        hidden_sizes=hidden_sizes,
    )
    if path is None:
        return model
    state = torch.load(path, map_location=torch.device('cpu'))
    # if saved dict contains 'state_dict', assume Lightning-style
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    return model


def get_model(
        context_len, 
        hidden_sizes=None,
        lr=1e-4, 
        wd=1e-5
    ):
    # feature_preparation_next_gen.py currently generates 16 input features (no RSSI)
    # and 10 output features for PRED models
    feature_len_in = FEATURE_LEN  # = 16 
    
    return MLP_PRED_REL_POS.MLP_PRED_REL_POS(context_len=context_len, feature_len=feature_len_in, hidden_sizes=hidden_sizes, lr=lr, wd=wd)



def get_train_dataloader(
        dataset_type,
        context_len, 
        use_rssi=True,
        batch_size = 64,
        scenario=["highway", "urban"],
        traffic_density=[2,7],
        attack_types=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        use_sliding_window=False,
        sliding_window_step=1,
        ground_truth_only=True
    ):

    dataset = VeReMiNextGenPredictionDatasetRelPos(context_len=context_len, training=True, scenario=scenario, attack_types = attack_types, traffic_densitys=traffic_density, data_type="Train", dataset_type=dataset_type, use_sliding_window=use_sliding_window, sliding_window_step=sliding_window_step, ground_truth_only=ground_truth_only)
        

    # Optimized for 48-core CPU server (conservative settings)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=2
    )
    return train_loader

def  get_val_dataloader(
        dataset_type,
        context_len, 
        use_rssi=True,
        batch_size = 64,
        scenario=["highway", "urban"],
        traffic_density=[2,7],
        attack_types=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        use_sliding_window=False,
        sliding_window_step=1,
        ground_truth_only=True
    ):


    dataset = VeReMiNextGenPredictionDatasetRelPos(context_len=context_len, training=True, scenario=scenario, attack_types = attack_types, traffic_densitys=traffic_density, data_type="Validation", dataset_type=dataset_type, use_sliding_window=use_sliding_window, sliding_window_step=sliding_window_step, ground_truth_only=ground_truth_only)
    
    # Optimized for 48-core CPU server (conservative settings)
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=2
    )
    return val_loader

def get_test_dataloader(
        dataset_type,
        context_len, 
        use_rssi=True,
        batch_size = 64,
        scenario=["highway", "urban"],
        traffic_density=[2,7],
        attack_types=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        use_sliding_window=False,
        sliding_window_step=1,
        ground_truth_only=False
    ):
    dataset = VeReMiNextGenPredictionDatasetRelPos(context_len=context_len, training=False, scenario=scenario, attack_types = attack_types, traffic_densitys=traffic_density, data_type="Test", dataset_type=dataset_type, use_sliding_window=use_sliding_window, sliding_window_step=sliding_window_step, ground_truth_only=ground_truth_only)
       
    # Optimized for 48-core CPU server
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=2
    )
    return test_loader


def get_opt_dataloader(
    dataset_type,
    context_len, 
    use_rssi=True,
    batch_size = 64,
    scenario=["highway", "urban"],
    traffic_density=[2,7],
    attack_types=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    use_sliding_window=False,
    sliding_window_step=1,
    ground_truth_only=False
):
    dataset = VeReMiNextGenPredictionDatasetRelPos(context_len=context_len, training=False, scenario=scenario, attack_types = attack_types, traffic_densitys=traffic_density, data_type="Optimization", dataset_type=dataset_type, use_sliding_window=use_sliding_window, sliding_window_step=sliding_window_step, ground_truth_only=ground_truth_only)

    # Optimized for 48-core CPU server
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=2
    )
    return test_loader
