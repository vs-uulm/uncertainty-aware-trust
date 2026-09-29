import os
import torch
import json
from NN_datasets.VeReMiNextGenPredictionDatasetRelPos import VeReMiNextGenPredictionDatasetRelPos
from test_train_helper import get_model
from NN_datasets.AttackType import AttackType

CONTEXT_LENGTH = 15
MODEL_PATH = "model/MLP_PRED_NO_RSSI_VeReMiNextGenNew_0_feature_output.pth"
MODE = 2 
SCENARIOS = ["urban", "highway"]
TRAFFIC_DENSITIES = [2,7]
ATTACK_TYPES = [0,1,2] # Adjust attack type

    
def generate_feature_output(dataset, datatype):
    model = get_model(context_len=CONTEXT_LENGTH)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    for x, y, label, attack_type, receiver, messageId, scenario, traffic_density, drivers_profile, sender_id in dataset:
        x = x.unsqueeze(0)        
        y = y.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            y_hat = model(x)
            per_sample_per_feature_abs = torch.abs(y_hat - y.float())

            errors_group = per_sample_per_feature_abs[:, :6].reshape(per_sample_per_feature_abs.size(0), 3, 2)  # (B, 3, 2)
            l2_errors = torch.norm(errors_group, dim=2)  # (B, 3)

            # Direct absolute errors for distance_to_road_edge and rcv_time
            distance_error = torch.abs(per_sample_per_feature_abs[:, 6]).unsqueeze(1)
            time_error = torch.abs(per_sample_per_feature_abs[:, 7]).unsqueeze(1)
            
            # Angular error for heading (features 8-9 are normalized sin/cos)
            sin_pred = (y_hat[:, 8] * 2) - 1
            cos_pred = (y_hat[:, 9] * 2) - 1
            sin_true = (y[:, 8] * 2) - 1
            cos_true = (y[:, 9] * 2) - 1
            
            # Geodesic distance on unit circle, normalized to [0, 1]
            dot_product = torch.clamp(sin_pred * sin_true + cos_pred * cos_true, -1.0, 1.0)
            angular_error = torch.acos(dot_product) / torch.pi  # Normalize: [0, π] → [0, 1]
            angular_error = angular_error.unsqueeze(1)
            
            grouped_errors = torch.cat([l2_errors, distance_error, time_error, angular_error], dim=1)


            save_feature_output(grouped_errors.mean(dim=0), f"grouped_feature_outputs/{scenario}_{traffic_density}_{AttackType(attack_type).name}/{datatype}/{receiver}.json", messageId, label)

    
def save_feature_output(feature_output, filename, messageId, label):
    output_data = {
        "messageId": messageId,
        "attacker": label,
        "relative_position_error": feature_output[0].item(),
        "sender_speed_error": feature_output[1].item(),
        "sender_acceleration_error": feature_output[2].item(),
        "distance_to_road_edge_error": feature_output[3].item(),
        "receiver_time_error": feature_output[4].item(),
        "sender_heading_error": feature_output[5].item(),
    }
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    with open(filename, 'a+', encoding='utf-8') as f:
        f.seek(0)
        existing = f.read()
        if existing and existing.strip():
            # file not empty -> append separator before the new object
            f.seek(0, os.SEEK_END)
            f.write(',\n')
        else:
            # file empty
            f.seek(0)
            f.write('[')
        f.write(json.dumps(output_data))

def add_closing_brackets_to_files(datatypes=None):
    # Add closing bracket to all files
    for scenario in SCENARIOS:
        for traffic_density in TRAFFIC_DENSITIES:
            for attack_type in ATTACK_TYPES:
                 for datatype in datatypes:
                    dir_path = f"grouped_feature_outputs/{scenario}_{traffic_density}_{AttackType(attack_type).name}/{datatype}"
                    if os.path.exists(dir_path):
                        for filename in os.listdir(dir_path):
                            if filename.endswith(".json"):
                                file_path = os.path.join(dir_path, filename)
                                with open(file_path, 'a+', encoding='utf-8') as f:
                                    f.seek(0, os.SEEK_END)
                                    f.write(']')


if __name__ == "__main__":
    train_dataset = VeReMiNextGenPredictionDatasetRelPos(context_len=CONTEXT_LENGTH, training=False, scenario=SCENARIOS, attack_types = ATTACK_TYPES, traffic_densitys=TRAFFIC_DENSITIES, data_type="Train", dataset_type="VeReMiNextGenNew", use_sliding_window=True, sliding_window_step=1, ground_truth_only=False)
    generate_feature_output(train_dataset, datatype="Train")
    add_closing_brackets_to_files(["Train"])

    val_dataset = VeReMiNextGenPredictionDatasetRelPos(context_len=CONTEXT_LENGTH, training=False, scenario=SCENARIOS, attack_types = ATTACK_TYPES, traffic_densitys=TRAFFIC_DENSITIES, data_type="Optimization", dataset_type="VeReMiNextGenNew", use_sliding_window=True, sliding_window_step=1, ground_truth_only=False)
    generate_feature_output(val_dataset, datatype="Validation")
    add_closing_brackets_to_files(["Validation"])

    test_dataset = VeReMiNextGenPredictionDatasetRelPos(context_len=CONTEXT_LENGTH, training=False, scenario=SCENARIOS, attack_types = ATTACK_TYPES, traffic_densitys=TRAFFIC_DENSITIES, data_type="Test", dataset_type="VeReMiNextGenNew", use_sliding_window=True, sliding_window_step=1, ground_truth_only=False)
    generate_feature_output(test_dataset, datatype="Test")
    add_closing_brackets_to_files(["Test"])