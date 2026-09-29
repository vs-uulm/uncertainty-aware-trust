import torch 
import torch.nn as nn

import numpy as np
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, BinaryConfusionMatrix

import pytorch_lightning as pl



class MLP_PRED_REL_POS(pl.LightningModule):
    def __init__(self, context_len, feature_len=16, log_path=None, hidden_sizes=None, lr=1e-3, wd=0.0):
        super().__init__()
        output_feature_len = 10
        # If n_layers and hidden_sizes are provided, build configurable MLP, otherwise fallback to original small net
        if hidden_sizes is not None:
            in_features = feature_len * (context_len - 1)
            layers = []
            layers.append(nn.Linear(in_features, in_features))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(p=0.2))
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(nn.ReLU())
                #layers.append(nn.Dropout(p=0.2))
                in_features = hidden_size
            layers.append(nn.Linear(in_features, output_feature_len))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Sequential(
                nn.Linear(feature_len*(context_len-1), feature_len*(context_len-1)),
                nn.ReLU(),
                nn.Linear(feature_len*(context_len-1), 3*(context_len-1)),
                nn.ReLU(),
                nn.Linear(3*(context_len-1), output_feature_len)
            )
        self.context_len = context_len
        self.save_hyperparameters()

        self.test_conf_matrix = BinaryConfusionMatrix()
        
        # Fallback
        self.register_buffer("val_quantil_threshold", torch.tensor(1.0))
        self.register_buffer("grouped_thresholds", torch.ones(6))

        self.test_accuracy = Accuracy(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.val_reconstruction_errors_grouped = []
        self.val_reconstruction_errors_single = []
        self.test_threshold_exceedances = []
        self.test_attack_type_predictions = {}  # {attack_type: [(pred, label), ...]}
        # optional file path (not used automatically). Use explicit method to write attack metrics to file.
        self.log_path = log_path
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer

    def forward(self, input):
        x = input
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y, label, attack, receiver, messageId, scenario, traffic_density, drivers_profile, sender_id = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)

        y = y.float()
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, label, attack, receiver, messageId, scenario, traffic_density, drivers_profile, sender_id = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)


        y = y.float()
        loss = nn.functional.mse_loss(y_hat, y)

        # Compute per-group errors:
        # - L2 (Euclidean) norm for the 3 groups with 2 components
        # - absolute error for single-dimension features
        errors = (y_hat - y)
        errors_group = errors[:, :6].reshape(errors.size(0), 3, 2)  # (B, 3, 2)
        l2_errors = torch.norm(errors_group, dim=2)  # (B, 3)

        # Direct absolute errors for distance_to_road_edge and rcv_time
        distance_error = torch.abs(errors[:, 6]).unsqueeze(1)
        time_error = torch.abs(errors[:, 7]).unsqueeze(1)
        
        # Angular error for heading (features 8-9 are normalized sin/cos)
        sin_pred = (y_hat[:, 8] * 2) - 1
        cos_pred = (y_hat[:, 9] * 2) - 1
        sin_true = (y[:, 8] * 2) - 1
        cos_true = (y[:, 9] * 2) - 1
        
        # Geodesic distance on unit circle
        dot_product = torch.clamp(sin_pred * sin_true + cos_pred * cos_true, -1.0, 1.0)
        angular_error = torch.acos(dot_product) / torch.pi  # Normalize: [0, π] → [0, 1]
        angular_error = angular_error.unsqueeze(1)
        
        grouped_errors = torch.cat([l2_errors, distance_error, time_error, angular_error], dim=1)
        self.val_reconstruction_errors_grouped.append(grouped_errors.detach().cpu().numpy())
        errors = torch.abs(errors)
        self.val_reconstruction_errors_single.append(errors.detach().cpu().numpy())

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        if self.val_reconstruction_errors_grouped:
            
            errors_array = np.concatenate(self.val_reconstruction_errors_grouped, axis=0)

            # Compute grouped thresholds (95% quantile)
            grouped_thresholds = np.quantile(errors_array, 0.95, axis=0)
            
            self.grouped_thresholds.copy_(
                torch.from_numpy(grouped_thresholds).float().to(self.grouped_thresholds.device)
            )

            self.val_quantil_threshold.copy_(
                torch.tensor(np.mean(grouped_thresholds), device=self.val_quantil_threshold.device)
            )

            self.last_val_grouped_errors = errors_array.copy()

        self.last_val_single_errors = np.concatenate(self.val_reconstruction_errors_single, axis=0) if self.val_reconstruction_errors_single else None

        self.val_reconstruction_errors_grouped = []
        self.val_reconstruction_errors_single = []

        print(f"Validation grouped thresholds (rel_pos + sender): {self.grouped_thresholds.detach().cpu().numpy()}")
    
    
    def _quantil_threshold(self, val_errors):
        return np.quantile(val_errors, 0.99)

    def set_grouped_thresholds(self, thresholds):
        """rel_pos, sender_spd, sender_acc, distance_to_road, rcv_time, sender_heading"""
        self.grouped_thresholds.copy_(torch.tensor(thresholds).float().to(self.grouped_thresholds.device))

    
    #def test_step(self, batch, batch_idx):
    #    x, y = batch
    #    y_hat = self(x)
    #    loss = nn.functional.mse_loss(y_hat, y)
    #    self.log("test_loss", loss)
    #    return loss

    def test_step(self, batch, batch_idx):
        x, y, label, attack, receiver, messageId, scenario, traffic_density, drivers_profile, sender_id = batch
        x = x.view(x.size(0), -1)

        with torch.no_grad():
            y_hat = self(x)


        # L2 errors per group (rel_pos, sender spd, acc) and angular error (heading)
        errors = (y_hat - y)
        errors_group = errors[:, :6].reshape(errors.size(0), 3, 2)
        l2_errors = torch.norm(errors_group, dim=2)

        # Direct absolute errors for distance_to_road_edge and rcv_time
        distance_error = torch.abs(errors[:, 6]).unsqueeze(1)
        time_error = torch.abs(errors[:, 7]).unsqueeze(1)
        
        # Angular error for heading (otherwise large error between 0 and 359 degrees)
        sin_pred = (y_hat[:, 8] * 2) - 1
        cos_pred = (y_hat[:, 9] * 2) - 1
        sin_true = (y[:, 8] * 2) - 1
        cos_true = (y[:, 9] * 2) - 1
        
        dot_product = torch.clamp(sin_pred * sin_true + cos_pred * cos_true, -1.0, 1.0)
        angular_error = torch.acos(dot_product) / torch.pi  # Normalize: [0, π] → [0, 1]
        angular_error = angular_error.unsqueeze(1)
        
        grouped_errors = torch.cat([l2_errors, distance_error, time_error, angular_error], dim=1)

        # Anomaly if ANY group exceeds its threshold (OR rule)
        exceeds = grouped_errors > self.grouped_thresholds
        is_anomaly = exceeds.any(dim=1)

        # Log threshold exceedances for analysis
        group_names = ['rel_pos', 'sender_spd', 'sender_acc', 'distance_to_road', 'rcv_time', 'sender_heading']
        exceedence_groups_per_sample = [[] for _ in range(grouped_errors.size(0))]
        for sample_idx in range(grouped_errors.size(0)):
            if is_anomaly[sample_idx]:
                for group_idx in range(6):
                    if exceeds[sample_idx, group_idx]:
                        exceedence_groups_per_sample[sample_idx].append(group_names[group_idx])
                        error_val = grouped_errors[sample_idx, group_idx].item()
                        threshold_val = self.grouped_thresholds[group_idx].item()
                        excess = error_val - threshold_val
                        self.test_threshold_exceedances.append({
                            'group': group_names[group_idx],
                            'error': error_val,
                            'threshold': threshold_val,
                            'excess': excess,
                            'is_true_anomaly': label[sample_idx].item() == 1,
                            'attack_type': attack[sample_idx].item()
                        })
           
        # Update metrics
        self.test_conf_matrix.update(is_anomaly, label.int())
        self.test_accuracy.update(is_anomaly, label.int())
        self.test_precision.update(is_anomaly, label.int())
        self.test_recall.update(is_anomaly, label.int())
        self.test_f1.update(is_anomaly, label.int())
        
        # Track predictions per attack type
        for i in range(len(attack)):
            att_type_val = attack[i].item()
            if att_type_val not in self.test_attack_type_predictions:
                self.test_attack_type_predictions[att_type_val] = []
            self.test_attack_type_predictions[att_type_val].append((
                is_anomaly[i].item(),
                label[i].item(),
                messageId[i].item(),
                drivers_profile[i],
                sender_id[i],
                exceedence_groups_per_sample[i]
            ))
        return
    
    
    def get_val_error(self):
        if self.grouped_thresholds.shape[0] > 1:
            return self.grouped_thresholds.mean().item()
        return self.val_quantil_threshold.item()


    def on_test_epoch_end(self):
        acc = self.test_accuracy.compute()
        prec = self.test_precision.compute()
        rec = self.test_recall.compute()
        f1 = self.test_f1.compute()

        cm = self.test_conf_matrix.compute()
        tn, fp, fn, tp = cm.ravel()

        self.log_dict({
            "test/accuracy": acc,
            "test/precision": prec,
            "test/recall": rec,
            "test/f1": f1
        })

        self.log_dict({
            "test/true_positives": float(tp),
            "test/true_negatives": float(tn),
            "test/false_positives": float(fp),
            "test/false_negatives": float(fn),
        })

        # Log threshold exceedance statistics
        if self.test_threshold_exceedances:
            print("\n=== Threshold Exceedance Summary ===")
            group_counts = {}
            group_excess_sum = {}
            group_true_attacks = {}
            
            for exc in self.test_threshold_exceedances:
                group = exc['group']
                group_counts[group] = group_counts.get(group, 0) + 1
                group_excess_sum[group] = group_excess_sum.get(group, 0) + exc['excess']
                if exc['is_true_anomaly']:
                    group_true_attacks[group] = group_true_attacks.get(group, 0) + 1
            
            total_true = sum(1 for exc in self.test_threshold_exceedances if exc['is_true_anomaly'])
            print(f"Total anomalies detected: {len(self.test_threshold_exceedances)} ({total_true} true attacks, {len(self.test_threshold_exceedances) - total_true} false positives)")
            for group in ['rel_pos', 'sender_spd', 'sender_acc', 'distance_to_road', 'rcv_time', 'sender_heading']:
                if group in group_counts:
                    count = group_counts[group]
                    true_count = group_true_attacks.get(group, 0)
                    false_count = count - true_count
                    avg_excess = group_excess_sum[group] / count
                    print(f"  {group:15s}: {count:5d} exceedances ({true_count} true, {false_count} FP), avg excess: {avg_excess:.4f}")
            
            self.test_threshold_exceedances = []
        
        # Log per-attack-type metrics (print only). Writing to file is performed explicitly
        # by calling `write_attack_type_metrics(log_path)` at final test time.
        if self.test_attack_type_predictions:
            print("\n=== Per-Attack-Type Metrics ===")
            for att_type in sorted(self.test_attack_type_predictions.keys()):
                predictions = self.test_attack_type_predictions[att_type]
                tp = sum(1 for pred, label, _, _, _, _ in predictions if pred == 1 and label == 1)
                fp = sum(1 for pred, label, _, _, _, _ in predictions if pred == 1 and label == 0)
                tn = sum(1 for pred, label, _, _, _, _ in predictions if pred == 0 and label == 0)
                fn = sum(1 for pred, label, _, _, _, _ in predictions if pred == 0 and label == 1)

                total = len(predictions)
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                line = (f"Attack Type {att_type:2d}: TP={tp:5d}, FP={fp:5d}, TN={tn:5d}, FN={fn:5d} | "
                        f"Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
                print(line)

            print("================================\n")

            # keep `self.test_attack_type_predictions` available until explicitly written

    def write_attack_type_metrics(self, log_path):
        """Append the per-attack-type metrics to `log_path`.

        This is intended to be called once after the final test run.
        """
        if not self.test_attack_type_predictions:
            return
        try:
            with open(log_path, 'a') as lf:
                lf.write("\n=== Per-Attack-Type Metrics ===\n")
                for att_type in sorted(self.test_attack_type_predictions.keys()):
                    predictions = self.test_attack_type_predictions[att_type]
                    tp = sum(1 for pred, label, _, _, _, _ in predictions if pred == 1 and label == 1)
                    fp = sum(1 for pred, label, _, _, _, _ in predictions if pred == 1 and label == 0)
                    tn = sum(1 for pred, label, _, _, _, _ in predictions if pred == 0 and label == 0)
                    fn = sum(1 for pred, label, _, _, _, _ in predictions if pred == 0 and label == 1)

                    fp_messages = [(msg_id, drivers_profile, sender_id, exceedence_groupes) for pred, label, msg_id, drivers_profile, sender_id, exceedence_groupes in predictions if pred == 1 and label == 0]

                    total = len(predictions)
                    accuracy = (tp + tn) / total if total > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    line = (f"Attack Type {att_type:2d}: TP={tp:5d}, FP={fp:5d}, TN={tn:5d}, FN={fn:5d} | "
                            f"Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}\n")
                    lf.write(line)

                    with open(f"test_fp/fp_message_ids_{att_type}.txt", "a") as fp_file:
                        fp_file.write("Attack Type {}: FP Message IDs:\n".format(att_type))
                        fp_file.write("-" * 40 + "\n")
                        for msg_id, drivers_profile, sender_id, exceedence_groupes in fp_messages:
                            fp_file.write(f"{sender_id},{drivers_profile},{msg_id},{exceedence_groupes}\n")

        except Exception:
            pass
        finally:
            # clear stored predictions after writing to file
            self.test_attack_type_predictions = {}



