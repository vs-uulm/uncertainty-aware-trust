import torch
import numpy as np
import math

def generate_nn_input_data(
    data,
    context_len,
    remove_attacker=False,
    attack_type=0,
    data_type="train",
    use_sliding_window=False,
    sliding_window_step=1,
    scenario=None,
    traffic_density=None
):
    all_input_data = []
        
    for vehicle_id in data:
        vehicle_data = data[vehicle_id]
        labeled_vehicle_input_data = _create_labeled_vehicle_input_data(vehicle_data, context_len, remove_attacker, attack_type, data_type, vehicle_id, use_sliding_window=use_sliding_window, sliding_window_step=sliding_window_step, scenario=scenario, traffic_density=traffic_density)
        all_input_data.extend(labeled_vehicle_input_data)
    return all_input_data
        

def _create_labeled_vehicle_input_data(
    vehicle_data,
    context_len,
    remove_attacker,
    attack_type,
    data_type="train",
    receiver_id=None,
    use_sliding_window=False,
    sliding_window_step=1,
    scenario=None,
    traffic_density=None
):
    messages_by_sender = _get_messages_per_sender(vehicle_data, remove_attacker, data_type)
    all_labeled_sender_windows = []
    for sender in messages_by_sender:
        labeled_feature_vecs = _create_NN_data_for_sender(messages_by_sender[sender], context_len, attack_type, data_type, receiver_id=receiver_id, use_sliding_window=use_sliding_window, sliding_window_step=sliding_window_step, scenario=scenario, traffic_density=traffic_density)

        all_labeled_sender_windows.extend(labeled_feature_vecs)
    return all_labeled_sender_windows


def _create_NN_data_for_sender(
        sender_messages,
        context_len, 
        attack_type,
        data_type="train",
        receiver_id=None,
        use_sliding_window=False,
        sliding_window_step=1,
        scenario=None,
        traffic_density=None
        ):
    labeled_sender_windows = create_labeled_sender_windows(sender_messages, context_len, data_type, attack_type, use_sliding_window, sliding_window_step=sliding_window_step)

    labeled_feature_vecs = extract_labeled_feature_vecs(labeled_sender_windows, receiver_id, scenario, traffic_density)
    return labeled_feature_vecs


def extract_labeled_feature_vecs(
        labeled_sender_windows,
        receiver_id=None,
        scenario=None,
        traffic_density=None
        ):
    labeled_feature_vecs = []
    for labeled_window in labeled_sender_windows:
        window, label, attack_type = labeled_window

        labeled_feature_vec = _window_to_pred_input_no_rssi(label, window, attack_type, receiver_id, scenario, traffic_density)

        labeled_feature_vecs.append(labeled_feature_vec)
    return labeled_feature_vecs


def _get_messages_per_sender(vehicle_data, remove_attacker, data_type="train"):
    # creates dict: {sender_id: [messages of sender]}
    messages_by_sender = {}
    for message in vehicle_data:
        if message["sender_id"] in messages_by_sender:
            messages_by_sender[message["sender_id"]].append(message)
        else:
            messages_by_sender[message["sender_id"]] = [message]
       
    return messages_by_sender

def create_labeled_sender_windows(messages, context_len, data_type="train", attack_type=0, use_sliding_window=False, sliding_window_step=1):
    windows = []
    window = []
    i = 0
 
    if use_sliding_window:
        for datapoint in messages:
            if (i < context_len):
                window.append(datapoint)
                i += 1
            else:
                if "attacker" in datapoint:
                    label = datapoint["attacker"]
                else:
                    label = 0  # no attackers in training/val

                windows.append((window.copy(), label, attack_type))
                window = window[sliding_window_step:]  # slide the window
                window.append(datapoint)
                i = len(window)
    else:
        for datapoint in messages:
            if (i < context_len):
                window.append(datapoint)
                i += 1
            else:
                # only add complete windows
                if data_type == "Train" or data_type == "Validation":
                    label = 0  # no attackers in training/val
                else:
                    label = datapoint["attacker"]
                windows.append((window, label, attack_type))
                window = [datapoint]
                i = 1
    return windows



def _window_to_pred_input_no_rssi(label, window, attack_type, receiver_id=None, scenario=None, traffic_density=None):
    normalized_in = []

    prev_rcv_time = 0
    for i, message in enumerate(window[0:-1]):
        receiver_gps_pos = [float(s) for s in message["receiver"]["pos"].split(",")][:2]
        sender_gps_pos = [float(s) for s in message["sender"]["pos"].split(",")][:2]
        abs_pos_zip = zip(receiver_gps_pos, sender_gps_pos)
        rel_pos = [a-b for (a,b) in abs_pos_zip]

        normalized_in.extend(_normalize_relative_position(rel_pos))
        normalized_in.extend(_normalize_spd(message["receiver"]["spd"], message["receiver"]["hed"]))
        normalized_in.extend(_normalize_spd(message["sender"]["spd"], message["sender"]["hed"]))
        normalized_in.extend(_normalize_acceleration(message["receiver"]["acl"], message["receiver"]["hed"]))
        normalized_in.extend(_normalize_acceleration(message["sender"]["acl"], message["sender"]["hed"]))
        normalized_in.extend(_normalize_distance_to_road_edge(message["sender"]["distance_to_road_edge"]))

        if prev_rcv_time == 0:
            # Use time diff of first two messages in window for normalization of first message
            initial_time_difference = int(window[1]["rcvTime"]) - int(message["rcvTime"])
            normalized_in.extend(_normalize_rcv_time(initial_time_difference))
        else:
            normalized_in.extend(_normalize_rcv_time(int(message["rcvTime"]) - prev_rcv_time))
        prev_rcv_time = int(message["rcvTime"])

        normalized_in.extend(_normalize_heading(message["sender"]["hed"]))
        normalized_in.extend(_normalize_heading(message["receiver"]["hed"]))

    normalized_out = []
    last_msg = window[-1]

    receiver_gps_pos = [float(s) for s in last_msg["receiver"]["pos"].split(",")][:2]
    sender_gps_pos = [float(s) for s in last_msg["sender"]["pos"].split(",")][:2]
    abs_pos_zip = zip(receiver_gps_pos, sender_gps_pos)
    rel_pos = [a-b for (a,b) in abs_pos_zip]

    normalized_out.extend(_normalize_relative_position(rel_pos))
    normalized_out.extend(_normalize_spd(last_msg["sender"]["spd"], last_msg["sender"]["hed"]))
    normalized_out.extend(_normalize_acceleration(last_msg["sender"]["acl"], last_msg["sender"]["hed"]))
    normalized_out.extend(_normalize_distance_to_road_edge(last_msg["sender"]["distance_to_road_edge"]))
    normalized_out.extend(_normalize_rcv_time(int(last_msg["rcvTime"]) - prev_rcv_time))
    normalized_out.extend(_normalize_heading(last_msg["sender"]["hed"]))

    return (torch.tensor(normalized_in), torch.tensor(normalized_out), label, attack_type, receiver_id, last_msg['messageID'], scenario, traffic_density, last_msg["sender"]["driversProfile"], last_msg["sender_id"])

def _normalize_spd(spd, hed):
    # convert speed to cartesian
    spd = float(spd)
    hed = float(hed)
    hed_rad = math.radians(hed)
    spd_x = spd * math.sin(hed_rad)
    spd_y = spd * math.cos(hed_rad)

    # clamp to [-20,20] m/s and map to [0,1]
    vx = max(-20.0, min(spd_x, 20.0))
    vy = max(-20.0, min(spd_y, 20.0))
    nx = (vx / 20.0 + 1.0) / 2.0
    ny = (vy / 20.0 + 1.0) / 2.0
    return [nx, ny]
   

def _normalize_relative_position(pos):
    dist_threshold = 1000
    squared_sum = 0
    for val in pos:
        squared_sum += val**2
    dist = float(np.sqrt(squared_sum))
    if dist <= dist_threshold:
        f = (0.9*(dist/dist_threshold))
    else:
        f = (1 - (0.1* (1 / (1+dist-dist_threshold))))
    norm_pos = []
    for val in pos:
        norm_pos.append((val/dist)*f)
    return norm_pos

def _normalize_acceleration(acl, hed):
    acl = float(acl)
    hed = float(hed)
    hed_rad = math.radians(hed)
    
    # Convert longitudinal acceleration to cartesian components
    acl_x = acl * math.sin(hed_rad)
    acl_y = acl * math.cos(hed_rad)
    
    # Normalization: typical acceleration -10 to +10 m/s²
    max_acl = 10.0  # m/s²
    # Clamp components to [-max_acl, max_acl] to avoid values outside [0,1]
    acl_x_clamped = max(-max_acl, min(acl_x, max_acl))
    acl_y_clamped = max(-max_acl, min(acl_y, max_acl))
    return [((acl_x_clamped + max_acl) / (2 * max_acl)), ((acl_y_clamped + max_acl) / (2 * max_acl))]

def _normalize_heading(hed):
    # Normalize with sin & cos curves (as heading is circular -> 0° = 360°)
    hed = float(hed)
    hed_rad = math.radians(hed)
    return [(math.sin(hed_rad)+1)/2, (math.cos(hed_rad)+1)/2]

def _normalize_rcv_time(rcvTime):
    # Normalize time difference to a range of 0 to 1, where 0 corresponds to 0 ns and 1 corresponds to a defined maximum time window (e.g., 500 ms = 500,000,000 ns).
    time_window = 2_000_000_000  # ns
    try:
        r = float(rcvTime)
    except Exception:
        return [0.0]
    # negative deltas are treated as zero
    if r < 0.0:
        r = 0.0
    normalized_time = max(0.0, min(r / time_window, 1.0))
    return [normalized_time]

def _normalize_distance_to_road_edge(diff):
    diff = float(diff)
    min_distance = -100.0  # meters
    max_distance = 20.0   # meters
    normalized_diff = (diff - min_distance) / (max_distance - min_distance)
    normalized_diff = min(max(normalized_diff, 0), 1)
    return [normalized_diff]