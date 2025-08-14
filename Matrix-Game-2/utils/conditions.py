
import torch
import random

def combine_data(data, num_frames=57, keyboard_dim=6):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    
    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]['keyboard_condition']
        
        if current_frame == 0:
            keyboard_condition[:1] = k[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame:current_frame+rd_frame] = k.repeat(repeat_time, 1)
            current_frame += rd_frame
    return {"keyboard_condition": keyboard_condition}

def Bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
    ]
    actions_to_test = actions_single_action * 1

    base_action = actions_single_action

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3
    }

    data = []

    for action_name in actions_to_test:
        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)]

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
        })
    return combine_data(data, num_frames, keyboard_dim=4)