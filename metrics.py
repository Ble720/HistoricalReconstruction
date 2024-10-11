import os

def score(source_name, pred):
    correct_matches = os.listdir(f'./labels/{source_name}')
    for correct_img in correct_matches:
        if correct_img in pred:
            return True
    return False


def score_folder(pred_folder_path):
    correct = 0
    source_names = os.listdir(pred_folder_path)
    topk = 15
    for source_name in source_names:
        correct_matches = os.listdir(f'./labels/{source_name}')
        preds = os.listdir(f'{pred_folder_path}/{source_name}')
        intersect = set(correct_matches).intersection(preds)
        if len(intersect) > 0:
            correct += 1

    print(f'Top-{topk} Accuracy : {correct}/{len(source_names)}')
    return correct/len(source_names)
