import argparse
import os
import torch
import clip
import json
import operator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import ImageNet2p, ImageNet, ImageNetSketch, ImageNetR
from utils import get_model_from_sd, test_model_on_dataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('~/data'))
    parser.add_argument("--model-location", type=str, default=os.path.expanduser('~/ssd/checkpoints/soups'))
    parser.add_argument("--download-models", action="store_true", default=False)
    parser.add_argument("--eval-individual-models", action="store_true", default=False)
    parser.add_argument("--uniform-soup", action="store_true", default=False)
    parser.add_argument("--greedy-soup", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    NUM_MODELS = 72
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'greedy_soup_results.jsonl'

    # Skip download and use already available models
    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)]

    # Skip models that do not exist
    model_paths = [path for path in model_paths if os.path.exists(path)]
    if len(model_paths) != NUM_MODELS:
        print(f"Skipping {NUM_MODELS - len(model_paths)} models that are missing.")
    
    if args.eval_individual_models or args.uniform_soup or args.greedy_soup:
        base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)

    # Evaluate individual models
    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                model = get_model_from_sd(state_dict, base_model)

                results = {'model_name': f'model_{j}'}
                for dataset_cls in [ImageNet2p, ImageNet, ImageNetSketch, ImageNetR]:  
                    print(f'Evaluating model {j} on {dataset_cls.__name__}')
                    dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                    accuracy = test_model_on_dataset(model, dataset)
                    results[dataset_cls.__name__] = accuracy
                    print(accuracy)

                with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                    f.write(json.dumps(results) + '\n')
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")

    # Uniform soup
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)

        uniform_soup = None
        for j, model_path in enumerate(model_paths):
            print(f'Adding model {j} to uniform soup')
            try:
                state_dict = torch.load(model_path)
                if uniform_soup is None:
                    uniform_soup = {k: v * (1. / NUM_MODELS) for k, v in state_dict.items()}
                else:
                    uniform_soup = {k: v * (1. / NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")

        if uniform_soup:
            model = get_model_from_sd(uniform_soup, base_model)
            results = {'model_name': 'uniform_soup'}
            for dataset_cls in [ImageNet2p, ImageNet, ImageNetSketch, ImageNetR]: 
                print(f'Evaluating on {dataset_cls.__name__}')
                dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                accuracy = test_model_on_dataset(model, dataset)
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(UNIFORM_SOUP_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

        # Greedy soup
    if args.greedy_soup:
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            os.remove(GREEDY_SOUP_RESULTS_FILE)

        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = {
            row['model_name']: row['ImageNetR'] for _, row in individual_model_db.iterrows()
        }

        existing_model_val_accs = []
        for model_name, acc in individual_model_val_accs.items():
            model_path = os.path.join(args.model_location, f'{model_name}.pt')
            if os.path.exists(model_path):
                existing_model_val_accs.append((model_name, acc))
            else:
                print(f"Skipping missing model: {model_name}")

        if not existing_model_val_accs:
            print("No models available for greedy soup. Exiting.")
            exit(0)

        existing_model_val_accs.sort(key=operator.itemgetter(1), reverse=True)
        sorted_models = [x[0] for x in existing_model_val_accs]

        greedy_soup_ingredients = [sorted_models[0]]
        greedy_soup_params = torch.load(os.path.join(args.model_location, f'{sorted_models[0]}.pt'))
        best_val_acc_so_far = existing_model_val_accs[0][1]
        held_out_val_set = ImageNetR(preprocess, args.data_location, args.batch_size, args.workers)

        for i in range(1, len(sorted_models)):
            print(f'Testing model {i} of {len(sorted_models)}')
            model_path = os.path.join(args.model_location, f'{sorted_models[i]}.pt')
            new_ingredient_params = torch.load(model_path)

            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) +
                   new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            model = get_model_from_sd(potential_greedy_soup_params, base_model)
            held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)

            print(f'Potential val acc: {held_out_val_accuracy}, best so far: {best_val_acc_so_far}')
            if held_out_val_accuracy > best_val_acc_so_far:
                greedy_soup_ingredients.append(sorted_models[i])
                best_val_acc_so_far = held_out_val_accuracy
                greedy_soup_params = potential_greedy_soup_params
                print(f'Added. Current greedy soup: {greedy_soup_ingredients}')

        model = get_model_from_sd(greedy_soup_params, base_model)
        results = {'model_name': 'greedy_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetSketch, ImageNetR]:
            print(f'Evaluating on {dataset_cls.__name__}')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(GREEDY_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')


    # Plot results
    if args.plot:
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_db['OOD'] = 1. / 4 * (
            individual_model_db['ImageNet2p'] +
            individual_model_db['ImageNetR'] +
            individual_model_db['ImageNetSketch'] 
        )
        uniform_soup_db = pd.read_json(UNIFORM_SOUP_RESULTS_FILE, lines=True)
        uniform_soup_db['OOD'] = 1. / 4 * (
            uniform_soup_db['ImageNet2p'] +
            uniform_soup_db['ImageNetR'] +
            uniform_soup_db['ImageNetSketch'] 
        )
        greedy_soup_db = pd.read_json(GREEDY_SOUP_RESULTS_FILE, lines=True)
        greedy_soup_db['OOD'] = 1. / 4 * (
            greedy_soup_db['ImageNet2p'] +
            greedy_soup_db['ImageNetR'] +
            greedy_soup_db['ImageNetSketch'] 
        )

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        ax = fig.subplots()

        ax.scatter(
            greedy_soup_db['ImageNet'],
            greedy_soup_db['OOD'],
            marker='*',
            color='C4',
            s=400,
            label='Greedy Soup',
            zorder=10
        )

        ax.scatter(
            uniform_soup_db['ImageNet'],
            uniform_soup_db['OOD'],
            marker='o',
            color='C0',
            s=200,
            label='Uniform Soup',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[0],
            individual_model_db['OOD'].values[0],
            marker='h',
            color='slategray',
            s=150,
            label='Initialization (LP)',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[1:],
            individual_model_db['OOD'].values[1:],
            marker='d',
            color='C2',
            s=130,
            label='Various hyperparameters',
            zorder=10
        )

        ax.set_ylabel('Avg. accuracy on 4 distribution shifts', fontsize=16)
        ax.set_xlabel('ImageNet Accuracy (top-1%)', fontsize=16)
        ax.grid()
        ax.legend(fontsize=13)
        plt.savefig('figure.png', bbox_inches='tight')

