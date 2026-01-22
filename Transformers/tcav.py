import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

def select_concept_images(folder_path, concept, n): 
    """
    Returns a list with up to `n` random .jpg fiel paths from `folder_path` that start with `concept`.
    """
    matching_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if (f.lower().startswith(concept.lower()) and f.lower().endswith('.jpg'))
    ]
    
    # Shuffle and select
    if len(matching_files) < n:
        print(f"Only found {len(matching_files)} files matching concept '{concept}'. Returning all.")
        return matching_files
    else:
        return random.sample(matching_files, n)

def sample_images(folder_path, n):
    """
    Returns a list with up to `n` random files from `folder_path`.
    """
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]

    # Shuffle and select
    if len(files) < n:
        print(f"Only found {len(files)} files in folder. Returning all.")
        return files
    else:
        return random.sample(files, n)

def get_vit_dataloader(image_paths, processor, batch_size=16): 
    dataset = ViTImageDataset(image_paths, processor)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader 

class ViTImageDataset(Dataset):
    def __init__(self, paths, processor=None):
        self.processor = processor
        self.image_paths = paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

def make_activation_hook(activations): 
    def fwd_hook(module, input, output):
        out = output[0]
        activations.append(out.squeeze(0).detach().cpu())
    return fwd_hook

def make_gradient_hook(grads):
    def bwd_hook(module, grad_input, grad_output):
        grad = grad_output[0]
        grads.append(grad.squeeze(0).detach().cpu())
    return bwd_hook

def get_activations(model, data_loader, device, layer):
    model.eval()
    all_activations = []

    for batch_inputs in data_loader:
        for i in range(batch_inputs.size(0)):

            sample_input = batch_inputs[i].unsqueeze(0).to(device)

            activations = []
            hook = layer.register_forward_hook(make_activation_hook(activations))

            with torch.no_grad():
                _ = model(sample_input)

            hook.remove()

            all_activations.append(activations[0])

    return torch.stack(all_activations)

def get_grads(model, data_loader, device, layer, target_idx=None):
    model.eval()
    all_grads = []

    for batch_inputs in data_loader:
        for i in range(batch_inputs.size(0)):
            sample_input = batch_inputs[i].unsqueeze(0).to(device)

            grads = []
            hook = layer.register_full_backward_hook(make_gradient_hook(grads))

            model.zero_grad()
            output = model(sample_input)

            logits = output.logits[0]

            if target_idx is None:
                logit = logits[logits.argmax()]
            else:
                logit = logits[target_idx]

            logit.backward()

            hook.remove()

            all_grads.append(grads[0])

    return torch.stack(all_grads)

def flatten(acts):
    return acts.view(acts.size(0), -1).numpy()


    
def train_cav(positive_activations, negative_activations, seed=1337):

    X = np.concatenate([positive_activations, negative_activations], axis=0)
    y = np.concatenate([np.ones(len(positive_activations)), np.zeros(len(negative_activations))])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=seed)
    X_train, y_train = shuffle(X, y, random_state=seed)
    classifier = LogisticRegression(random_state=seed, max_iter=1000) 
    classifier.fit(X_train, y_train)

    cav = classifier.coef_[0]
    unit_cav = cav / np.linalg.norm(cav)

    return unit_cav

def tcav(cav, grads):

    directional_derivatives = np.dot(grads, cav)
    positive_counts = np.sum(directional_derivatives > 0)
    tcav_score = positive_counts / directional_derivatives.shape[0]

    return tcav_score

