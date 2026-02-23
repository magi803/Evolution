"""
Neural Network Experiments for Evolution Theory
Complete H1+H2+H3 on Fashion-MNIST (cleaner dataset)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d
import warnings
import sys
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def effective_rank(weight_matrix, energy_threshold=0.95):
    u, s, v = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
    s = s.cpu().numpy()
    total_energy = np.sum(s**2)
    cum_energy = np.cumsum(s**2) / total_energy
    idx = np.where(cum_energy >= energy_threshold)[0]
    if len(idx) == 0:
        return len(s)
    if idx[0] == 0:
        return 1.0
    return idx[0] + (energy_threshold - cum_energy[idx[0]-1]) / (cum_energy[idx[0]] - cum_energy[idx[0]-1])

def compute_order_quantity(model, layer_name='patch_embed', energy_threshold=0.95):
    for name, module in model.named_modules():
        if name == layer_name:
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                w_flat = w.view(w.size(0), -1)
                er = effective_rank(w_flat, energy_threshold)
                return -er
            elif isinstance(module, nn.Linear):
                w = module.weight.data
                er = effective_rank(w, energy_threshold)
                return -er
    raise ValueError(f"Layer {layer_name} not found")

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, correct / total

# LightViT_smaller adapted for Fashion-MNIST (1 channel, 28x28)
class LightViT_smaller(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10,
                 embed_dim=64, depth=3, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super(LightViT_smaller, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in (self.head,):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x

def load_fashion_mnist(batch_size=256, num_workers=4, subset_ratio=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    if subset_ratio < 1.0:
        num_samples = int(len(full_trainset) * subset_ratio)
        indices = list(range(num_samples))
        trainset = Subset(full_trainset, indices)
    else:
        trainset = full_trainset
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

# -------------------- H1 --------------------
def run_h1(model_class, trainloader, testloader, epochs=80, layer_name='patch_embed',
           lr=0.01, search_window=(-30,15), patience=5, smooth_B=True, smooth_window=3,
           save_prefix='h1_fashion'):
    print("\n" + "="*60)
    print("H1: Overfitting and Internal博弈 (Fashion-MNIST)")
    print("="*60)
    model = model_class(dropout=0.0).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)

    M_list, M_steps = [], []
    grad_norms_per_epoch = []
    val_losses = []
    steps_per_epoch = len(trainloader)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = correct = total = 0
        epoch_grad_norms = []
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False, ncols=100)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = compute_gradient_norm(model)
                epoch_grad_norms.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = compute_gradient_norm(model)
                epoch_grad_norms.append(gn)
                optimizer.step()

            M = compute_order_quantity(model, layer_name=layer_name)
            M_list.append(M)
            M_steps.append(global_step)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
            global_step += 1

        grad_norms_per_epoch.append(epoch_grad_norms)
        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, testloader, criterion)
        val_losses.append(val_loss)
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, M: {M_list[-1]:.4f}")

    M_epoch = [np.mean(M_list[i*steps_per_epoch:(i+1)*steps_per_epoch]) for i in range(epochs)]
    M_epoch = np.array(M_epoch)

    B_raw = np.array([np.max(g) for g in grad_norms_per_epoch])  # use max
    if smooth_B and smooth_window > 1:
        B_epoch = uniform_filter1d(B_raw, size=smooth_window, mode='reflect')
    else:
        B_epoch = B_raw

    val_losses_arr = np.array(val_losses)
    best_epoch = np.argmin(val_losses_arr)
    best_val_loss = val_losses_arr[best_epoch]
    overfit_start = epochs
    for i in range(best_epoch+1, epochs):
        if val_losses[i] < best_val_loss - 1e-4:
            best_val_loss = val_losses[i]
            best_epoch = i
        elif i - best_epoch >= patience:
            overfit_start = i - patience
            break

    search_start = max(0, overfit_start + search_window[0])
    search_end = min(epochs, overfit_start + search_window[1])
    B_window = B_epoch[search_start:search_end]
    if len(B_window) == 0 or np.all(np.isnan(B_window)):
        lead, p_val = np.nan, np.nan
    else:
        peak_local = np.nanargmax(B_window)
        peak_global = search_start + peak_local
        lead = overfit_start - peak_global
        n_perm = 10000
        leads_perm = []
        for _ in range(n_perm):
            B_shuffled = np.random.permutation(B_epoch)
            B_win_shuf = B_shuffled[search_start:search_end]
            if len(B_win_shuf) == 0 or np.all(np.isnan(B_win_shuf)):
                continue
            peak_shuf_local = np.nanargmax(B_win_shuf)
            peak_shuf_global = search_start + peak_shuf_local
            leads_perm.append(overfit_start - peak_shuf_global)
        p_val = (np.sum(np.array(leads_perm) >= lead) + 1) / (n_perm + 1)

    print(f"Min val loss at epoch {best_epoch} (loss={best_val_loss:.4f})")
    print(f"Overfit start epoch: {overfit_start}")
    print(f"B peak epoch: {peak_global}")
    print(f"Lead: {lead}, p-value: {p_val:.4f}")

    results = {
        'min_val_epoch': best_epoch, 'overfit_start': overfit_start,
        'B_peak': peak_global, 'lead': lead, 'p_value': p_val,
        'M': M_epoch.tolist(), 'val_loss': val_losses, 'B_epoch': B_epoch.tolist()
    }
    torch.save(results, f'{save_prefix}_results.pth')
    return results

# -------------------- H2 --------------------
def estimate_omega(seq, sample_rate=1):
    seq = seq - np.mean(seq)
    n = len(seq)
    fft_vals = np.abs(fft(seq))
    fft_freq = fftfreq(n, d=sample_rate)[:n//2]
    fft_vals = fft_vals[:n//2]
    valid = fft_freq > 1/n
    if not np.any(valid):
        return 0.1
    peak_freq = fft_freq[valid][np.argmax(fft_vals[valid])]
    return 2 * np.pi * peak_freq

def run_h2(model_class, trainloader, testloader, base_epochs=30, tune_epochs=15,
           layer_name='patch_embed', save_prefix='h2_fashion'):
    print("\n" + "="*60)
    print("H2: Resonance Effect (Fashion-MNIST)")
    print("="*60)

    # Phase 1
    model = model_class(dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    M_list, val_losses = [], []
    for epoch in range(base_epochs):
        model.train()
        pbar = tqdm(trainloader, desc=f'Phase1 Epoch {epoch+1}/{base_epochs}', leave=False, ncols=100)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        val_loss, _ = evaluate(model, testloader, criterion)
        val_losses.append(val_loss)
        M = compute_order_quantity(model, layer_name=layer_name)
        M_list.append(M)
        tqdm.write(f"Phase1 Epoch {epoch+1}/{base_epochs}, Val Loss: {val_loss:.4f}, M: {M:.4f}")

    M_arr = np.array(M_list)
    dM = np.diff(M_arr)
    omega = estimate_omega(dM, sample_rate=1)
    print(f"Estimated omega: {omega:.4f} rad/epoch")

    search_start = base_epochs // 2
    dM_search = dM[search_start-1:] if search_start-1 < len(dM) else dM
    crit_epoch = search_start + np.argmax(dM_search) if len(dM_search) > 0 else base_epochs//2
    print(f"Critical point at epoch {crit_epoch}")

    freqs = [0.0, omega, 2*omega]
    total_epochs = base_epochs + tune_epochs
    modulation_window = 8
    results = {}

    for freq in freqs:
        print(f"\nTesting frequency: {freq:.4f}")
        model = model_class(dropout=0.1).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        M_vals = []
        for epoch in range(total_epochs):
            if abs(epoch - crit_epoch) <= modulation_window//2 and freq > 0:
                mod_lr = 0.01 * (1 + 0.3 * np.sin(2*np.pi*freq*epoch))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = mod_lr
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01

            model.train()
            pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{total_epochs}', leave=False, ncols=100)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            M = compute_order_quantity(model, layer_name=layer_name)
            M_vals.append(M)
            if (epoch+1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{total_epochs}, M: {M:.4f}")

        post_start = crit_epoch + modulation_window//2 + 1
        if post_start < len(M_vals):
            M_post = M_vals[post_start:]
            var_M = np.var(M_post)
        else:
            var_M = np.nan
        results[freq] = {'M': M_vals, 'var': var_M}
        print(f"Variance after modulation: {var_M:.6f}")

    torch.save(results, f'{save_prefix}_results.pth')
    return results

# -------------------- H3 --------------------
def run_h3(model_class, trainloader, testloader, base_epochs=60, noise_epochs=20,
           layer_name='patch_embed', save_prefix='h3_fashion'):
    print("\n" + "="*60)
    print("H3: Capacity Release (Fashion-MNIST)")
    print("="*60)

    model = model_class(dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    M_list, val_losses = [], []
    for epoch in range(base_epochs):
        model.train()
        pbar = tqdm(trainloader, desc=f'Phase1 Epoch {epoch+1}/{base_epochs}', leave=False, ncols=100)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        val_loss, _ = evaluate(model, testloader, criterion)
        val_losses.append(val_loss)
        M = compute_order_quantity(model, layer_name=layer_name)
        M_list.append(M)
        tqdm.write(f"Phase1 Epoch {epoch+1}/{base_epochs}, Val Loss: {val_loss:.4f}, M: {M:.4f}")

    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    noise_strengths = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    final_M, final_val_loss = [], []

    for noise in noise_strengths:
        print(f"\nTesting noise strength: {noise}")
        model.load_state_dict(initial_state)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        if noise > 0:
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * noise)
        for epoch in range(noise_epochs):
            model.train()
            pbar = tqdm(trainloader, desc=f'Noise Epoch {epoch+1}/{noise_epochs}', leave=False, ncols=100)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        val_loss, _ = evaluate(model, testloader, criterion)
        final_val_loss.append(val_loss)
        M = compute_order_quantity(model, layer_name=layer_name)
        final_M.append(M)
        print(f"Final M: {M:.4f}, Final Val Loss: {val_loss:.4f}")

    best_idx = np.argmax(final_M)
    best_noise = noise_strengths[best_idx]
    print(f"\nBest noise strength: {best_noise}, final M: {final_M[best_idx]:.4f}")

    results = {
        'noise_strengths': noise_strengths,
        'final_M': final_M,
        'final_val_loss': final_val_loss,
        'best_noise': best_noise,
        'best_M': final_M[best_idx]
    }
    torch.save(results, f'{save_prefix}_results.pth')
    return results

def main():
    original_stdout = sys.stdout
    with open('TT.txt', 'w', encoding='utf-8') as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            print("="*60)
            print("NEURAL NETWORK EXPERIMENTS FOR EVOLUTION THEORY (FASHION-MNIST COMPLETE)")
            print("="*60)

            trainloader, testloader = load_fashion_mnist(batch_size=256, subset_ratio=0.5)

            # H1
            print("\n" + "="*60)
            print("Running H1")
            print("="*60)
            h1_results = run_h1(LightViT_smaller, trainloader, testloader,
                                epochs=80, lr=0.01, search_window=(-30,15),
                                patience=5, smooth_B=True, smooth_window=3,
                                save_prefix='h1_fashion')

            # H2
            print("\n" + "="*60)
            print("Running H2")
            print("="*60)
            h2_results = run_h2(LightViT_smaller, trainloader, testloader,
                                base_epochs=30, tune_epochs=15,
                                layer_name='patch_embed', save_prefix='h2_fashion')

            # H3
            print("\n" + "="*60)
            print("Running H3")
            print("="*60)
            h3_results = run_h3(LightViT_smaller, trainloader, testloader,
                                base_epochs=60, noise_epochs=20,
                                layer_name='patch_embed', save_prefix='h3_fashion')

            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"H1 p-value = {h1_results['p_value']:.4f}")
            print("H2 variances:")
            for freq, res in h2_results.items():
                print(f"  freq={freq:.3f}: var={res['var']:.6f}")
            print(f"H3 best noise = {h3_results['best_noise']}, best M = {h3_results['best_M']:.4f}")
        finally:
            sys.stdout = original_stdout
    print("All outputs saved to TT.txt")

if __name__ == "__main__":
    main()