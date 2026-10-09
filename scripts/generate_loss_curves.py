"""
Script to parse training_logs.txt and generate high-resolution publication-quality
telemetry curves for Usaid AI (500M) Pretraining and SFT Alignment.
Created by Mohamed Usaid.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

log_path = "training_logs.txt"
if not os.path.exists(log_path):
    raise FileNotFoundError(f"Cannot find {log_path}")

# 1. Parse Pretraining Data
pretrain_steps = []
pretrain_loss = []
pretrain_lr = []
pretrain_speed = []

eval_steps = []
eval_loss = []
eval_ppl = []

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m_train = re.search(r'Step\s+(\d+)/2000\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Grad Norm:\s+([0-9.]+)\s+\|\s+LR:\s+([0-9.e+-]+)\s+\|\s+Speed:\s+([0-9,]+)\s+tok/s', line)
        if m_train:
            pretrain_steps.append(int(m_train.group(1)))
            pretrain_loss.append(float(m_train.group(2)))
            pretrain_lr.append(float(m_train.group(4)))
            pretrain_speed.append(float(m_train.group(5).replace(',', '')))
        
        m_eval = re.search(r'\[EVAL\]\s+Step\s+(\d+)\s+\|\s+Val Loss:\s+([0-9.]+)\s+\|\s+Val PPL:\s+([0-9.]+)', line)
        if m_eval:
            eval_steps.append(int(m_eval.group(1)))
            eval_loss.append(float(m_eval.group(2)))
            eval_ppl.append(float(m_eval.group(3)))

# 2. Parse SFT Data (from line 3690 onwards)
sft_steps = []
sft_loss = []
sft_epoch = []

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if 'USAID AI (500M) SUPERVISED FINE-TUNING' in line:
            start_idx = i

    seen_steps = set()
    for line in lines[start_idx:]:
        m_sft = re.search(r'Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)/441\s+\|\s+SFT Loss:\s+([0-9.]+)\s+\|\s+Grad Norm:\s+([0-9.]+)\s+\|\s+LR:\s+([0-9.e+-]+)', line)
        if m_sft:
            ep = int(m_sft.group(1))
            st = int(m_sft.group(2))
            if st not in seen_steps:
                seen_steps.add(st)
                sft_epoch.append(ep)
                sft_steps.append(st)
                sft_loss.append(float(m_sft.group(3)))

# Create Multi-Panel Figure
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], top=0.86, bottom=0.07, left=0.07, right=0.92, hspace=0.30, wspace=0.32)

# Top Left: Pretraining Loss & Validation Convergence
ax1 = fig.add_subplot(gs[0, 0])
# Moving average for smooth visualization
window = 25
loss_smooth = np.convolve(pretrain_loss, np.ones(window)/window, mode='valid')
smooth_steps = pretrain_steps[window-1:]

ax1.plot(pretrain_steps, pretrain_loss, color='#93c5fd', alpha=0.35, label='Train Loss (Raw Steps)')
ax1.plot(smooth_steps, loss_smooth, color='#1d4ed8', linewidth=2.0, label=f'Train Loss ({window}-Step Moving Avg)')
ax1.plot(eval_steps, eval_loss, color='#dc2626', marker='o', markersize=5, linewidth=2.2, label='Validation Loss (Evaluated)')

# Highlight best val loss
best_eval_idx = np.argmin(eval_loss)
best_step = eval_steps[best_eval_idx]
best_val = eval_loss[best_eval_idx]
best_p = eval_ppl[best_eval_idx]
ax1.annotate(f'Best Val Loss: {best_val:.4f}\n(Val PPL: {best_p:.2f})',
             xy=(best_step, best_val), xytext=(best_step - 500, best_val + 1.2),
             arrowprops=dict(facecolor='#dc2626', shrink=0.08, width=1.5, headwidth=7),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.9),
             fontweight='bold', fontsize=10, color='#991b1b')

ax1.set_title('1. Foundational Pretraining Convergence (131M Tokens)', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Pretraining Steps (Effective Batch: 65,536 tokens)', fontsize=11)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax1.set_ylim(2.5, 11.5)
ax1.legend(loc='upper right', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# Top Right: SFT Alignment Convergence
ax2 = fig.add_subplot(gs[0, 1])
sft_window = 10
sft_smooth = np.convolve(sft_loss, np.ones(sft_window)/sft_window, mode='valid')
sft_smooth_steps = sft_steps[sft_window-1:]

ax2.plot(sft_steps, sft_loss, color='#e9d5ff', alpha=0.5, label='SFT Loss (Raw Steps)')
ax2.plot(sft_smooth_steps, sft_smooth, color='#7c3aed', linewidth=2.0, label=f'SFT Loss ({sft_window}-Step Moving Avg)')

# Highlight best SFT step
best_sft_idx = np.argmin(sft_loss)
best_sft_step = sft_steps[best_sft_idx]
best_sft_val = sft_loss[best_sft_idx]
ax2.annotate(f'All-Time Best SFT Loss: {best_sft_val:.4f}\n(Step {best_sft_step})',
             xy=(best_sft_step, best_sft_val), xytext=(best_sft_step - 160, best_sft_val + 0.8),
             arrowprops=dict(facecolor='#7c3aed', shrink=0.08, width=1.5, headwidth=7),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ede9fe', edgecolor='#7c3aed', alpha=0.9),
             fontweight='bold', fontsize=10, color='#5b21b6')

# Epoch vertical lines
epoch_boundaries = [147, 294]
for ep_step in epoch_boundaries:
    ax2.axvline(x=ep_step, color='#6b7280', linestyle=':', alpha=0.7)
ax2.text(70, 3.3, 'Epoch 1', ha='center', fontsize=9, color='#4b5563', style='italic')
ax2.text(220, 3.3, 'Epoch 2', ha='center', fontsize=9, color='#4b5563', style='italic')
ax2.text(360, 3.3, 'Epoch 3', ha='center', fontsize=9, color='#4b5563', style='italic')

ax2.set_title('2. Supervised Fine-Tuning (SFT) Alignment (4,731 Pairs)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('SFT Steps (Masked Target Loss)', fontsize=11)
ax2.set_ylabel('SFT Cross-Entropy Loss', fontsize=11)
ax2.set_ylim(1.2, 3.6)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

# Bottom Left: Pretraining Learning Rate Schedule & Perplexity
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(pretrain_steps, [lr * 1e4 for lr in pretrain_lr], color='#059669', linewidth=2.0, label='Learning Rate (x1e-4)')
ax3.set_title('3. Cosine Learning Rate Schedule with Linear Warmup', fontsize=13, fontweight='bold', pad=10)
ax3.set_xlabel('Pretraining Steps', fontsize=11)
ax3.set_ylabel('Learning Rate (x 10⁻⁴)', fontsize=11, color='#059669')
ax3.tick_params(axis='y', labelcolor='#059669')
ax3.set_ylim(0, 3.3)
ax3.grid(True, linestyle='--', alpha=0.6)

# Dual y-axis for perplexity
ax3_twin = ax3.twinx()
ax3_twin.plot(eval_steps, eval_ppl, color='#d97706', linewidth=2.0, linestyle='--', marker='s', markersize=4, label='Validation Perplexity (PPL)')
ax3_twin.set_ylabel('Validation Perplexity (Log Scale)', fontsize=11, color='#d97706')
ax3_twin.tick_params(axis='y', labelcolor='#d97706')
ax3_twin.set_yscale('log')
ax3_twin.set_ylim(15, 350)
ax3_twin.grid(False)

# Bottom Right: Hardware Execution & Training Throughput
ax4 = fig.add_subplot(gs[1, 1])
smooth_speed = np.convolve(pretrain_speed, np.ones(30)/30, mode='valid')
ax4.plot(pretrain_steps[29:], smooth_speed, color='#0891b2', linewidth=2.0, label='DDP Cluster Speed (Tokens / Sec)')
mean_speed = np.mean(pretrain_speed)
ax4.axhline(y=mean_speed, color='#0e7490', linestyle='--', alpha=0.8, label=f'Mean Throughput: ~{mean_speed:.0f} tok/s')
ax4.set_title('4. Hardware Execution Throughput (2x Tesla T4 GPUs)', fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('Pretraining Steps', fontsize=11)
ax4.set_ylabel('Tokens Processed / Second', fontsize=11)
ax4.set_ylim(4500, 7000)
ax4.legend(loc='lower right', frameon=True)
ax4.grid(True, linestyle='--', alpha=0.6)

fig.suptitle('Usaid AI (500M) — Empirical Pretraining & Alignment Telemetry Audit\nArchitected, Pretrained & SFT Aligned from Scratch by Mohamed Usaid', fontsize=15, fontweight='bold', y=0.95)

# Save high-resolution PNG
out_model_card = os.path.join("exported_usaid_ai_aligned", "training_telemetry.png")
out_root = "training_telemetry.png"
os.makedirs("exported_usaid_ai_aligned", exist_ok=True)

fig.savefig(out_model_card, dpi=300, bbox_inches='tight')
fig.savefig(out_root, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Generated successfully:\n  1. {out_model_card}\n  2. {out_root}")
