import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==========================================
# BALANCED SETTINGS (Speed + Accuracy)
# ==========================================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ==========================================
# 1. BALANCED ARCHITECTURE (Fixed)
# ==========================================

class BalancedUNet(nn.Module):
    """Balanced U-Net - Good accuracy with reasonable speed"""
    def __init__(self):
        super(BalancedUNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Balanced channels
        self.enc1 = conv_block(3, 48)
        self.enc2 = conv_block(48, 96)
        self.enc3 = conv_block(96, 192)
        self.enc4 = conv_block(192, 384)
        self.pool = nn.MaxPool2d(2)
        
        self.up3 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.dec3 = conv_block(384, 192)
        self.up2 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec2 = conv_block(192, 96)
        self.up1 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec1 = conv_block(96, 48)
        self.final = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        
        up_s4 = self.up3(s4)
        if up_s4.shape != s3.shape:
            up_s4 = F.interpolate(up_s4, size=s3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up_s4, s3], dim=1))
        
        up_d3 = self.up2(d3)
        if up_d3.shape != s2.shape:
            up_d3 = F.interpolate(up_d3, size=s2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up_d3, s2], dim=1))
        
        up_d2 = self.up1(d2)
        if up_d2.shape != s1.shape:
            up_d2 = F.interpolate(up_d2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up_d2, s1], dim=1))
        
        return torch.sigmoid(self.final(d1))

class BalancedCBAMAttention(nn.Module):
    """Balanced CBAM attention"""
    def __init__(self, channels, reduction=16):
        super(BalancedCBAMAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att

class BalancedHybridSystem(nn.Module):
    def __init__(self):
        super(BalancedHybridSystem, self).__init__()
        self.unet = BalancedUNet()
        
        # Use EfficientNet-B3 but get proper feature dimensions
        # We'll use it as a feature extractor without the classifier
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
        
        # Remove the classifier head
        self.backbone.reset_classifier(0)
        
        # Get the feature dimension from the backbone
        # For EfficientNet-B3, the final conv layer outputs 1536 channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            features = self.backbone.forward_features(dummy)
            self.feature_dim = features.shape[1]
            print(f"Detected feature dimension: {self.feature_dim}")
        
        self.attention = BalancedCBAMAttention(self.feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Balanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Linear(384, 2)
        self.stage2 = nn.Linear(384, 3)
        self.stage3 = nn.Linear(384, 2)

    def forward(self, x):
        mask = self.unet(x)
        segmented_x = x * mask
        
        # Extract features using forward_features method
        features = self.backbone.forward_features(segmented_x)
        
        attended = self.attention(features)
        pooled = self.pool(attended).flatten(1)
        
        shared = self.classifier(pooled)
        return self.stage1(shared), self.stage2(shared), self.stage3(shared), mask

# ==========================================
# 2. OPTIMIZED DATASET
# ==========================================

class LungDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        categories = ['Normal', 'Corona Virus Disease', 'Tuberculosis', 'Bacterial Pneumonia', 'Viral Pneumonia']
        for idx, cat in enumerate(categories):
            path = os.path.join(root_dir, cat)
            if not os.path.exists(path): 
                continue
                
            for img in os.listdir(path):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, img)
                    if idx == 0:
                        self.samples.append((img_path, 0, -1, -1))
                    elif idx == 1:
                        self.samples.append((img_path, 1, 0, -1))
                    elif idx == 2:
                        self.samples.append((img_path, 1, 1, -1))
                    elif idx == 3:
                        self.samples.append((img_path, 1, 2, 0))
                    elif idx == 4:
                        self.samples.append((img_path, 1, 2, 1))

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, s1, s2, s3 = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, s1, s2, s3

# Balanced resolution
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. EVALUATION
# ==========================================

@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    res = {'s1': {'p': [], 't': []}, 's2': {'p': [], 't': []}, 's3': {'p': [], 't': []}}
    
    for imgs, s1, s2, s3 in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device, non_blocking=True)
        o1, o2, o3, _ = model(imgs)
        
        p1 = torch.argmax(o1, 1).cpu().numpy()
        t1 = s1.numpy()
        res['s1']['p'].extend(p1)
        res['s1']['t'].extend(t1)
        
        mask2 = (s1 == 1)
        if mask2.any():
            res['s2']['p'].extend(torch.argmax(o2[mask2], 1).cpu().numpy())
            res['s2']['t'].extend(s2[mask2].numpy())
        
        mask3 = (s2 == 2)
        if mask3.any():
            res['s3']['p'].extend(torch.argmax(o3[mask3], 1).cpu().numpy())
            res['s3']['t'].extend(s3[mask3].numpy())
        
        del imgs, o1, o2, o3

    stages = [('s1', 'Normal/Abnormal', ['Normal', 'Abnormal']),
              ('s2', 'Disease Type', ['COVID', 'TB', 'Pneumonia']),
              ('s3', 'Pneumonia Type', ['Bacterial', 'Viral'])]
    
    results = {}
    for k, title, names in stages:
        if len(res[k]['t']) > 0:
            acc = accuracy_score(res[k]['t'], res[k]['p'])
            results[title] = acc
            print(f"\n{'='*30}\n{title} | Accuracy: {acc:.2%}\n{'='*30}")
            print(classification_report(res[k]['t'], res[k]['p'], target_names=names))
            
            cm = confusion_matrix(res[k]['t'], res[k]['p'])
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
            plt.title(f'Confusion Matrix - {title}')
            plt.show()
    
    return results

# ==========================================
# 4. TRAINING
# ==========================================

def train_epoch(model, loader, optimizer, device, s3_weight, scaler, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for imgs, s1, s2, s3 in pbar:
        imgs = imgs.to(device, non_blocking=True)
        s1 = s1.to(device, non_blocking=True)
        s2 = s2.to(device, non_blocking=True)
        s3 = s3.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            o1, o2, o3, _ = model(imgs)
            
            l1 = F.cross_entropy(o1, s1)
            
            m2 = (s1 == 1)
            l2 = F.cross_entropy(o2[m2], s2[m2]) if m2.any() else 0
            
            m3 = (s2 == 2)
            l3 = F.cross_entropy(o3[m3], s3[m3], weight=s3_weight) if m3.any() else 0
            
            # Progressive weighting
            if epoch < 5:
                loss = l1 + l2 + l3 * 0.5
            elif epoch < 10:
                loss = l1 + l2 + l3 * 0.8
            else:
                loss = l1 + l2 + l3
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        del imgs, s1, s2, s3, o1, o2, o3, loss
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for imgs, s1, _, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            s1 = s1.to(device, non_blocking=True)
            o1, _, _, _ = model(imgs)
            correct += (torch.argmax(o1, 1) == s1).sum().item()
            total += s1.size(0)
            del imgs, o1
    
    return correct / total

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    KAGGE_ROOT = "/kaggle/input/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types/Lung Disease Dataset"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'balanced_hybrid_best.pth'
    
    print(f"Using device: {device}")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading datasets...")
    
    train_dataset = LungDataset(os.path.join(KAGGE_ROOT, 'train'), TRAIN_TRANSFORM)
    val_dataset = LungDataset(os.path.join(KAGGE_ROOT, 'val'), VAL_TRANSFORM)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=12,
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=12,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    model = BalancedHybridSystem().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')  # Fixed deprecated warning
    
    s3_weight = torch.tensor([1.5, 1.0]).to(device)
    
    best_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(20):
        print(f"\n{'='*50}")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        train_loss = train_epoch(model, train_loader, optimizer, device, s3_weight, scaler, epoch)
        val_acc = validate(model, val_loader, device)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.2%}, LR: {current_lr:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_save_path)
            print(f"⭐ Best model saved with {best_acc:.2%} accuracy")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 5 and current_lr < 1e-6:
            print("Early stopping triggered")
            break
    
    print("\n🚀 Loading Best Model for Final evaluation...")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    results = run_evaluation(model, val_loader, device)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS:")
    for task, acc in results.items():
        print(f"{task}: {acc:.2%}")

if __name__ == "__main__":
    main()
