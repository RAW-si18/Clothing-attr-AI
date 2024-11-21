import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import timm
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from PIL import Image
import os
from tqdm import tqdm
import time
from datetime import datetime

warnings.filterwarnings('ignore')

class ProductDataset(Dataset):
    def __init__(self, df, img_dir, category_attrs, transform=None, is_test=False, attr_encoders=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.category_attrs = category_attrs
        self.attr_encoders = {} if attr_encoders is None else attr_encoders
        
        # Create category to attribute mapping
        self.category_to_attrs = {}
        for _, row in category_attrs.iterrows():
            self.category_to_attrs[row['Category']] = row['No_of_attribute']
        
        if not is_test and attr_encoders is None:
            print("Initializing attribute encoders...")
            for category, n_attrs in self.category_to_attrs.items():
                cat_data = df[df['Category'] == category]
                for i in range(n_attrs):
                    col = f'attr_{i+1}'
                    encoder_key = f"{category}_attr_{i+1}"
                    
                    # Get unique values and add unknown
                    values = cat_data[col].fillna('unknown').astype(str).unique()
                    values = np.append(values, 'unknown')
                    values = np.unique(values)  # Remove duplicates
                    
                    # Create and fit encoder
                    self.attr_encoders[encoder_key] = LabelEncoder()
                    self.attr_encoders[encoder_key].fit(values)
                    
                    # Print debug info
                    print(f"Encoder {encoder_key}: {len(values)} classes")
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Load and process image
            try:
                image_path = os.path.join(self.img_dir, f"{int(row['id']):06d}.jpg")
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    image = np.array(image)
                else:
                    print(f"Warning: Image not found at {image_path}")
                    image = np.ones((224, 224, 3), dtype=np.uint8) * 255
            except Exception as e:
                print(f"Error loading image for id {row['id']}: {str(e)}")
                image = np.ones((224, 224, 3), dtype=np.uint8) * 255
                
            if self.transform:
                try:
                    transformed = self.transform(image=image)
                    image = transformed['image']
                except Exception as e:
                    print(f"Error in transform for id {row['id']}: {str(e)}")
                    image = torch.ones((3, 224, 224))
            
            category = row['Category']
            n_attrs = self.category_to_attrs.get(category, 0)
            
            sample = {
                'image': image,
                'category': category,
                'id': row['id'],
                'n_attrs': n_attrs
            }
            
            if not self.is_test:
                attrs = []
                attr_lengths = []
                
                for i in range(n_attrs):
                    col = f'attr_{i+1}'
                    encoder_key = f"{category}_attr_{i+1}"
                    
                    # Handle missing values
                    value = str(row[col]) if pd.notna(row[col]) else 'unknown'
                    
                    try:
                        encoded_value = self.attr_encoders[encoder_key].transform([value])[0]
                    except Exception as e:
                        print(f"Error encoding value for {encoder_key}: {str(e)}")
                        encoded_value = self.attr_encoders[encoder_key].transform(['unknown'])[0]
                    
                    attrs.append(encoded_value)
                    attr_lengths.append(len(self.attr_encoders[encoder_key].classes_))
                
                sample['attributes'] = torch.tensor(attrs, dtype=torch.long)
                sample['attr_lengths'] = torch.tensor(attr_lengths, dtype=torch.long)
            
            return sample
        
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # Return a valid but empty sample
            return {
                'image': torch.zeros((3, 224, 224)),
                'category': 'unknown',
                'id': -1,
                'n_attrs': 0,
                'attributes': torch.tensor([], dtype=torch.long),
                'attr_lengths': torch.tensor([], dtype=torch.long)
            }

def custom_collate_fn(batch):
    """Modified collate function with additional error checking"""
    try:
        batch_size = len(batch)
        max_attrs = max(item['n_attrs'] for item in batch)
        
        images = torch.stack([item['image'] for item in batch])
        categories = [item['category'] for item in batch]
        ids = [item['id'] for item in batch]
        n_attrs = torch.tensor([item['n_attrs'] for item in batch])
        
        result = {
            'image': images,
            'category': categories,
            'id': ids,
            'n_attrs': n_attrs
        }
        
        if 'attributes' in batch[0]:
            # Initialize with -1 for invalid/padding values
            attributes = torch.full((batch_size, max_attrs), -1, dtype=torch.long)
            attr_lengths = torch.zeros((batch_size, max_attrs), dtype=torch.long)
            
            for i, item in enumerate(batch):
                n = item['n_attrs']
                if n > 0:  # Check if there are any attributes
                    attributes[i, :n] = item['attributes']
                    attr_lengths[i, :n] = item['attr_lengths']
            
            # Verify data integrity
            assert (attributes >= -1).all(), "Invalid negative values in attributes"
            assert (attr_lengths >= 0).all(), "Invalid negative values in attr_lengths"
            
            result['attributes'] = attributes
            result['attr_lengths'] = attr_lengths
        
        return result
    
    except Exception as e:
        print(f"Error in collate function: {str(e)}")
        raise

class ProductAttributeModel(nn.Module):
    def __init__(self, category_attrs, dataset):
        super().__init__()
        self.category_attrs = category_attrs
        self.dataset = dataset
        
        self.backbone = timm.create_model('efficientnet_b5', pretrained=True)
        feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 1024),  # Increased hidden layer size
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),  # Slightly increased dropout
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )
        
        # Store the maximum number of classes for each attribute position
        self.max_classes_per_position = {}
        self.heads = nn.ModuleDict()
        
        for category, n_attrs in dataset.category_to_attrs.items():
            category_heads = nn.ModuleList()
            for i in range(n_attrs):
                encoder_key = f"{category}_attr_{i+1}"
                num_classes = len(dataset.attr_encoders[encoder_key].classes_)
                
                # Update max classes for this position
                if i not in self.max_classes_per_position:
                    self.max_classes_per_position[i] = num_classes
                else:
                    self.max_classes_per_position[i] = max(
                        self.max_classes_per_position[i], 
                        num_classes
                    )
                head = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.GELU(),
                    nn.LayerNorm(512),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.LayerNorm(256),
                    nn.Dropout(0.1),
                    nn.Linear(256, num_classes)
                )
                category_heads.append(head)
            self.heads[category] = category_heads

    @torch.cuda.amp.autocast()
    def forward(self, x, categories, n_attrs):
        features = self.backbone(x)
        features = self.feature_processor(features)
        batch_size = len(categories)
        
        # Group samples by category
        category_indices = {}
        for i, category in enumerate(categories):
            if category not in category_indices:
                category_indices[category] = []
            category_indices[category].append(i)
        
        max_attrs = max(n_attrs).item()
        outputs = []
        
        for attr_idx in range(max_attrs):
            if attr_idx not in self.max_classes_per_position:
                outputs.append(None)
                continue
                
            max_classes = self.max_classes_per_position[attr_idx]
            attr_outputs = []
            
            for category, indices in category_indices.items():
                if attr_idx < len(self.heads[category]):
                    cat_features = features[indices]
                    head = self.heads[category][attr_idx]
                    cat_output = head(cat_features)
                    
                    # Pad output if necessary to match max_classes
                    if cat_output.shape[1] < max_classes:
                        padding = torch.full(
                            (cat_output.shape[0], max_classes - cat_output.shape[1]),
                            float('-inf'),
                            device=cat_output.device,
                            dtype=cat_output.dtype
                        )
                        cat_output = torch.cat([cat_output, padding], dim=1)
                    
                    attr_outputs.append((indices, cat_output))
            
            if attr_outputs:
                # Create combined tensor with proper size
                combined = torch.full(
                    (batch_size, max_classes),
                    float('-inf'),
                    device=x.device,
                    dtype=attr_outputs[0][1].dtype
                )
                
                for indices, output in attr_outputs:
                    combined[indices, :output.shape[1]] = output[:, :output.shape[1]]
                
                outputs.append(combined)
            else:
                outputs.append(None)
        
        return outputs

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, config):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["max_epochs"]} [Train]')
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            images = batch['image'].to(device)
            attributes = batch['attributes'].to(device)
            categories = batch['category']
            n_attrs = batch['n_attrs']
            attr_lengths = batch['attr_lengths'].to(device)
            
            # Zero gradients at the start
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                outputs = model(images, categories, n_attrs)
                batch_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                valid_predictions = 0
                
                # Calculate loss for each attribute
                for i, output in enumerate(outputs):
                    if output is not None:
                        current_attr_length = attr_lengths[:, i]
                        valid_mask = (attributes[:, i] >= 0) & (attributes[:, i] < current_attr_length)
                        
                        if valid_mask.any():
                            valid_outputs = output[valid_mask]
                            valid_targets = attributes[valid_mask, i]
                            num_classes = current_attr_length[valid_mask][0].item()
                            
                            if torch.all((valid_targets >= 0) & (valid_targets < num_classes)):
                                valid_outputs = valid_outputs[:, :num_classes].float()
                                valid_targets = valid_targets.long()
                                
                                attr_loss = criterion(valid_outputs, valid_targets)
                                if torch.isfinite(attr_loss).all():
                                    batch_loss += attr_loss.mean()
                                    valid_predictions += 1
                
                if valid_predictions > 0:
                    batch_loss = batch_loss / valid_predictions
            
            # Skip batch if loss is invalid
            if not torch.isfinite(batch_loss):
                continue
                
            # Backward pass with gradient scaling
            scaler.scale(batch_loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {str(e)}")
            continue
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / max(1, num_batches)
    
    return avg_loss, epoch_time

def validate(model, val_loader, device, epoch, config):
    model.eval()
    all_preds = []
    all_labels = []
    category_scores = {}
    
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["max_epochs"]} [Val]')
    val_start_time = time.time()

    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)
            attributes = batch['attributes']
            categories = batch['category']
            n_attrs = batch['n_attrs'].to(device)

            outputs = model(images, categories, n_attrs)

            max_attrs = max(n_attrs).item()
            for i in range(max_attrs):
                attr_mask = attributes[:, i] != -1
                if attr_mask.sum() > 0:
                    valid_outputs = outputs[i][attr_mask] if i < len(outputs) else None
                    valid_attributes = attributes[attr_mask, i]

                    if valid_outputs is not None and valid_outputs.shape[0] == valid_attributes.shape[0]:
                        preds = valid_outputs.argmax(dim=1).cpu()
                        all_preds.extend(preds.numpy())
                        all_labels.extend(valid_attributes.numpy())

                        category = categories[0]
                        if category not in category_scores:
                            category_scores[category] = []
                        f1 = f1_score(valid_attributes.numpy(), preds.numpy(), average='macro')
                        category_scores[category].append(f1)

    val_time = time.time() - val_start_time
    
    if len(all_preds) > 0:
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        harmonic_mean = 2 * (micro_f1 * macro_f1) / (micro_f1 + macro_f1)
        overall_score = sum(np.mean(scores) for scores in category_scores.values()) / len(category_scores)

        metrics = {
            'Micro F1': micro_f1,
            'Macro F1': macro_f1,
            'Harmonic Mean': harmonic_mean,
            'Overall Score': overall_score,
            'Time': val_time
        }

        return metrics

    return {
        'Micro F1': 0.0,
        'Macro F1': 0.0,
        'Harmonic Mean': 0.0,
        'Overall Score': 0.0,
        'Time': val_time
    }

def main():
    # Set default dtype to float32
    torch.set_default_dtype(torch.float32)
    
    # Memory optimization settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    config = {
        'batch_size': 32,  # Increased batch size
        'num_workers': 4,
        'learning_rate': 1e-4,  # Reduced learning rate
        'patience': 7,
        'max_epochs': 50,  # More epochs
        'train_val_split': 0.85,
        'image_size': 256  # Larger image size
    }
    
    train_transform = A.Compose([
        A.RandomResizedCrop(config['image_size'], config['image_size'], scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(config['image_size'], config['image_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Data loading
    BASE_PATH = '/kaggle/input/visual-taxonomy'
    TRAIN_PATH = os.path.join(BASE_PATH, 'train_images')
    TEST_PATH = os.path.join(BASE_PATH, 'test_images')
    
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    category_attrs = pd.read_parquet(os.path.join(BASE_PATH, 'category_attributes.parquet'))
    
    full_train_dataset = ProductDataset(
        train_df,
        TRAIN_PATH,
        category_attrs,
        transform=train_transform
    )
    
    train_size = int(config['train_val_split'] * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProductAttributeModel(category_attrs, full_train_dataset).to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(reduction='none')
    # Use AdamW with more aggressive weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.05,  # Increased weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # More sophisticated LR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['max_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Longer warm-up
        div_factor=10,  # Wider learning rate range
        final_div_factor=1e4  # Slower final decay
    )
    
    # Initialize gradient scaler
    scaler = GradScaler()
    
    # Create output directory for models and logs
    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('outputs', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Training loop
    best_val_score = 0
    patience_counter = 0
    best_model_path = os.path.join(run_dir, "best_model.pth")
    history = []
    
    print(f"\nStarting training at {timestamp}")
    print(f"Model checkpoints and logs will be saved to: {run_dir}")
    print(f"Training on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    try:
        for epoch in range(config['max_epochs']):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_time = train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, config)
            
            # Validation phase
            val_metrics = validate(model, val_loader, device, epoch, config)
            val_score = val_metrics['Harmonic Mean']
            
            # Update learning rate
            scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_time': train_time,
                'val_time': val_metrics['Time'],
                'total_time': epoch_time,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **{k: v for k, v in val_metrics.items() if k != 'Time'}
            }
            history.append(metrics)
            
            # Save metrics to CSV
            pd.DataFrame(history).to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{config['max_epochs']} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s")
            print("Validation Metrics:")
            for metric, value in val_metrics.items():
                if metric != 'Time':
                    print(f"  {metric}: {value:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                print(f"New best model! Score: {val_score:.4f}")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_score': best_val_score,
                    'config': config,
                    'history': history
                }, best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"\nEarly stopping triggered! Best validation score: {best_val_score:.4f}")
                    break
            
            # Save backup checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                backup_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_score': best_val_score,
                    'config': config,
                    'history': history
                }, backup_path)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Save final model state
        final_path = os.path.join(run_dir, "final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'config': config
        }, final_path)
        print(f"\nFinal model saved to {final_path}")
        
        # Plot training history
        try:
            import matplotlib.pyplot as plt
            
            # Plot loss
            plt.figure(figsize=(10, 6))
            plt.plot([x['train_loss'] for x in history], label='Train Loss')
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
            plt.close()
            
            # Plot metrics
            plt.figure(figsize=(12, 8))
            for metric in ['Micro F1', 'Macro F1', 'Harmonic Mean', 'Overall Score']:
                plt.plot([x[metric] for x in history], label=metric)
            plt.title('Validation Metrics Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(os.path.join(run_dir, 'metrics_plot.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error creating plots: {str(e)}")

def predict_test(model, test_loader, device):
    model.eval()
    predictions = []
    
    # Create progress bar for test predictions
    progress_bar = tqdm(test_loader, desc='Generating predictions')
    
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)
            categories = batch['category']
            ids = batch['id']
            n_attrs = batch['n_attrs'].to(device)
            
            outputs = model(images, categories, n_attrs)
            
            for b in range(len(categories)):
                category = categories[b]
                id_ = ids[b]
                n_attr = n_attrs[b].item()
                
                batch_preds = []
                for i in range(n_attr):
                    pred_idx = outputs[i][b].argmax().cpu().item()
                    encoder_key = f"{category}_attr_{i+1}"
                    pred_value = model.dataset.attr_encoders[encoder_key].inverse_transform([pred_idx])[0]
                    batch_preds.append(pred_value)
                
                while len(batch_preds) < 10:
                    batch_preds.append('dummy_value')
                
                predictions.append({
                    'id': id_,
                    'Category': category,
                    **{f'attr_{i+1}': pred for i, pred in enumerate(batch_preds)}
                })
    
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        
        main()
    except Exception as e:
        print(f"Program terminated with error: {str(e)}")
