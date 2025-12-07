"""
Utility functions for model evaluation and metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path


class MetricsCalculator:
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        per_class_metrics = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        if y_pred_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true_bin, y_pred_proba, 
                    average='macro', 
                    multi_class='ovr'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true_bin, y_pred_proba, 
                    average='weighted', 
                    multi_class='ovr'
                )
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
        
        return metrics, per_class_metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path, title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def plot_normalized_confusion_matrix(self, y_true, y_pred, save_path, 
                                        title='Normalized Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_classification_report(self, per_class_metrics, save_path, 
                                  title='Per-Class Performance'):
        classes = [c for c in self.class_names if c in per_class_metrics]
        metrics_data = {
            'Precision': [per_class_metrics[c]['precision'] for c in classes],
            'Recall': [per_class_metrics[c]['recall'] for c in classes],
            'F1-Score': [per_class_metrics[c]['f1-score'] for c in classes]
        }
        
        df = pd.DataFrame(metrics_data, index=classes)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        df.plot(kind='bar', ax=ax, width=0.8)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Attack Class', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_path, title='ROC Curves'):
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, history, save_path, title='Training History'):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(alpha=0.3)
        
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics_to_json(self, metrics, per_class_metrics, save_path):
        output = {
            'overall_metrics': metrics,
            'per_class_metrics': per_class_metrics
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)
    
    def create_summary_report(self, metrics, per_class_metrics, save_path):
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MULTI-CLASS INTRUSION DETECTION SYSTEM - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append("OVERALL METRICS:")
        report_lines.append("-" * 80)
        report_lines.append(f"Accuracy:                {metrics['accuracy']:.4f}")
        report_lines.append(f"Precision (Macro):       {metrics['precision_macro']:.4f}")
        report_lines.append(f"Precision (Weighted):    {metrics['precision_weighted']:.4f}")
        report_lines.append(f"Recall (Macro):          {metrics['recall_macro']:.4f}")
        report_lines.append(f"Recall (Weighted):       {metrics['recall_weighted']:.4f}")
        report_lines.append(f"F1-Score (Macro):        {metrics['f1_macro']:.4f}")
        report_lines.append(f"F1-Score (Weighted):     {metrics['f1_weighted']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            report_lines.append(f"ROC AUC (Macro):         {metrics['roc_auc_macro']:.4f}")
            report_lines.append(f"ROC AUC (Weighted):      {metrics['roc_auc_weighted']:.4f}")
        
        report_lines.append("")
        report_lines.append("PER-CLASS METRICS:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        report_lines.append("-" * 80)
        
        for class_name in self.class_names:
            if class_name in per_class_metrics:
                metrics_dict = per_class_metrics[class_name]
                report_lines.append(
                    f"{class_name:<15} "
                    f"{metrics_dict['precision']:<12.4f} "
                    f"{metrics_dict['recall']:<12.4f} "
                    f"{metrics_dict['f1-score']:<12.4f} "
                    f"{metrics_dict['support']:<10.0f}"
                )
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text


def calculate_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    
    return class_weight_dict
