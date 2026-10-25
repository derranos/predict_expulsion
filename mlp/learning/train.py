# подбор гиперпараметров, вычисления метрик и сохранение модели 
import json
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlp.model.mlp import MLP, train_model


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data' / 'processed_data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 80
N_TRIALS = 40


PARAMETER_OPTIONS = {
    'hidden_dim': [32, 64, 128, 256],
    'n_layers': [2, 3, 4, 5, 6, 7, 8],
    'dropout': [0.0, 0.2, 0.5],
    'lr': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128, 256],
}


def load_dataset(split):
    features = pd.read_csv(DATA_DIR / f'X_{split}.csv')
    labels = pd.read_csv(DATA_DIR / f'y_{split}.csv').squeeze('columns')
    return TensorDataset(
        torch.tensor(features.to_numpy(), dtype=torch.float32),
        # crossentropy ожидает целочисленные метки классов
        torch.tensor(labels.to_numpy(), dtype=torch.long),
    )


def objective(trial, train_data, val_data):
    torch.manual_seed(42)
    params = {
        name: trial.suggest_categorical(name, values)
        for name, values in PARAMETER_OPTIONS.items()
    }
    train_loader = DataLoader(
        train_data,
        batch_size=params['batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(val_data, batch_size=params['batch_size'])
    model = MLP(
        input_dim=train_data.tensors[0].shape[1],
        hidden_dim=params['hidden_dim'],
        n_layers=params['n_layers'],
        dropout=params['dropout'],
    )
    _, val_loss = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=params['lr'],
        device=DEVICE,
        trial=trial,
    )
    return val_loss


def evaluate(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    labels_all = []
    risk_scores = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            labels_all.append(labels.cpu())
            # вероятность класса 1 используется как оценка риска отчисления
            risk_scores.append(torch.softmax(logits, dim=1)[:, 1].cpu())

    labels_all = torch.cat(labels_all)
    risk_scores = torch.cat(risk_scores)
    positive_count = labels_all.sum().item()
    negative_count = total - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError('ROC-AUC requires both classes in the test data.')

    # берём верхнюю пятую часть студентов по прогнозируемому риску
    top_count = max(1, (total + 4) // 5)
    top_indices = torch.topk(risk_scores, k=top_count).indices
    recall_at_top_20 = labels_all[top_indices].sum().item() / positive_count

    sorted_scores, sorted_indices = torch.sort(risk_scores, descending=True)
    sorted_labels = labels_all[sorted_indices].float()
    # одинаковые оценки риска относятся к одному порогу ROC-кривой
    _, score_groups = torch.unique_consecutive(
        sorted_scores,
        return_inverse=True,
    )
    true_positives = torch.zeros(
        score_groups.max().item() + 1,
        dtype=torch.float32,
    ).scatter_add_(0, score_groups, sorted_labels)
    group_sizes = torch.bincount(score_groups)
    false_positives = group_sizes - true_positives
    cumulative_true_positives = true_positives.cumsum(dim=0)
    cumulative_false_positives = false_positives.cumsum(dim=0)
    true_positive_rate = torch.cat(
        (torch.zeros(1), cumulative_true_positives / positive_count)
    )
    false_positive_rate = torch.cat(
        (torch.zeros(1), cumulative_false_positives / negative_count)
    )
    # площадь под ROC-кривой считается методом трапеций
    roc_auc = torch.trapezoid(true_positive_rate, false_positive_rate).item()

    return total_loss / total, correct / total, roc_auc, recall_at_top_20


def main():
    print(f'Устройство: {DEVICE}')
    print('Загрузка данных')
    train_data = load_dataset('train')
    val_data = load_dataset('val')
    test_data = load_dataset('test')
    print('Данные загружены')

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    print(f'Подбор гиперпараметров: {N_TRIALS} trials')
    study.optimize(
        lambda trial: objective(trial, train_data, val_data),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f'Лучшие параметры: {best_params}')
    print(f'Лучший validation loss: {study.best_value:.4f}')

    torch.manual_seed(42)
    train_loader = DataLoader(
        train_data,
        batch_size=best_params['batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(val_data, batch_size=best_params['batch_size'])
    test_loader = DataLoader(test_data, batch_size=best_params['batch_size'])
    model = MLP(
        input_dim=train_data.tensors[0].shape[1],
        hidden_dim=best_params['hidden_dim'],
        n_layers=best_params['n_layers'],
        dropout=best_params['dropout'],
    )
    print('Финальное обучение модели')
    model, val_loss = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=best_params['lr'],
        device=DEVICE,
    )

    print('Оценка модели')
    test_loss, test_accuracy, test_roc_auc, test_recall_at_top_20 = evaluate(
        model,
        test_loader,
    )
    checkpoint_path = Path(__file__).with_name('mlp_best.pth')
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_dim': train_data.tensors[0].shape[1],
            'params': best_params,
            'best_val_loss': val_loss,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_roc_auc': test_roc_auc,
            'test_recall_at_top_20': test_recall_at_top_20,
        },
        checkpoint_path,
    )
    metrics_path = Path(__file__).with_name('mlp_metrics.json')
    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        json.dump(
            {
                'best_params': best_params,
                'best_validation_loss': val_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'test_roc_auc': test_roc_auc,
                'test_recall_at_top_20': test_recall_at_top_20,
            },
            metrics_file,
            indent=2,
        )

    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test ROC-AUC: {test_roc_auc:.4f}')
    print(f'Test Recall@Top20%: {test_recall_at_top_20:.4f}')
    print(f'Модель сохранена: {checkpoint_path}')
    print(f'Метрики сохранены: {metrics_path}')


if __name__ == '__main__':
    main()
