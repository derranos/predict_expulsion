# нормализация и сохранение данных для обучения, валидации и тестирования модели
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('./data/processed_data/studentInfo_MLP.csv')

target_column = 'final_result'
group_column = 'id_student'
scale_columns = [
    'highest_education',
    'imd_band',
    'age_band',
    'num_of_prev_attempts',
    'studied_credits',
    'date_registration',
    'final_score',
    'avg_score',
    'assessment_type_CMA',
    'assessment_type_TMA',
    'module_presentation_length',
    'clicks_collaboration',
    'clicks_content',
    'clicks_discussions',
    'clicks_interactive',
    'clicks_navigation',
    'clicks_tests',
]

X = data.drop(columns=[target_column, group_column])
y = data[target_column]
groups = data[group_column]

splitter = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=42,
)
folds = list(splitter.split(X, y, groups))
test_indices = [index for _, index in folds[:3]]
val_indices = [index for _, index in folds[3:6]]
train_indices = [index for _, index in folds[6:]]

train_indices = pd.Index([index for fold in train_indices for index in fold])
val_indices = pd.Index([index for fold in val_indices for index in fold])
test_indices = pd.Index([index for fold in test_indices for index in fold])

X_train, y_train = X.loc[train_indices].copy(), y.loc[train_indices].copy()
X_val, y_val = X.loc[val_indices].copy(), y.loc[val_indices].copy()
X_test, y_test = X.loc[test_indices].copy(), y.loc[test_indices].copy()

scaler = StandardScaler()
X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
X_val[scale_columns] = scaler.transform(X_val[scale_columns])
X_test[scale_columns] = scaler.transform(X_test[scale_columns])

for name, indices, labels in (
    ('train', train_indices, y_train),
    ('val', val_indices, y_val),
    ('test', test_indices, y_test),
):
    class_counts = labels.value_counts().sort_index()
    class_percentages = (labels.value_counts(normalize=True).sort_index() * 100)
    print(f'{name}: {len(indices)} строк')
    print(f'  классы: {class_counts.to_dict()}')
    print(f'  баланс классов (%): {class_percentages.round(2).to_dict()}')

train_students = set(groups.loc[train_indices])
val_students = set(groups.loc[val_indices])
test_students = set(groups.loc[test_indices])
assert not (train_students & val_students)
assert not (train_students & test_students)
assert not (val_students & test_students)
X_train.to_csv('./data/processed_data/X_train.csv', index=False)
X_val.to_csv('./data/processed_data/X_val.csv', index=False)
X_test.to_csv('./data/processed_data/X_test.csv', index=False)
y_train.to_csv('./data/processed_data/y_train.csv', index=False)
y_val.to_csv('./data/processed_data/y_val.csv', index=False)
y_test.to_csv('./data/processed_data/y_test.csv', index=False)