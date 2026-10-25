# подготовка данных для MLP модели
import numpy as np
import pandas as pd


def prepare_student_info():
    student_info = pd.read_csv('./data/raw_data/studentInfo.csv')

    student_info['gender'] = student_info['gender'].apply(
        lambda x: int(x == 'M')
    )
    student_info['highest_education'] = student_info['highest_education'].map(
        {
            'No Formal quals': 0,
            'Lower Than A Level': 1,
            'A Level or Equivalent': 2,
            'HE Qualification': 3,
            'Post Graduate Qualification': 4,
        }
    )
    student_info['age_band'] = student_info['age_band'].map(
        {
            '0-35': 0,
            '35-55': 1,
            '55<=': 2,
        }
    )
    student_info['disability'] = student_info['disability'].apply(
        lambda x: int(x == 'Y')
    )
    student_info['final_result'] = student_info['final_result'].apply(
        lambda x: int(x == 'Withdrawn' or x == 'Fail')
    )
    student_info = pd.get_dummies(
        student_info,
        columns=['region'],
        dtype=int,
    )
    student_info['imd_band'] = student_info['imd_band'].map(
        {
            '0-10%': 0.0,
            '10-20': 1.0,
            '20-30%': 2.0,
            '30-40%': 3.0,
            '40-50%': 4.0,
            '?': 4.5,
            '50-60%': 5.0,
            '60-70%': 6.0,
            '70-80%': 7.0,
            '80-90%': 8.0,
            '90-100%': 9.0,
        }
    )
    print(student_info.head())

    missing_counts = student_info.isna().sum()
    student_info = student_info.dropna(subset=['imd_band'])
    print(student_info.shape)

    return student_info


def add_registration_data(student_info):
    student_registration = pd.read_csv(
        './data/raw_data/studentRegistration.csv'
    )
    student_info = student_info.merge(
        student_registration,
        on=['code_module', 'code_presentation', 'id_student'],
        how='left',
    )

    # пустых значений нет
    student_info['date_unregistration'] = student_info[
        'date_unregistration'
    ].apply(lambda x: int(x) if x != '?' else 28)
    student_info['date_registration'] = student_info[
        'date_registration'
    ].apply(lambda x: int(x) if x != '?' else 0)
    student_info = student_info[
        (student_info['date_unregistration'].astype(int) >= 28)
        & (student_info['date_registration'].astype(int) < 28)
    ]
    student_info = student_info.drop(columns=['date_unregistration'])
    print(student_info.shape)

    return student_info


def prepare_assessments():
    student_assessment = pd.read_csv(
        './data/raw_data/studentAssessment.csv'
    )
    assessments = pd.read_csv('./data/raw_data/assessments.csv')

    student_assessment = student_assessment[
        student_assessment['date_submitted'] < 28
    ]
    student_assessment = student_assessment.drop(
        columns=['date_submitted', 'is_banked']
    )
    assessments = assessments.drop(columns=['date'])
    assessments = assessments.merge(
        student_assessment,
        on=['id_assessment'],
        how='inner',
    )
    assessments = pd.get_dummies(
        assessments,
        columns=['assessment_type'],
        dtype=int,
    )
    assessments = assessments.drop(columns=['id_assessment'])
    assessments['score'] = assessments['score'].apply(
        lambda x: int(x) if x != '?' else 0
    )
    print(assessments.head())
    print(assessments.describe())

    assessments['weighted_score'] = (
        assessments['score'] * assessments['weight'] / 100
    )
    assessments = (
        assessments.groupby(
            ['code_module', 'code_presentation', 'id_student']
        )
        .agg(
            final_score=('weighted_score', 'sum'),
            avg_score=('score', 'mean'),
            assessment_type_CMA=('assessment_type_CMA', 'sum'),
            assessment_type_TMA=('assessment_type_TMA', 'sum'),
        )
        .reset_index()
    )
    print(assessments.head())

    return assessments


def add_assessment_features(student_info, assessments):
    student_info = student_info.merge(
        assessments,
        on=['code_module', 'code_presentation', 'id_student'],
        how='left',
    )
    print(student_info.shape)
    student_info = student_info.fillna(0)
    print(student_info.info())

    courses = pd.read_csv('./data/raw_data/courses.csv')
    student_info = student_info.merge(
        courses,
        on=['code_module', 'code_presentation'],
        how='left',
    )

    return student_info


def prepare_activity_features():
    vle = pd.read_csv('./data/raw_data/vle.csv')
    vle = vle.drop(columns=['week_from', 'week_to'])
    resource_groups = {
        # Контент
        'oucontent': 'content',
        'resource': 'content',
        'subpage': 'content',
        'page': 'content',
        'url': 'content',
        'htmlactivity': 'content',
        'sharedsubpage': 'content',
        'folder': 'content',
        # Тесты
        'quiz': 'tests',
        'externalquiz': 'tests',
        'questionnaire': 'tests',
        # Обсуждения
        'forumng': 'discussions',
        # Совместная работа
        'ouwiki': 'collaboration',
        'oucollaborate': 'collaboration',
        'ouelluminate': 'collaboration',
        # Навигация
        'homepage': 'navigation',
        'glossary': 'navigation',
        'dualpane': 'navigation',
        # Интерактив
        'dataplus': 'interactive',
        'repeatactivity': 'interactive',
    }
    vle['activity_type'] = vle['activity_type'].map(resource_groups)
    vle = pd.get_dummies(vle, columns=['activity_type'], dtype=int)
    print(vle.head())
    print(vle.describe())

    student_vle = pd.read_csv('./data/raw_data/studentVle.csv')
    student_vle = student_vle.merge(
        vle,
        on=['code_module', 'code_presentation', 'id_site'],
        how='left',
    )
    student_vle = student_vle.drop(columns=['id_site'])
    student_vle = student_vle[student_vle['date'] < 28]
    student_vle = student_vle.drop(columns=['date'])
    print(student_vle.head())

    type_cols = [
        'activity_type_collaboration',
        'activity_type_content',
        'activity_type_discussions',
        'activity_type_interactive',
        'activity_type_navigation',
        'activity_type_tests',
    ]
    grouped = (
        student_vle.groupby(
            ['code_module', 'code_presentation', 'id_student']
        )
        .apply(
            lambda g: pd.Series(
                {
                    col.replace('activity_type_', 'clicks_'): np.log1p(
                        (g['sum_click'] * g[col]).sum()
                    )
                    for col in type_cols
                }
            )
        )
        .reset_index()
    )
    print(grouped.head())
    print(grouped.describe())

    return grouped


def build_mlp_dataset():
    student_info = prepare_student_info()
    student_info = add_registration_data(student_info)

    assessments = prepare_assessments()
    student_info = add_assessment_features(student_info, assessments)

    grouped_activity = prepare_activity_features()
    student_info = student_info.merge(
        grouped_activity,
        on=['code_module', 'code_presentation', 'id_student'],
        how='left',
    )
    student_info = student_info.fillna(0)
    student_info['code_presentation'] = student_info['code_presentation'].apply(
        lambda x: 0 if x[-1] == 'J' else 1
    )
    student_info = pd.get_dummies(
        student_info,
        columns=['code_module'],
        dtype=int,
    )
    print(student_info.isna().sum())
    student_info.to_csv(
        './data/processed_data/studentInfo_MLP.csv',
        index=False,
    )

    return student_info


studentInfo = build_mlp_dataset()
