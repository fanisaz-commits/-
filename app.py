import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ===================================================================
# КОНФИГУРАЦИЯ И НАУЧНО ОБОСНОВАННЫЕ РЕФЕРЕНСЫ
# ===================================================================
CONFIG = {
    'THRESHOLDS': {
        'sbp_critical': 250,          # Löllgen: абсолютное прекращение
        'sbp_warning': 220,            # Löllgen: чрезмерный подъём
        'hr_rec_min': 12,              # Löllgen: снижение ЧСС за 1 мин < 12 — плохой прогноз
        'chronotropic_ratio': 0.85,    # Löllgen: < 85% от макс. ЧСС — недостаточность
    },
    'WEIGHTS': {
        'vo2': 0.4,       # Вес VO2peak в индексе
        'power': 0.3,     # Вес мощности (LT2)
        'recovery': 0.2,  # Вес восстановления
        'o2_pulse': 0.1   # Вес кислородного пульса
    }
}

# Референсы для VO2peak (Rapp et al. 2018)
RAPP_VO2_MEDIANS = {
    'Male': {(18,29): 42.0, (30,39): 38.0, (40,49): 36.0, (50,59): 32.0, (60,69): 28.0},
    'Female': {(18,29): 34.0, (30,39): 31.0, (40,49): 30.0, (50,59): 26.0, (60,69): 23.0}
}
RAPP_VO2_SD = {'Male': 7.7, 'Female': 6.7}

# Референсы для мощности на пороге LT2 (Fiedler et al. 2025)
FIEDLER_LT2_MEDIANS = {
    'Male': {(14,24): 1.91, (25,34): 1.53, (35,44): 1.47, (45,54): 1.42, (55,64): 1.36},
    'Female': {(14,24): 1.65, (25,34): 1.40, (35,44): 1.36, (45,54): 1.29, (55,64): 1.19}
}
FIEDLER_LT2_SD = {'Male': 0.35, 'Female': 0.30}

# ===================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===================================================================
def get_ref_median(ref_dict, sex, age):
    gender_data = ref_dict.get(sex, {})
    for (age_min, age_max), val in gender_data.items():
        if age_min <= age <= age_max: return val
    return list(gender_data.values())[0] if age < 20 else list(gender_data.values())[-1]

def calculate_percentile_score(value, median, sd):
    """Рассчитывает балл (0-100) на основе нормального распределения."""
    z = (value - median) / sd
    score = norm.cdf(z) * 100
    return round(score, 1)

# ===================================================================
# ЛОГИКА АНАЛИЗА
# ===================================================================
class StressTestAnalyzer:
    @staticmethod
    def calculate(data):
        # 1. Расчет VO2peak (ACSM)
        if data['type'] == 'bike':
            vo2_rel = (10.8 * data['power'] / data['weight']) + 7.0
        else: # treadmill
            speed_m_min = (data['power'] * 1000) / 60
            vo2_rel = (0.2 * speed_m_min) + (0.9 * speed_m_min * (data['grade'] / 100)) + 3.5
        
        vo2_abs = vo2_rel * data['weight'] / 1000
        vo2_median = get_ref_median(RAPP_VO2_MEDIANS, data['sex'], data['age'])
        vo2_score = calculate_percentile_score(vo2_rel, vo2_median, RAPP_VO2_SD[data['sex']])

        # 2. Мощность LT2 (Fiedler)
        rel_power = data['power'] / data['weight']
        p_median = get_ref_median(FIEDLER_LT2_MEDIANS, data['sex'], data['age'])
        p_score = calculate_percentile_score(rel_power, p_median, FIEDLER_LT2_SD[data['sex']])

        # 3. Восстановление и Кислородный пульс
        hr_rec_drop = data['hr_peak'] - data['hr_rec']
        rec_score = min(100, (hr_rec_drop / 40) * 100) # 40 уд. падения = 100 баллов
        
        o2_pulse = (vo2_abs * 1000) / data['hr_peak']
        o2_score = min(100, (o2_pulse / 15) * 100) # 15 мл/уд = 100 баллов

        # 4. ИТОГОВЫЙ ИНТЕГРАЛЬНЫЙ ИНДЕКС
        perf_index = (vo2_score * CONFIG['WEIGHTS']['vo2'] + 
                      p_score * CONFIG['WEIGHTS']['power'] + 
                      rec_score * CONFIG['WEIGHTS']['recovery'] + 
                      o2_score * CONFIG['WEIGHTS']['o2_pulse'])

        # 5. Алерты безопасности (Löllgen) - ОТДЕЛЬНО
        alerts = []
        if data['sbp'] >= CONFIG['THRESHOLDS']['sbp_critical']:
            alerts.append(f"🛑 КРИТИЧЕСКОЕ АД: {data['sbp']} мм рт.ст. Немедленно прекратить тест!")
        elif data['sbp'] >= CONFIG['THRESHOLDS']['sbp_warning']:
            alerts.append(f"⚠️ ГИПЕРТОНИЧЕСКАЯ РЕАКЦИЯ: {data['sbp']} мм рт.ст. (Löllgen)")
        
        if hr_rec_drop < CONFIG['THRESHOLDS']['hr_rec_min']:
            alerts.append(f"⚠️ ПЛОХОЕ ВОССТАНОВЛЕНИЕ: падение лишь на {hr_rec_drop} уд/мин (норма > 12)")

        # 6. Зоны ЧСС (Карвонен)
        hr_max_pred = 208 - 0.7 * data['age']
        hr_reserve = hr_max_pred - data['rest_hr']
        hr_zones = {f"Z{i+1}": (round(hr_reserve * p1 + data['rest_hr']), round(hr_reserve * p2 + data['rest_hr'])) 
                    for i, (p1, p2) in enumerate([(0.5,0.6), (0.6,0.7), (0.7,0.8), (0.8,0.9), (0.9,1.0)])}

        return {
            'index': round(perf_index, 1),
            'vo2': round(vo2_rel, 1),
            'vo2_pct': round(vo2_score),
            'mets': round(vo2_rel / 3.5, 1),
            'hr_zones': hr_zones,
            'alerts': alerts,
            'rec_drop': hr_rec_drop,
            'o2_pulse': round(o2_pulse, 1)
        }

# ===================================================================
# ИНТЕРФЕЙС STREAMLIT
# ===================================================================
st.set_page_config(page_title="Performance Analyzer", layout="wide")
st.title("🧬 Анализ кардиореспираторной производительности")

with st.sidebar:
    st.header("📥 Ввод данных теста")
    sex = st.selectbox("Пол", ["Male", "Female"])
    age = st.number_input("Возраст (лет)", 10, 90, 35)
    weight = st.number_input("Вес (кг)", 30.0, 150.0, 75.0)
    test_type = st.radio("Тип эргометра", ["bike", "tm"], format_func=lambda x: "Вело" if x=="bike" else "Бег")
    power = st.number_input("Макс. нагрузка (Вт или км/ч)", 0.0, 600.0, 200.0)
    grade = st.number_input("Уклон % (только для бега)", 0.0, 25.0, 0.0)
    hr_peak = st.number_input("Пиковая ЧСС", 100, 220, 175)
    hr_rest = st.number_input("ЧСС покоя", 40, 110, 60)
    hr_rec = st.number_input("ЧСС через 1 мин отдыха", 40, 200, 140)
    sbp = st.number_input("САД на пике (мм рт.ст.)", 100, 260, 180)

if st.sidebar.button("🔬 РАССЧИТАТЬ", type="primary"):
    # Формируем данные для анализа
    test_data = {
        'sex': sex, 'age': age, 'weight': weight, 'type': test_type,
        'power': power, 'grade': grade, 'hr_peak': hr_peak,
        'rest_hr': hr_rest, 'hr_rec': hr_rec, 'sbp': sbp
    }
    
    res = StressTestAnalyzer.calculate(test_data)
    
    # Секция метрик
    c1, c2, c3 = st.columns(3)
    c1.metric("ИНДЕКС ПОДГОТОВКИ", f"{res['index']}/100", help="Общий балл на основе VO2, мощности и восстановления")
    c2.metric("VO2peak", f"{res['vo2']} мл/кг", f"{res['vo2_pct']}% нормы")
    c3.metric("METs", res['mets'])

    # Вывод алертов
    if res['alerts']:
        for a in res['alerts']: st.warning(a)
    else:
        st.success("✅ Критерии безопасности в норме (Löllgen)")

    # Вкладки
    t_zones, t_plots = st.tabs(["🎯 Тренировочные зоны", "📊 Визуализация"])
    
    with t_zones:
        st.subheader("Зоны интенсивности по Карвонену")
        z_cols = st.columns(5)
        for i, (name, val) in enumerate(res['hr_zones'].items()):
            z_cols[i].info(f"**{name}**\n\n{val[0]}–{val[1]} bpm")

    with t_plots:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # График 1: Состав индекса
        ax[0].bar(["Индекс"], [res['index']], color='#3498db')
        ax[0].set_ylim(0, 100)
        ax[0].set_title("Общая оценка (0-100)")
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # График 2: Сердце
        ax[1].bar(["Восст.", "O2-Пульс"], [res['rec_drop'], res['o2_pulse']], color='#2ecc71')
        ax[1].set_title("Гемодинамические показатели")
        ax[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
