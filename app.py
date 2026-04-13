import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import datetime

# ===================================================================
# КОНФИГУРАЦИЯ И НАУЧНО ОБОСНОВАННЫЕ РЕФЕРЕНСЫ
# ===================================================================
CONFIG = {
    'THRESHOLDS': {
        'sbp_critical': 250,          # Löllgen: абсолютное прекращение
        'sbp_warning': 220,            # Löllgen: чрезмерный подъём
        'hr_rec_min': 12,              # Löllgen: снижение ЧСС за 1 мин <12 — плохой прогноз
        'chronotropic_ratio': 0.85,    # Löllgen: <85% возрастной ЧСС — хронотропная недостаточность
        'vo2_risk_low': 35,            # ~10 MET
        'vo2_risk_mod': 25,            # ~7 MET (высокий риск ниже этого порога)
    },
    'WEIGHTS': {
        'vo2': 0.4, 'power': 0.3, 'recovery': 0.2, 'o2_pulse': 0.1
    }
}

# Референсы Rapp et al. 2018 и Fiedler et al. 2025 (сокращено для примера)
RAPP_VO2_MEDIANS = {
    'Male': {(18,29): 42.0, (30,39): 38.0, (40,49): 36.0, (50,59): 32.0, (60,69): 28.0},
    'Female': {(18,29): 34.0, (30,39): 31.0, (40,49): 30.0, (50,59): 26.0, (60,69): 23.0}
}
RAPP_VO2_SD = {'Male': 7.7, 'Female': 6.7}

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

def calculate_score(value, median, sd, reverse=False):
    """Рассчитывает балл от 0 до 100 на основе Z-оценки."""
    z = (value - median) / sd
    score = norm.cdf(z) * 100
    return round(score, 1)

# ===================================================================
# АНАЛИЗАТОР
# ===================================================================
class StressTestAnalyzer:
    @staticmethod
    def calculate(data):
        # 1. Базовые расчеты
        hr_max_pred = 208 - 0.7 * data['age']
        hr_reserve = hr_max_pred - data['rest_hr']
        
        # 2. VO2peak (ACSM)
        if data['type'] == 'bike':
            vo2_rel = (10.8 * data['power'] / data['weight']) + 7.0
        else:
            speed_m_min = (data['power'] * 1000) / 60
            vo2_rel = (0.2 * speed_m_min) + (0.9 * speed_m_min * (data['grade'] / 100)) + 3.5
        
        vo2_abs = vo2_rel * data['weight'] / 1000
        vo2_median = get_ref_median(RAPP_VO2_MEDIANS, data['sex'], data['age'])
        vo2_score = calculate_score(vo2_rel, vo2_median, RAPP_VO2_SD[data['sex']])

        # 3. Мощность (Fiedler)
        rel_power = data['power'] / data['weight']
        p_median = get_ref_median(FIEDLER_LT2_MEDIANS, data['sex'], data['age'])
        p_score = calculate_score(rel_power, p_median, FIEDLER_LT2_SD[data['sex']])

        # 4. Гемодинамика
        o2_pulse = (vo2_abs * 1000) / data['hr_peak']
        hr_rec_drop = data['hr_peak'] - data['hr_rec']
        rec_score = min(100, (hr_rec_drop / 40) * 100)
        o2_score = min(100, (o2_pulse / 15) * 100) # Условная норма 15 мл/уд

        # 5. ИТОГОВЫЙ ИНДЕКС (Взвешенный)
        perf_index = (vo2_score * CONFIG['WEIGHTS']['vo2'] + 
                      p_score * CONFIG['WEIGHTS']['power'] + 
                      rec_score * CONFIG['WEIGHTS']['recovery'] + 
                      o2_score * CONFIG['WEIGHTS']['o2_pulse'])

        # 6. Алерты (Löllgen)
        alerts = []
        if data['sbp'] >= CONFIG['THRESHOLDS']['sbp_critical']:
            alerts.append(f"🛑 КРИТИЧЕСКОЕ АД: {data['sbp']} мм рт.ст. Прекратить тест!")
        elif data['sbp'] >= CONFIG['THRESHOLDS']['sbp_warning']:
            alerts.append(f"⚠️ ГИПЕРТОНИЧЕСКАЯ РЕАКЦИЯ: {data['sbp']} мм рт.ст.")
        
        if hr_rec_drop < CONFIG['THRESHOLDS']['hr_rec_min']:
            alerts.append(f"⚠️ СЛАБОЕ ВОССТАНОВЛЕНИЕ: -{hr_rec_drop} уд/мин (норма >12)")

        # 7. Зоны Карвонена
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
# ИНТЕРФЕЙС
# ===================================================================
st.set_page_config(page_title="Анализатор Тестирования", layout="wide")
st.title("🧬 Система оценки нагрузочного тестирования")

with st.sidebar:
    st.header("📥 Ввод данных")
    sex = st.selectbox("Пол", ["Male", "Female"])
    age = st.number_input("Возраст", 10, 90, 35)
    weight = st.number_input("Вес (кг)", 30.0, 150.0, 75.0)
    test_type = st.radio("Тип теста", ["bike", "tm"])
    power = st.number_input("Нагрузка (Вт или км/ч)", 0.0, 500.0, 200.0)
    grade = st.number_input("Уклон % (для тредмила)", 0.0, 20.0, 0.0)
    hr_peak = st.number_input("Пиковая ЧСС", 100, 220, 170)
    hr_rest = st.number_input("ЧСС покоя", 40, 100, 60)
    hr_rec = st.number_input("ЧСС через 1 мин", 40, 200, 130)
    sbp = st.number_input("САД на пике", 100, 260, 180)

if st.sidebar.button("Рассчитать", type="primary"):
    res = StressTestAnalyzer.calculate(locals())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ИНДЕКС ПОДГОТОВКИ", f"{res['index']}/100")
    col2.metric("VO2peak", f"{res['vo2']} мл/кг", f"{res['vo2_pct']}% нормы")
    col3.metric("METs", res['mets'])

    if res['alerts']:
        for a in res['alerts']: st.warning(a)
    else:
        st.success("✅ Отклонений по безопасности не обнаружено")

    tab_zones, tab_charts = st.tabs(["🎯 Тренировочные зоны", "📊 Графики"])
    
    with tab_zones:
        st.subheader("Зоны ЧСС (по Карвонену)")
        cols = st.columns(5)
        for i, (name, range_val) in enumerate(res['hr_zones'].items()):
            cols[i].info(f"**{name}**\n\n{range_val[0]}-{range_val[1]} уд/мин")

    with tab_charts:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # Индекс
        ax[0].bar(["Индекс"], [res['index']], color='skyblue')
        ax[0].set_ylim(0, 100)
        ax[0].set_title("Общая оценка")
        # Гемодинамика
        ax[1].bar(["Восст.", "О2-пульс"], [res['rec_drop'], res['o2_pulse']], color='lightgreen')
        ax[1].set_title("Показатели сердца")
        st.pyplot(fig)
