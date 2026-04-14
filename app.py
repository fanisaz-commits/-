import streamlit as st
import matplotlib.pyplot as plt
import datetime

# ===================================================================
# КОНФИГУРАЦИЯ И СПРАВОЧНЫЕ ДАННЫЕ
# ===================================================================
CONFIG = {
    'WEIGHTS': {'vo2': 0.4, 'lt2': 0.3, 'hr_rec': 0.2, 'o2_pulse': 0.1},
    'THRESHOLDS': {
        'perf_low': 70, 'perf_med': 90, 'hr_rec_min': 12,
        'rpe_high': 17, 'sbp_warn': 220, 'sbp_crit': 250
    },
    'VALIDATION': {
        'age': (10, 90), 'weight': (30, 200), 'hr': (40, 220),
        'sbp': (90, 260), 'rpe': (6, 20)
    }
}

RAPP_VO2_REF = {
    'Male': {(18, 29): 46.5, (30, 39): 42.0, (40, 49): 38.5, (50, 59): 34.0, (60, 69): 30.0},
    'Female': {(18, 29): 37.5, (30, 39): 33.5, (40, 49): 29.5, (50, 59): 26.0, (60, 69): 22.5}
}

FIEDLER_LT2_REF = {
    'Male': {(14, 24): 2.4, (25, 34): 2.3, (35, 44): 2.1, (45, 54): 1.9, (55, 64): 1.7},
    'Female': {(14, 24): 1.9, (25, 34): 1.8, (35, 44): 1.7, (45, 54): 1.5, (55, 64): 1.4}
}

# ===================================================================
# ЛОГИЧЕСКОЕ ЯДРО
# ===================================================================
class StressTestAnalyzer:
    @staticmethod
    def get_ref_value(data_dict, sex, age):
        gender_data = data_dict.get(sex, {})
        for (age_min, age_max), val in gender_data.items():
            if age_min <= age <= age_max:
                return val
        return list(gender_data.values())[0] if age < 18 else list(gender_data.values())[-1]

    @staticmethod
    def calculate(data):
        sex, age, weight = data['sex'], int(data['age']), float(data['weight'])
        power, hr_peak, sbp = float(data['power']), float(data['hr']), float(data['sbp'])
        hr_rest = float(data['rest_hr']) if data.get('rest_hr') else 60
        rpe, grade, test_type = int(data['rpe']), float(data.get('grade', 0)), data['type']
        hr_rec_1min = float(data['hr_rec']) if data.get('hr_rec') else None

        hr_max_pred = 208 - 0.7 * age
        hr_reserve = hr_max_pred - hr_rest

        # 2. VO2peak (Исправленная профессиональная формула)
        if test_type == 'bike':
            # 12 мл/Вт + 3.5 мл/кг (базовый метаболизм)
            vo2_abs_ml = (power * 12) + (3.5 * weight)
            vo2_rel = vo2_abs_ml / weight
        else:
            speed_m_min = (power * 1000) / 60
            vo2_rel = (0.1 * speed_m_min) + (1.8 * speed_m_min * (grade / 100)) + 3.5

        vo2_abs = vo2_rel * weight / 1000
        vo2_norm = StressTestAnalyzer.get_ref_value(RAPP_VO2_REF, sex, age)
        vo2_pct = (vo2_rel / vo2_norm) * 100

        rel_power = power / weight
        lt2_norm = StressTestAnalyzer.get_ref_value(FIEDLER_LT2_REF, sex, age)
        lt2_pct = (rel_power / lt2_norm) * 100

        o2_pulse = (vo2_abs * 1000) / hr_peak
        rec_idx = (hr_peak - hr_rec_1min) if hr_rec_1min else None

        def norm_comp(val): return min(val, 130) / 130 * 100
        comp_vo2, comp_lt2 = norm_comp(vo2_pct), norm_comp(lt2_pct)
        comp_rec = (min(rec_idx, 40) / 40 * 100) if rec_idx else 50
        comp_o2 = min(o2_pulse * 5, 100)

        perf_idx = (CONFIG['WEIGHTS']['vo2'] * comp_vo2 + CONFIG['WEIGHTS']['lt2'] * comp_lt2 +
                    CONFIG['WEIGHTS']['hr_rec'] * comp_rec + CONFIG['WEIGHTS']['o2_pulse'] * comp_o2)

        fit_level = 'low' if perf_idx < 70 else 'medium' if perf_idx < 90 else 'high'
        def karvonen(pct): return round((hr_reserve * pct) + hr_rest)
        
        zones = {
            'Z1': (karvonen(0.50), karvonen(0.60)), 'Z2': (karvonen(0.60), karvonen(0.70)),
            'Z3': (karvonen(0.70), karvonen(0.80)), 'Z4': (karvonen(0.80), karvonen(0.90)),
            'Z5': (karvonen(0.90), int(hr_max_pred))
        }

        alerts = []
        if sbp >= CONFIG['THRESHOLDS']['sbp_crit']: 
            alerts.append(f"🛑 КРИТИЧЕСКОЕ АД: {sbp} достигло порога прекращения теста (Löllgen).")
        elif sbp >= CONFIG['THRESHOLDS']['sbp_warn']:
            alerts.append(f"⚠️ РИСК АД: Систолическое давление {sbp} выше нормы.")
        
        if rec_idx is not None and rec_idx < CONFIG['THRESHOLDS']['hr_rec_min']: 
            alerts.append("⚠️ ВОССТАНОВЛЕНИЕ: ЧСС снижается медленно (<12 уд/мин). Риск переутомления.")
        
        if rpe > CONFIG['THRESHOLDS']['rpe_high'] and vo2_pct < 85:
            alerts.append("⚠️ ДЕЗАДАПТАЦИЯ: Высокое RPE при невысоких показателях.")

        return {**data, 'hr_max_pred': round(hr_max_pred), 'vo2_rel': round(vo2_rel, 1), 'vo2_pct': round(vo2_pct, 1),
                'vo2_norm': vo2_norm, 'rel_power': round(rel_power, 2), 'lt2_pct': round(lt2_pct, 1), 
                'o2_pulse': round(o2_pulse, 1), 'rec_idx': rec_idx, 'performance_index': round(perf_idx, 1), 
                'fitness_level': fit_level, 'zones': zones, 'alerts': alerts, 'sbp_val': sbp, 'rpe_val': rpe}

# ===================================================================
# ИНТЕРФЕЙС STREAMLIT
# ===================================================================
st.set_page_config(page_title="Expert Physiology Analyzer", layout="wide")

st.title("🧬 Анализатор нагрузочного тестирования")
st.caption("Expert Physiology Edition | Formulas: Löllgen, Rapp, Fiedler")

tab1, tab2, tab3 = st.tabs(["📝 Ввод и Отчет", "📊 Графики", "📖 Справочник"])

with tab1:
    with st.form("main_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            sex = st.radio("Пол:", ["Male", "Female"], horizontal=True)
            test_type = st.radio("Тип теста:", ["bike", "tm"], format_func=lambda x: "Велоэргометр (Вт)" if x=="bike" else "Тредмил (км/ч)", horizontal=True)
            age = st.number_input("Возраст (лет):", 10, 90, 35)
            weight = st.number_input("Вес (кг):", 30.0, 200.0, 75.0)
            rest_hr = st.number_input("ЧСС покоя:", 40, 120, 60)
        with col_b:
            power = st.number_input("Макс. нагрузка (Вт или км/ч):", 0.0, 1000.0, 250.0)
            grade = st.number_input("Уклон (% для тредмила):", 0.0, 25.0, 0.0)
            hr_peak = st.number_input("ЧСС пик (уд/мин):", 50, 220, 180)
            sbp = st.number_input("САД пик (mmHg):", 80, 260, 170)
            hr_rec = st.number_input("ЧСС 1 мин отдыха:", 40, 200, 140)
            rpe = st.slider("Borg RPE (6-20):", 6, 20, 15)
        
        submit = st.form_submit_button("🚀 ЗАПУСТИТЬ АНАЛИЗ", type="primary")

    if submit:
        # Валидация типов для тредмила
        if test_type == 'tm' and power > 30:
            st.error("Ошибка: Для тредмила введите скорость в км/ч (напр. 12), а не Ватты.")
        else:
            raw_data = {'sex': sex, 'type': test_type, 'age': age, 'weight': weight, 'rest_hr': rest_hr,
                        'power': power, 'grade': grade, 'hr': hr_peak, 'sbp': sbp, 'hr_rec': hr_rec, 'rpe': rpe}
            res = StressTestAnalyzer.calculate(raw_data)

            for alert in res['alerts']:
                st.warning(alert)
            if not res['alerts']:
                st.success("✅ Противопоказаний по результатам теста не выявлено.")

            st.subheader("📊 Ключевые показатели")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("VO2peak", f"{res['vo2_rel']} мл/кг", f"{res['vo2_pct']}% нормы")
            m2.metric("Perf. Index", f"{res['performance_index']}/100", res['fitness_level'].upper())
            m3.metric("O2 Pulse", f"{res['o2_pulse']} мл/уд")
            m4.metric("Восстановление", f"-{res['rec_idx'] if res['rec_idx'] else 'N/A'} уд")

            st.subheader("🏃 Тренировочный план")
            if res['sbp_val'] >= 220 or res['rpe_val'] > 18:
                st.error("⚠️ РЕЖИМ ОГРАНИЧЕНИЯ: Требуется консультация кардиолога.")
            else:
                plans = {
                    'low': "📉 БАЗОВЫЙ ПЛАН: Аэробные нагрузки в Z2 (30-45 мин) 3 раза в неделю.",
                    'medium': "📈 РАЗВИВАЮЩИЙ ПЛАН: 2 базы (Z2) + 1 темповая тренировка (Z3/Z4).",
                    'high': "🏆 ПРОФЕССИОНАЛЬНЫЙ ПЛАН: Поляризованный тренинг (80% Z2, 20% Z5)."
                }
                st.info(plans[res['fitness_level']])
            
            # Сохранение результата для графика
            st.session_state['last_res'] = res

with tab2:
    if 'last_res' in st.session_state:
        res = st.session_state['last_res']
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # График 1
        axs[0].bar(['Ваш VO2', 'Норма'], [res['vo2_rel'], res['vo2_norm']], color=['#3498db', '#2ecc71'])
        axs[0].set_title("Сравнение VO2peak (мл/кг/мин)")

        # График 2
        z_labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
        colors = ['#a1c4fd', '#7ed957', '#f1c40f', '#e67e22', '#e74c3c']
        for i, z in enumerate(z_labels):
            low, high = res['zones'][z]
            axs[1].barh(z, high-low, left=low, color=colors[i], edgecolor='black')
            axs[1].text(low+(high-low)/2, i, f"{low}-{high}", ha='center', va='center', weight='bold')
        axs[1].set_title("Зоны ЧСС (Карвонена)")
        st.pyplot(fig)
    else:
        st.info("Запустите анализ, чтобы увидеть графики.")

with tab3:
    st.markdown("""
    ### 📖 Справочное пособие
    1. **VO2peak**: Рассчитан по формуле: (W*12 + kg*3.5)/kg.
    2. **ЧСС Recovery**: Снижение за 1 мин. Если < 12 уд/мин — маркер риска ССЗ (Löllgen).
    3. **САД (Давление)**: Порог 250 mmHg — абсолютное прекращение теста.
    4. **Нормативы**: Rapp (2018), Fiedler (2025).
    """)
