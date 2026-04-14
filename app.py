import streamlit as st
import matplotlib.pyplot as plt

# ===================================================================
# КОНФИГУРАЦИЯ И ДАННЫЕ ИЗ СТАТЬИ LÖLLGEN 2018
# ===================================================================
CONFIG = {
    'WEIGHTS': {'power': 0.35, 'hr': 0.25, 'lactate': 0.20, 'rpe': 0.10, 'recovery': 0.10},
    'THRESHOLDS': {'sbp_crit': 250, 'sbp_warn': 220, 'hr_rec_min': 12, 'rpe_high': 17},
    'VALIDATION': {'age': (10, 90), 'weight': (30, 200), 'hr': (40, 220)}
}

# Таблицы Löllgen (bike, 4-минутные ступени, 200 Вт) — eTable 3a и 4a
LOLLGEN_200W = {
    'groups': [160, 200, 240, 280, 320],  # max performance groups
    'hr': {  # p05 / p95
        160: (163, 209), 200: (154, 182), 240: (150, 182),
        280: (138, 160), 320: (138, 160)
    },
    'lactate': {  # p05 / p95
        160: (5.7, 18.2), 200: (4.6, 15.8), 240: (3.3, 11.4),
        280: (3.0, 5.4), 320: (3.0, 5.4)
    }
}

# Нормы максимальной мощности (eTable 2) — упрощённо для примера
# Можно расширить полностью, если нужно

class StressTestAnalyzer:
    @staticmethod
    def percentile_score(value, p05, p95):
        """Нормализация 0-100 по процентилям Löllgen (чем ниже — тем лучше)"""
        if value <= p05:
            return 100
        if value >= p95:
            return 0
        return 100 * (p95 - value) / (p95 - p05)

    @staticmethod
    def calculate(data):
        sex = data['sex']
        age = int(data['age'])
        weight = float(data['weight'])
        test_type = data['type']
        power = float(data['power'])          # max или субмакс (200 Вт)
        hr_peak = float(data['hr'])
        sbp = float(data['sbp'])
        hr_rest = float(data.get('rest_hr', 60))
        hr_rec = float(data.get('hr_rec', 0)) or None
        rpe = int(data['rpe'])
        submax_load = 200 if test_type == 'bike' else 13  # км/ч для тредмила

        # 1. Power Score (из eTable 2 — упрощённо)
        # Здесь можно добавить полную таблицу eTable 2
        power_score = min(power / (3.5 * weight) * 100, 100)  # относительная мощность

        # 2. HR и Lactate Score на субмакс. нагрузке (Löllgen eTable 3/4)
        # Предполагаем, что пользователь ввёл значения именно на 200 Вт
        hr_p05, hr_p95 = LOLLGEN_200W['hr'].get(240, (150, 182))  # default 240W group
        lac_p05, lac_p95 = LOLLGEN_200W['lactate'].get(240, (3.3, 11.4))

        hr_score = StressTestAnalyzer.percentile_score(hr_peak, hr_p05, hr_p95)
        lactate_score = StressTestAnalyzer.percentile_score(data.get('lactate', 6.0), lac_p05, lac_p95)

        # 3. Recovery
        rec_idx = (hr_peak - hr_rec) if hr_rec else 0
        recovery_score = min(rec_idx / 40 * 100, 100) if rec_idx else 50

        # 4. RPE
        rpe_score = max(100 - (rpe - 6) * (100 / 14), 0)

        # 5. Итоговый CEPS
        ceps = (
            CONFIG['WEIGHTS']['power'] * power_score +
            CONFIG['WEIGHTS']['hr'] * hr_score +
            CONFIG['WEIGHTS']['lactate'] * lactate_score +
            CONFIG['WEIGHTS']['rpe'] * rpe_score +
            CONFIG['WEIGHTS']['recovery'] * recovery_score
        )

        # Уровень
        fitness_level = 'low' if ceps < 70 else 'medium' if ceps < 85 else 'high'

        # Зоны Карвонена
        hr_max_pred = 208 - 0.7 * age
        hr_reserve = hr_max_pred - hr_rest
        zones = {
            'Z1': (round(hr_rest + 0.5 * hr_reserve), round(hr_rest + 0.6 * hr_reserve)),
            'Z2': (round(hr_rest + 0.6 * hr_reserve), round(hr_rest + 0.7 * hr_reserve)),
            'Z3': (round(hr_rest + 0.7 * hr_reserve), round(hr_rest + 0.8 * hr_reserve)),
            'Z4': (round(hr_rest + 0.8 * hr_reserve), round(hr_rest + 0.9 * hr_reserve)),
            'Z5': (round(hr_rest + 0.9 * hr_reserve), int(hr_max_pred))
        }

        # Алёрты
        alerts = []
        if sbp >= CONFIG['THRESHOLDS']['sbp_crit']:
            alerts.append(f"🛑 КРИТИЧЕСКОЕ АД {sbp} мм рт. ст. — тест должен быть прекращён (Löllgen Box 2)")
        elif sbp >= CONFIG['THRESHOLDS']['sbp_warn']:
            alerts.append(f"⚠️ Высокое АД {sbp} мм рт. ст.")
        if hr_rec and hr_rec < CONFIG['THRESHOLDS']['hr_rec_min']:
            alerts.append("⚠️ Медленное восстановление ЧСС (<12 уд/мин) — риск переутомления")
        if rpe > CONFIG['THRESHOLDS']['rpe_high'] and ceps < 80:
            alerts.append("⚠️ Высокое RPE при среднем CEPS — возможна дезадаптация")

        return {
            'ceps': round(ceps, 1),
            'fitness_level': fitness_level,
            'power_score': round(power_score, 1),
            'hr_score': round(hr_score, 1),
            'lactate_score': round(lactate_score, 1),
            'rpe_score': round(rpe_score, 1),
            'recovery_score': round(recovery_score, 1),
            'zones': zones,
            'alerts': alerts,
            'hr_max_pred': round(hr_max_pred),
            **data
        }

# ===================================================================
# STREAMLIT ИНТЕРФЕЙС (оставлен почти без изменений)
# ===================================================================
st.set_page_config(page_title="CEPS Analyzer — Löllgen 2018", layout="wide")
st.title("🧬 CEPS Analyzer")
st.caption("Единая шкала работоспособности 0–100 по данным Löllgen & Leyk 2018")

tab1, tab2, tab3 = st.tabs(["📝 Анализ", "📊 Графики", "📖 Справочник"])

with tab1:
    with st.form("main_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            sex = st.radio("Пол", ["Male", "Female"], horizontal=True)
            test_type = st.radio("Тест", ["bike", "tm"], horizontal=True,
                                 format_func=lambda x: "Велоэргометр (Вт)" if x == "bike" else "Тредмил (км/ч)")
            age = st.number_input("Возраст", 18, 65, 35)
            weight = st.number_input("Вес (кг)", 40.0, 150.0, 75.0)
            rest_hr = st.number_input("ЧСС покоя", 40, 120, 60)
        with col_b:
            power = st.number_input("Нагрузка (200 Вт или 13 км/ч)", 50.0, 400.0, 200.0)
            hr_peak = st.number_input("ЧСС на этой нагрузке", 80, 220, 165)
            lactate = st.number_input("Лактат (ммоль/л) на этой нагрузке", 0.5, 15.0, 5.0)
            sbp = st.number_input("САД пик", 100, 260, 180)
            hr_rec = st.number_input("ЧСС через 1 мин отдыха", 40, 200, 145)
            rpe = st.slider("Borg RPE", 6, 20, 14)

        if st.form_submit_button("🚀 Рассчитать CEPS", type="primary"):
            raw_data = {
                'sex': sex, 'type': test_type, 'age': age, 'weight': weight,
                'power': power, 'hr': hr_peak, 'lactate': lactate,
                'sbp': sbp, 'rest_hr': rest_hr, 'hr_rec': hr_rec, 'rpe': rpe
            }
            res = StressTestAnalyzer.calculate(raw_data)

            for alert in res['alerts']:
                st.warning(alert)
            if not res['alerts']:
                st.success("✅ Всё в норме")

            st.subheader("📈 Результат")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CEPS", f"{res['ceps']}/100", res['fitness_level'].upper())
            col2.metric("Power Score", f"{res['power_score']}")
            col3.metric("HR Score", f"{res['hr_score']}")
            col4.metric("Lactate Score", f"{res['lactate_score']}")

            st.session_state['last_res'] = res

with tab2:
    if 'last_res' in st.session_state:
        res = st.session_state['last_res']
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].bar(['Power', 'HR', 'Lactate', 'RPE', 'Recovery'], 
                   [res['power_score'], res['hr_score'], res['lactate_score'], 
                    res['rpe_score'], res['recovery_score']], color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6'])
        axs[0].set_title("Вклад компонентов в CEPS")
        axs[0].set_ylim(0, 100)

        z_labels = list(res['zones'].keys())
        for i, z in enumerate(z_labels):
            low, high = res['zones'][z]
            axs[1].barh(z, high - low, left=low, color=['#a1c4fd', '#7ed957', '#f1c40f', '#e67e22', '#e74c3c'][i])
            axs[1].text(low + (high - low) / 2, i, f"{low}-{high}", ha='center', va='center', weight='bold')
        axs[1].set_title("Зоны ЧСС (Карвонена)")
        st.pyplot(fig)

with tab3:
    st.markdown("**CEPS** — Composite Ergometric Performance Score по данным Löllgen 2018\n\n"
                "• 0–70 — низкий уровень\n"
                "• 70–85 — средний\n"
                "• 85–100 — высокий / элитный")
