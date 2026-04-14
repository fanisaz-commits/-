import streamlit as st
import matplotlib.pyplot as plt

# ===================================================================
# ДАННЫЕ ИЗ LÖLLGEN 2018 (eTable 3a/4a — bike, 200 Вт)
# ===================================================================
LOLLGEN_200W = {
    'hr': {160: (163, 209), 200: (154, 182), 240: (150, 182),
           280: (138, 160), 320: (138, 160)},
    'lactate': {160: (5.7, 18.2), 200: (4.6, 15.8), 240: (3.3, 11.4),
                280: (3.0, 5.4), 320: (3.0, 5.4)}
}

CONFIG = {
    'WEIGHTS': {'power': 0.35, 'hr': 0.25, 'lactate': 0.20, 'rpe': 0.10, 'recovery': 0.10},
    'THRESHOLDS': {'sbp_crit': 250, 'sbp_warn': 220, 'hr_rec_min': 12, 'rpe_high': 17},
}

class StressTestAnalyzer:
    @staticmethod
    def percentile_score(value, p05, p95):
        if value <= p05: return 100.0
        if value >= p95: return 0.0
        return 100 * (p95 - value) / (p95 - p05)

    @staticmethod
    def calculate(data):
        test_type = data['type']
        power = float(data['power'])          # введённая нагрузка
        hr_peak = float(data['hr'])
        lactate = float(data.get('lactate', 6.0))
        rpe = int(data['rpe'])
        hr_rec = float(data.get('hr_rec', 0)) or None
        age = int(data['age'])
        weight = float(data['weight'])
        rest_hr = float(data.get('rest_hr', 60))
        sbp = float(data['sbp'])

        # --- 1. Определяем режим расчёта ---
        use_lollgen = False
        if test_type == 'bike' and 150 <= power <= 250:
            use_lollgen = True
        elif test_type == 'tm' and 11 <= power <= 15:
            use_lollgen = True

        if use_lollgen:
            # Точный режим Löllgen (200 Вт)
            group = 240  # ближайшая референсная группа
            hr_p05, hr_p95 = LOLLGEN_200W['hr'][group]
            lac_p05, lac_p95 = LOLLGEN_200W['lactate'][group]

            hr_score = StressTestAnalyzer.percentile_score(hr_peak, hr_p05, hr_p95)
            lactate_score = StressTestAnalyzer.percentile_score(lactate, lac_p05, lac_p95)
            power_score = min(power / (3.5 * weight) * 100, 100)   # относительная мощность
        else:
            # Обобщённый режим (как в твоём оригинальном коде)
            st.warning(f"⚠️ Нагрузка {power} не в диапазоне Löllgen (150-250 Вт). Используем обобщённый расчёт.")
            power_score = min(power / (3.5 * weight) * 100, 100)
            hr_score = max(100 - (hr_peak - 120) * 0.8, 0)          # упрощённо
            lactate_score = max(100 - (lactate - 4) * 8, 0)

        # Recovery
        rec_idx = (hr_peak - hr_rec) if hr_rec else 0
        recovery_score = min(rec_idx / 40 * 100, 100) if rec_idx else 50

        # RPE
        rpe_score = max(100 - (rpe - 6) * (100 / 14), 0)

        # CEPS
        ceps = (
            CONFIG['WEIGHTS']['power'] * power_score +
            CONFIG['WEIGHTS']['hr'] * hr_score +
            CONFIG['WEIGHTS']['lactate'] * lactate_score +
            CONFIG['WEIGHTS']['rpe'] * rpe_score +
            CONFIG['WEIGHTS']['recovery'] * recovery_score
        )

        fitness_level = 'low' if ceps < 70 else 'medium' if ceps < 85 else 'high'

        # Зоны Карвонена
        hr_max_pred = 208 - 0.7 * age
        hr_reserve = hr_max_pred - rest_hr
        zones = {f'Z{i}': (round(hr_rest + low*hr_reserve), round(hr_rest + high*hr_reserve))
                 for i, (low, high) in enumerate([(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)], 1)}

        # Алёрты
        alerts = []
        if sbp >= CONFIG['THRESHOLDS']['sbp_crit']:
            alerts.append(f"🛑 КРИТИЧЕСКОЕ АД {sbp} мм рт. ст.")
        if hr_rec and hr_rec < CONFIG['THRESHOLDS']['hr_rec_min']:
            alerts.append("⚠️ Медленное восстановление ЧСС")
        if rpe > CONFIG['THRESHOLDS']['rpe_high'] and ceps < 80:
            alerts.append("⚠️ Высокое RPE при среднем CEPS")

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
            'mode': 'Löllgen точный' if use_lollgen else 'Обобщённый'
        }

# ====================== STREAMLIT ======================
st.set_page_config(page_title="CEPS Analyzer v2.1", layout="wide")
st.title("🧬 CEPS Analyzer v2.1")
st.caption("Гибридный режим: 150–250 Вт — точные таблицы Löllgen | любой диапазон — обобщённый расчёт")

# ... (весь твой интерфейс tab1, tab2, tab3 остаётся почти без изменений)

# В форме добавь поле для лактата (если ещё нет)
# lactate = st.number_input("Лактат (ммоль/л)", 0.5, 15.0, 5.0)
